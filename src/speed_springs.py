from sympy import *
from sympy.physics.mechanics import *
# from symengine import *

import numpy as np

import torch
import torch.nn as nn

from itertools import product


class GGS3DE(nn.Module):
    def __init__(
        self, n_pieces, u_i, y_i, friction=0, temp=0, k=1, M=1, kb=1.38064852e-23, k2=0
    ):
        """
        n_pices (int, NDArray): If int, all directions have same number of pieces. If NDArray, each direction has a different number of pieces. If NDArray, it must have the same length as the dimensions of u_i.

        If n_pieces is [1,1,1,1] it uses one stick per dimension.
        """
        super().__init__()
        self.n_pieces = torch.tensor(n_pieces, dtype=int) if isinstance(n_pieces, (list, tuple)) else torch.ones(u_i.shape[0], dtype=int)*n_pieces
        assert self.n_pieces.shape[0] == u_i.shape[0], "n_pieces and u_i dimensions mismatch"

        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.u_min = u_i.min(axis=1).values
        self.u_max = u_i.max(axis=1).values
        self.ell = ((self.u_max - self.u_min) / n_pieces)

        self.y_i = y_i
        self.u_i = u_i

        self.k = k
        self.k2 = k2
        self.M = M / torch.prod((self.n_pieces))
        self.kb = kb

        self.friction = friction
        self.temp = temp
        # self.eta_cte = np.sqrt(2 * friction * temp * kb / (M * np.prod(n_pieces)))
        self.eta_cte = float(np.sqrt(2 * friction * temp * kb / M))

        self._init_symbols()
        self._init_kinetic_energy()
        self._init_potential_energy()
        self._init_lagrangian()

    def _init_symbols(self):
        shape = tuple([int(n) for n in np.append(self.n_pieces, self.y_i.shape[0])])

        self.symbols_shape = shape
        self.N = np.prod(shape)

        self.x_symbols = np.empty(
            np.array([n for n in np.append(self.n_pieces + 1, self.y_i.shape[0])]), dtype=object
        )
        self.dx_symbols = np.empty_like(self.x_symbols, dtype=object)
        self.ddx_symbols = np.empty_like(self.x_symbols, dtype=object)

        self.N = np.prod(self.x_symbols.shape)

        # Iterate over all indices and assign symbols
        print("Creating symbols...")
        for index in np.ndindex(self.x_symbols.shape):
            # Create symbol and assign it to the corresponding index in the array
            self.x_symbols[index] = dynamicsymbols(f"x_{''.join(map(str, index))}")
            self.dx_symbols[index] = self.x_symbols[index].diff()
            self.ddx_symbols[index] = self.dx_symbols[index].diff()

    def _init_kinetic_energy(self):
        print("Calculating kinetic energy...")
        print(f"Iterating over {len(self.symbols_shape)} dimensions")
        ktr = 0
        for i in range(len(self.symbols_shape)-1):
            print(f"Calculating kinetic energy for dimension {i}...")
            slice_front = [slice(None)] * len(self.symbols_shape)
            slice_back = [slice(None)] * len(self.symbols_shape)
            slice_front[i] = slice(1, None)
            slice_back[i] = slice(None, -1)
            difference = (self.dx_symbols[tuple(slice_front)] + self.dx_symbols[tuple(slice_back)]) ** 2
            difference = difference.ravel()
            ktr = Add(ktr, (self.M.item() / 8) * Add(*difference))

        self.ktr = ktr

        # Kinetic energy (rotational)
        krot = 0
        print("Calculating rotational kinetic energy...")
        for i in range(len(self.symbols_shape)-1):
            print(f"Calculating rotational kinetic energy for dimension {i}...")
            slice_front = [slice(None)] * len(self.symbols_shape)
            slice_back = [slice(None)] * len(self.symbols_shape)
            slice_front[i] = slice(1, None)
            slice_back[i] = slice(None, -1)
            difference = (self.dx_symbols[tuple(slice_front)] - self.dx_symbols[tuple(slice_back)]) ** 2
            difference = difference.ravel()
            krot = Add(krot, (self.M.item()/ 24) * Add(*difference))

        self.krot = krot

    def _init_potential_energy(self):
        print("Calculating potential energy...")
        # Elastic energy (we don't want xi - xi+1 to be so far from ell)
        uelastic = 0

        if self.k2 != 0:
            for i in range(len(self.x_symbols.shape)-1):
                slice_front = [slice(None)] * len(self.x_symbols.shape)
                slice_back = [slice(None)] * len(self.x_symbols.shape)
                slice_front[i] = slice(1, None)
                slice_back[i] = slice(None, -1)

                difference = (self.x_symbols[tuple(slice_front)] - self.x_symbols[tuple(slice_back)] - self.ell[i]) ** 2
                difference = difference.ravel()
                uelastic = Add(uelastic, self.k2 / 2 * Add(*difference))

        self.uelastic_symbols = uelastic

        # Potential energy (cost function)
        U = 0
        for j in range(self.u_i.shape[1]):
            print(f"Calculating potential energy for point {j}...")
            i = self.find_box(self.u_i[:, j])
            point_pred = self.y_prediction(self.u_i[:, j], i)
            difference = (point_pred - self.y_i[:, j].detach().cpu().numpy()) ** 2
            difference = difference.ravel()
            U = Add(U, (self.k/ 2 * Add(*difference)))
        
        self.U = U

    def _init_lagrangian(self):
        print("Calculating Lagrangian...")
        # Lagrangian
        self.lagrangian = Add(self.ktr, self.krot, -self.U, -self.uelastic_symbols)
        self.LM = LagrangesMethod(self.lagrangian, self.x_symbols.flatten())
        self.LM.form_lagranges_equations()
   
        self.mass_matrix = self.LM.mass_matrix
        self.forcing_vector = self.LM.forcing
        try:
            self.evol_dynamics = self.mass_matrix.inv() @ self.forcing_vector
        except Exception as e:
            print(e)
            self.evol_dynamics = self.mass_matrix.pinv() * self.forcing_vector

        self.lambdified_dyn = lambdify(
            [*self.x_symbols.flatten(), *self.dx_symbols.flatten()], self.evol_dynamics - self.friction * self.dx_symbols.reshape(-1,1) / self.N 
        )

        self.ypred = lambda u: lambdify(
            [*self.x_symbols.flatten()], self.y_prediction(u)
        )

        self.ue = lambdify(
            [*self.x_symbols.flatten()], self.U
        )

        print("Done initializing.")

    def lamd(self, i, u, flip_ind=None, div_by_ell=True):
        """Returns the lambda function for the i-th dimension."""
        val = u - i * self.ell - self.u_min
        if div_by_ell:
            val /= self.ell
        if flip_ind is not None:
            val[flip_ind] *= -1

        # set vals less than 1e-8 to 0
        val[np.abs(val) < 1e-8] = 0
        return val
    
    def find_box(self, u):
        """Returns the box where the input u is located."""
        i = ((u - self.u_min) / self.ell).to(torch.int)
        return torch.clip(i, torch.tensor(0), self.n_pieces - 1)

    def y_prediction(self, u, i=None):
        return self._y_prediction(u, i)
    
    def _y_prediction(self, u, i=None):
        if i is None:
            i = self.find_box(u) 
        
        ld = self.lamd(i, u) 

        dimension = len(i)
        offsets = torch.tensor(list(product([0, 1], repeat=dimension))) 

        grid_points = i + offsets 

        weights = torch.prod((1 - ld) * (offsets == 0) + ld * (offsets != 0), axis=1)

        assert torch.sum(weights) - 1 < 1e-5, "Weights do not sum to 1"

        lin_combinations = np.array([weights[j].item() * self.x_symbols[tuple(grid_points[j])] for j in range(len(weights))])
        pred = np.array([Add(*lin_combinations[:,j]) for j in range(lin_combinations.shape[1])])

        return pred

    def num_y_prediction(self, u, y):
        q = y[: self.N].T
        ypred = self.ypred(u)
        return ypred(*q)

    def f(self, t, y):
        """y is a tensor of shape (batch_size, state_dim)

        Returns a tensor of shape (batch_size, state_dim)
        """
        q = y[:, : self.N].T
        dq = y[:, self.N :].T


        ddqdt = torch.tensor(self.lambdified_dyn(*q, *dq))
        ddqdt_shape = ddqdt.shape
        ddqdt = ddqdt.squeeze()
        if ddqdt_shape[-1] == 1:
            ddqdt = ddqdt.unsqueeze(-1)

        step = torch.vstack([dq, ddqdt])

        return step.T

    def g(self, t, y):
        return self.eta_cte * torch.ones_like(y)

    def cost(self, y):
        q = y[:, : self.N].T
        return self.ue(*q)