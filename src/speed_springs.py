from sympy import *
from sympy.physics.mechanics import *
# from symengine import *

import numpy as np

import torch
import torch.nn as nn

from src.utils import verbose_display
from itertools import product

from concurrent.futures import ThreadPoolExecutor, as_completed


class GroupGS3DE(nn.Module):
    """Grouped GS3DE models. It breaks the learning into M groups of GG3DE models and outputs the average of the predictions."""
    def __init__(
        self, max_features, n_sticks, u_i, y_i, friction=0, temp=0, k=1, M=1, kb=1.38064852e-23, k2=0, boundaries=None, verbose=False, group_strategy="sequential", parallel=False
    ):
        """
        max_features (int or float): If int, it should be greater than 1. If float, it should be between 0 and 1 and will represent the fraction of the total features that will be used in each group.

        group_strategy (str): Strategy to group the features. Can be "sequential" or "random".
        """
        assert group_strategy in ["sequential", "random"], "group_strategy must be 'sequential' or 'random'"

        match max_features:
            case int():
                assert max_features >= 1, "max_features must be greater than or equal to 1"
            case float():
                assert 0 < max_features < 1, "max_features must be between 0 and 1"
            case _:
                raise ValueError("max_features must be an int or a float")

        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"

        self.u_i = u_i
        self.y_i = y_i

        self.num_features = u_i.shape[0]
        self.num_labels = y_i.shape[0]

        self.max_features = max_features if isinstance(max_features, int) else int(max_features * self.num_features)
        self.group_strategy = group_strategy

        self.groups, self.groups_idx = self.split_features(u_i, strategy=group_strategy)

        models = []

        if parallel:
            def create_gs3de_model(idx, group):
                return GS3DE(n_sticks[self.groups_idx[idx]], group, y_i, friction, temp, k, M, kb, k2, boundaries=(boundaries[0][self.groups_idx[idx]],boundaries[1][self.groups_idx[idx]]))
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(create_gs3de_model, idx, group): idx for idx, group in enumerate(self.groups)}                
                for future in as_completed(futures):
                    model = future.result()
                    models.append(model)
        else:
            for idx, group in enumerate(self.groups):
                msg = f"Creating GS3DE for group {idx}/{len(self.groups)}..."
                with verbose_display(msg, verbose=verbose):
                    model = GS3DE(n_sticks[self.groups_idx[idx]], group, y_i, friction, temp, k, M, kb, k2, boundaries=(boundaries[0][self.groups_idx[idx]],boundaries[1][self.groups_idx[idx]]))
                models.append(model)

        self.models = nn.ModuleList(models)

        self.state_size = sum([model.state_size for model in self.models])
        self.state_sizes = torch.tensor([model.state_size for model in self.models])
        self.cstate_sizes = torch.cat((torch.tensor([0]), self.state_sizes.cumsum(dim=0)))  # Precompute cumulative sums

    def split_features(self, u_i, strategy="sequential"):
        num_samples = u_i.shape[0]
        indices = torch.arange(num_samples)

        if strategy == "sequential":
            split_tensors = torch.split(u_i, self.max_features, dim=0)
            split_indices = torch.split(indices, self.max_features, dim=0)
        elif strategy == "random":
            permuted_indices = torch.randperm(num_samples)
            split_tensors = torch.split(u_i[permuted_indices], self.max_features, dim=0)
            split_indices = torch.split(permuted_indices, self.max_features, dim=0)
        elif strategy == "conv":
            raise NotImplementedError("Convolutional strategy not implemented yet.")
        else:
            raise ValueError("strategy must be 'sequential' or 'random'")

        return split_tensors, split_indices
        

    def f(self, t, theta):
        return torch.cat([model.f(t, self.get_theta_group(theta, i)) for i, model in enumerate(self.models)], dim=1)

    def g(self, t, theta):
        return torch.cat([model.g(t, self.get_theta_group(theta, i)) for i, model in enumerate(self.models)], dim=1)
    
    def cost(self, theta):
        return torch.sum(torch.stack([model.cost(self.get_theta_group(theta, i)) for i, model in enumerate(self.models)]), dim=0)
    
    def loss(self, theta):
        loss = 0
        for ui, yi in zip(self.u_i.t(), self.y_i.t()):
            loss += torch.sum((self.num_y_prediction(ui, theta) - yi) ** 2)
        return loss/self.u_i.shape[1]
    
    def y_prediction(self, u, output_type="mean"):
        if output_type == "stack":
            return torch.stack([torch.tensor(model.y_prediction(u[indices])) for model, indices in zip(self.models, self.groups_idx)], dim=0)
        elif output_type == "mean":
            return torch.mean(torch.stack([torch.tensor(model.y_prediction(u[indices])) for model, indices in zip(self.models, self.groups_idx)]), dim=0)
        else:
            raise ValueError("output_type must be 'stack' or 'mean'")
    
    def num_y_prediction(self, u, theta, output_type="mean"):
        if output_type == "stack":
            return torch.stack([torch.tensor(model.num_y_prediction(u[indices], self.get_theta_group(theta, i, predict=True))) for i, (model, indices) in enumerate(zip(self.models, self.groups_idx))], dim=0)
        elif output_type == "mean":
            return torch.mean(torch.stack([torch.tensor(model.num_y_prediction(u[indices], self.get_theta_group(theta, i, predict=True))) for i, (model, indices) in enumerate(zip(self.models, self.groups_idx))]), dim=0)
        else:
            raise ValueError("output_type must be 'stack' or 'mean'")

    def get_theta_group(self, theta, group, predict=False):
        if predict:
            return theta[self.cstate_sizes[group]:self.cstate_sizes[group + 1]]
        return theta[:, self.cstate_sizes[group]:self.cstate_sizes[group + 1]]
    

class GS3DE(nn.Module):
    def __init__(
        self, n_sticks, u_i, y_i, friction=0, temp=0, k=1, M=1, kb=1.38064852e-23, k2=0, boundaries=None, verbose=False
    ):
        """
        Generalized Springs and Sticks Stochastic Differential Equation (GS3DE) model.

        n_sticks (int, NDArray): If int, all directions have same number of pieces. If NDArray, each direction has a different number of pieces. If NDArray, it must have the same length as the dimensions of u_i.

        If n_sticks is [1,1,1,1] it uses one stick per dimension.

        boundaries (Tuple): Tuple with two NDArrays. The first NDArray contains the minimum values for each dimension. The second NDArray contains the maximum values for each dimension.
        """
        super().__init__()
        self.n_sticks = torch.tensor(n_sticks, dtype=int) if isinstance(n_sticks, (list, tuple)) else torch.ones(u_i.shape[0], dtype=int)*n_sticks
        assert self.n_sticks.shape[0] == u_i.shape[0], "n_sticks and u_i dimensions mismatch"

        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.u_min = u_i.min(axis=1).values if boundaries is None else boundaries[0]
        self.u_max = u_i.max(axis=1).values if boundaries is None else boundaries[1]
        self.ell = ((self.u_max - self.u_min) / n_sticks)

        self.state_size = torch.prod(self.n_sticks + 1) * 2 * y_i.shape[0]

        self.y_i = y_i
        self.u_i = u_i

        self.k = k
        self.k2 = k2
        self.M = M / torch.prod((self.n_sticks))
        self.kb = kb

        self.friction = friction
        self.temp = temp
        # self.eta_cte = np.sqrt(2 * friction * temp * kb / (M * np.prod(n_sticks)))
        self.eta_cte = float(np.sqrt(2 * friction * temp * kb / M))

        self._init_symbols()
        self._init_kinetic_energy()
        self._init_potential_energy()
        self._init_lagrangian()


    def _init_symbols(self):
        shape = tuple([int(n) for n in np.append(self.n_sticks, self.y_i.shape[0])])

        self.symbols_shape = shape
        self.N = np.prod(shape)

        self.x_symbols = np.empty(
            np.array([n for n in np.append(self.n_sticks + 1, self.y_i.shape[0])]), dtype=object
        )
        self.dx_symbols = np.empty_like(self.x_symbols, dtype=object)
        self.ddx_symbols = np.empty_like(self.x_symbols, dtype=object)

        self.N = np.prod(self.x_symbols.shape)

        # Iterate over all indices and assign symbols
        # print("Creating symbols...")
        for index in np.ndindex(self.x_symbols.shape):
            # Create symbol and assign it to the corresponding index in the array
            self.x_symbols[index] = dynamicsymbols(f"x_{''.join(map(str, index))}")
            self.dx_symbols[index] = self.x_symbols[index].diff()
            self.ddx_symbols[index] = self.dx_symbols[index].diff()

    def _init_kinetic_energy(self):
        # print("Calculating kinetic energy...")
        # print(f"Iterating over {len(self.symbols_shape)} dimensions")
        ktr = 0
        for i in range(len(self.symbols_shape)-1):
            # print(f"Calculating kinetic energy for dimension {i}...")
            slice_front = [slice(None)] * len(self.symbols_shape)
            slice_back = [slice(None)] * len(self.symbols_shape)
            slice_front[i] = slice(1, None)
            slice_back[i] = slice(None, -1)
            difference = (self.dx_symbols[tuple(slice_front)] + self.dx_symbols[tuple(slice_back)]) ** 2
            difference = difference.ravel()
            ktr = Add(ktr, (self.M.item() / 8) * Add(*difference))

        self.ktr = simplify(ktr)

        # Kinetic energy (rotational)
        krot = 0
        # print("Calculating rotational kinetic energy...")
        for i in range(len(self.symbols_shape)-1):
            # print(f"Calculating rotational kinetic energy for dimension {i}...")
            slice_front = [slice(None)] * len(self.symbols_shape)
            slice_back = [slice(None)] * len(self.symbols_shape)
            slice_front[i] = slice(1, None)
            slice_back[i] = slice(None, -1)
            difference = (self.dx_symbols[tuple(slice_front)] - self.dx_symbols[tuple(slice_back)]) ** 2
            difference = difference.ravel()
            krot = Add(krot, (self.M.item()/ 24) * Add(*difference))

        self.krot = simplify(krot)

    def _init_potential_energy(self):
        # print("Calculating potential energy...")
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

        self.uelastic_symbols = simplify(uelastic)

        # Potential energy (cost function)
        U = 0
        for j in range(self.u_i.shape[1]):
            # print(f"Calculating potential energy for point {j}...")
            i = self.find_box(self.u_i[:, j])
            point_pred = self.y_prediction(self.u_i[:, j], i)
            difference = (point_pred - self.y_i[:, j].detach().cpu().numpy()) ** 2
            difference = difference.ravel()
            U = Add(U, (self.k/ 2 * Add(*difference)))
        
        self.U = simplify(U)

    def _init_lagrangian(self):
        # print("Calculating Lagrangian...")
        # Lagrangian
        self.lagrangian = simplify(Add(self.ktr, self.krot, -self.U, -self.uelastic_symbols))
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

        # print("Done initializing.")

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
        return torch.clip(i, torch.tensor(0), self.n_sticks - 1)

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

        assert torch.sum(weights) - 1 < 1e-5, f"Weights do not sum to 1. Sum: {torch.sum(weights)}"

        lin_combinations = np.array([weights[j].item() * self.x_symbols[tuple(grid_points[j])] for j in range(len(weights))])
        pred = np.array([Add(*lin_combinations[:,j]) for j in range(lin_combinations.shape[1])])

        return pred

    def num_y_prediction(self, u, theta):
        q = theta[: self.N].t()
        ypred = self.ypred(u)
        return ypred(*q)

    def f(self, t, theta):
        """y is a tensor of shape (batch_size, state_dim)

        Returns a tensor of shape (batch_size, state_dim)
        """
        q = theta[:, : self.N].t()
        dq = theta[:, self.N :].t()


        ddqdt = torch.tensor(self.lambdified_dyn(*q, *dq))
        ddqdt_shape = ddqdt.shape
        ddqdt = ddqdt.squeeze()
        if ddqdt_shape[-1] == 1:
            ddqdt = ddqdt.unsqueeze(-1)

        step = torch.vstack([dq, ddqdt])

        return step.t()

    def g(self, t, theta):
        return self.eta_cte * torch.ones_like(theta)

    def cost(self, theta):
        q = theta[:, : self.N].t()
        return self.ue(*q)
    
    def loss(self, theta):
        """Returns the MSE loss."""
        loss = torch.sum((self.num_y_prediction(self.u_i, theta) - self.y_i) ** 2)
        return loss/self.u_i.shape[1]