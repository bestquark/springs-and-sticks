
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
    """Grouped GS3DE models. It splits the input features into groups and instantiates a GS3DE model for each group."""
    def __init__(
        self, max_features, n_sticks, boundaries, n_labels, friction=0, temp=0, k=1, M=1,
        kb=1.38064852e-23, k2=0, verbose=False, group_strategy="sequential", parallel=False
    ):
        """
        Parameters:
          max_features (int or float): If int, the maximum number of features per group.
            If float, a fraction of the total features.
          n_sticks: int or tensor/array defining the discretization for each input dimension.
          boundaries: tuple (u_min, u_max) with each a tensor of shape (d,), where d is the number of input features.
          n_labels: int, number of labels (output dimensions).
          group_strategy: "sequential" or "random".
          parallel: whether to create underlying models in parallel.
        """
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        
        # Determine feature dimension from boundaries.
        self.num_features = boundaries[0].shape[0]
        self.num_labels = n_labels
        
        # Determine max_features based on type.
        self.max_features = max_features if isinstance(max_features, int) else int(max_features * self.num_features)
        self.group_strategy = group_strategy

        # Split the feature indices into groups.
        self.groups, self.groups_idx = self.split_features(self.num_features, strategy=group_strategy)

        models = []
        if parallel:
            def create_gs3de_model(idx, group_idx):
                group_boundaries = (boundaries[0][group_idx], boundaries[1][group_idx])
                return GS3DE(n_sticks[group_idx], group_boundaries, n_labels, friction, temp, k, M, kb, k2, verbose)
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(create_gs3de_model, idx, group_idx): idx for idx, group_idx in enumerate(self.groups_idx)}
                for future in as_completed(futures):
                    models.append(future.result())
        else:
            for idx, group_idx in enumerate(self.groups_idx):
                if verbose:
                    print(f"Creating GS3DE for group {idx+1}/{len(self.groups_idx)}...")
                group_boundaries = (boundaries[0][group_idx], boundaries[1][group_idx])
                model = GS3DE(n_sticks[group_idx], group_boundaries, n_labels, friction, temp, k, M, kb, k2, verbose)
                models.append(model)

        self.models = nn.ModuleList(models)
        self.state_size = sum([model.state_size for model in self.models])
        self.state_sizes = torch.tensor([model.state_size for model in self.models])
        self.cstate_sizes = torch.cat((torch.tensor([0]), self.state_sizes.cumsum(dim=0)))  # Cumulative state sizes

    def split_features(self, num_features, strategy="sequential"):
        """
        Splits feature indices [0, 1, ..., num_features-1] into groups.
        Returns a tuple (groups, groups_idx), where each element in groups_idx is a tensor of indices.
        """
        indices = torch.arange(num_features)
        if strategy == "sequential":
            split_indices = torch.split(indices, self.max_features)
        elif strategy == "random":
            permuted_indices = torch.randperm(num_features)
            split_indices = torch.split(permuted_indices, self.max_features)
        else:
            raise ValueError("strategy must be 'sequential' or 'random'")
        return split_indices, split_indices

    def f(self, t, theta):
        return torch.cat([model.f(t, self.get_theta_group(theta, i)) for i, model in enumerate(self.models)], dim=1)

    def g(self, t, theta):
        return torch.cat([model.g(t, self.get_theta_group(theta, i)) for i, model in enumerate(self.models)], dim=1)

    def cost(self, theta):
        return torch.sum(torch.stack([model.cost(self.get_theta_group(theta, i)) for i, model in enumerate(self.models)]), dim=0)

    def loss(self, theta, u_i, y_i):
        loss = torch.sum((self.num_y_prediction(u_i, theta) - y_i) ** 2)
        return loss / u_i.shape[0]

    def y_prediction(self, u, output_type="mean"):
        if output_type == "stack":
            return torch.stack([torch.tensor(model.y_prediction(u[:, indices])) for model, indices in zip(self.models, self.groups_idx)], dim=0)
        elif output_type == "mean":
            return torch.mean(torch.stack([torch.tensor(model.y_prediction(u[:, indices])) for model, indices in zip(self.models, self.groups_idx)]), dim=0)
        else:
            raise ValueError("output_type must be 'stack' or 'mean'")

    def num_y_prediction(self, u, theta, output_type="mean"):
        if output_type == "stack":
            return torch.stack([torch.tensor(model.num_y_prediction(u[:, indices], self.get_theta_group(theta, i, predict=True)))
                                for i, (model, indices) in enumerate(zip(self.models, self.groups_idx))], dim=0)
        elif output_type == "mean":
            return torch.mean(torch.stack([torch.tensor(model.num_y_prediction(u[:, indices], self.get_theta_group(theta, i, predict=True)))
                                             for i, (model, indices) in enumerate(zip(self.models, self.groups_idx))]), dim=0)
        else:
            raise ValueError("output_type must be 'stack' or 'mean'")

    def get_theta_group(self, theta, group, predict=False):
        if predict:
            return theta[self.cstate_sizes[group]:self.cstate_sizes[group + 1]]
        return theta[:, self.cstate_sizes[group]:self.cstate_sizes[group + 1]]

    def update_data(self, new_u_i, new_y_i):
        """
        Update the potential energy terms in each underlying GS3DE model based on new data.
        new_u_i is expected to be a tensor of shape (num_features, ...), and new_y_i of shape (n_labels, ...).
        If the number of features changes, the grouping is re-computed.
        """
        # if new_u_i.shape[0] != self.num_features:
        #     self.num_features = new_u_i.shape[0]
        #     self.groups, self.groups_idx = self.split_features(self.num_features, strategy=self.group_strategy)
        for idx, group_idx in enumerate(self.groups_idx):
            new_group = new_u_i[:, group_idx]
            self.models[idx].update_data(new_group, new_y_i)



class GS3DE(nn.Module):
    def __init__(self, n_sticks, boundaries, n_labels, friction=0, temp=0, k=1, M=1,
                 kb=1.38064852e-23, k2=0, verbose=False):
        """
        GS3DE model where the non-potential Lagrangian is computed once.
        
        Parameters:
          n_sticks: int or array-like. If int, all input dimensions use the same number of pieces.
                    If array-like, it must have the same length as the input dimension.
          boundaries: tuple of two tensors/arrays, (u_min, u_max), each of shape (d,), where d is the input dimension.
          n_labels: int, number of labels (used to determine the size of the output space).
          friction, temp, k, M, kb, k2: physical parameters.
          verbose: optional verbosity flag.
          
        Note: u_i and y_i are not used in initialization. They must be supplied via update_data.
        """
        super().__init__()
        # Determine input dimension from boundaries.
        d = boundaries[0].shape[0]
        if isinstance(n_sticks, int):
            self.n_sticks = torch.ones(d, dtype=int) * n_sticks
        elif isinstance(n_sticks, torch.Tensor):
            self.n_sticks = n_sticks.clone().detach().to(torch.int)
        else:
            self.n_sticks = torch.tensor(n_sticks, dtype=torch.int)

        assert self.n_sticks.shape[0] == d, "n_sticks must have the same length as input dimension"

        self.noise_type = "diagonal"
        self.sde_type = "ito"
        # Boundaries must be provided.
        self.u_min = boundaries[0]
        self.u_max = boundaries[1]
        self.ell = ((self.u_max - self.u_min) / self.n_sticks)

        # Set state_size using n_labels.
        self.state_size = torch.prod(self.n_sticks + 1) * 2 * n_labels

        self.k = k
        self.k2 = k2
        self.M = M / torch.prod(self.n_sticks)
        self.kb = kb

        self.friction = friction
        self.temp = temp
        self.eta_cte = float(np.sqrt(2 * friction * temp * kb / M))

        self.n_labels = n_labels  # used for symbolic variable shapes

        # Initialize symbolic variables, kinetic and elastic energies.
        self._init_symbols(n_labels)
        self._init_kinetic_energy()
        self._init_elastic_energy()
        self._init_lagrangian() 

    def _init_symbols(self, n_labels):
        # Create symbolic arrays with shape: (n_sticks+1) x ... x (n_labels)
        shape = tuple([int(n) for n in np.append(self.n_sticks + 1, n_labels)])
        self.symbols_shape = shape
        self.N = np.prod(shape)
        self.x_symbols = np.empty(shape, dtype=object)
        self.dx_symbols = np.empty_like(self.x_symbols, dtype=object)
        self.ddx_symbols = np.empty_like(self.x_symbols, dtype=object)

        for index in np.ndindex(shape):
            self.x_symbols[index] = dynamicsymbols(f"x_{''.join(map(str, index))}")
            self.dx_symbols[index] = self.x_symbols[index].diff()
            self.ddx_symbols[index] = self.dx_symbols[index].diff()

    def _init_kinetic_energy(self):
        # Translational kinetic energy.
        ktr = 0
        for i in range(len(self.symbols_shape) - 1):
            slice_front = [slice(None)] * len(self.symbols_shape)
            slice_back  = [slice(None)] * len(self.symbols_shape)
            slice_front[i] = slice(1, None)
            slice_back[i]  = slice(None, -1)
            difference = (self.dx_symbols[tuple(slice_front)] + self.dx_symbols[tuple(slice_back)]) ** 2
            difference = difference.ravel()
            ktr = Add(ktr, (self.M.item() / 8) * Add(*difference))
        self.ktr = simplify(ktr)

        # Rotational kinetic energy.
        krot = 0
        for i in range(len(self.symbols_shape) - 1):
            slice_front = [slice(None)] * len(self.symbols_shape)
            slice_back  = [slice(None)] * len(self.symbols_shape)
            slice_front[i] = slice(1, None)
            slice_back[i]  = slice(None, -1)
            difference = (self.dx_symbols[tuple(slice_front)] - self.dx_symbols[tuple(slice_back)]) ** 2
            difference = difference.ravel()
            krot = Add(krot, (self.M.item() / 24) * Add(*difference))
        self.krot = simplify(krot)

    def _init_elastic_energy(self):
        # Elastic energy to keep neighboring states near the rest length.
        uelastic = 0
        if self.k2 != 0:
            for i in range(len(self.x_symbols.shape) - 1):
                slice_front = [slice(None)] * len(self.x_symbols.shape)
                slice_back  = [slice(None)] * len(self.x_symbols.shape)
                slice_front[i] = slice(1, None)
                slice_back[i]  = slice(None, -1)
                difference = (self.x_symbols[tuple(slice_front)] - self.x_symbols[tuple(slice_back)] - self.ell[i]) ** 2
                difference = difference.ravel()
                uelastic = Add(uelastic, self.k2 / 2 * Add(*difference))
        self.uelastic_symbols = simplify(uelastic)

    def _init_lagrangian(self):
        # Compute non-potential Lagrangian (kinetic + elastic).
        self.L_nonpot = simplify(Add(self.ktr, self.krot, -self.uelastic_symbols))
        # Initially, set the potential energy U to zero.
        self.U = 0  
        LM_nonpot = LagrangesMethod(self.L_nonpot, self.x_symbols.flatten())
        LM_nonpot.form_lagranges_equations()
        self.mass_matrix = simplify(LM_nonpot.mass_matrix)
        try:
            self.inv_mass_matrix = simplify(LM_nonpot.mass_matrix.inv())
        except Exception as e:
            print(e)
            self.inv_mass_matrix = simplify(LM_nonpot.mass_matrix.pinv())
        self.forcing_nonpot = simplify(LM_nonpot.forcing)
        self._update_total_lagrangian()
        self.ypred = lambda u: lambdify([*self.x_symbols.flatten()], self.y_prediction(u))

    def _compute_potential_energy(self, u_i, y_i):
        """Compute the potential energy U that depends on new data u_i and y_i."""
        i_idx = self.find_box(u_i)
        point_pred = self.y_prediction(u_i, i_idx)
        difference = (point_pred - y_i.cpu().numpy()) ** 2
        difference = difference.ravel()
        U = Add(self.k / 2 * Add(*difference))
        self.U = simplify(U)
    
    def _update_total_lagrangian(self):
        """Update the full Lagrangian and derived dynamics using the current U."""
        self.lagrangian = Add(self.L_nonpot, -self.U)
        self.forcing_pot = Matrix([-diff(self.U, q) for q in self.x_symbols.flatten()])
        self.forcing_vector = Add(self.forcing_nonpot, self.forcing_pot)
        self.evol_dynamics = simplify(self.inv_mass_matrix * self.forcing_vector)
        self.lambdified_dyn = lambdify(
            [*self.x_symbols.flatten(), *self.dx_symbols.flatten()],
            self.evol_dynamics - self.friction * self.dx_symbols.reshape(-1, 1) / self.N
        )
        self.ue = lambdify([*self.x_symbols.flatten()], self.U)

    def update_data(self, new_u_i, new_y_i):
        """
        Update the model's potential energy based on new data.
        This recomputes U (and its derived dynamics) without redoing the non-potential computations.
        """
        self._compute_potential_energy(new_u_i, new_y_i)
        self._update_total_lagrangian()

    def lamd(self, i, u, flip_ind=None, div_by_ell=True):
        """Return the lambda function for the i-th dimension."""
        val = u - i * self.ell - self.u_min
        if div_by_ell:
            val /= self.ell
        if flip_ind is not None:
            val[flip_ind] *= -1
        val[np.abs(val) < 1e-8] = 0
        return val

    def find_box(self, u):
        """Return the box where the input u is located."""
        i = ((u - self.u_min) / self.ell).to(torch.int)
        return torch.clip(i, torch.tensor(0), self.n_sticks - 1)

    def y_prediction(self, u, i=None):
        return self._y_prediction(u, i)

    def _y_prediction(self, u, i=None):
        u = torch.clamp(u, self.u_min.unsqueeze(0), self.u_max.unsqueeze(0))
        if i is None:
            i = self.find_box(u)
        ld = self.lamd(i, u)
        dimension = i.shape[1]
        offsets = torch.tensor(list(product([0, 1], repeat=dimension)))
        grid_points = i.unsqueeze(1) + offsets.unsqueeze(0)
        weights = torch.prod((1 - ld.unsqueeze(1)) * (offsets.unsqueeze(0) == 0) + ld.unsqueeze(1) * (offsets.unsqueeze(0) != 0), axis=2)
        assert torch.all(weights.sum(axis=1) - 1 < 1e-5), f"Weights do not sum to 1. Sum: {weights.sum(axis=1)}"
        grid_points_np = grid_points.cpu().numpy().astype(int)
        weights_np = weights.cpu().numpy()
        index_tuple = tuple(grid_points_np[..., i] for i in range(grid_points_np.shape[-1]))
        symbols = self.x_symbols[index_tuple]
        lin_combinations = weights_np[..., None] * symbols
        pred = np.apply_along_axis(lambda col: Add(*col), axis=1, arr=lin_combinations)
        return pred

    def num_y_prediction(self, u, theta):
        q = theta[:self.N].t()
        ypred = self.ypred(u)
        return ypred(*q)

    def f(self, t, theta):
        """Compute the time derivative f(t, theta)."""
        q = theta[:, :self.N].t()
        dq = theta[:, self.N:].t()
        ddqdt = torch.tensor(self.lambdified_dyn(*q, *dq))
        ddqdt_shape = ddqdt.shape
        ddqdt = ddqdt.squeeze()
        if ddqdt_shape[-1] == 1:
            ddqdt = ddqdt.unsqueeze(-1)
        step = torch.vstack([dq, ddqdt])
        return step.t()

    def g(self, t, theta):
        return torch.cat([self.eta_cte * torch.zeros_like(theta[:, :self.N]), torch.ones_like(theta[:, self.N:])], dim=1)

    def cost(self, theta):
        q = theta[:, :self.N].t()
        return self.ue(*q)

    def loss(self, theta, u_i, y_i):
        """Compute the MSE loss using provided data."""        
        loss = torch.sum((torch.tensor(self.num_y_prediction(u_i, theta)) - y_i) ** 2)
        return loss / u_i.shape[0]