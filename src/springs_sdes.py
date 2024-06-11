from sympy import *
from sympy.physics.mechanics import *

import numpy as np

import torch
import torch.nn as nn


class S3DE(nn.Module):
    def __init__(self, u_i, y_i, friction=0, temp=0, k=1, M=1, kb=1.38064852e-23):
        """
        Makes S3DE where the langevin dynamics are given by
        ddxi = -dU/dxi - friction * dxi + sqrt(2 friction kT) * eta_i
        where eta_i is a white noise process with mean 0 and variance 1
        """

        super().__init__()

        self.noise_type = "diagonal"
        self.sde_type = "ito"

        x1, xN = dynamicsymbols("x_1 x_N")
        dx1, dxN = dynamicsymbols("x_1 x_N", 1)
        ddx1, ddxN = dx1.diff(), dxN.diff()

        ell = max(u_i) - min(u_i)

        self.y_i = y_i
        self.u_i = u_i
        self.ell = ell

        self.x1 = x1
        self.xN = xN
        self.dx1 = dx1
        self.dxN = dxN
        self.ddx1 = ddx1
        self.ddxN = ddxN

        # Constants
        self.k = k
        self.M = M

        # Kinetic energy (translational)
        ktr = M / 8 * (dx1 + dxN) ** 2
        self.ktr = ktr

        # Kinetic energy (rotational)
        tdxN, tdx1, tell = symbols("temp_dxN temp_dx1 temp_ell")
        # krot = M*tell**2 / 24 * asin((tdxN - tdx1)/tell)**2
        # krot = M*tell**2 / 24 * (
        # approx_krot = series(krot, tdxN - tdx1, 0, 3).removeO().subs({tdxN: dxN, tdx1: dx1, tell: ell})
        approx_krot = M * ell**2 / 24 * ((dxN - dx1) / (ell)) ** 2
        krot = approx_krot
        self.krot = krot

        # Potential energy (cost function)
        U = (
            k
            / 2
            * sum(
                [
                    (x1 + (uj - u_i[0]) * (xN - x1) / ell - y_i[i]) ** 2
                    for i, uj in enumerate(u_i)
                ]
            )
        )
        self.U = U

        # Lagrangian
        self.lagrangian = ktr + krot - U

        self.LM = LagrangesMethod(self.lagrangian, [x1, xN])
        self.elm = solve(self.LM.form_lagranges_equations(), (ddx1, ddxN))
        self.fddx1 = lambdify([x1, xN, dx1, dxN], self.elm[ddx1] - friction * dx1)
        self.fddxN = lambdify([x1, xN, dx1, dxN], self.elm[ddxN] - friction * dxN)

        self.friction = friction
        self.temp = temp
        self.kb = kb
        self.eta_cte = float(np.sqrt(2 * friction * temp * kb / M**2))

    def f(self, t, y):
        """y is a tensor of shape (batch_size, state_dim)

        Returns a tensor of shape (batch_size, state_dim)
        """
        q0, q1, dq0, dq1 = y.T

        dq0dt = dq0
        dq1dt = dq1
        ddq0dt = self.fddx1(q0, q1, dq0, dq1)
        ddq1dt = self.fddxN(q0, q1, dq0, dq1)

        step = torch.stack([dq0dt, dq1dt, ddq0dt, ddq1dt], dim=0).T

        return step

    def g(self, t, y):
        return self.eta_cte * torch.ones_like(
            y
        )  # * torch.normal(mean=0, std=1, size=y.shape)

    def cost(self, y):
        engf = lambdify([self.x1, self.xN, self.dx1, self.dxN], self.U)
        return engf(*y.T)

    def model_params(self, y):
        """Returns the slope and intercept of the linear fit of the potential energy."""
        x0, x1, _, _ = y.T

        slope = (x1 - x0) / self.ell
        intercept = x0 - (x1 - x0) / self.ell

        return slope, intercept


class GS3DE(nn.Module):
    def __init__(
        self, n_pieces, u_i, y_i, friction=0, temp=0, k=1, M=1, kb=1.38064852e-23
    ):
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"

        u_min = min(u_i)
        u_max = max(u_i)
        ell = u_max - u_min

        self.y_i = y_i
        self.u_i = u_i
        self.ell = ell

        self.k = k
        self.M = M / n_pieces
        self.kb = kb

        self.friction = friction
        self.temp = temp
        self.eta_cte = float(np.sqrt(2 * friction * temp * kb / M))

        self.n_pieces = n_pieces
        self.N = n_pieces + 1
        self.x_symbols = [dynamicsymbols(f"x{i}") for i in range(self.N)]
        self.dx_symbols = [dynamicsymbols(f"x{i}", 1) for i in range(self.N)]
        self.ddx_symbols = [dx.diff() for dx in self.dx_symbols]

        # Kinetic energy (translational)
        ktr = (
            self.M
            / 8
            * sum(
                [
                    (final_i + initial_i) ** 2
                    for initial_i, final_i in zip(
                        self.dx_symbols[:-1], self.dx_symbols[1:]
                    )
                ]
            )
        )
        self.ktr = ktr

        # Kinetic energy (rotational)
        # tdxN, tdx1, tell = symbols('temp_dxN temp_dx1 temp_ell')
        # krot = M*tell**2 / 24 * asin((tdxN - tdx1)/tell)**2
        # approx_krot = series(krot, tdxN - tdx1, 0, 3).removeO()

        # krot = sum([approx_krot.subs({tdxN: self.dx_symbols[i+1], tdx1: self.dx_symbols[i], tell: ell}) for i in range(self.N-1)])

        krot = (
            self.M
            * ell**2
            / 24
            * sum(
                [
                    ((self.dx_symbols[i + 1] - self.dx_symbols[i]) / ell) ** 2
                    for i in range(self.N - 1)
                ]
            )
        )

        self.krot = krot

        # Potential energy (cost function)
        U = 0

        interval_length = ell / n_pieces

        for ui, yi in zip(u_i, y_i):
            interval = int(float(ui - u_min) // float(interval_length))

            if interval == n_pieces:
                U += k / 2 * (self.x_symbols[-1] - yi) ** 2
            else:
                init_i = self.x_symbols[interval]
                final_i = self.x_symbols[interval + 1]

                slope = (final_i - init_i) / interval_length
                dif = ui - interval * interval_length - u_min
                U += k / 2 * (init_i + dif * slope - yi) ** 2

        self.U = U
        # Lagrangian
        self.lagrangian = ktr + krot - U
        self.LM = LagrangesMethod(self.lagrangian, self.x_symbols)
        self.elm = solve(self.LM.form_lagranges_equations(), self.ddx_symbols)
        self.fddx = [
            lambdify(
                [*self.x_symbols, *self.dx_symbols],
                self.elm[ddx] - friction * dx / n_pieces,
            )
            for ddx, dx in zip(self.ddx_symbols, self.dx_symbols)
        ]

    def f(self, t, y):
        """y is a tensor of shape (batch_size, state_dim)

        Returns a tensor of shape (batch_size, state_dim)
        """
        q = y[:, : self.N].T
        dq = y[:, self.N :].T

        dqdt = dq
        ddqdt = torch.stack([f(*q, *dq).T for f in self.fddx])
        step = torch.hstack([dqdt.T, ddqdt.T])

        return step

    def g(self, t, y):
        return self.eta_cte * torch.ones_like(
            y
        )  # * torch.normal(mean=0, std=1, size=y.shape)

    def cost(self, y):
        q = y[:, : self.N].T
        engf = lambdify(self.x_symbols, self.U)
        return engf(*q)


class GGS3DE(nn.Module):
    def __init__(
        self, n_pieces, u_i, y_i, friction=0, temp=0, k=1, M=1, kb=1.38064852e-23, k2=0
    ):
        """
        n_pices (int, NDArray): If int, all directions have same number of pieces. If NDArray, each direction has a different number of pieces. If NDArray, it must have the same length as the dimensions of u_i.
        """
        super().__init__()
        self.noise_type = "diagonal"
        self.sde_type = "ito"

        if isinstance(n_pieces, int):
            n_pieces = np.ones(u_i.shape[0], dtype=int) * n_pieces

        assert (
            n_pieces.shape[0] == u_i.shape[0]
        ), f"n_pieces must have the same length as the dimensions of u_i. Expected {u_i.shape[0]}, got {n_pieces.shape[0]}"

        self.u_min = u_i.min(axis=1)
        self.u_max = u_i.max(axis=1)
        ell = (self.u_max - self.u_min) / (n_pieces - np.ones_like(n_pieces))

        self.y_i = y_i
        self.u_i = u_i
        self.ell = ell.astype(np.float64)

        self.k = k
        self.k2 = k2
        self.M = M / np.prod((n_pieces - np.ones_like(n_pieces)))
        self.kb = kb

        self.friction = friction
        self.temp = temp
        # self.eta_cte = np.sqrt(2 * friction * temp * kb / (M * np.prod(n_pieces)))
        self.eta_cte = float(np.sqrt(2 * friction * temp * kb / M))

        x_symbols = np.empty(
            np.array([n for n in np.append(n_pieces, y_i.shape[0])]), dtype=object
        )

        self.N = np.prod(x_symbols.shape)

        # Iterate over all indices and assign symbols
        print("Creating symbols...")
        for index in np.ndindex(x_symbols.shape):
            # Create symbol and assign it to the corresponding index in the array
            x_symbols[index] = dynamicsymbols(f"x_{''.join(map(str, index))}")

        self.x_symbols = x_symbols
        self.dx_symbols = np.empty_like(x_symbols, dtype=object)
        self.ddx_symbols = np.empty_like(x_symbols, dtype=object)

        for index in np.ndindex(x_symbols.shape):
            self.dx_symbols[index] = x_symbols[index].diff()
            self.ddx_symbols[index] = self.dx_symbols[index].diff()

        # Kinetic energy (translational)
        print("Calculating kinetic energy...")
        ktr = 0
        for i in range(len(x_symbols.shape) - 1):
            # Create slices to shift the view one position along the i-th dimension
            slice_front = [slice(None)] * len(x_symbols.shape)
            slice_back = [slice(None)] * len(x_symbols.shape)
            slice_front[i] = slice(1, None)
            slice_back[i] = slice(None, -1)

            # Ensure that the operation is only applied to corresponding spatial components
            for component in range(x_symbols.shape[-1]):
                # Add the squared difference for the same component across neighboring spatial points
                ktr += (self.M / 8) * np.sum(
                    (
                        self.dx_symbols[tuple(slice_front)][..., component]
                        + self.dx_symbols[tuple(slice_back)][..., component]
                    )
                    ** 2
                )

        self.ktr = ktr

        # Kinetic energy (rotational)
        krot = 0

        for i in range(len(x_symbols.shape) - 1):
            slice_front = [slice(None)] * len(x_symbols.shape)
            slice_back = [slice(None)] * len(x_symbols.shape)
            slice_front[i] = slice(1, None)
            slice_back[i] = slice(None, -1)

            for component in range(x_symbols.shape[-1]): # sum over output dimensions
                krot += (self.M / 24) * np.sum(
                    (
                        self.dx_symbols[tuple(slice_front)][..., component]
                        - self.dx_symbols[tuple(slice_back)][..., component]
                    )
                    ** 2
                )

        self.krot = krot

        print("Calculating potential energy...")
        # Elastic energy (we don't want xi - xi+1 to be so far from ell)
        uelastic = 0

        if k2 != 0:
            for i in range(len(x_symbols.shape) - 1):
                slice_front = [slice(None)] * len(x_symbols.shape)
                slice_back = [slice(None)] * len(x_symbols.shape)
                slice_front[i] = slice(1, None)
                slice_back[i] = slice(None, -1)

                for component in range(x_symbols.shape[-1]):
                    uelastic += self.k2 / 2 * np.sum(
                        (self.x_symbols[tuple(slice_front)][..., component]
                        - self.x_symbols[tuple(slice_back)][..., component]
                            - self.ell[i])** 2
                    )

        # Potential energy (cost function)
        U = 0
        for j in range(u_i.shape[1]):
            found_box = False
            for i in np.ndindex(
                tuple(
                    (x_symbols.shape[:-1] - np.ones_like(x_symbols.shape[:-1])).astype(
                        np.int64
                    )
                ) # Iterate over all boxes (removing -1 from the shape to avoid having an extra box)
            ):
                hvs = self.vec_heaviside(
                    self.lamd(i, u_i[:, j], div_by_ell=True) 
                ) * np.prod(
                    [
                        self.vec_heaviside(
                            self.lamd(i + eb, u_i[:, j], flpind, div_by_ell=True) # Flip ind does the same as (\vec{1}-2e^b)*\lambda_i+e^b
                        )
                        for flpind, eb in enumerate(np.eye(u_i.shape[0]))
                    ]
                )

                if abs(hvs) > 0.5:

                    found_box = True
                    U += (
                        k
                        / 2
                        * sum((self.y_prediction(u_i[:, j], i) - y_i[:, j]) ** 2)
                    )
                    break

            if not found_box:
                print(
                    "Not found. Values of H:",
                    self.vec_heaviside(self.lamd(i, u_i[:, j], div_by_ell=True)),
                    f"Point {j}:",
                    u_i[:, j],
                    "ProdHvs:",
                    [
                        self.lamd(i + eb, u_i[:, j], flpind, div_by_ell=True)
                        for flpind, eb in enumerate(np.eye(u_i.shape[0]))
                    ],
                )

                raise ValueError(
                    f"Point {j} ({u_i[:, j]-self.u_min}) not found in any box"
                    f"\n min: {self.u_min}"
                    f"\n max: {self.u_max}"
                )
        self.U = U

        print("Calculating Lagrangian...")
        # Lagrangian
        self.lagrangian = ktr + krot - U - uelastic
        self.LM = LagrangesMethod(self.lagrangian, self.x_symbols.flatten())
        self.elm = solve(self.LM.form_lagranges_equations(), self.ddx_symbols.flatten())
        self.fddx = [
            lambdify(
                [*self.x_symbols.flatten(), *self.dx_symbols.flatten()],
                self.elm[ddx] - friction * dx / np.prod(n_pieces),
            )
            for ddx, dx in zip(self.ddx_symbols.flatten(), self.dx_symbols.flatten())
        ]

        self.ke = lambdify(
            [*self.x_symbols.flatten(), *self.dx_symbols.flatten()],
            self.ktr + self.krot,
        )
        self.ue = lambdify([*self.x_symbols.flatten()], self.U)
        self.uelastic = lambdify([*self.x_symbols.flatten()], uelastic) 
        self.ypred = lambda u: lambdify(
            [*self.x_symbols.flatten()], self.y_prediction(u)
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

    def vec_heaviside(self, x):
        """Returns the vectorized heaviside function."""
        return np.prod(np.heaviside(x, 1), axis=0)

    def find_box(self, u):
        """Returns the box where the input u is located."""
        for i in np.ndindex(
            tuple(
                (
                    self.x_symbols.shape[:-1] - np.ones_like(self.x_symbols.shape[:-1])
                ).astype(np.int64)
            )
        ):
            hvs = self.vec_heaviside(self.lamd(i, u, div_by_ell=False)) * np.prod(
                [
                    self.vec_heaviside(self.lamd(i + eb, u, flpind, div_by_ell=False))
                    for flpind, eb in enumerate(np.eye(u.shape[0]))
                ]
            )
            if abs(hvs) > 0.5:
                return i
        raise ValueError(f"Point {u-self.u_min} not found in any box")

    def y_prediction(self, u, i=None):
        """Returns the predicted y given the input u."""
        pred = None
        if i is None:
            i = self.find_box(u)

        ld = self.lamd(i, u)

        for indx, l in enumerate(np.ndindex(tuple(2 * np.ones_like(i)))):
            interpolation = 1
            ind = tuple(np.array(i) + np.array(l))
            flind = np.array(l) == 0
            lds = (1 - ld) * flind + ld * (1 - flind)
            interpolation = np.prod(lds)

            if indx == 0:
                pred = self.x_symbols[ind] * interpolation
            else:
                pred += self.x_symbols[ind] * interpolation
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

        dqdt = dq
        ddqdt = torch.stack([f(*q, *dq).T for f in self.fddx])
        step = torch.hstack([dqdt.T, ddqdt.T])

        return step

    def g(self, t, y):
        return self.eta_cte * torch.ones_like(y)

    def cost(self, y):
        q = y[:, : self.N].T
        return self.ue(*q)

    def kinetic_energy(self, y):
        q = y[:, : self.N].T
        dq = y[:, self.N :].T

        return self.ke(*q, *dq)


# class SSModel(nn.Module):
#     def __init__(self, data, friction=0, temp=0, k=1, M=1):
#         super().__init__()
#         self.noise_type = "diagonal"
#         self.sde_type = "ito"

#         self.kb = 1.38064852e-23
#         self.friction = friction
#         self.temp = temp
#         self.k = k
#         self.M = M

#     def f(self, t, y):
#         pass

if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser(description="Spring SDEs")
    # parser.add_argument("example", type=int, help="Example number")

    # # args = parser.parse_args()

    # # if args.example == 1:
    # S3DE with input of 2 dims and output of 1 dim
    # u_i = np.array(np.random.rand(5))
    # y_i = np.array(np.random.rand(5))

    # sde = GS3DE(4, u_i, y_i, friction=0.1, temp=0.1, k=1, M=1)
    # print(sde)

    # if args.example == 2:
    #     # GGS3DE with input of 3 dims and output of 2 dims
    u_i = np.random.rand(3, 5)
    y_i = np.random.rand(2, 5)

    n_pieces = np.array([3, 3, 3])
    gen_sde = GGS3DE(n_pieces, u_i, y_i, friction=0.1, temp=0.1, k=1, M=1)
    print(gen_sde)
