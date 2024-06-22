import torch
import numpy as np
import sympy as sp

def get_xxt_avg(ys):
    yse = ys[:, :, :, None]
    yse_t = yse.transpose(2, 3)

    xxt = torch.matmul(yse, yse_t)
    xxt_avg = torch.mean(xxt, dim=1)
    return xxt_avg

def get_x_avg(ys):
    return torch.mean(ys, dim=1)

def get_theta(ys):
    xxt_avg = get_xxt_avg(ys)
    x_avg_e = get_x_avg(ys)[:, :, None]
    x_avg_t = x_avg_e.transpose(1, 2)
    return xxt_avg - torch.matmul(x_avg_e, x_avg_t)

def get_Dtensor(diag):
    # turn diag float to tensor
    diag = torch.tensor(diag)
    diag_tensor = torch.diag(diag)
    Dtensor = diag_tensor @ diag_tensor.T / 2
    return Dtensor

def get_Ab(sde):
    minvf = sde.LM.mass_matrix_full.pinv() @ sde.LM.forcing_full
    vecs = sp.Matrix(np.vstack((sde.x_symbols.reshape(-1,1), sde.dx_symbols.reshape(-1,1))))

    n = vecs.shape[0]

    A = np.zeros((minvf.shape[0], n))
    b = np.zeros((minvf.shape[0], 1))

    for i in range(minvf.shape[0]):
        for j in range(n):
            A[i, j] = -minvf[i].coeff(vecs[j])
        b[i] = minvf[i].subs({vi: 0 for vi in vecs})
    return A, b

def get_Abqp(sde):
    A, b = get_Ab(sde)
    n = A.shape[1]

    A_irr = np.zeros_like(A)
    b_irr = np.zeros_like(b)
    A_rev = np.zeros_like(A)
    b_rev = np.zeros_like(b)

    A_irr[:n//2, :n//2] = A[:n//2, :n//2]
    A_irr[n//2:, n//2:] = A[n//2:, n//2:]
    b_irr[:n//2] = b[:n//2]

    A_rev[:n//2, n//2:] = A[:n//2, n//2:]
    A_rev[n//2:, :n//2] = A[n//2:, :n//2]
    b_rev[n//2:] = b[n//2:]
    return A_irr, b_irr, A_rev, b_rev

def getPit(ys, sde):
    theta = get_theta(ys)
    diag = sde.eta_cte * torch.ones(ys.shape[2])
    Dtensor = get_Dtensor(diag)
    # Aq = get_Aq(sde)
    return torch.vmap(torch.trace)(Dtensor @ theta.inverse())

def get_dsdt(ys, sde):
    theta = get_theta(ys)
    diag = sde.eta_cte * torch.ones(ys.shape[2])
    Dtensor = get_Dtensor(diag)
    A, b = get_Ab(sde)

    return torch.vmap(torch.trace)(theta.inverse() @ Dtensor - A)