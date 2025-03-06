import torch
import numpy as np
import sympy as sp
from sympy.physics.mechanics import LagrangesMethod
from scipy.integrate import simpson


def get_xxt_avg(ys):
    yse = ys[:, :, :, None]
    yse_t = yse.transpose(2, 3)
    xxt = torch.matmul(yse, yse_t)
    return torch.mean(xxt, dim=1)

def get_x_avg(ys):
    return torch.mean(ys, dim=1).to(torch.float64)

def get_theta(ys):
    xxt_avg = get_xxt_avg(ys)
    x_avg_e = get_x_avg(ys)[:, :, None]
    x_avg_t = x_avg_e.transpose(1, 2)
    return (xxt_avg - torch.matmul(x_avg_e, x_avg_t)).to(torch.float64)

def get_Dtensor(diag):
    # turn diag float to tensor and compute D = diag^2/2
    diag = torch.tensor(diag)
    diag_tensor = torch.diag(diag)
    Dtensor = diag_tensor @ diag_tensor.T / 2
    return Dtensor.to(torch.float64)

def get_Ab(sde):
    # Cache result to avoid repeated computation.
    if hasattr(sde, '_Ab'):
        return sde._Ab

    # Compute dynamics using LagrangesMethod
    lagr = sde.lagrangian
    LM = LagrangesMethod(lagr, sde.x_symbols.flatten())
    LM.form_lagranges_equations()
    minvf = LM.mass_matrix_full.pinv() @ LM.forcing_full

    # Add friction term
    fric_term = - sde.friction * sde.dx_symbols.reshape(-1, 1) 
    fric_term = np.vstack((np.zeros((sde.x_symbols.shape[0], 1)), fric_term))
    minvf += fric_term

    vecs = sp.Matrix(np.vstack((sde.x_symbols.reshape(-1, 1), sde.dx_symbols.reshape(-1, 1))))
    n = vecs.shape[0]

    A = np.zeros((minvf.shape[0], n))
    b = np.zeros((minvf.shape[0], 1))

    for i in range(minvf.shape[0]):
        for j in range(n):
            A[i, j] = -minvf[i].coeff(vecs[j])
        b[i] = minvf[i].subs({vi: 0 for vi in vecs})
        
    sde._Ab = (A, b)
    return A, b

def get_Abqp(sde):
    # Cache result to avoid repeated computation.
    if hasattr(sde, '_Abqp'):
        return sde._Abqp

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
    
    sde._Abqp = (A_irr, b_irr, A_rev, b_rev)
    return A_irr, b_irr, A_rev, b_rev

def getPit(ys, sde):
    theta = get_theta(ys)
    # diag = sde.eta_cte * torch.ones(ys.shape[2])
    diag = torch.cat([torch.zeros(ys.shape[2]//2), sde.eta_cte * torch.ones(ys.shape[2]//2)])
    Dtensor = get_Dtensor(diag)
    Aq, bq, _, _ = get_Abqp(sde)
    x_avg = get_x_avg(ys)
    
    Aq = torch.tensor(Aq)
    bq = torch.tensor(bq).squeeze()
    vec = torch.vmap(lambda x: Aq @ x - bq)(x_avg)

    Dtensor_inv = torch.linalg.pinv(Dtensor)
    t1 = torch.vmap(torch.trace)(Dtensor @ theta.inverse() - Aq)
    t2 = torch.vmap(torch.trace)(Aq.T @ Dtensor_inv @ Aq @ theta - Aq)
    t3 = torch.vmap(lambda x: x.T @ Dtensor_inv @ x)(vec)

    return t1 + t2 + t3

def getPhit(ys, sde):
    theta = get_theta(ys).to(torch.float64)
    # diag = sde.eta_cte * torch.ones(ys.shape[2])
    diag = torch.cat([torch.zeros(ys.shape[2]//2), sde.eta_cte * torch.ones(ys.shape[2]//2)])
    Dtensor = get_Dtensor(diag).to(torch.float64)
    _, _, Aq, bq = get_Abqp(sde)
    x_avg = get_x_avg(ys).to(torch.float64)

    Aq = torch.tensor(Aq)
    bq = torch.tensor(bq).squeeze()
    vec = torch.vmap(lambda x: Aq @ x - bq)(x_avg)

    Dtensor_inv = torch.linalg.pinv(Dtensor)
    t1 = torch.vmap(torch.trace)(Aq.T @ Dtensor_inv @ Aq @ theta - Aq)
    t2 = torch.vmap(lambda x: x.T @ Dtensor_inv @ x)(vec)

    return t1 + t2

def getdSdt(ys, sde):
    theta = get_theta(ys)
    # diag = sde.eta_cte * torch.ones(ys.shape[2])
    diag = torch.cat([torch.zeros(ys.shape[2]//2), sde.eta_cte * torch.ones(ys.shape[2]//2)])
    Dtensor = get_Dtensor(diag)
    Aq, bq, _, _ = get_Abqp(sde)
    x_avg = get_x_avg(ys)
    
    Aq = torch.tensor(Aq)
    bq = torch.tensor(bq).squeeze()
    t1 = torch.vmap(torch.trace)(Dtensor @ theta.inverse() - Aq)

    return t1

def get_entropy_rates(ys, sde):
    "Returns Pi(t), Phi(t), and dS/dt = Pi(t) - Phi(t)"
    pit = getPit(ys, sde)
    phit = getPhit(ys, sde)
    dSdt = getdSdt(ys, sde)
    return pit, phit, dSdt

# def get_free_energy_rate(ys, sde):
#     "Returns the free energy rate dF/dt"
#     T = sde.temp
#     kb = sde.kb
#     dSdt = getdSdt(ys, sde)
#     dUdt = torch.mean(torch.stack([sde.dcost(ys[:, i, :]) for i in range(ys.shape[1])]), dim=0)
    
#     return dUdt -kb * T * dSdt

# def get_free_energy(time, ys, sde):
#     dFdt = get_free_energy_rate(ys, sde)
#     F = simpson(dFdt, x=time)
#     return F

def _DF_from_work(work, sde):
    T = sde.temp
    kb = sde.kb
    beta = 1.0 / (kb * T)
    work = torch.tensor(work, dtype=torch.float64)
    N = work.numel()
    log_mean_exp = torch.logsumexp(-beta * work, dim=0) - torch.log(torch.tensor(N, dtype=torch.float64))
    DF = - (1.0 / beta) * log_mean_exp
    return DF


def get_free_energy(time, ys, sde):
    "Returns the free energy using the Jarzynski equality"
    batch_size = ys.shape[1]
    dworks = torch.stack([sde.dw(ys[:, i, :]) for i in range(batch_size)], dim=1)
    work = simpson(dworks, x=time, axis=0)
    return _DF_from_work(work, sde)

# import torch
# import numpy as np
# import sympy as sp

# from sympy.physics.mechanics import LagrangesMethod

# def get_xxt_avg(ys):
#     yse = ys[:, :, :, None]
#     yse_t = yse.transpose(2, 3)

#     xxt = torch.matmul(yse, yse_t)
#     xxt_avg = torch.mean(xxt, dim=1)
#     return xxt_avg

# def get_x_avg(ys):
#     return torch.mean(ys, dim=1).to(torch.float64)

# def get_theta(ys):
#     xxt_avg = get_xxt_avg(ys)
#     x_avg_e = get_x_avg(ys)[:, :, None]
#     x_avg_t = x_avg_e.transpose(1, 2)
#     return (xxt_avg - torch.matmul(x_avg_e, x_avg_t)).to(torch.float64)

# def get_Dtensor(diag):
#     # turn diag float to tensor
#     diag = torch.tensor(diag)
#     diag_tensor = torch.diag(diag)
#     Dtensor = diag_tensor @ diag_tensor.T / 2
#     return Dtensor.to(torch.float64)

# def get_Ab(sde):
#     # Breaks dynamics into Ax + b
#     minvf = sde.LM.mass_matrix_full.pinv() @ sde.LM.forcing_full

#     # add friction term
#     fric_term = - sde.friction * sde.dx_symbols.reshape(-1,1) / sde.N 
#     fric_term = np.vstack((np.zeros((sde.x_symbols.shape[0], 1)), fric_term))
#     minvf += fric_term

#     vecs = sp.Matrix(np.vstack((sde.x_symbols.reshape(-1,1), sde.dx_symbols.reshape(-1,1))))

#     n = vecs.shape[0]

#     A = np.zeros((minvf.shape[0], n))
#     b = np.zeros((minvf.shape[0], 1))

#     for i in range(minvf.shape[0]):
#         for j in range(n):
#             A[i, j] = -minvf[i].coeff(vecs[j])
#         b[i] = minvf[i].subs({vi: 0 for vi in vecs})
#     return A, b


# def get_Abqp(sde):
#     A, b = get_Ab(sde)
#     n = A.shape[1]

#     A_irr = np.zeros_like(A)
#     b_irr = np.zeros_like(b)
#     A_rev = np.zeros_like(A)
#     b_rev = np.zeros_like(b)

#     A_irr[:n//2, :n//2] = A[:n//2, :n//2]
#     A_irr[n//2:, n//2:] = A[n//2:, n//2:]
#     b_irr[:n//2] = b[:n//2]

#     A_rev[:n//2, n//2:] = A[:n//2, n//2:]
#     A_rev[n//2:, :n//2] = A[n//2:, :n//2]
#     b_rev[n//2:] = b[n//2:]
#     return A_irr, b_irr, A_rev, b_rev

# def getPit(ys, sde):
#     theta = get_theta(ys)
#     diag = sde.eta_cte * torch.ones(ys.shape[2])
#     Dtensor = get_Dtensor(diag)
#     Aq, bq, _, _ = get_Abqp(sde)
#     x_avg = get_x_avg(ys)
    
#     Aq = torch.tensor(Aq)
#     bq = torch.tensor(bq).squeeze()

#     vec = torch.vmap(lambda x: Aq @ x - bq)(x_avg)

#     t1 = torch.vmap(torch.trace)(Dtensor @ theta.inverse() - Aq)
#     t2 = torch.vmap(torch.trace)(Aq.T @ Dtensor.inverse() @ Aq @ theta - Aq)
#     t3 = torch.vmap(lambda x: x.T @ Dtensor.inverse() @ x)(vec)


#     print(t1.shape, t2.shape, t3.shape)

#     return t1 + t2 + t3
    
# def getPhit(ys, sde):
#     theta = get_theta(ys).to(torch.float64)
#     diag = sde.eta_cte * torch.ones(ys.shape[2])
#     Dtensor = get_Dtensor(diag).to(torch.float64)
#     _, _, Aq, bq = get_Abqp(sde)
#     x_avg = get_x_avg(ys).to(torch.float64)

#     Aq = torch.tensor(Aq)
#     bq = torch.tensor(bq).squeeze()

#     vec = torch.vmap(lambda x: Aq @ x - bq)(x_avg)

#     t1 = torch.vmap(torch.trace)(Aq.T @ Dtensor.inverse() @ Aq @ theta - Aq)
#     t2 = torch.vmap(lambda x: x.T @ Dtensor.inverse() @ x)(vec)

#     return t1 + t2

# def get_entropy_rates(ys, sde):
#     "Returns Pi(t), Phi(t), and dS/dt = Pi(t) - Phi(t)"
#     pit = getPit(ys, sde)
#     phit = getPhit(ys, sde)
#     return pit, phit, pit - phit