import os 

import numpy as np
import pickle

import torch
import torchsde

from src.ff_springs import GS3DE
from src.entropy import get_entropy_rates, get_free_energy

from concurrent.futures import ProcessPoolExecutor, as_completed

# functions_1d = {
#     r'x^2': lambda x: x**2,
#     r'x^3': lambda x: x**3,
#     r'\sin(x)': lambda x: torch.sin(x),
#     r'\exp(x)': lambda x: torch.exp(x/np.pi),
# }

# functions_2d = {
#     r"\sin(x)+\cos(y)": lambda u: torch.sin(u[:, 0]) + torch.cos(u[:, 1]),
#     r"x \cdot y": lambda u: u[:, 0] * u[:, 1],
#     r"x^2 - y^2": lambda u: u[:, 0]**2 - u[:, 1]**2,
#     r"\exp(-x^2-y^2)": lambda u: torch.exp(- (u[:, 0]**2 + u[:, 1]**2)),
#     r"\log(1+|x|)-\log(1+|y|)": lambda u: torch.log1p(torch.abs(u[:, 0])) - torch.log1p(torch.abs(u[:, 1])),
#     r"\tanh(x \cdot y)": lambda u: torch.tanh(u[:, 0] * u[:, 1]),
#     r"\sin(x \cdot y)": lambda u: torch.sin(u[:, 0] * u[:, 1]),
#     r"\cos(x)+\sin(y)": lambda u: torch.cos(u[:, 0]) + torch.sin(u[:, 1]),
# }

def f_0(x):
    return torch.zeros_like(x)

def f_x(x):
    return x

def f_x2(x):
    return x**2

def f_x3(x):
    return x**3

def f_sin_x(x):
    return torch.sin(x)

def f_cos_x(x):
    return torch.cos(x)

def f_exp_x(x):
    return torch.exp(x/np.pi)

functions_1d = {
    r'0': f_0,
    r'x': f_x,
    r'x^2': f_x2,
    r'x^3': f_x3,
    r'\sin(x)': f_sin_x,
    r'\cos(x)': f_cos_x,
    r'\exp(x)': f_exp_x,
}

# Top-level functions for 2D functions
def f_sin_x_plus_cos_y(u):
    return torch.sin(u[:, 0]) + torch.cos(u[:, 1])

def f_x_dot_y(u):
    return u[:, 0] * u[:, 1]

def f_x2_minus_y2(u):
    return u[:, 0]**2 - u[:, 1]**2

def f_exp_neg_x2_y2(u):
    return torch.exp(- (u[:, 0]**2 + u[:, 1]**2))

def f_log_diff(u):
    return torch.log1p(torch.abs(u[:, 0])) - torch.log1p(torch.abs(u[:, 1]))

def f_tanh_x_dot_y(u):
    return torch.tanh(u[:, 0] * u[:, 1])

def f_sin_x_dot_y(u):
    return torch.sin(u[:, 0] * u[:, 1])

def f_cos_x_plus_sin_y(u):
    return torch.cos(u[:, 0]) + torch.sin(u[:, 1])

functions_2d = {
    r"\sin(x)+\cos(y)": f_sin_x_plus_cos_y,
    r"x \cdot y": f_x_dot_y,
    r"x^2 - y^2": f_x2_minus_y2,
    r"\exp(-x^2-y^2)": f_exp_neg_x2_y2,
    r"\log(1+|x|)-\log(1+|y|)": f_log_diff,
    r"\tanh(x \cdot y)": f_tanh_x_dot_y,
    r"\sin(x \cdot y)": f_sin_x_dot_y,
    r"\cos(x)+\sin(y)": f_cos_x_plus_sin_y,
}

def run_simulation(dim, n_points, temp, k, M, fric, target_function):
    if dim == 1:
        u_i = torch.linspace(0, 2 * torch.pi, n_points).reshape(-1, 1)
    elif dim == 2:
        x = torch.linspace(0, 2 * torch.pi, n_points)
        y = torch.linspace(0, 2 * torch.pi, n_points)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        u_i = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    else:
        raise ValueError("Only dimensions 1 or 2 are supported.")

    # Compute y_i using the provided target function
    y_i = target_function(u_i)
    if y_i.dim() == 1:
        y_i = y_i.unsqueeze(1)
    
    # Compute boundaries and number of labels from the data
    boundaries = (torch.min(u_i, dim=0).values, torch.max(u_i, dim=0).values)
    n_labels = y_i.shape[1]

    n_sticks = [1] if dim == 1 else [1, 1]
    print(f"Running simulation with parameters: n={n_points}, temp={temp}, k={k:.2e}, M={M:.2e}, friction={fric}")
    sde = GS3DE(n_sticks, boundaries, n_labels, friction=fric, temp=temp, k=k, M=M)
    sde.update_data(u_i, y_i)
    
    batch_size, t_size = 100, 10000
    ts = torch.linspace(0, 10, t_size)
    y0 = (torch.rand(size=(batch_size, sde.state_size))) * 3

    with torch.no_grad():
        thetas = torchsde.sdeint(sde, y0, ts, method='euler')
    
    # Compute costs and entropy rates
    # all_costs = torch.stack([torch.tensor([sde.loss(thetas[t, I, :], u_i, y_i) for t in  range(t_size)]) for I in range(batch_size)], dim=1).detach().numpy()
    all_costs = torch.stack([sde.loss(thetas[:, I, :], u_i, y_i) for I in range(batch_size)], dim=1).detach().numpy()
    costs_mean = all_costs.mean(axis=1)
    costs2_mean = (all_costs**2).mean(axis=1)
    costs_std = all_costs.std(axis=1)
    costs_var = all_costs.var(axis=1)
    pit, phit, dSdt = get_entropy_rates(thetas, sde)
    # dF = get_free_energy_rate(thetas, sde)
    FE = get_free_energy(ts, thetas, sde)
    

    # costs_mean = np.array(costs_mean)
    # costs2_mean = np.array(costs2_mean)
    # costs_std = np.array(costs_std)
    # costs_var = np.array(costs_var)
    # cost_snr = 10 * np.log10(costs2_mean / costs_var)
    
    results = {
        'ts': ts,
        'pit': pit,
        'phit': phit,
        'dsdt': dSdt,
        # 'df': dF,
        'fe': FE,
        # 'cost_snr': cost_snr,
        'costs_mean': costs_mean,
        'costs_std': costs_std,
        'all_costs': all_costs,
        'thetas': thetas,   
    }

    return results

def simulation_task(params):
    """
    Wrapper to run a simulation given a parameter tuple:
    (dim, k, M, fric, target_function)
    """
    dim, n, temp, k, M, fric, target_function, func_name = params
    # Run the simulation (this calls your run_simulation function)
    result = run_simulation(dim, n, temp, k, M, fric, target_function)
    return (dim, n, temp, k, M, fric, result)


if __name__ == '__main__':

    # k_values = np.logspace(-10, -30, 10)
    # M_values = np.logspace(-10, -30, 10)
    # fric_values = [0.2, 0.5, 1, 2, 5, 10, 15, 20, 100]
    # n_points = [10, 20, 40]

    # k_values = np.logspace(-10, -30, 3)
    # M_values = np.logspace(-10, -30, 3)
    # fric_values = [0.1, 1, 5]

    # function_names = [r'\sin(x)', r'\cos(x)', r'\sin(x)+\cos(y)', r'x \cdot y']
    # function_names = [r'0', r'x', r'x^2', r'x^3', r'\sin(x)', r'\cos(x)', r'\exp(x)']
    
    # k_values = [0.01, 0.05, 0.1, 0.5, 1]
    # M_values = [0.01, 0.05, 0.1, 0.5, 1]

    k_values = np.logspace(-10, -30, 10)
    M_values = np.logspace(-10, -30, 10)
    n_points = [20, 40, 80]
    fric_values = [4, 8, 16, 32]
    temps = [0.1, 1, 10]
    function_names = [r'x', r'x^2', r'x^3', r'\sin(x)', r'\cos(x)', r'\exp(x)']

    os.makedirs('data', exist_ok=True)
    result_file = 'data/entropy_results.pkl'

    for func_name in function_names:
        if func_name in functions_1d:
            dim = 1
            func = functions_1d[func_name]
        elif func_name in functions_2d:
            dim = 2
            func = functions_2d[func_name]
        else:
            raise ValueError("Invalid function name: " + func_name)

        tasks = [(dim, n, temp, k, M, fric, func, func_name)
                 for k, M in zip(k_values, M_values)
                 for fric in fric_values
                 for n in n_points
                 for temp in temps]

        results = {}
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(simulation_task, params) for params in tasks]
            for future in as_completed(futures):
                d, n, temp, k, M, fric, result = future.result()
                results[(n, temp, k, M, fric)] = result

        safe_func_name = func_name.replace('\\', '')
        data_to_append = {"function": safe_func_name, "results": results}

        # Append the data to the pickle file without loading previous data
        with open(result_file, 'ab') as f:
            pickle.dump(data_to_append, f)

        print(f"Results for {safe_func_name} appended to {result_file}.")
        print("Order of keys: (n, temp, k, M, fric)")