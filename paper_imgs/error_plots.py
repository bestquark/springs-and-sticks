import torch
import numpy as np
import torchsde
import matplotlib.pyplot as plt
import multiprocessing

from model import GS3DE

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Times New Roman",
    })
plt.rc('text.latex', preamble=r'\usepackage{amsmath, upgreek}')

def f_x2(x):
    return x**2

def f_x3(x):    
    return x**3

def f_sin(x):
    return torch.sin(x)

def f_exp(x):
    return torch.exp(x/np.pi)



def target_functions():
    return {
        'x^2': f_x2,
        'x^3': f_x3,
        'sin(x)': f_sin,
        'exp(x)': f_exp,
    }

def compute_errors(function_name, function, n_sticks_list, u_i):
    errors, errors_std = [], []
    y_i = function(u_i)
    boundaries = (torch.min(u_i, dim=0).values, torch.max(u_i, dim=0).values)
    n_labels = y_i.shape[0]
    
    for ns in n_sticks_list:
        print(f"Processing {function_name} with {ns} sticks")
        batch_size, t_size = 8, 100
        sde = GS3DE(ns, boundaries, n_labels, friction=10, temp=0.01, k=1, M=1)
        sde.update_data(u_i, y_i)
        ts = torch.linspace(0, 100, t_size)
        theta0 = (torch.rand(size=(batch_size, sde.state_size)) - 0.5) * 3
        
        with torch.no_grad():
            thetas = torchsde.sdeint(sde, theta0, ts, method='euler')
        
        x_linspace = torch.linspace(0, 2*np.pi, 1000)
        y_pred = np.array([
            [sde.num_y_prediction(x, thetas[-1, i, :]) for x in x_linspace]
            for i in range(batch_size)
        ])
        final_errors = np.sum(np.abs(y_pred.squeeze() - function(x_linspace).detach().numpy()), axis=1) * (2*np.pi/1000)
        
        errors.append(np.mean(final_errors))
        errors_std.append(np.std(final_errors))
    
    return function_name, np.array(errors), np.array(errors_std)

def plot_errors(n_sticks_list, errors_dict, errors_std_dict):
    from scipy.stats import linregress  

    plt.figure()
    
    markers = ['o', 's', '^', 'd']  # Different markers for each function
    colors = plt.cm.inferno(np.linspace(0, 1, len(errors_dict)+1))  # Different colors for each function
    
    for i, (func_name, errors) in enumerate(errors_dict.items()):
        err = errors_std_dict[func_name]
        
        # Perform linear regression on log-log data
        log_x = np.log10(n_sticks_list)
        log_y = np.log10(errors)
        slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
        
        # Update function name with slope and its standard error
        if func_name == 'sin(x)':
            func_name = r'\sin(x)'
        elif func_name == 'exp(x)':
            func_name = r'\exp(x/\pi)'
        
        label = f"${func_name}$ [{slope:.2f} ± {std_err:.2f}]"
        
        # Plot data with error bars
        plt.errorbar(n_sticks_list, errors, yerr=err, capsize=5,
                     color=colors[i], marker=markers[i], linestyle='-', label=label)
    
    plt.xlabel(r'$N_{\text{s}}$', fontsize=20)
    plt.ylabel(r'$E$', fontsize=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(np.arange(min(n_sticks_list), max(n_sticks_list) + 1, 1), fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.legend(
        title="Function [Slope ± Std.]", title_fontsize=18, alignment="left",
        framealpha=0.8, handlelength=1.5, handleheight=1.0, fontsize=18,
        loc='center left', bbox_to_anchor=(1, 0.5)
    )

    plt.savefig('figs/error_analysis_multiple.pdf', bbox_inches='tight')

def main():
    n_sticks_list = [3, 4, 5, 6, 7, 8, 9]
    torch.manual_seed(0)
    u_i = torch.linspace(0, 2*torch.pi, 20).reshape(1, -1)
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(compute_errors, [(func_name, func, n_sticks_list, u_i) for func_name, func in target_functions().items()])
    
    errors_dict = {func_name: errors for func_name, errors, _ in results}
    errors_std_dict = {func_name: errors_std for func_name, _, errors_std in results}
    
    # Save dicts
    np.save('data/errors_dict.npy', errors_dict)
    np.save('data/errors_std_dict.npy', errors_std_dict)

    # Load dicts
    errors_dict = np.load('data/errors_dict.npy', allow_pickle=True).item()
    errors_std_dict = np.load('data/errors_std_dict.npy', allow_pickle=True).item()
    
    plot_errors(n_sticks_list, errors_dict, errors_std_dict)

if __name__ == "__main__":
    main()
