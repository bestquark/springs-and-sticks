import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import inferno
from matplotlib.ticker import LogLocator
from scipy.integrate import simpson

# Path to the pickle file with appended results
result_file = 'data/entropy_results.pkl'

# Load all appended pickle objects from the file
all_entries = []
if os.path.exists(result_file):
    with open(result_file, 'rb') as f:
        while True:
            try:
                entry = pickle.load(f)
                all_entries.append(entry)
            except EOFError:
                break
else:
    raise FileNotFoundError(f"{result_file} does not exist.")

def plot_vary_parameter(all_entries, param_name):
    """
    Creates a transposed figure with each column corresponding to a function.
    
    The top row shows free energy plots (-fe vs. k) and the bottom row shows mean error plots (avg cost vs. k).
    
    For each function, only the entries where the non-varying parameters are fixed to a default value
    (the first sorted unique value) are used. The parameter specified by param_name is varied.
    The other two parameters (from n_points, temp, and fric) are fixed.
    
    Parameters:
        all_entries : list of dict
            Each dict must have:
              - 'function': a string representing the function (e.g. r'$x$', r'$x^2$')
              - 'results': a dict with keys that are tuples (n, temp, k, M, fric) and values containing:
                     'fe'         : free energy value
                     'all_costs'  : list/array of cost values (we compute the mean over the last 5000 entries)
        param_name : str
            Which parameter to vary. Must be one of: 'temp', 'fric', or 'n_points'
    """
    num_funcs = len(all_entries)
    fig, axes = plt.subplots(2, num_funcs, figsize=(5*num_funcs, 8), squeeze=False)
    cmap = plt.cm.inferno

    for i, entry in enumerate(all_entries):
        func_name = entry.get('function', f"Function {i}")
        
        # Initialize lists for parameters and data
        ns, temps, ks, Ms, frics = [], [], [], [], []
        fes, avg_costs_late = [], []
        
        for key, results in entry['results'].items():
            # key is assumed to be (n, temp, k, M, fric)
            n, temp, k, M, fric = key
            ns.append(n)
            temps.append(temp)
            ks.append(k)
            Ms.append(M)
            frics.append(fric)
            fes.append(results['fe'])
            avg_costs_late.append(np.mean(results['all_costs'][-5000:]))
        
        # Convert to numpy arrays for easier filtering and sorting
        ns = np.array(ns)
        temps = np.array(temps)
        ks = np.array(ks)
        fes = np.array(fes)
        avg_costs_late = np.array(avg_costs_late)
        frics = np.array(frics)
        
        # Depending on which parameter is varied, fix the other two.
        if param_name == 'temp':
            param_array = temps
            fixed_n = np.sort(np.unique(ns))[0]
            fixed_fric = np.sort(np.unique(frics))[0]
            fixed_mask = (ns == fixed_n) & (frics == fixed_fric)
        elif param_name == 'fric':
            param_array = frics
            fixed_n = np.sort(np.unique(ns))[0]
            fixed_temp = np.sort(np.unique(temps))[0]
            fixed_mask = (ns == fixed_n) & (temps == fixed_temp)
        elif param_name == 'n_points':
            param_array = ns
            fixed_temp = np.sort(np.unique(temps))[0]
            fixed_fric = np.sort(np.unique(frics))[0]
            fixed_mask = (temps == fixed_temp) & (frics == fixed_fric)
        else:
            raise ValueError("param_name must be one of 'temp', 'fric', or 'n_points'")
        
        # Apply the mask to all arrays to filter out only the fixed combination
        ks = ks[fixed_mask]
        fes = fes[fixed_mask]
        avg_costs_late = avg_costs_late[fixed_mask]
        param_array = param_array[fixed_mask]
        
        # Get unique values for the varying parameter and assign colors
        unique_params = np.unique(param_array)
        colors = [cmap(i/len(unique_params)) for i in range(len(unique_params))]
        
        # Get the axes for free energy (top) and mean error (bottom) for this function
        ax_fe = axes[0, i]
        ax_cost = axes[1, i]
        
        # Plot each curve for each unique value of the varying parameter
        for j, p_val in enumerate(unique_params):
            mask = (param_array == p_val)
            sort_idx = np.argsort(ks[mask])
            k_sorted = ks[mask][sort_idx]
            fes_sorted = fes[mask][sort_idx]
            cost_sorted = avg_costs_late[mask][sort_idx]
            
            ax_fe.plot(k_sorted, -fes_sorted, 'o-', color=colors[j],
                       label=f"{param_name}={p_val}")
            ax_cost.plot(k_sorted, cost_sorted, 'o-', color=colors[j],
                         label=f"{param_name}={p_val}")
            # ax_cost.plot(-fes_sorted, cost_sorted, 'o-', color=colors[j],
            #                 label=f"{param_name}={p_val}")
            # ax_cost.plot(k_sorted,-cost_sorted/fes_sorted, 'o-', color=colors[j],
            #              label=f"{param_name}={p_val}")
        
        # Set log scales on both axes
        for ax in (ax_fe, ax_cost):
            ax.set_xscale('log')
            ax.set_yscale('log')
        
        # Remove redundant labels: only left column gets y-labels; only bottom row gets x-labels
        if i == 0:
            ax_fe.set_ylabel('$-\\Delta F$')
            ax_cost.set_ylabel('$\\langle L \\rangle$')
        else:
            ax_fe.set_ylabel('')
            ax_cost.set_ylabel('')
        ax_fe.set_xlabel('')
        ax_cost.set_xlabel('$k$')
        
        # Set the title for the column to the function name
        ax_fe.set_title(f"${func_name}$")
        
        # Optionally remove legends if they are redundant
        ax_fe.legend(fontsize='small')
        ax_cost.legend(fontsize='small')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f'figs/free_energy_{param_name}.pdf')

if __name__ == '__main__':
    plot_vary_parameter(all_entries, 'temp')
    plot_vary_parameter(all_entries, 'fric')
    plot_vary_parameter(all_entries, 'n_points')