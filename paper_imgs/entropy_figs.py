import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os
import pickle

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Times New Roman",
    })
plt.rc('text.latex', preamble=r'\usepackage{amsmath, upgreek}')

def plot_vary_parameter(all_entries, param_name):
    num_funcs = len(all_entries)
    cmap = plt.cm.inferno

    fig, axes = plt.subplots(num_funcs, 2, figsize=(12, 5 * num_funcs), squeeze=False)

    for i, entry in enumerate(all_entries):
        func_name = entry.get('function', f"Function {i}")

        ns, temps, ks, Ms, frics, fes, avg_costs_late = [], [], [], [], [], [], []

        for key, results in entry['results'].items():
            n, temp, k, M, fric = key
            ns.append(n)
            temps.append(temp)
            ks.append(k)
            Ms.append(M)
            frics.append(fric)
            fes.append(results['fe'])
            avg_costs_late.append(np.mean(results['costs_mean'][-5000:]))

        ns, temps, ks = map(np.array, [ns, temps, ks])
        frics, fes, avg_costs_late = map(np.array, [frics, fes, avg_costs_late])

        if param_name == 'temp':
            param_array, fixed_n, fixed_fric = temps, np.min(ns), np.min(frics)
            fixed_mask = (ns == fixed_n) & (frics == fixed_fric)
            param_label = 'T'
        elif param_name == 'fric':
            param_array, fixed_n, fixed_temp = frics, np.min(ns), np.min(temps)
            fixed_mask = (ns == fixed_n) & (temps == fixed_temp)
            param_label = '\gamma'
        elif param_name == 'n_points':
            param_array, fixed_temp, fixed_fric = ns, np.min(temps), np.min(frics)
            fixed_mask = (temps == fixed_temp) & (frics == fixed_fric)
            param_label = 'N'
        else:
            raise ValueError("param_name must be 'temp', 'fric', or 'n_points'")

        ks = ks[fixed_mask]
        fes = fes[fixed_mask]
        avg_costs_late = avg_costs_late[fixed_mask]
        param_array = param_array[fixed_mask]

        unique_params = np.unique(param_array)
        colors = [cmap(j / len(unique_params)) for j in range(len(unique_params))]

        ax_cost, ax_costfe = axes[i, :]

        for j, p_val in enumerate(unique_params):
            mask = param_array == p_val
            sort_idx = np.argsort(ks[mask])

            k_sorted = ks[mask][sort_idx]
            fes_sorted = fes[mask][sort_idx]
            cost_sorted = avg_costs_late[mask][sort_idx]

            ax_cost.plot(k_sorted, cost_sorted, 'o-', color=colors[j], label=f"${p_val:.2f}$")
            ax_costfe.plot(-fes_sorted, cost_sorted, 'o-', color=colors[j], label=f"${p_val:.2f}$")

        for ax in [ax_cost, ax_costfe]:
            ax.set_xscale('log')
            ax.set_yscale('log')

        ax_cost.set_xlabel(r'$k$', fontsize=20)
        ax_cost.set_ylabel(r'$\langle L \rangle$', fontsize=20)
        ax_costfe.set_xlabel(r'$\Delta F$', fontsize=20)
        ax_costfe.set_yticklabels([])

        # set ticks larger
        ax_cost.tick_params(axis='both', which='major', labelsize=20)
        ax_costfe.tick_params(axis='both', which='major', labelsize=20)

        # add \ if the function does not begin with x
        func_name_m = func_name if func_name[0] == 'x' else f"\\{func_name}"
        # ax_cost.set_title(f"${func_name_m}$", fontsize=20)
        ax_cost.legend(title=f"${param_label}$", fontsize=20, title_fontsize=20)
        
        # for ax_cost set loc very right
        ax_cost.set_title(f"$f(x) = {func_name_m}$", fontsize=28, loc='right', x=1.2, y=1.05)
        

    norm = matplotlib.colors.LogNorm(vmin=unique_params.min(), vmax=unique_params.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', pad=0.02)
    # cbar.set_label(f"${param_label}$", fontsize=16, rotation=0, labelpad=15)
    # cbar.ax.tick_params(labelsize=14)

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(f'figs/{param_name}_vary_all.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':

    # Path to the pickle file with appended results
    result_file = 'data/entropy_results_4sticks.pkl'

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

    plot_vary_parameter(all_entries, 'temp')
    plot_vary_parameter(all_entries, 'fric')
    plot_vary_parameter(all_entries, 'n_points')