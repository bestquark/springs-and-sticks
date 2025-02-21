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

# Process each function's data one by one
for entry in all_entries:
    func_name = entry['function']
    results = entry['results']  # keys: (k, M, friction); value: (ts, pit, phit, dSdt, costs_mean, costs_std, all_costs)

    # Determine unique friction values in the data
    fric_set = {fr for (_, _, fr) in results.keys()}
    fric_list = sorted(list(fric_set))
    
    # Initialize dictionaries to store computed quantities for each friction value
    heat_values = {fr: [] for fr in fric_list}
    heat_values_diff = {fr: [] for fr in fric_list}
    snr_values = {fr: [] for fr in fric_list}
    k_m_values = []

    # Loop over each parameter combination in the results
    for (k, M, fr), sim_data in results.items():
        ts, pit, phit, dSdt, costs_mean, costs_std, all_costs = sim_data


        kb = 1.38064852e-23
        # kb = 1.0 

        # Calculate dq = phit * kb and then the heat Q via Simpson's rule
        ts = np.array(ts)
        dq = np.array(phit) * kb
        Q = simpson(dq, x=ts)

        # Calculate SNR using costs_mean and costs_std
        costs_mean = np.array(costs_mean)
        costs_std = np.array(costs_std)
        SNR = costs_mean / (costs_std+1e-10)
        # Use the last 40% of SNR data
        SNR = SNR[-int(0.4 * len(SNR)):]
        avg_SNR = np.mean(SNR)

        # Compute energy difference from all_costs
        all_costs = np.array(all_costs)
        max_energy = np.mean(np.max(all_costs, axis=1))
        min_energy = np.mean(np.min(all_costs, axis=1))

        # Store the computed quantities for this friction value
        heat_values[fr].append((k, M, Q))
        heat_values_diff[fr].append((k, M, max_energy - min_energy))
        snr_values[fr].append((k, M, avg_SNR))

        # Collect unique (k, M) pairs
        if (k, M) not in k_m_values:
            k_m_values.append((k, M))

    # Sort the unique (k, M) pairs by k for plotting (assuming k == M)
    k_m_values.sort(key=lambda x: x[0])
    
    # Set up the colormap for plotting different friction values
    cmap = inferno
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot heat (Q) and SNR for each friction value
    for indx, fr in enumerate(fric_list):
        color = cmap(indx / len(fric_list))
        # Sort the results by k to align with k_m_values
        heat_curve = [val for (_, _, val) in sorted(heat_values[fr], key=lambda x: x[0])]
        snr_curve = [val for (_, _, val) in sorted(snr_values[fr], key=lambda x: x[0])]
        k_vals = [k for (k, _) in sorted(k_m_values, key=lambda x: x[0])]

        ax1.plot(k_vals, heat_curve, label=rf'$\gamma={fr}$', linestyle='-', marker='o', color=color)
        ax2.plot(k_vals, snr_curve, label=rf'$\gamma={fr}$', linestyle='--', marker='x', color=color)

    # Formatting for the heat plot
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel(r'$|k|=|M|$', fontsize=20)
    ax1.set_ylabel(r'$Q \ (J)$', fontsize=20)

    # Formatting for the SNR plot
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$|k|=|M|$', fontsize=20)
    ax2.set_ylabel('SNR', fontsize=20)

    # Add legends to both subplots
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    heat_legend = ax1.legend(lines_1, labels_1, fontsize=20, title=r'$\gamma$', loc='upper left')
    snr_legend = ax2.legend(lines_2, labels_2, fontsize=20, title=r'$\gamma$', loc='upper left')
    plt.setp(heat_legend.get_title(), fontsize=20)
    plt.setp(snr_legend.get_title(), fontsize=20)

    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.tick_params(axis='both', which='minor', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='minor', labelsize=20)
    ax1.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax2.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle(f'Heat and SNR for function {func_name}', fontsize=24)

    os.makedirs('figs', exist_ok=True)
    plt.savefig(f'figs/heat_snr_{func_name}.pdf')
    plt.show()