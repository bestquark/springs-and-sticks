import numpy as np
import matplotlib.pyplot as plt


def evolution_plot(ts, samples, xlabel, ylabel, title=""):
    ts = ts.cpu()
    samples = samples.squeeze().t().cpu()
    plt.figure()
    coord = {0: r"$x_0$", 1: r"$x_1$", 2: r"$\dot{x}_0$", 3: r"$\dot{x}_1$"}
    for i, sample in enumerate(samples):
        plt.plot(ts, sample, marker="x", markersize=5, label=coord[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def plot_evolution_ud_vs_od(
    ts, ys_ud, ys_od, title=None, savefig_path="figs/over_under_damped.pdf"
):
    """
    Plot the evolution of the underdamped and overdamped systems.

    Args:
        ts (torch.Tensor): Time steps.
        ys_ud (torch.Tensor): Underdamped system states.
        ys_od (torch.Tensor): Overdamped system states.
    """
    ts = ts.cpu()
    ys_ud = ys_ud.squeeze().t().cpu()
    ys_od = ys_od.squeeze().t().cpu()
    plt.figure()
    plt.plot(ts, ys_ud[0], label=r"$x_0$ (underdamped)")
    plt.plot(ts, ys_ud[1], label=r"$x_1$ (underdamped)")
    plt.plot(ts, ys_od[0], label=r"$x_0$ (overdamped)")
    plt.plot(ts, ys_od[1], label=r"$x_1$ (overdamped)")
    plt.xlabel("$t$")
    plt.ylabel("$x_i$")
    plt.legend()
    plt.title(title if title else "Underdamped vs Overdamped")
    plt.savefig(savefig_path)
    plt.show()


def evolution_plot(ts, samples, xlabel, ylabel, n_pieces=1, title=""):
    ts = ts.cpu()
    samples = samples.squeeze().t().cpu()
    plt.figure()
    # coord = {
    #     i: rf"$x_{i}$" if i < n_pieces + 1 else rf"$\dot{{x}}_{i - n_pieces}$"
    #     for i in range(2 * n_pieces)
    # }
    for i, sample in enumerate(samples):
        plt.plot(ts, sample, marker="x", markersize=5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
