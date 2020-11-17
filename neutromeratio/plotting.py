import torch
import logging
import random
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .analysis import bootstrap_rmse_r

logger = logging.getLogger(__name__)
plt.style.use("seaborn-deep")


def plot_correlation_analysis(
    names: list,
    x_: list,
    y_: list,
    title: str,
    x_label: str,
    y_label: str,
    fontsize: int = 15,
    nsamples: int = 5000,
    yerror: list = [],
    mark_point_by_name=[],
):
    """Plot correlation between x and y.

    Parameters
    ----------
    df : pd.Dataframe
        the df contains columns with colum names 'names', 'x', 'y', 'y-error'
    title : str
        to put above plot. use '' (empty string) for no title.
    nsamples : int
        number of samples to draw for bootstrap
    """

    plt.figure(figsize=[8, 8], dpi=300)
    ax = plt.gca()
    ax.set_title(title, fontsize=fontsize)

    rmse, mae, r = bootstrap_rmse_r(np.array(x_), np.array(y_), 1000)

    plt.text(-9.0, 22.0, r"MAE$ = {}$".format(mae), fontsize=fontsize)
    plt.text(-9.0, 20.0, r"RMSE$ = {}$".format(rmse), fontsize=fontsize)
    plt.text(
        -9.0, 18.0, r"Nr of tautomer pairs$ = {}$".format(len(names)), fontsize=fontsize
    )

    if yerror:
        logger.info("Plotting with y-error bars")
        for X, Y, name, error in zip(x_, y_, names, yerror):
            ax.errorbar(
                X,
                Y,
                yerr=error,
                mfc="blue",
                mec="blue",
                ms=4,
                fmt="o",
                capthick=2,
                capsize=2,
                alpha=0.6,
                ecolor="red",
            )

    else:
        logger.info("Plotting without y-error bars")
        for X, Y, name in zip(x_, y_, names):
            if name in mark_point_by_name:
                ax.scatter(X, Y, color="red", s=13, alpha=0.6)
            else:
                ax.scatter(X, Y, color="blue", s=13, alpha=0.6)

    # draw lines +- 1kcal/mol
    ax.plot((-10.0, 25.0), (-10.0, 25.0), "k--", zorder=-1, linewidth=1.0, alpha=0.5)
    ax.plot((-9.0, 25.0), (-10.0, 24.0), "gray", zorder=-1, linewidth=1.0, alpha=0.5)
    ax.plot((-10.0, 24.0), (-9.0, 25.0), "gray", zorder=-1, linewidth=1.0, alpha=0.5)

    ax.plot((-10.0, 25.0), (0.0, 0.0), "r--", zorder=-1, linewidth=1.0, alpha=0.5)
    ax.plot((0.0, 0.0), (-10.0, 25.0), "r--", zorder=-1, linewidth=1.0, alpha=0.5)

    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    plt.tight_layout()
    plt.fill(
        0.0,
    )
    plt.subplots_adjust(bottom=0.3),  # left=1.3, right=0.3)
    # make sure that we plot a square
    ax.set_aspect("equal", "box")
    # color quadrants
    x = np.arange(0.01, 25, 0.1)
    y = -30  # np.arange(0.01,30,0.1)
    plt.fill_between(x, y, color="#539ecd", alpha=0.2)

    x = -np.arange(0.01, 25, 0.1)
    y = 30  # np.arange(0.01,30,0.1)

    plt.fill_between(x, y, color="#539ecd", alpha=0.2)
    plt.axvspan(-10, 0, color="grey")
    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)

    ax.set_xlim([-10, 25])
    ax.set_ylim([-10, 25])
    return plt


def plot_dist():
    fontsize = 17
    from neutromeratio.analysis import compute_kl_divergence, bootstrap_rmse_r
    import seaborn as sns

    sns.distplot(x_list, kde=True, rug=True, bins=15, label="befor optimization")
    sns.distplot(y_list, kde=True, rug=True, bins=15, label="befor optimization")
    rmse, mae, rho = bootstrap_rmse_r(np.array(x_list), np.array(y_list), 1000)
    kl = compute_kl_divergence(np.array(x_list), np.array(y_list))
    plt.text(8.0, 0.10, f"MAE$ = {mae}$", fontsize=fontsize)
    plt.text(8.0, 0.09, f"KL$ = {kl:.2f}$", fontsize=fontsize)
    plt.xlabel("$\Delta_{r}G_{solv}$", fontsize=fontsize)
    plt.ylabel("Probability", fontsize=fontsize)
    plt.show()
