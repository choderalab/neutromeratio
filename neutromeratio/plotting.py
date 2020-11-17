import torch
import logging
import random
from collections import OrderedDict
import pickle

import matplotlib.pyplot as plt
from simtk import unit
import numpy as np
import pandas as pd
from .analysis import bootstrap_rmse_r, _get_exp_results
from .constants import _get_names, kT

logger = logging.getLogger(__name__)
plt.style.use("seaborn-deep")


def analyse_optimization(base_path: str):
    import seaborn as sns

    plt.style.use("seaborn-deep")
    from scipy.stats import entropy
    from neutromeratio.analysis import compute_kl_divergence, bootstrap_rmse_r

    all_names = _get_names()
    # load the model perofrmance before optimization on the training/validation set
    results_before_retraining = pickle.load(
        open(base_path + "/results_before_training.pickle", "rb+")
    )
    # results_before_retraining is a dictionary of lists, the keys are the mol names, the values are a list of [ddG, validation/training]

    # load the ddG values for the test set before and after training
    results_test_set = pickle.load(
        open(base_path + "/results_AFTER_training_for_test_set.pickle", "rb+")
    )
    # results_test_set is a dictionary of lists, the key are the mol anmes, the values are [experimental_ddG, before_optimization_ddG, after_optimization_ddG]

    # get everythin in four lists
    original_ = []
    reweighted_ = []
    exp_ = []
    names_ = []

    exp_results = _get_exp_results()

    for n in results_test_set:
        original, reweighted = results_test_set[n]
        exp = (
            exp_results[n]["energy"] * unit.kilocalorie_per_mole
        )  # already in kcal/mol
        names_.append(n)
        original_.append((original * kT).value_in_unit(unit.kilocalorie_per_mole))
        reweighted_.append((reweighted * kT).value_in_unit(unit.kilocalorie_per_mole))
        exp_.append(exp)

    # plot the distribution of ddG before and after optimization

    sns.set(color_codes=True)
    plt.figure(figsize=[8, 8], dpi=300)
    fontsize = 25
    delta_exp_original = np.array(exp_) - np.array(original_)
    kl = compute_kl_divergence(np.array(exp_), np.array(original_))
    rmse, mae, rho = bootstrap_rmse_r(np.array(exp_), np.array(original_), 1000)
    plt.text(
        -28.0,
        0.175,
        f"RMSE$ = {rmse}$",
        fontsize=fontsize,
        color=sns.xkcd_rgb["denim blue"],
    )
    plt.text(
        -28.0,
        0.16,
        f"MAE$ = {mae}$",
        fontsize=fontsize,
        color=sns.xkcd_rgb["denim blue"],
    )
    plt.text(
        -28.0,
        0.145,
        f"KL$ = {kl:.2f}$",
        fontsize=fontsize,
        color=sns.xkcd_rgb["denim blue"],
    )

    sns.distplot(
        delta_exp_original,
        kde=True,
        rug=True,
        bins=15,
        label="ANI1ccx native",
        color=sns.xkcd_rgb["denim blue"],
    )
    delta_exp_reweighted = np.array(exp_) - np.array(reweighted_)
    rmse, mae, rho = bootstrap_rmse_r(np.array(exp_), np.array(reweighted_), 1000)
    kl = compute_kl_divergence(np.array(exp_), np.array(reweighted_))
    plt.text(
        -28.0,
        0.12,
        f"RMSE$ = {rmse}$",
        fontsize=fontsize,
        color=sns.xkcd_rgb["pale red"],
    )
    plt.text(
        -28.0,
        0.105,
        f"MAE$ = {mae}$",
        fontsize=fontsize,
        color=sns.xkcd_rgb["pale red"],
    )
    plt.text(
        -28.0,
        0.09,
        f"KL$ = {kl:.2f}$",
        fontsize=fontsize,
        color=sns.xkcd_rgb["pale red"],
    )
    sns.distplot(
        delta_exp_reweighted,
        kde=True,
        rug=True,
        bins=15,
        label="ANI1ccx optimized",
        color=sns.xkcd_rgb["pale red"],
    )
    plt.legend(fontsize=fontsize - 5)
    plt.ylabel("Probability", fontsize=fontsize)
    plt.xlabel(
        "$\Delta_{r} G_{solv}^{exp} -  \Delta_{r} G_{vac}^{calc}$ [kcal/mol]",
        fontsize=fontsize,
    )
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.yticks([])
    plt.show()


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
