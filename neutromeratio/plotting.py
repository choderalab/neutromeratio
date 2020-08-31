import logging
import random
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .analysis import bootstrap_rmse_r

logger = logging.getLogger(__name__)
plt.style.use('seaborn-deep')


def plot_correlation_analysis(
    df: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    nsamples=5000,
    mark_tautomer_names: list = [],
    remove_list :list = []
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
    mark_tautomer_names : list of str
        mark specific tautomers with labels in the plot
    """

    plt.figure(figsize=[8, 8], dpi=300)
    fontsize = 15
    ax = plt.gca()
    ax.set_title(title, fontsize=fontsize)

    for n in remove_list:
        df = df[df.names != n]

    rmse, r = bootstrap_rmse_r(df, 1000)

    plt.text(-29.0, 27.0, r"$\rho = {}$".format(round(float(r), 2)), fontsize=fontsize)
    plt.text(-29.0, 24.0, r"RMSE$ = {}$".format(rmse), fontsize=fontsize)
    plt.text(-29.0, 21.0, r"Nr of tautomer pairs$ = {}$".format(len(df['names'])), fontsize=fontsize)

    try:
        logger.info('Plotting with y-error bars')   
        for X, Y, name, error in zip(df.x, df.y, df.names, df.yerror):
            ax.errorbar(X, Y, yerr=error,  mfc='blue', ms=3, fmt='o', capthick=2, alpha=0.6, ecolor='r')
            # mark tautomer pairs that behave strangly
            if name in mark_tautomer_names:
                ax.annotate(str(name), (X, Y), fontsize=10)

    except AttributeError:
        logger.info('Plotting without y-error bars')   
        for X, Y, name in zip(df.x, df.y, df.names):
            ax.scatter(X, Y, color='blue', s=13, alpha=0.6)
            # mark tautomer pairs that behave strangly
            if name in mark_tautomer_names:
                ax.annotate(str(name), (X, Y), fontsize=10)
        logger.info('Plotting without y-error bars')   

    # draw lines +- 1kcal/mol
    ax.plot((-32.0, 32.0), (-32.0, 32.0), "k--", zorder=-1, linewidth=1., alpha=0.5)
    ax.plot((-31.0, 32.0), (-32.0, 31.0), "gray", zorder=-1, linewidth=1., alpha=0.5)
    ax.plot((-32.0, 31.0), (-31.0, 32.0), "gray", zorder=-1, linewidth=1., alpha=0.5)
    
    ax.plot((-32.0, 32.0), (0.0, 0.0), "r--", zorder=-1, linewidth=1., alpha=0.5)
    ax.plot((0.0, 0.0), (-32.0, 32.0), "r--", zorder=-1, linewidth=1., alpha=0.5)

    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    plt.tight_layout()
    plt.fill(0.0, )
    plt.subplots_adjust(bottom=0.3),  # left=1.3, right=0.3)
    # make sure that we plot a square
    ax.set_aspect('equal', 'box')
    # color quadrants
    x = np.arange(0.01, 30, 0.1)
    y = -30  # np.arange(0.01,30,0.1)
    plt.fill_between(x, y, color='#539ecd', alpha=0.2)

    x = -np.arange(0.01, 30, 0.1)
    y = 30  # np.arange(0.01,30,0.1)

    plt.fill_between(x, y, color='#539ecd', alpha=0.2)
    ax.set_xlim([-30, 30])
    ax.set_ylim([-30, 30])
    return plt


