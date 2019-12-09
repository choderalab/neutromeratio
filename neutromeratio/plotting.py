import random
import matplotlib.pyplot as plt
from collections import OrderedDict
import logging
import pandas as pd
import numpy as np
import scipy.stats as scs

logger = logging.getLogger(__name__)

def plot_correlation_analysis(
    df:pd.DataFrame,
    title:str,
    x_label:str,
    y_label:str,
    nsamples=5000,
    mark_tautomer_names:list=[],
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

    plt.figure(figsize=[8,8], dpi=300)
    fontsize=15
    ax = plt.gca()
    ax.set_title(title, fontsize=fontsize)

    rmse, r = bootstrap_rmse_r(df, 1000)

    plt.text(-20.0 , 20.0, r"$\rho = {}$".format(r), fontsize=fontsize)
    plt.text(-20.0 , 17.0, r"RMSE$ = {}$".format(rmse), fontsize=fontsize)

    for X, Y, name in zip(df.x, df.y, df.names):
        ax.scatter(X, Y, color='blue', s=15, alpha=0.6)
        # mark tautomer pairs that behave strangly
        if name in mark_tautomer_names:
            ax.annotate(str(name), (X, Y), fontsize=10)

    # draw lines +- 1kcal/mol
    ax.plot((-22.0, 22.0), (-22.0, 22.0), "k--", zorder=-1, linewidth=1., alpha=0.5)
    ax.plot((-21.0, 22.0), (-22.0, 21.0), "gray", zorder=-1, linewidth=1., alpha=0.5)
    ax.plot((-22.0, 21.0), (-21.0, 22.0), "gray", zorder=-1, linewidth=1., alpha=0.5)
    
    ax.plot((-22.0, 22.0), (0.0, 0.0), "r--", zorder=-1, linewidth=1., alpha=0.5)
    ax.plot((0.0, 0.0), (-22.0, 22.0), "r--", zorder=-1, linewidth=1., alpha=0.5)

    ax.set_ylabel(x_label, fontsize=fontsize)
    ax.set_xlabel(y_label, fontsize=fontsize)
    plt.tight_layout()
    plt.fill(0.0, )
    plt.subplots_adjust(bottom=0.3), #left=1.3, right=0.3) 
    # make sure that we plot a square
    ax.set_aspect('equal', 'box')
    # color quadrants
    x = np.arange(0.01,30,0.1)
    y = -30 #np.arange(0.01,30,0.1)
    plt.fill_between(x, y, color='#539ecd', alpha=0.2)

    x = -np.arange(0.01,30,0.1)
    y = 30 #np.arange(0.01,30,0.1)

    plt.fill_between(x, y, color='#539ecd', alpha=0.2)
    ax.set_xlim([-22, 22])
    ax.set_ylim([-22, 22])
    return plt


def array_rmse(x1: np.ndarray, x2: np.ndarray) -> float:
    """Returns the root mean squared error between two arrays."""
    return np.sqrt(((x1 - x2) ** 2).mean())


def bootstrap_tautomer_exp_predict_results(original_df: pd.DataFrame) -> pd.DataFrame:
    """Perform empirical bootstrap over rows for correlation analysis."""
    size = original_df.shape[0]
    rows = np.random.choice(np.arange(size), size=size)
    return original_df.iloc[rows].copy()


def bootstrap_rmse_r(df: pd.DataFrame, nsamples: int):
    """Perform a bootstrap correlation analysis for a dataframe
    
    Parameters
    ----------
    df - the original pandas dataframe with preidcted data.
    nsamples - number of bootstrap samples to draw    
    """


    rmse_list = list()
    rs_list = list()
    for _ in range(nsamples):
        bootstrap_df = bootstrap_tautomer_exp_predict_results(df)
        exp = bootstrap_df.x
        pred = bootstrap_df.y
        rmse_list.append(array_rmse(exp, pred))
        rs_list.append(scs.pearsonr(exp, pred)[0])

    rmse_array = np.asarray(rmse_list)
    rs_array = np.asarray(rs_list)

    rmse = array_rmse(df.x, df.y)
    rs = scs.pearsonr(df.x, df.y)[0]

    return (
        BootstrapDistribution(rmse, rmse_array),
        BootstrapDistribution(rs, rs_array),
    )

class BootstrapDistribution:
    """Represents a bootstrap distribution, and allows calculation of useful statistics."""

    def __init__(self, sample_estimate: float, bootstrap_estimates: np.array) -> None:
        """
        Parameters
        ----------
        sample_estimate - estimated value from original sample
        bootstrap_estimates - estimated values from bootstrap samples
        """
        self._sample = sample_estimate
        self._bootstrap = bootstrap_estimates
        # approximation of δ = X - μ
        # denoted as δ* = X* - X
        self._delta = bootstrap_estimates - sample_estimate
        self._significance: int = 16
        return

    def __float__(self):
        return float(self._sample)

    def empirical_bootstrap_confidence_intervals(self, percent: float):
        """Return % confidence intervals.
        
        Uses the approximation that the variation around the center of the 
        bootstrap distribution is the same as the variation around the 
        center of the true distribution. E.g. not sensitive to a bootstrap
        distribution is biased in the median.

        Note
        ----
        Estimates are rounded to the nearest significant digit.
        """
        if not 0.0 < percent < 100.0:
            raise ValueError("Percentage should be between 0.0 and 100.0")
        elif 0.0 < percent < 1.0:
            raise ValueError("Received a fraction but expected a percentile.")

        a = (100.0 - percent) / 2
        upper = np.percentile(self._delta, a)
        lower = np.percentile(self._delta, 100 - a)
        return self._sig_figures(self._sample, self._sample - lower, self._sample - upper)

    def bootstrap_percentiles(self, percent: float):
        """Return percentiles of the bootstrap distribution.
        
        This assumes that the bootstrap distribution is similar to the real 
        distribution. This breaks down when the median of the bootstrap 
        distribution is very different from the sample/true variable median.

        Note
        ----
        Estimates are rounded to the nearest significant digit.
        """
        if not 0.0 < percent < 100.0:
            raise ValueError("Percentage should be between 0.0 and 100.0")
        elif 0.0 < percent < 1.0:
            raise UserWarning("Received a fraction but expected a percentage.")

        a = (100.0 - percent) / 2
        lower = np.percentile(self._bootstrap, a)
        upper = np.percentile(self._bootstrap, 100 - a)
        return self._sig_figures(self._sample, lower, upper)

    def standard_error(self) -> float:
        """Return the standard error for the bootstrap estimate."""
        # Is calculated by taking the standard deviation of the bootstrap distribution
        return self._bootstrap.std()

    def _sig_figures(self, mean: float, lower: float, upper: float, max_sig: int = 16):
        """Find the lowest number of significant figures that distinguishes
        the mean from the lower and upper bound.
        """
        i=16
        for i in range(1, max_sig):
            if (round(mean, i) != round(lower, i)) and (round(mean, i) != round(upper, i)):
                break
        self._significance = i
        return round(mean, i), round(lower, i), round(upper, i)

    def __repr__(self) -> str:
        try:
            return "{0:.{3}f}; [{1:.{3}f}, {2:.{3}f}]".format(*self.bootstrap_percentiles(95),
                                                          self._significance)
        except:
            return str(self.__dict__)

    def _round_float(self, flt):
        return (flt * pow(10, self._significance)) / pow(10, self._significance)

    def to_tuple(self, percent=95):
        """Return the mean, lower and upper percentiles rounded to significant digits"""
        mean, lower, upper = self.bootstrap_percentiles(percent)
        mean = self._round_float(mean)
        lower = self._round_float(lower)
        upper = self._round_float(upper)
        return mean, lower, upper
