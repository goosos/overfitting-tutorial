"""
overfitting.py — Part 4 of "Build Your Own Quant Research System".

Backtest overfitting: the statistics of not fooling yourself when you
test many strategy variants and report the best one.

Two tools:
  deflated_sharpe() — Bailey & López de Prado (2014). Adjusts an observed
      Sharpe ratio for the fact that it was selected as the best of K trials.
  pbo() — simplified Probability of Backtest Overfitting (López de Prado,
      AFML ch. 11). Estimates how likely the in-sample winner is to
      disappoint out-of-sample, via combinatorial symmetric cross-validation.

Tutorial: https://goosos.com/backtest-overfitting-pbo
Code: https://github.com/goosos/overfitting-tutorial
"""
import numpy as np
import pandas as pd
from scipy import stats

_EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant


def _annualized_sharpe(returns, periods_per_year=252):
    """Annualized Sharpe ratio of a return series (risk-free = 0)."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return np.nan
    return (
        returns.mean() / returns.std(ddof=1)
        * np.sqrt(periods_per_year)
    )


def deflated_sharpe(observed_sr, trial_srs, n_obs,
                    periods_per_year=252, skew=None, kurt=None):
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Answers: "I tried K strategies and the best had Sharpe `observed_sr`.
    What's the probability this Sharpe is *genuinely* positive, after
    correcting for selection bias?"

    Parameters
    ----------
    observed_sr : float
        Annualized Sharpe ratio of the selected (best) strategy.
    trial_srs : array-like
        Annualized Sharpe ratios of ALL K trials (including the best).
        Used to estimate the expected Sharpe under the null.
    n_obs : int
        Number of return observations (bars) behind the Sharpe ratios.
    periods_per_year : int
        Annualization factor (252 = daily data). Used to convert n_obs
        to years for the test statistic.
    skew : float, optional
        Skewness of the selected strategy's returns. Estimated from
        returns if not given — but this function takes Sharpe ratios,
        not returns, so pass it explicitly when you can.
    kurt : float, optional
        Kurtosis of the selected strategy's returns (Pearson, normal=3).

    Returns
    -------
    dsr : float
        Probability (0–1) that the true Sharpe ratio is positive,
        after deflating for K trials. Rule of thumb: DSR > 0.95 means
        the Sharpe is statistically significant; DSR < 0.5 means the
        "great" backtest is likely luck.

    Notes
    -----
    DSR = Phi( (SR - SR0) * sqrt(T-1)
               / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2) )
    where SR0 is the expected Sharpe under the null, estimated from
    the K trials as:
        SR0 = sqrt(V) * ((1-g)*Phi^{-1}(1-1/K) + g*Phi^{-1}(1-1/(K*e)))
    V = variance of trial Sharpe ratios, g = Euler-Mascheroni constant.
    """
    trial_srs = np.asarray(trial_srs, dtype=float)
    trial_srs = trial_srs[~np.isnan(trial_srs)]
    k = len(trial_srs)
    if k < 2:
        raise ValueError("need at least 2 trial Sharpe ratios")
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")

    # Expected Sharpe under the null (best of K draws from noise)
    var_srs = trial_srs.var(ddof=1)
    if var_srs <= 0:
        return 1.0 if observed_sr > 0 else 0.0
    sr0 = np.sqrt(var_srs) * (
        (1 - _EULER_GAMMA) * stats.norm.ppf(1 - 1 / k)
        + _EULER_GAMMA * stats.norm.ppf(1 - 1 / (k * np.e))
    )

    # Test statistic, adjusted for non-normality
    skew = 0.0 if skew is None else skew
    kurt = 3.0 if kurt is None else kurt
    t_years = n_obs / periods_per_year
    denom = np.sqrt(
        max(1e-12, 1 - skew * observed_sr + (kurt - 1) / 4 * observed_sr ** 2)
    )
    stat = (observed_sr - sr0) * np.sqrt(max(t_years - 1, 1e-9)) / denom
    return float(stats.norm.cdf(stat))


def _cscv_partitions(n_periods, n_partitions):
    """
    Yield (is_idx, oos_idx) index pairs for Combinatorial Symmetric
    Cross-Validation: split the timeline into `n_partitions` contiguous
    blocks; each fold trains on half the blocks (in-sample) and tests on
    the other half (out-of-sample), cycling through combinations.
    """
    blocks = np.array_split(np.arange(n_periods), n_partitions)
    half = n_partitions // 2
    # Use a manageable subset of combinations for large n_partitions
    from itertools import combinations
    combos = list(combinations(range(n_partitions), half))
    # Cap at 16 folds to keep it fast; sample evenly if more
    if len(combos) > 16:
        step = len(combos) / 16
        combos = [combos[int(i * step)] for i in range(16)]
    for is_blocks in combos:
        is_idx = np.concatenate([blocks[b] for b in is_blocks])
        oos_blocks = [b for b in range(n_partitions) if b not in is_blocks]
        oos_idx = np.concatenate([blocks[b] for b in oos_blocks])
        yield np.sort(is_idx), np.sort(oos_idx)


def pbo(returns_matrix, n_partitions=8, periods_per_year=252):
    """
    Simplified Probability of Backtest Overfitting (PBO).

    Answers: "If I pick the best strategy in-sample, what's the
    probability it ranks *below median* out-of-sample?" PBO near 0.5+
    means your selection process is close to random — the in-sample
    winner is likely overfit.

    This is a simplified CSCV (López de Prado, AFML ch. 11): instead of
    the full combinatorial machinery with logit regression, we directly
    measure the rank degradation of the in-sample winner out-of-sample
    across folds, and report the fraction of folds where the IS winner
    lands below the OOS median.

    Parameters
    ----------
    returns_matrix : (n_strategies, n_periods) array-like
        Each row is one strategy variant's per-period returns.
    n_partitions : int
        Timeline blocks for CSCV (must be even, >= 4).
    periods_per_year : int
        Annualization factor for Sharpe ratios.

    Returns
    -------
    dict with keys:
        pbo : float
            Fraction of folds where the IS-optimal strategy ranked below
            the OOS median. > 0.5 suggests overfitting.
        n_folds : int
            Number of CSCV folds evaluated.
        is_sharpes / oos_sharpes : arrays
            Per-fold Sharpe of the IS winner, in-sample and out-of-sample.

    Notes
    -----
    Simplified vs the paper: the original PBO fits a logit on
    (IS rank, OOS rank) pairs and integrates; we use the direct
    below-median frequency, which converges to the same intuition with
    fewer moving parts. For production decisions, read AFML ch. 11.
    """
    r = np.asarray(returns_matrix, dtype=float)
    if r.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (n_strategies, n_periods)")
    n_strat, n_periods = r.shape
    if n_strat < 2:
        raise ValueError("need at least 2 strategies")
    if n_partitions < 4 or n_partitions % 2:
        raise ValueError("n_partitions must be even and >= 4")
    if n_partitions > n_periods:
        raise ValueError("n_partitions exceeds n_periods")

    below_median = 0
    n_folds = 0
    is_list, oos_list = [], []
    for is_idx, oos_idx in _cscv_partitions(n_periods, n_partitions):
        is_sr = np.array(
            [_annualized_sharpe(r[s, is_idx], periods_per_year)
             for s in range(n_strat)]
        )
        oos_sr = np.array(
            [_annualized_sharpe(r[s, oos_idx], periods_per_year)
             for s in range(n_strat)]
        )
        valid = ~(np.isnan(is_sr) | np.isnan(oos_sr))
        if valid.sum() < 2:
            continue
        is_sr, oos_sr = is_sr[valid], oos_sr[valid]
        winner = int(np.nanargmax(is_sr))
        # Rank of the IS winner in OOS (1 = best)
        oos_rank = int((oos_sr >= oos_sr[winner]).sum())
        median_rank = (len(oos_sr) + 1) / 2
        if oos_rank > median_rank:
            below_median += 1
        n_folds += 1
        is_list.append(float(is_sr[winner]))
        oos_list.append(float(oos_sr[winner]))

    pbo_value = below_median / n_folds if n_folds else np.nan
    return {
        "pbo": float(pbo_value),
        "n_folds": n_folds,
        "is_sharpes": np.array(is_list),
        "oos_sharpes": np.array(oos_list),
    }
