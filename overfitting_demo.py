"""
overfitting_demo.py — Part 4 companion code.

Reality check: take Part 1's MA-crossover parameter grid, compute the
Deflated Sharpe Ratio and (simplified) PBO for the best-looking result.

Run:  python overfitting_demo.py
Needs: vectorbt, yfinance, pandas, numpy, matplotlib, scipy
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import vectorbt as vbt

from overfitting import deflated_sharpe, pbo

SYMBOL = "SPY"
COMMISSION = 0.001  # 0.1% per side


def load_data():
    """3 years of daily SPY closes (same as Part 1)."""
    import yfinance as yf
    df = yf.download(SYMBOL, period="3y", interval="1d",
                     auto_adjust=True, progress=False)
    close = df["Close"].iloc[:, 0] if df["Close"].ndim > 1 else df["Close"]
    close = close.dropna()
    print(f"Data: {len(close)} daily bars of {SYMBOL} "
          f"({close.index[0].date()} -> {close.index[-1].date()})")
    return close


def main():
    price = load_data()
    n = len(price)

    # Grid: fast 5..50 step 5, slow 20..100 step 5, fast < slow
    # ~80 combos — enough to demonstrate selection bias, fast to run
    fast_windows = np.arange(5, 55, 5)
    slow_windows = np.arange(20, 105, 5)
    combos = [(f, s) for f in fast_windows for s in slow_windows if f < s]
    print(f"Grid: {len(combos)} (fast, slow) combinations")

    sharpes = []
    returns_list = []
    labels = []
    for fast, slow in combos:
        fast_ma = vbt.MA.run(price, window=fast)
        slow_ma = vbt.MA.run(price, window=slow)
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        # No lookahead: trade on next bar's signal
        entries = entries.vbt.signals.fshift(1)
        exits = exits.vbt.signals.fshift(1)
        pf = vbt.Portfolio.from_signals(
            price, entries, exits, freq="1D",
            fees=COMMISSION, init_cash=10000,
        )
        sr = pf.sharpe_ratio()
        sharpes.append(float(sr) if not np.isnan(sr) else -99)
        # Per-period strategy returns for PBO
        rets = pf.returns()
        returns_list.append(rets.values)
        labels.append(f"({fast},{slow})")

    sharpes = np.array(sharpes)
    # Filter out degenerate (no-trade) results for DSR
    valid = sharpes > -99
    trial_srs = sharpes[valid]
    k = len(trial_srs)
    best_idx = int(np.argmax(trial_srs))
    best_sr = float(trial_srs[best_idx])
    best_label = labels[np.where(valid)[0][best_idx]]
    print(f"\nBest in-sample: MA{best_label} Sharpe = {best_sr:.2f} "
          f"(best of {k} trials)")

    # --- Deflated Sharpe Ratio ---
    dsr = deflated_sharpe(best_sr, trial_srs, n_obs=n)
    print(f"Deflated Sharpe Ratio: {dsr:.3f}")
    if dsr > 0.95:
        verdict = "significant — likely a real edge"
    elif dsr > 0.5:
        verdict = "marginal — could be luck, needs more evidence"
    else:
        verdict = "not significant — the 'best' backtest is likely luck"
    print(f"Verdict: DSR {dsr:.2f} → {verdict}")

    # --- PBO (simplified CSCV) ---
    # Use the valid strategies' returns
    rets_matrix = np.array([returns_list[i] for i in np.where(valid)[0]])
    # Trim to common length (drop NaN columns)
    mask = ~np.isnan(rets_matrix).any(axis=0)
    rets_matrix = rets_matrix[:, mask]
    print(f"\nPBO input: {rets_matrix.shape[0]} strategies × "
          f"{rets_matrix.shape[1]} periods")
    res = pbo(rets_matrix, n_partitions=8)
    print(f"PBO (simplified): {res['pbo']:.3f} over {res['n_folds']} folds")
    if res["pbo"] > 0.5:
        print("Verdict: PBO > 0.5 — selection process looks overfit")
    else:
        print("Verdict: PBO <= 0.5 — IS winner holds up reasonably OOS")

    # --- Diagram 1: Sharpe distribution with DSR threshold ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(trial_srs, bins=30, color="#4a90d9", alpha=0.7,
            edgecolor="white", linewidth=0.5)
    ax.axvline(best_sr, color="#e74c3c", linestyle="--", linewidth=2,
               label=f"Best in-sample: {best_sr:.2f}")
    # Expected Sharpe under null (best of K noise draws)
    var_srs = trial_srs.var(ddof=1)
    from scipy import stats as _st
    _g = 0.5772156649015329
    sr0 = np.sqrt(var_srs) * (
        (1 - _g) * _st.norm.ppf(1 - 1 / k)
        + _g * _st.norm.ppf(1 - 1 / (k * np.e))
    )
    ax.axvline(sr0, color="#f39c12", linestyle=":", linewidth=2,
               label=f"Expected under null: {sr0:.2f}")
    ax.set_xlabel("Annualized Sharpe ratio")
    ax.set_ylabel("Number of parameter combos")
    ax.set_title(
        f"Sharpe distribution across {k} MA combinations\n"
        f"DSR = {dsr:.2f} — P(best is real) after correcting for {k} trials"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig("sharpe_dist.png", dpi=150)
    print("\nSaved: sharpe_dist.png")

    # --- Diagram 2: IS vs OOS Sharpe per PBO fold ---
    fig, ax = plt.subplots(figsize=(10, 5))
    is_s = res["is_sharpes"]
    oos_s = res["oos_sharpes"]
    x = np.arange(len(is_s))
    width = 0.35
    ax.bar(x - width / 2, is_s, width, label="IS winner (in-sample)",
           color="#4a90d9")
    ax.bar(x + width / 2, oos_s, width, label="Same strategy (out-of-sample)",
           color="#e74c3c", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("CSCV fold")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title(
        f"In-sample winner vs out-of-sample reality "
        f"(PBO = {res['pbo']:.2f})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig("pbo_folds.png", dpi=150)
    print("Saved: pbo_folds.png")


if __name__ == "__main__":
    main()
