# Backtest Overfitting Tutorial (Part 4)

**Part 4 of [Build Your Own Quant Research System](https://goosos.com/tutorials/)** — the statistics of not fooling yourself when you test many strategy variants.

Your best backtest Sharpe of 1.67 looks great. But you ran 142 trials to find it. The **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014) and **PBO** (López de Prado, AFML ch. 11) tell you how much of that 1.67 is real.

## What you get

| File | Purpose |
|---|---|
| `overfitting.py` | `deflated_sharpe()` + simplified `pbo()` — the Part 4 toolkit module |
| `overfitting_demo.py` | End-to-end: 142-combo MA grid → DSR + PBO → honest verdict |
| `article.md` | Full tutorial text |

## Quick start

```bash
pip install -r requirements.txt
python overfitting_demo.py
```

Expected output (real SPY data, Oct 2023–Oct 2026):

```
Best in-sample: MA(20,25) Sharpe = 1.67 (best of 142 trials)
Deflated Sharpe Ratio: 0.840
Verdict: DSR 0.84 → marginal — could be luck, needs more evidence

PBO (simplified): 0.750 over 16 folds
Verdict: PBO > 0.5 — selection process looks overfit
```

## The toolkit

This tutorial contributes `overfitting.py` to [goosos/quant-toolkit](https://github.com/goosos/quant-toolkit):

```python
from quant_toolkit.overfitting import deflated_sharpe, pbo
```

## References

- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio.*
- López de Prado, M. *Advances in Financial Machine Learning* (2018), ch. 11.
