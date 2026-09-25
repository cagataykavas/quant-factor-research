# Execution-capacity audit

The baseline factor backtest applies a transparent constant-bps transaction
cost. That is useful for sensitivity analysis, but it cannot determine whether a
target book is executable at a stated capital level. `audit_execution_capacity`
adds that admission boundary without claiming access to real market liquidity.

For every timezone-aware rebalance timestamp it consumes aligned matrices of:

- target portfolio weights;
- average daily dollar volume (ADV);
- executable half-spread estimates in basis points; and
- optional annualized short-borrow rates in basis points.

The first target is traded from cash. Subsequent turnover uses every asset in
the governed universe, so closing a name that leaves the active book is not
silently omitted. For trade weight `|Δw|` and requested capital `C`, the audit
uses:

```text
trade notional = |Δw| × C
participation  = trade notional / ADV
cost bps       = |Δw| × (commission + half spread + impact × √participation)
```

Short borrow is added separately as `short weight × annual borrow bps / 252`.
The capacity for a period is the largest capital for which every non-zero trade
stays below the configured participation ceiling.

## Operational contract

Inputs fail closed unless timestamps are unique, increasing and timezone-aware;
asset identifiers and matrix alignment are exact; every value is finite; ADV is
positive; costs are non-negative; and each target satisfies gross leverage,
dollar neutrality and active-name policies. Period and universe budgets bound
work before any calculation begins.

The JSON-ready report includes aggregate capacity, participation, turnover and
cost diagnostics; bounded period evidence; stable finding codes; a SHA-256
identity for all normalized inputs; and a separate digest for the decision
artifact. The CLI returns `0` for a completed audit, `2` for a policy rejection
when `--require-pass` is supplied, and `1` for malformed evidence.

```bash
python -m quantfactor.execution_audit evidence.json --require-pass
```

## Limitations

The square-root impact formula is an explicit approximation, not an execution
simulator. ADV, spread and borrow inputs must be point-in-time and should cover
the actual execution venue and rebalance window. The audit does not model order
book depth, intraday volume curves, volatility-dependent impact, partial fills,
borrow recalls, taxes, financing, crossed orders, corporate actions or signal
decay. Its capacity is a policy-bound estimate, not guaranteed deployable AUM.

The next production step is calibrating impact coefficients from broker fills
and storing the liquidity snapshot version with each research run.
