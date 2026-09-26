# Purged walk-forward split audit

`quantfactor.purged_cv` validates that a factor backtest's expanding-window folds do not leak future label information into training. Each observation records when its features were available and when its forward-looking label became complete. The audit reconstructs fold membership and rejects label overlap, inadequate purge/embargo gaps, non-prefix training windows, non-contiguous test blocks, reused test observations and backward fold order.

Artifacts bind the point-in-time dataset, label definition and split configuration by SHA-256. Strict parsing rejects duplicate fields, unknown observations, duplicate membership, naive timestamps, excessive label horizons and resource-budget violations. Reports hash audit/fold identities and never emit raw observation IDs.

```bash
python -m quantfactor.purged_cv split-artifact.json --output split-report.json
```

CLI exits are `0` for acceptance, `2` for a well-formed policy rejection and `3` for malformed evidence. Report output uses atomic replacement.

## Trust boundaries and limitations

Passing proves only the supplied timestamp and membership contract. It does not verify that market features were genuinely point-in-time, that vendor revisions and corporate actions were handled correctly, or that the recorded label horizon matches the implemented target. SHA-256 binds evidence but does not authenticate its producer.

Purging prevents direct interval overlap; it does not remove serial dependence, selection bias, universe survivorship, multiple testing or regime instability. Purge and embargo gaps must be prespecified from the maximum information horizon. Cross-sectional data also require point-in-time asset membership and availability timestamps.

## Next integration step

Emit this artifact directly from the factor experiment runner, sign it with the dataset manifest, and require an accepted digest before performance metrics can enter the factor promotion report.
