from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_factor_panel(
    dates: int = 500,
    assets: int = 80,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-02", periods=dates, freq="B")
    names = [f"asset_{i:03d}" for i in range(assets)]

    size_exposure = rng.normal(0, 1, assets)
    sector_exposure = rng.normal(0, 1, assets)
    latent_quality = rng.normal(0, 1, assets)
    latent_value = rng.normal(0, 1, assets)
    latent_momentum = np.zeros(assets)

    signal_rows = []
    return_rows = []
    exposure_rows = []
    for _ in index:
        latent_momentum = 0.92 * latent_momentum + rng.normal(0, 0.35, assets)
        value = latent_value + rng.normal(0, 0.30, assets)
        quality = latent_quality + rng.normal(0, 0.25, assets)
        momentum = latent_momentum + rng.normal(0, 0.20, assets)
        composite = 0.45 * momentum + 0.30 * value + 0.25 * quality
        forward = 0.0015 * composite - 0.0004 * size_exposure + rng.normal(0, 0.018, assets)
        signal_rows.append(composite)
        return_rows.append(forward)
        exposure_rows.append(size_exposure + rng.normal(0, 0.03, assets))

    signal = pd.DataFrame(signal_rows, index=index, columns=names)
    forward_returns = pd.DataFrame(return_rows, index=index, columns=names)
    size = pd.DataFrame(exposure_rows, index=index, columns=names)
    return signal, forward_returns, size
