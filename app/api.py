from __future__ import annotations

from fastapi import FastAPI

from quantfactor.engine import FactorResearchEngine
from quantfactor.synthetic import synthetic_factor_panel

app = FastAPI(title="Quant Factor Research", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/demo")
def demo(seed: int = 42, dates: int = 300, assets: int = 60, cost_bps: float = 5.0) -> dict[str, object]:
    signal, forward, size = synthetic_factor_panel(dates=dates, assets=assets, seed=seed)
    return FactorResearchEngine().evaluate(signal, forward, size_exposure=size, cost_bps=cost_bps)
