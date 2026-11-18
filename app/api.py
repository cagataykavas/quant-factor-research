from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from quantfactor.engine import FactorResearchEngine
from quantfactor.synthetic import synthetic_factor_panel

app = FastAPI(title="Quant Factor Research", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/demo")
def demo(
    seed: int = 42,
    dates: int = Query(default=300, ge=3, le=5000),
    assets: int = Query(default=60, ge=5, le=2000),
    cost_bps: float = Query(default=5.0, ge=0, le=1000),
    ic_hac_lag: int | None = Query(default=None, ge=0),
) -> dict[str, object]:
    if ic_hac_lag is not None and ic_hac_lag >= dates:
        raise HTTPException(
            status_code=422,
            detail="ic_hac_lag must be smaller than dates",
        )
    signal, forward, size = synthetic_factor_panel(
        dates=dates, assets=assets, seed=seed
    )
    return FactorResearchEngine().evaluate(
        signal,
        forward,
        size_exposure=size,
        cost_bps=cost_bps,
        ic_hac_lag=ic_hac_lag,
    )
