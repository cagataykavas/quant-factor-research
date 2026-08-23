from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import FactorResearchEngine
from .report import render_report
from .synthetic import synthetic_factor_panel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-sectional factor research reference project")
    parser.add_argument("--dates", type=int, default=500)
    parser.add_argument("--assets", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    signal, forward, size = synthetic_factor_panel(args.dates, args.assets, args.seed)
    result = FactorResearchEngine().evaluate(signal, forward, size_exposure=size, cost_bps=args.cost_bps)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "factor_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    render_report(result, args.output / "factor_report.html")
    print(json.dumps({
        "mean_daily_ic": result["mean_daily_ic"],
        "sharpe": result["backtest"]["sharpe"],
        "net_return": result["backtest"]["net_return"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
