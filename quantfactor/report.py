from __future__ import annotations

from pathlib import Path
from typing import Any


def render_report(result: dict[str, Any], output: str | Path) -> Path:
    bt = result["backtest"]
    cost_rows = "".join(
        f"<tr><td>{cost} bps</td><td>{float(value):.2%}</td></tr>"
        for cost, value in result["transaction_cost_sensitivity"].items()
    )
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quant Factor Research</title><style>
body{{background:#08101c;color:#eef5ff;font-family:Inter,system-ui,sans-serif;margin:0;padding:32px}}main{{max-width:1050px;margin:auto}}.muted{{color:#98a8c0}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}}.card{{background:#121c2d;border:1px solid #29384f;border-radius:14px;padding:18px}}.big{{font-size:26px;font-weight:800}}
table{{width:100%;border-collapse:collapse;background:#121c2d}}th,td{{padding:10px;border-bottom:1px solid #29384f;text-align:left}}th{{color:#b0c4ff}}@media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main><p class="muted">Synthetic cross-sectional research · not investment advice</p><h1>Quant Factor Research</h1>
<div class="cards"><div class="card"><div class="big">{result['mean_daily_ic']:.3f}</div><div>Mean daily IC</div></div><div class="card"><div class="big">{result['positive_ic_rate']:.0%}</div><div>Positive IC days</div></div><div class="card"><div class="big">{bt['sharpe']:.2f}</div><div>Net Sharpe</div></div><div class="card"><div class="big">{bt['max_drawdown']:.1%}</div><div>Max drawdown</div></div></div>
<h2>Backtest summary</h2><p>Net return: <strong>{bt['net_return']:.2%}</strong> · average turnover: <strong>{bt['average_turnover']:.2f}</strong> · observations: {bt['observations']}</p>
<h2>Transaction-cost sensitivity</h2><table><thead><tr><th>Round-trip cost assumption</th><th>Net cumulative return</th></tr></thead><tbody>{cost_rows}</tbody></table>
</main></body></html>"""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
