#!/usr/bin/env python3
"""Summarize Modal matrix result JSON files.

Usage:
    python scripts/summarize_results.py results/runs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(root: str) -> None:
    path = Path(root)
    rows = []
    for file in sorted(path.glob("*.json")):
        try:
            item = json.loads(file.read_text())
        except json.JSONDecodeError:
            continue
        eval_ = item.get("eval") or {}
        persona = item.get("persona") or {}
        rows.append(
            {
                "tag": item.get("tag", file.stem),
                "family": item.get("family", ""),
                "model": item.get("model", ""),
                "method": item.get("method", ""),
                "trait": eval_.get("trait_score"),
                "baseline": eval_.get("baseline_trait_score"),
                "ppl": eval_.get("ppl_ratio"),
                "scorer": eval_.get("scorer"),
                "neurons": len(persona.get("neurons") or []),
                "server": (item.get("server") or {}).get("status"),
                "error": item.get("error"),
            }
        )
    header = f"{'tag':32} {'method':8} {'trait':>6} {'base':>6} {'ppl':>5} {'n':>2} {'srv':>4}  error"
    print(header)
    print("-" * len(header))
    for row in rows:
        trait = f"{row['trait']:.3f}" if isinstance(row["trait"], (int, float)) else "  -  "
        base = f"{row['baseline']:.3f}" if isinstance(row["baseline"], (int, float)) else "  -  "
        ppl = f"{row['ppl']:.2f}" if isinstance(row["ppl"], (int, float)) else "  - "
        server = str(row["server"]) if row["server"] is not None else "-"
        print(
            f"{row['tag']:32} {row['method']:8} {trait:>6} {base:>6} {ppl:>5} "
            f"{row['neurons']:>2} {server:>4}  {row['error'] or ''}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/runs")
