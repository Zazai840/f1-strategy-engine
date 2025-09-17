from __future__ import annotations
import argparse
import os
import warnings
from typing import List

import numpy as np
import pandas as pd

from src.data import load_race_session, load_many_years, ensure_core_columns
from src.fit_models import fit_degradation, fit_pit_loss, validate_models
from src.hazards import lap_hazard_curves
from src.recommend import grid_search_strategies

OUTPUT_DIR = os.path.join("outputs")


def _ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="F1 Strategy Monte Carlo Optimizer")
    parser.add_argument("--gp", type=str, required=True, help="Grand Prix name or location")
    parser.add_argument("--years", type=int, nargs="+", required=True, help="Historical years for training")
    parser.add_argument("--target", type=int, required=True, help="Target year to optimize (for total laps)")
    parser.add_argument("--mode", type=str, choices=["1-stop", "2-stop"], default="1-stop")
    parser.add_argument("--sims", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    _ensure_dirs()

    try:
        target_session = load_race_session(args.target, args.gp)
        laps_target = ensure_core_columns(target_session.laps, year=args.target, event=target_session.event.get("EventName", args.gp))
        total_laps = int(laps_target["LapNumber"].max())
        if total_laps <= 0:
            raise ValueError("Invalid total_laps from target session")
    except Exception as exc:
        raise SystemExit(f"Failed to load target session {args.target} {args.gp}: {exc}")

    laps_all, pits_all = load_many_years(args.years, args.gp)
    models = fit_degradation(laps_all)
    pit_dists = fit_pit_loss(pits_all)
    if not validate_models(models, pit_dists):
        warnings.warn("Model validation failed; proceeding with caution")

    hazards = lap_hazard_curves(args.years, args.gp)

    table = grid_search_strategies(total_laps, models, pit_dists, hazards, mode=args.mode, n_sims=args.sims, seed=args.seed)

    if len(table) == 0:
        raise SystemExit("No strategies produced. Try adjusting years/mode.")

    csv_path = os.path.join(OUTPUT_DIR, "top_strategies.csv")
    table.head(50).to_csv(csv_path, index=False)

    top5 = table.head(5).copy()
    print("Top strategies (mean ± 90% CI):")
    for _, row in top5.iterrows():
        mean_s = row["mean_s"]
        lo = row["p10_s"]
        hi = row["p90_s"]
        print(f"- {row['strategy']}  ->  {mean_s:.3f}s  (90%: {lo:.3f}–{hi:.3f})  stops={row['stops']}  pit_mean={row['pit_mean_s']:.2f}s")


if __name__ == "__main__":
    main()


