

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data import load_race_session, load_many_years, ensure_core_columns
from src.fit_models import fit_degradation, fit_pit_loss
from src.hazards import lap_hazard_curves
from src.simulate import simulate_strategy
from src.recommend import grid_search_strategies

PLOT_DPI = 140

OUTPUT_DIR = os.path.join("outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")


def _ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def _plot_hist(ax, data: np.ndarray, title: str, bins: int = 30) -> None:
    ax.hist(data, bins=bins, color="#4C78A8", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Total time (s)")
    ax.set_ylabel("Count")


def _plot_box(ax, distributions: Dict[str, np.ndarray], title: str) -> None:
    labels = list(distributions.keys())
    series = [distributions[k] for k in labels]
    ax.boxplot(series, labels=labels, vert=True)
    ax.set_title(title)
    ax.set_ylabel("Total time (s)")
    ax.tick_params(axis='x', rotation=30)


def _plot_pacing_diagnostic(ax, laps: pd.DataFrame, models: Dict[str, Dict]) -> None:
    colors = {"SOFT": "#E45756", "MEDIUM": "#54A24B", "HARD": "#4C78A8", "UNK": "#B279A2"}
    for comp, grp in laps.groupby("Compound"):
        if len(grp) == 0:
            continue
        ax.scatter(grp["StintLap"], grp["LapTimeSeconds"], s=6, alpha=0.35, label=f"{comp} laps", color=colors.get(comp, "#888888"))
        params = models.get(comp, models.get("UNK", None))
        if params is None:
            continue
        x = np.linspace(1, grp["StintLap"].max() if len(grp) else 30, 50)
        y = params["base"] + params["slope"] * x
        ax.plot(x, y, color=colors.get(comp, "#555555"), linewidth=2, label=f"{comp} fit")
    ax.set_title("Pacing model diagnostic")
    ax.set_xlabel("StintLap")
    ax.set_ylabel("LapTimeSeconds")
    ax.legend(fontsize=8)


def run_experiment(gp: str, years: List[int], target: int, mode: str, sims: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()

    
    target_session = load_race_session(target, gp)
    laps_target = ensure_core_columns(target_session.laps, year=target, event=target_session.event.get("EventName", gp))
    total_laps = int(laps_target["LapNumber"].max())

    laps_all, pits_all = load_many_years(years, gp)
    models = fit_degradation(laps_all)
    pit_dists = fit_pit_loss(pits_all)
    hazards = lap_hazard_curves(years, gp)

    ranked = grid_search_strategies(total_laps, models, pit_dists, hazards, mode=mode, n_sims=sims, seed=seed)

    
    top_csv = os.path.join(OUTPUT_DIR, "top_strategies.csv")
    ranked.assign(gp=gp, years=" ".join(map(str, years)), mode=mode)[
        ["gp", "years", "mode", "strategy", "mean_s", "p10_s", "p90_s", "stops", "pit_mean_s"]
    ].head(50).to_csv(top_csv, index=False)

    
    top5 = ranked.head(5)
    all_sims = []
    for _, row in top5.iterrows():
        
        plan = []
        for tok in str(row["strategy"]).split("|"):
            comp, n = tok.split("-")
            plan.append((comp, int(n)))
        sims_df = simulate_strategy(total_laps, plan, models, pit_dists, hazards, n_sims=2000, seed=seed + 1)
        sims_df["gp"] = gp
        sims_df["years"] = " ".join(map(str, years))
        sims_df["mode"] = mode
        all_sims.append(sims_df)
    sim_df = pd.concat(all_sims, ignore_index=True) if all_sims else pd.DataFrame()
    sim_csv = os.path.join(OUTPUT_DIR, "sim_distributions.csv")
    if len(sim_df):
        sim_df.to_csv(sim_csv, index=False)

    
    if len(sim_df):
        strategies = sim_df["strategy"].unique().tolist()
        for stg in strategies:
            data = sim_df.loc[sim_df["strategy"] == stg, "total_time_s"].values
            fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=PLOT_DPI)
            _plot_hist(ax, data, title=f"Histogram: {gp} {mode} {stg}")
            fig.tight_layout()
            fig.savefig(os.path.join(PLOTS_DIR, f"hist_{gp}_{mode}_{stg.replace('|','_')}.png"))
            plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=PLOT_DPI)
        dists = {stg: sim_df.loc[sim_df["strategy"] == stg, "total_time_s"].values for stg in strategies}
        _plot_box(ax, dists, title=f"Top strategies: {gp} {mode}")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"box_{gp}_{mode}.png"))
        plt.close(fig)

    try:
        green = laps_all[laps_all["TrackStatus"] == "1"]
    except Exception:
        green = laps_all
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=PLOT_DPI)
    _plot_pacing_diagnostic(ax, green.sample(min(len(green), 3000), random_state=seed), models)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"pacing_diagnostic_{gp}.png"))
    plt.close(fig)

    return ranked, sim_df


if __name__ == "__main__":
    ranked, sim_df = run_experiment("Monza", [2021, 2022, 2023], 2024, mode="1-stop", sims=500, seed=42)
    assert len(ranked) >= 1
    print("✓ results runner completed; outputs saved in outputs/")


