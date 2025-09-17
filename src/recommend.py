from __future__ import annotations
import itertools
import warnings
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from .simulate import simulate_strategy, SC_LAP_SLOWDOWN, MIN_STINT, MAX_STINT, CANDIDATE_COMPOUNDS

def _enumerate_partitions(total_laps: int, num_stints: int) -> List[List[int]]:
    parts: List[List[int]] = []

    if num_stints == 2:
        for a in range(MIN_STINT, MAX_STINT + 1):
            b = total_laps - a
            if MIN_STINT <= b <= MAX_STINT:
                parts.append([a, b])
    elif num_stints == 3:
        for a in range(MIN_STINT, MAX_STINT + 1):
            for b in range(MIN_STINT, MAX_STINT + 1):
                c = total_laps - a - b
                if MIN_STINT <= c <= MAX_STINT:
                    parts.append([a, b, c])
    else:
        raise ValueError("Only 1-stop (2 stints) and 2-stop (3 stints) supported")

    unique = []
    seen = set()
    for p in parts:
        key = tuple(p)
        if key not in seen:
            unique.append(p)
            seen.add(key)
    return unique


def _feasible_compound_plan(stint_lengths: List[int], allow_soft_long: bool = False) -> List[List[str]]:
    num_stints = len(stint_lengths)
    all_combos = list(itertools.product(CANDIDATE_COMPOUNDS, repeat=num_stints))
    plans: List[List[str]] = []
    for combo in all_combos:
        combo = list(combo)
        plausible = True
        for i, (comp, length) in enumerate(zip(combo, stint_lengths)):
            if comp == "SOFT" and length > 22 and not allow_soft_long:
                plausible = False
                break
            if i > 0 and comp == "SOFT":
                plausible = False
                break
        if plausible and len(set(combo)) >= 2:
            plans.append(combo)
    return plans


def grid_search_strategies(
    total_laps: int,
    models: Dict[str, Dict],
    pit_dists: Dict[str, np.ndarray],
    hazards: Dict,
    mode: str,
    n_sims: int,
    seed: int | None,
) -> pd.DataFrame:
    if mode not in {"1-stop", "2-stop"}:
        raise ValueError("mode must be '1-stop' or '2-stop'")

    num_stints = 2 if mode == "1-stop" else 3
    partitions = _enumerate_partitions(total_laps, num_stints)
    if not partitions:
        raise ValueError("No feasible partitions found; check MIN_STINT/MAX_STINT against total_laps")

    rng = np.random.default_rng(seed)
    summaries: List[Dict] = []

    for stint_lengths in partitions:
        compound_plans = _feasible_compound_plan(stint_lengths)
        for plan in compound_plans:
            strategy = list(zip(plan, stint_lengths))
            stg_seed = int(rng.integers(0, 2**31 - 1))
            sims = simulate_strategy(total_laps, strategy, models, pit_dists, hazards, n_sims, stg_seed)
            if len(sims) == 0:
                continue
            total_times = sims["total_time_s"].values
            mean_s = float(np.mean(total_times))
            p10 = float(np.percentile(total_times, 10))
            p90 = float(np.percentile(total_times, 90))
            pit_mean = float(np.mean(sims["pit_time_s"]))
            summaries.append(
                {
                    "strategy": "|".join([f"{c}-{n}" for c, n in strategy]),
                    "mean_s": mean_s,
                    "p10_s": p10,
                    "p90_s": p90,
                    "stops": num_stints - 1,
                    "pit_mean_s": pit_mean,
                }
            )

    if not summaries:
        warnings.warn("No strategies could be simulated; returning empty DataFrame")
        return pd.DataFrame(columns=["strategy", "mean_s", "p10_s", "p90_s", "stops", "pit_mean_s"])

    df = pd.DataFrame(summaries)
    df = df.sort_values("mean_s", ascending=True, kind="mergesort").reset_index(drop=True)
    return df


if __name__ == "__main__":
    fake_models = {
        "SOFT": {"base": 90.0, "slope": 0.12, "sigma": 1.0},
        "MEDIUM": {"base": 91.0, "slope": 0.09, "sigma": 0.9},
        "HARD": {"base": 92.0, "slope": 0.06, "sigma": 0.8},
        "UNK": {"base": 90.0, "slope": 0.10, "sigma": 1.0},
    }
    fake_pits = {"green": np.array([24.0, 25.0, 23.5]), "neutral": np.array([18.0, 19.5])}
    fake_haz = {"per_lap_p": np.zeros(58)}
    out = grid_search_strategies(58, fake_models, fake_pits, fake_haz, mode="1-stop", n_sims=20, seed=42)
    assert set(["strategy", "mean_s", "p10_s", "p90_s", "stops", "pit_mean_s"]).issubset(out.columns)
    print("✓ recommend smoke test passed")


