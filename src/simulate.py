from __future__ import annotations
import warnings
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


SC_LAP_SLOWDOWN = 0.30
MIN_STINT = 8
MAX_STINT = 35
CANDIDATE_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def _format_strategy_label(strategy: List[Tuple[str, int]]) -> str:
    return "|".join([f"{c}-{n}" for c, n in strategy])


def _sample_from_empirical(rng: np.random.Generator, arr: np.ndarray, fallback_mean: float = 24.0) -> float:
    if arr is None or len(arr) == 0 or not np.all(np.isfinite(arr)):
        warnings.warn("Empty/invalid empirical array; using fallback mean")
        return float(rng.normal(fallback_mean, 1.8))
    idx = rng.integers(0, len(arr))
    return float(arr[int(idx)])


def simulate_strategy(
    total_laps: int,
    strategy: List[Tuple[str, int]],
    models: Dict[str, Dict],
    pit_dists: Dict[str, np.ndarray],
    hazards: Dict,
    n_sims: int,
    seed: int | None,
) -> pd.DataFrame:
    if sum(n for _, n in strategy) != int(total_laps):
        raise ValueError("Strategy stint lengths must sum to total_laps")

    for comp, length in strategy:
        if length <= 0:
            raise ValueError("Stint lengths must be positive integers")
        if comp not in models:
            warnings.warn(f"Compound {comp} missing in models; using 'UNK'")

    per_lap_p = hazards.get("per_lap_p", np.zeros(total_laps, dtype=float))
    if len(per_lap_p) < total_laps:
        pad_val = float(per_lap_p[-1]) if len(per_lap_p) > 0 else 0.0
        per_lap_p = np.concatenate([per_lap_p, np.full(total_laps - len(per_lap_p), pad_val)])
    else:
        per_lap_p = per_lap_p[:total_laps]

    rng = np.random.default_rng(seed)

    stop_laps: List[int] = []
    cum = 0
    for _, stint_len in strategy[:-1]:
        cum += int(stint_len)
        stop_laps.append(cum)

    results = []

    for sim_id in range(int(n_sims)):
       
        is_neutral = rng.random(total_laps) < per_lap_p

        green_time_total = 0.0
        slowdown_extra_total = 0.0

        lap_cursor = 0
        for comp, stint_len in strategy:
            params = models.get(comp, models.get("UNK", {"base": 90.0, "slope": 0.1, "sigma": 1.0}))
            base = float(params.get("base", 90.0))
            slope = float(params.get("slope", 0.1))
            sigma = float(params.get("sigma", 1.0))

            stint_len = int(stint_len)
            global_idx = np.arange(lap_cursor, lap_cursor + stint_len)
            stint_laps = np.arange(1, stint_len + 1)

            base_times = base + slope * stint_laps + rng.normal(0.0, sigma, size=stint_len)
            base_times = np.clip(base_times, 0.0, None)

            green_time_total += float(np.sum(base_times))

            neutral_mask = is_neutral[global_idx]
            slowdown_extra_total += float(np.sum(base_times[neutral_mask] * SC_LAP_SLOWDOWN))

            lap_cursor += stint_len

        pit_total = 0.0
        for stop_lap in stop_laps:
            neutral_for_pit = bool(is_neutral[stop_lap - 1]) 
            key = "neutral" if neutral_for_pit else "green"
            pit_total += _sample_from_empirical(rng, pit_dists.get(key, np.array([])))

        total_time = green_time_total + slowdown_extra_total + pit_total
        results.append(
            {
                "strategy": _format_strategy_label(strategy),
                "sim_id": sim_id,
                "total_time_s": float(total_time),
                "pit_time_s": float(pit_total),
                "green_time_s": float(green_time_total),
                "neutral_laps": int(np.sum(is_neutral)),
            }
        )

    return pd.DataFrame(results, columns=["strategy", "sim_id", "total_time_s", "pit_time_s", "green_time_s", "neutral_laps"])


if __name__ == "__main__":
    fake_models = {
        "SOFT": {"base": 90.0, "slope": 0.12, "sigma": 1.0},
        "MEDIUM": {"base": 91.0, "slope": 0.09, "sigma": 0.9},
        "HARD": {"base": 92.0, "slope": 0.06, "sigma": 0.8},
        "UNK": {"base": 90.0, "slope": 0.10, "sigma": 1.0},
    }
    fake_pits = {"green": np.array([24.0, 25.0, 23.5]), "neutral": np.array([18.0, 19.5])}
    fake_haz = {"per_lap_p": np.zeros(58)}
    stg = [("MEDIUM", 29), ("HARD", 29)]
    df = simulate_strategy(58, stg, fake_models, fake_pits, fake_haz, n_sims=10, seed=1)
    assert len(df) == 10 and {"total_time_s", "pit_time_s"}.issubset(df.columns)
    print("✓ simulate smoke test passed")


