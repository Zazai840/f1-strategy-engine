from __future__ import annotations
import warnings
from typing import Dict, List
import numpy as np
import pandas as pd
from .data import load_many_years


SC_LAP_SLOWDOWN = 0.30
MIN_STINT = 8
MAX_STINT = 35
CANDIDATE_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def _compute_year_lap_neutral_flags(laps: pd.DataFrame) -> pd.DataFrame:
    if len(laps) == 0:
        return pd.DataFrame(columns=["Year", "LapNumber", "is_neutral"]).astype(
            {"Year": int, "LapNumber": int, "is_neutral": bool}
        )
    ts = laps.copy()
    ts["TrackStatus"] = ts["TrackStatus"].fillna("1").astype(str)
    grp = (
        ts.groupby(["Year", "LapNumber"], as_index=False)["TrackStatus"]
        .apply(lambda s: np.any(s.values != "1"))
        .rename(columns={"TrackStatus": "is_neutral"})
    )
    return grp


def lap_hazard_curves(history_years: List[int], gp: str) -> Dict:
    try:
        laps_all, _ = load_many_years(history_years, gp)
    except Exception as exc:
        warnings.warn(
            f"Hazard: failed to load history for {gp} {history_years}: {exc}. Using flat zero hazards."
        )
        return {"per_lap_p": np.zeros(60, dtype=float), "max_lap": 60, "meta": {"source": "empty"}}

    if len(laps_all) == 0:
        warnings.warn("Hazard: no laps in history. Using flat zero hazards.")
        return {"per_lap_p": np.zeros(60, dtype=float), "max_lap": 60, "meta": {"source": "empty"}}

    yr_lap = _compute_year_lap_neutral_flags(laps_all)
    if len(yr_lap) == 0:
        warnings.warn("Hazard: could not compute year-lap flags. Using flat zero hazards.")
        return {"per_lap_p": np.zeros(60, dtype=float), "max_lap": 60, "meta": {"source": "empty"}}
    max_lap_by_year = yr_lap.groupby("Year")["LapNumber"].max()
    max_laps = int(max_lap_by_year.max()) if len(max_lap_by_year) else 60
    per_lap_counts = np.zeros(max_laps, dtype=float)
    per_lap_denoms = np.zeros(max_laps, dtype=float)

    for year, series_max in max_lap_by_year.items():
        year_flags = yr_lap[yr_lap["Year"] == year]
        neutral_map = dict(zip(year_flags["LapNumber"].astype(int), year_flags["is_neutral"].astype(bool)))
        for lap in range(1, int(series_max) + 1):
            per_lap_denoms[lap - 1] += 1.0
            if neutral_map.get(lap, False):
                per_lap_counts[lap - 1] += 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        per_lap_p = np.divide(per_lap_counts, per_lap_denoms, out=np.zeros_like(per_lap_counts), where=per_lap_denoms > 0)

    per_lap_p = np.clip(per_lap_p, 0.0, 0.6)

    return {
        "per_lap_p": per_lap_p.astype(float),
        "max_lap": int(max_laps),
        "meta": {"gp": gp, "years": history_years, "method": "per-lap Bernoulli"},
    }


if __name__ == "__main__":
    hz = lap_hazard_curves([], "Monza")
    assert "per_lap_p" in hz and isinstance(hz["per_lap_p"], np.ndarray)
    assert hz["per_lap_p"].ndim == 1
    print("✓ hazards smoke test passed")


