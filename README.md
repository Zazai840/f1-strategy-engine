Strategy Engine: Monte Carlo Pit-Stop Optimizer for F1
======================================================

Overview
--------
This project implements a production-grade Monte Carlo calculator for optimizing F1 pit-stop strategies using FastF1, NumPy, pandas, and scikit-learn. It fits simple degradation models per tyre compound, learns pit-lane loss distributions (green vs. neutralized), estimates safety-car/virtual-SC hazards, and simulates candidate 1-stop and 2-stop strategies to recommend the fastest expected race time with uncertainty bands.

Quickstart
----------
1) Install dependencies (Python 3.10+ recommended):
```bash
pip install -r requirements.txt
```

2) Run a single optimization:
```bash
python app.py --gp "Monza" --years 2021 2022 2023 --target 2024 --mode 1-stop --sims 1500 --seed 123
```

3) Batch/paper runner (saves CSV and plots):
```bash
python results.py
```

What it does
------------
- Loads FastF1 race sessions and cleans laps to a consistent schema.
- Fits robust linear degradation per compound on green-flag laps via HuberRegressor: LapTimeSeconds ~ a + b * StintLap.
- Extracts pit-lane time distributions for green vs. neutralized (SC/VSC) conditions.
- Builds per-lap neutralization hazards from history.
- Simulates strategies and ranks them by mean race time with 90% CIs.

Modeling assumptions
--------------------
- Degradation is linear within a stint, with Gaussian lap noise.
- Neutralized laps are slowed by +30% relative to green-flag pace.
- Pit loss is sampled from empirical distributions; if empty, synthetic Normal(24, 1.8) is used.
- Hazards are per-lap Bernoulli probabilities estimated from historical TrackStatus.

Outputs
-------
- `outputs/top_strategies.csv`: ranked table with mean and 90% CI.
- `outputs/sim_distributions.csv`: simulation draws for top strategies.
- `outputs/plots/`: histograms per strategy, box plot comparison, pacing diagnostic.

Example CLI output
------------------
```
Top strategies (mean ± 90% CI):
- MEDIUM-26|HARD-26  ->  4987.321s  (90%: 4970.112–5006.893)  stops=1  pit_mean=24.15s
- SOFT-18|HARD-34    ->  4992.532s  (90%: 4975.392–5012.740)  stops=1  pit_mean=24.07s
...
```

Known limitations
-----------------
- Linear degradation ignores tyre thermal windows and track evolution.
- Hazards are aggregated and do not distinguish SC vs. VSC.
- No fuel burn or traffic modeling; this is a first-order estimator.

References
----------
- FastF1 documentation (`https://theoehrly.github.io/Fast-F1/`).


