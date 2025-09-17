
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import io
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="F1 Strategy Monte Carlo",
    page_icon="🏎️",
    layout="wide",
)


try:
    from src.data import load_race_session, load_many_years
    from src.fit_models import fit_degradation, fit_pit_loss
    from src.hazards import lap_hazard_curves
    from src.recommend import grid_search_strategies
except Exception as e:
    st.stop()
    raise e


def get_training_years(target_year: int, n_years: int = 3) -> list[int]:
    """
    Use the previous n_years before target_year for model fitting.
    Ensures years are >= 2006 (FastF1 coverage varies; adjust if needed).
    """
    years = [y for y in range(target_year - n_years, target_year) if y >= 2006]
    return years


@st.cache_data(show_spinner=False)
def prepare_corpus(location: str, history_years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load laps and pits across history_years for a given location.
    Returns concatenated DataFrames.
    """
    laps_all, pits_all = load_many_years(history_years, location)
    if laps_all is None:
        laps_all = pd.DataFrame()
    if pits_all is None:
        pits_all = pd.DataFrame()
    return laps_all, pits_all


@st.cache_data(show_spinner=False)
def fit_all_models(location: str, history_years: list[int]) -> tuple[dict, dict, dict]:
    """
    Fit degradation models, pit loss distributions, and safety car / VSC hazards.
    Returns (models, pit_dists, hazards).
    """
    laps_all, pits_all = prepare_corpus(location, history_years)

    models = fit_degradation(laps_all)

    pit_dists = fit_pit_loss(pits_all)

    hazards = lap_hazard_curves(history_years, location)

    return models, pit_dists, hazards


def run_grid_search(total_laps: int, models: dict, pit_dists: dict, hazards: dict,
                    mode: str, n_sims: int, seed: int = 42) -> pd.DataFrame:
    
    top = grid_search_strategies(
        total_laps, models, pit_dists, hazards, mode=mode, n_sims=n_sims, seed=seed
    )
    if isinstance(top, pd.DataFrame):
        return top
    return pd.DataFrame(top)


def nice_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "mean_time": "Mean Race Time [s]",
        "p95_time": "P95 Race Time [s]",
        "std_time": "Std [s]",
        "Strategy": "Strategy",
        "n_pits": "# Pits",
        "best_lap": "Best Lap [s]",
    }
    out = df.copy()
    for k, v in rename_map.items():
        if k in out.columns:
            out.rename(columns={k: v}, inplace=True)
    return out



st.title("F1 Monte Carlo Strategy Finder")
st.caption("Pick **location**, **target year**, and **mode**. The app runs **2000 simulations** by default and shows the **Top 5 strategies**.")

with st.sidebar:
    st.header("Inputs")
    location = st.text_input("Grand Prix (name or location)", value="Monza", help="Examples: 'Monza', 'Silverstone', 'Spa', 'Bahrain'")
    target_year = st.number_input("Target Year", min_value=2006, max_value=2100, value=2024, step=1)
    mode = st.selectbox(
        "Strategy Mode",
        options=["1-stop", "2-stop", "free"],
        index=0,
        help="Constrain the search to 1-stop, 2-stop, or let it be 'free' to consider anything."
    )

    with st.expander("Advanced", expanded=False):
        n_sims = st.number_input("Simulations (Monte Carlo)", min_value=100, max_value=200_000, value=2000, step=100)
        n_hist = st.slider("Training Years (look-back)", min_value=1, max_value=6, value=3,
                           help="How many seasons before the target year to use for fitting models.")
        seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=42, step=1)

    run_button = st.button("Run Strategy Search", type="primary")

history_years = get_training_years(int(target_year), n_years=int(n_hist))
st.caption(f"Training on history years: {history_years}")

col_status, col_results = st.columns([1, 2], vertical_alignment="top")

with col_status:
    st.subheader("Status")
    if run_button:
        with st.status("Preparing data…", expanded=False) as status:
            try:
                ses_target = load_race_session(int(target_year), location)
                total_laps = int(getattr(ses_target, "total_laps", 0) or 0)
                if total_laps <= 0:
                    st.warning("Couldn't determine total laps from session. Falling back to 60 laps.")
                    total_laps = 60

                status.update(label="Fitting models…", state="running")
                models, pit_dists, hazards = fit_all_models(location, history_years)

                status.update(label="Running Monte Carlo search…", state="running")
                top_df = run_grid_search(total_laps, models, pit_dists, hazards, mode=mode, n_sims=int(n_sims), seed=int(seed))

                status.update(label="Done", state="complete", expanded=False)

            except Exception as e:
                status.update(label="Failed", state="error")
                st.exception(e)
                top_df = None
    else:
        st.info("Set your inputs on the left, then click **Run Strategy Search**.")

with col_results:
    st.subheader("Top 5 Strategies")
    if run_button and top_df is not None and not top_df.empty:
        top_5 = top_df.head(5).copy()
        top_5 = nice_metrics(top_5)

        st.dataframe(top_5, use_container_width=True, hide_index=True)

        csv = top_5.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Top 5 as CSV",
            data=csv,
            file_name=f"top5_strategies_{location}_{target_year}_{mode}.csv",
            mime="text/csv",
        )

        st.markdown(
            f"- **Location:** `{location}`  \n"
            f"- **Target year:** `{target_year}`  \n"
            f"- **Mode:** `{mode}`  \n"
            f"- **Simulations:** `{n_sims}`  \n"
            f"- **Training years:** `{history_years}`"
        )
    elif run_button:
        st.warning("No strategies returned. Try a different mode, add more training years, or increase simulations.")


