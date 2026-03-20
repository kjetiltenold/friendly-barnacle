from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import streamlit as st

try:
    from .api import AstarIslandApiError, AstarIslandClient
    from .baseline import build_all_predictions, overlay_initial_settlements
    from .cache import CacheStore
    from .config import AppConfig
    from .visuals import confidence_to_image, grid_to_image, mask_to_image
except ImportError:
    # Streamlit runs the target file as a script, so absolute imports are the safe fallback.
    from astar_island.api import AstarIslandApiError, AstarIslandClient
    from astar_island.baseline import build_all_predictions, overlay_initial_settlements
    from astar_island.cache import CacheStore
    from astar_island.config import AppConfig
    from astar_island.visuals import confidence_to_image, grid_to_image, mask_to_image


def get_runtime() -> tuple[AstarIslandClient, CacheStore]:
    base_config = AppConfig.from_env()

    st.sidebar.header("Connection")
    base_url = st.sidebar.text_input("Base URL", value=base_config.base_url)
    token = st.sidebar.text_input("Access token", value=base_config.access_token or "", type="password")
    data_dir = st.sidebar.text_input("Data directory", value=str(base_config.data_dir))

    config = AppConfig(
        access_token=token or None,
        base_url=base_url.rstrip("/"),
        data_dir=Path(data_dir),
    )
    return AstarIslandClient(config), CacheStore(config.data_dir)


def render_initial_state(detail, seed_index: int) -> None:
    state = detail.initial_states[seed_index]
    grid = overlay_initial_settlements(state)
    st.image(grid_to_image(grid), caption=f"Seed {seed_index} initial state", clamp=True)
    st.caption(f"{len(state.settlements)} starting settlements")


def render_observation_summary(detail, cache: CacheStore, seed_index: int) -> None:
    observations = cache.load_observations(detail.id, seed_index=seed_index)
    st.metric("Cached queries for seed", len(observations))
    if observations:
        last = observations[-1]
        st.caption(
            f"Last viewport: ({last.viewport.x}, {last.viewport.y}) {last.viewport.w}x{last.viewport.h} at {last.captured_at}"
        )


def main() -> None:
    st.set_page_config(page_title="Astar Island", layout="wide")
    st.title("Astar Island Explorer")
    st.write("Inspect rounds, query the simulator, build baseline predictions, and submit them.")

    client, cache = get_runtime()

    try:
        rounds = client.get_rounds()
    except AstarIslandApiError as exc:
        st.error(str(exc))
        return

    round_options = {f"Round {item.round_number} ({item.status})": item.id for item in rounds}
    if not round_options:
        st.warning("No rounds available.")
        return

    selected_label = st.sidebar.selectbox("Round", list(round_options.keys()))
    round_id = round_options[selected_label]

    try:
        detail = cache.load_round_detail(round_id) or client.get_round_detail(round_id)
        cache.save_round_detail(detail)
    except AstarIslandApiError as exc:
        st.error(str(exc))
        return

    st.sidebar.subheader("Round")
    st.sidebar.write(f"Round #{detail.round_number}")
    st.sidebar.write(f"Status: {detail.status}")
    st.sidebar.write(f"Map: {detail.map_width} x {detail.map_height}")

    if client.config.access_token:
        try:
            budget = client.get_budget()
            st.sidebar.metric("Queries used", f"{budget.queries_used}/{budget.queries_max}")
        except AstarIslandApiError as exc:
            st.sidebar.warning(str(exc))

    seed_index = st.selectbox("Seed", list(range(detail.seeds_count)), index=0)

    overview_tab, query_tab, predictor_tab, submit_tab = st.tabs(
        ["Overview", "Query", "Predictor", "Submit"]
    )

    with overview_tab:
        left, right = st.columns([2, 1])
        with left:
            render_initial_state(detail, seed_index)
        with right:
            render_observation_summary(detail, cache, seed_index)
            state = detail.initial_states[seed_index]
            st.json({"settlements": [asdict(item) for item in state.settlements]})

    with query_tab:
        col1, col2, col3, col4 = st.columns(4)
        x = col1.number_input("Viewport X", min_value=0, max_value=detail.map_width - 1, value=0, step=1)
        y = col2.number_input("Viewport Y", min_value=0, max_value=detail.map_height - 1, value=0, step=1)
        w = col3.slider("Viewport W", min_value=5, max_value=15, value=15)
        h = col4.slider("Viewport H", min_value=5, max_value=15, value=15)

        if st.button("Run simulation query", type="primary"):
            try:
                result = client.simulate(
                    round_id=detail.id,
                    seed_index=seed_index,
                    viewport_x=int(x),
                    viewport_y=int(y),
                    viewport_w=int(w),
                    viewport_h=int(h),
                )
            except AstarIslandApiError as exc:
                st.error(str(exc))
            else:
                cache.append_observation(result)
                st.success("Observation cached.")
                st.image(
                    grid_to_image(np.array(result.grid, dtype=int)),
                    caption="Simulated viewport after 50 years",
                    clamp=True,
                )
                st.json(
                    {
                        "viewport": asdict(result.viewport),
                        "queries_used": result.queries_used,
                        "queries_max": result.queries_max,
                        "settlements": [asdict(item) for item in result.settlements],
                    }
                )

    with predictor_tab:
        observations = cache.load_observations(detail.id)
        predictions = build_all_predictions(detail, observations)
        preview = predictions[seed_index]

        col_a, col_b, col_c = st.columns(3)
        col_a.image(
            grid_to_image(preview.argmax_grid.astype(int), use_class_palette=True),
            caption="Predicted argmax classes",
            clamp=True,
        )
        col_b.image(
            confidence_to_image(preview.confidence_grid),
            caption="Prediction confidence",
            clamp=True,
        )
        col_c.image(
            mask_to_image(preview.coverage_mask),
            caption="Observed coverage",
            clamp=True,
        )
        st.caption(f"Observed cells for this seed: {int(preview.coverage_mask.sum())}")

        if st.button("Cache predictions for every seed"):
            for prediction_seed, bundle in predictions.items():
                cache.save_prediction(detail.id, prediction_seed, bundle.prediction.tolist())
            st.success("Predictions cached to disk.")

    with submit_tab:
        st.write("Submit cached predictions to the active round.")
        submit_all = st.checkbox("Submit all seeds", value=False)
        if st.button("Submit", type="primary"):
            try:
                if submit_all:
                    targets = range(detail.seeds_count)
                else:
                    targets = [seed_index]

                for target in targets:
                    prediction = cache.load_prediction(detail.id, target)
                    if prediction is None:
                        st.error(f"Missing cached prediction for seed {target}.")
                        break
                    response = client.submit_prediction(
                        round_id=detail.id,
                        seed_index=target,
                        prediction=prediction,
                    )
                    st.success(f"Submitted seed {target}: {response['status']}")
            except AstarIslandApiError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
