from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import streamlit as st

try:
    from .api import AstarIslandApiError, AstarIslandClient
    from .baseline import build_all_predictions, overlay_initial_settlements
    from .cache import CacheStore
    from .config import AppConfig
    from .planner import PlannedQuery, execute_query_plan, plan_query_batch, run_iterative_autopilot
    from .visuals import confidence_to_image, grid_to_image, mask_to_image
except ImportError:
    # Streamlit runs the target file as a script, so absolute imports are the safe fallback.
    from astar_island.api import AstarIslandApiError, AstarIslandClient
    from astar_island.baseline import build_all_predictions, overlay_initial_settlements
    from astar_island.cache import CacheStore
    from astar_island.config import AppConfig
    from astar_island.planner import PlannedQuery, execute_query_plan, plan_query_batch, run_iterative_autopilot
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


def cache_predictions(cache: CacheStore, round_id: str, predictions) -> None:
    for prediction_seed, bundle in predictions.items():
        cache.save_prediction(round_id, prediction_seed, bundle.prediction.tolist())


def serialize_plan(plan: list[PlannedQuery]) -> list[dict[str, float | int]]:
    return [asdict(item) for item in plan]


def deserialize_plan(payload: list[dict[str, float | int]]) -> list[PlannedQuery]:
    return [PlannedQuery(**item) for item in payload]


def submit_cached_predictions(client: AstarIslandClient, cache: CacheStore, round_id: str, seeds_count: int) -> None:
    for seed_index in range(seeds_count):
        prediction = cache.load_prediction(round_id, seed_index)
        if prediction is None:
            raise AstarIslandApiError(f"Missing cached prediction for seed {seed_index}.")
        client.submit_prediction(round_id=round_id, seed_index=seed_index, prediction=prediction)
        if seed_index < seeds_count - 1:
            time.sleep(0.6)


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

    budget = None
    if client.config.access_token:
        try:
            budget = client.get_budget()
            st.sidebar.metric("Queries used", f"{budget.queries_used}/{budget.queries_max}")
        except AstarIslandApiError as exc:
            st.sidebar.warning(str(exc))

    remaining_budget = None if budget is None else max(0, budget.queries_max - budget.queries_used)
    can_query = bool(client.config.access_token) and (remaining_budget is None or remaining_budget > 0)
    seed_index = st.selectbox("Seed", list(range(detail.seeds_count)), index=0)
    observations = cache.load_observations(detail.id)
    predictions = build_all_predictions(detail, observations)
    preview = predictions[seed_index]
    plan_state_key = f"query-plan-{detail.id}"

    overview_tab, query_tab, autopilot_tab, predictor_tab, submit_tab = st.tabs(
        ["Overview", "Query", "Autopilot", "Predictor", "Submit"]
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
        if not client.config.access_token:
            st.info("Add an access token in the sidebar to enable manual queries.")
        elif remaining_budget == 0:
            st.warning("Your team has no remaining query budget for the active round.")
        col1, col2, col3, col4 = st.columns(4)
        x = col1.number_input("Viewport X", min_value=0, max_value=detail.map_width - 1, value=0, step=1)
        y = col2.number_input("Viewport Y", min_value=0, max_value=detail.map_height - 1, value=0, step=1)
        w = col3.slider("Viewport W", min_value=5, max_value=15, value=15)
        h = col4.slider("Viewport H", min_value=5, max_value=15, value=15)

        if st.button("Run simulation query", type="primary", key="manual-query", disabled=not can_query):
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

        suggested_plan = plan_query_batch(
            detail,
            observations,
            count=1,
            viewport_w=int(w),
            viewport_h=int(h),
            seed_indices=[seed_index],
        )
        if suggested_plan:
            suggestion = suggested_plan[0]
            st.caption(
                "Suggested next viewport: "
                f"seed {suggestion.seed_index}, x={suggestion.x}, y={suggestion.y}, "
                f"score={suggestion.adjusted_score:.2f}"
            )

    with autopilot_tab:
        st.write("Let the planner choose high-information windows and optionally execute them in a batch.")
        if not client.config.access_token:
            st.info("Add an access token in the sidebar to run automated queries.")
        elif remaining_budget == 0:
            st.warning("Your team has no remaining query budget for the active round.")
        planner_col1, planner_col2, planner_col3 = st.columns(3)
        batch_max = 10 if remaining_budget is None else max(1, remaining_budget)
        batch_size = planner_col1.number_input("Batch size", min_value=1, max_value=batch_max, value=min(5, batch_max))
        batch_w = planner_col2.slider("Viewport width", min_value=5, max_value=15, value=15, key="auto-w")
        batch_h = planner_col3.slider("Viewport height", min_value=5, max_value=15, value=15, key="auto-h")
        budget_target_default = min(batch_max, remaining_budget) if remaining_budget is not None else min(25, batch_max)
        target_queries = st.number_input(
            "Queries to spend",
            min_value=1,
            max_value=batch_max,
            value=max(1, budget_target_default),
            help="The autopilot will replan every batch until it spends this many queries or runs out of useful queries.",
        )
        selected_seeds = st.multiselect(
            "Seeds to plan over",
            list(range(detail.seeds_count)),
            default=list(range(detail.seeds_count)),
        )
        auto_cache_after_run = st.checkbox("Cache fresh predictions after executing a batch", value=True)
        auto_submit_after_run = st.checkbox("Auto-submit all seeds after rebuilding", value=False)

        button_col1, button_col2, button_col3 = st.columns(3)
        if button_col1.button("Plan automated batch", key="plan-batch"):
            plan = plan_query_batch(
                detail,
                observations,
                count=min(int(batch_size), int(target_queries)),
                viewport_w=int(batch_w),
                viewport_h=int(batch_h),
                seed_indices=selected_seeds or None,
            )
            st.session_state[plan_state_key] = serialize_plan(plan)

        if button_col2.button("Plan and run now", type="primary", key="plan-run-batch", disabled=not can_query):
            try:
                run = run_iterative_autopilot(
                    client,
                    cache,
                    detail,
                    round_id=detail.id,
                    total_queries=int(target_queries),
                    viewport_w=int(batch_w),
                    viewport_h=int(batch_h),
                    seed_indices=selected_seeds or None,
                    replan_every=int(batch_size),
                )
                st.session_state[plan_state_key] = serialize_plan(run.batches[-1] if run.batches else [])
                if run.executed_queries == 0:
                    st.warning("Planner did not find any useful queries.")
                else:
                    if auto_cache_after_run:
                        updated_predictions = build_all_predictions(detail, cache.load_observations(detail.id))
                        cache_predictions(cache, detail.id, updated_predictions)
                        if auto_submit_after_run:
                            submit_cached_predictions(client, cache, detail.id, detail.seeds_count)
                    st.success(f"Executed {run.executed_queries} queries across {len(run.batches)} replanned batches.")
                    st.rerun()
            except AstarIslandApiError as exc:
                st.error(str(exc))

        if button_col3.button("Use remaining budget", key="run-remaining-budget", disabled=not can_query or remaining_budget == 0):
            try:
                run = run_iterative_autopilot(
                    client,
                    cache,
                    detail,
                    round_id=detail.id,
                    total_queries=int(remaining_budget or 0),
                    viewport_w=int(batch_w),
                    viewport_h=int(batch_h),
                    seed_indices=selected_seeds or None,
                    replan_every=int(batch_size),
                )
                st.session_state[plan_state_key] = serialize_plan(run.batches[-1] if run.batches else [])
                if run.executed_queries == 0:
                    st.warning("Planner did not find any useful queries.")
                else:
                    if auto_cache_after_run:
                        updated_predictions = build_all_predictions(detail, cache.load_observations(detail.id))
                        cache_predictions(cache, detail.id, updated_predictions)
                        if auto_submit_after_run:
                            submit_cached_predictions(client, cache, detail.id, detail.seeds_count)
                    st.success(f"Executed {run.executed_queries} queries across {len(run.batches)} replanned batches.")
                    st.rerun()
            except AstarIslandApiError as exc:
                st.error(str(exc))

        plan_payload = st.session_state.get(plan_state_key, [])
        if plan_payload:
            st.subheader("Planned queries")
            st.dataframe(plan_payload, hide_index=True, use_container_width=True)
            if st.button("Run planned batch", key="run-existing-plan", disabled=not can_query):
                try:
                    plan = deserialize_plan(plan_payload)
                    results = execute_query_plan(client, cache, detail.id, plan)
                    if auto_cache_after_run:
                        updated_predictions = build_all_predictions(detail, cache.load_observations(detail.id))
                        cache_predictions(cache, detail.id, updated_predictions)
                        if auto_submit_after_run:
                            submit_cached_predictions(client, cache, detail.id, detail.seeds_count)
                    st.success(f"Executed {len(results)} planned queries.")
                    st.rerun()
                except AstarIslandApiError as exc:
                    st.error(str(exc))

    with predictor_tab:
        col_a, col_b, col_c, col_d = st.columns(4)
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
        col_d.image(
            confidence_to_image(preview.entropy_grid / np.log(6.0)),
            caption="Model entropy",
            clamp=True,
        )
        st.caption(f"Observed cells for this seed: {int(preview.coverage_mask.sum())}")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Mean confidence", f"{float(np.mean(preview.confidence_grid)):.3f}")
        metric_col2.metric("Mean entropy", f"{float(np.mean(preview.entropy_grid)):.3f}")
        metric_col3.metric("Observed samples", int(np.sum(preview.sample_count_grid)))

        if st.button("Cache predictions for every seed", key="cache-all-predictions"):
            cache_predictions(cache, detail.id, predictions)
            st.success("Predictions cached to disk.")

    with submit_tab:
        st.write("Submit cached predictions to the active round.")
        submit_all = st.checkbox("Submit all seeds", value=False)
        if st.button("Submit", type="primary"):
            try:
                if submit_all:
                    submit_cached_predictions(client, cache, detail.id, detail.seeds_count)
                    st.success(f"Submitted all {detail.seeds_count} seeds.")
                else:
                    prediction = cache.load_prediction(detail.id, seed_index)
                    if prediction is None:
                        st.error(f"Missing cached prediction for seed {seed_index}.")
                    else:
                        response = client.submit_prediction(
                            round_id=detail.id,
                            seed_index=seed_index,
                            prediction=prediction,
                        )
                        st.success(f"Submitted seed {seed_index}: {response['status']}")
            except AstarIslandApiError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
