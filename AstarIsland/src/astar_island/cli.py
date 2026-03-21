from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from .analysis import (
    coordinate_search,
    evaluate_cases,
    evaluation_report,
    load_cached_evaluation_cases,
    run_training_preflight,
    sync_completed_analyses,
)
from .api import AstarIslandClient, AstarIslandApiError
from .baseline import ModelParameters, build_all_predictions
from .cache import CacheStore
from .config import AppConfig
from .planner import determine_stage, plan_query_batch, run_iterative_autopilot


def build_runtime(args: argparse.Namespace) -> tuple[AstarIslandClient, CacheStore]:
    config = AppConfig.from_env()
    if getattr(args, "base_url", None):
        config.base_url = args.base_url.rstrip("/")
    if getattr(args, "token", None):
        config.access_token = args.token
    if getattr(args, "data_dir", None):
        config.data_dir = Path(args.data_dir)
    return AstarIslandClient(config), CacheStore(config.data_dir)


def resolve_round_id(client: AstarIslandClient, requested_round_id: str | None) -> str:
    if requested_round_id:
        return requested_round_id
    active_round = client.get_active_round()
    if active_round is None:
        raise AstarIslandApiError("No active round found.")
    return active_round.id


def load_model_parameters(cache: CacheStore) -> ModelParameters:
    return cache.load_model_parameters() or ModelParameters()


def command_status(args: argparse.Namespace) -> int:
    client, _ = build_runtime(args)
    rounds = client.get_rounds()
    active_round = next((item for item in rounds if item.status == "active"), None)
    print(json.dumps([asdict(item) for item in rounds], indent=2))
    if active_round and client.config.access_token:
        budget = client.get_budget()
        print(json.dumps(asdict(budget), indent=2))
    return 0


def command_pull_round(args: argparse.Namespace) -> int:
    client, cache = build_runtime(args)
    round_id = resolve_round_id(client, args.round_id)
    detail = client.get_round_detail(round_id)
    path = cache.save_round_detail(detail)
    print(f"Cached round {detail.round_number} to {path}")
    return 0


def command_simulate(args: argparse.Namespace) -> int:
    client, cache = build_runtime(args)
    round_id = resolve_round_id(client, args.round_id)
    result = client.simulate(
        round_id=round_id,
        seed_index=args.seed,
        viewport_x=args.x,
        viewport_y=args.y,
        viewport_w=args.w,
        viewport_h=args.h,
    )
    cache.append_observation(result)
    print(json.dumps(asdict(result), indent=2, default=str))
    return 0


def command_build(args: argparse.Namespace) -> int:
    client, cache = build_runtime(args)
    round_id = resolve_round_id(client, args.round_id)
    detail = cache.load_round_detail(round_id) or client.get_round_detail(round_id)
    cache.save_round_detail(detail)
    observations = cache.load_observations(round_id)
    model_params = load_model_parameters(cache)
    predictions = build_all_predictions(detail, observations, params=model_params)

    for seed_index, preview in predictions.items():
        if args.seed is not None and seed_index != args.seed:
            continue
        path = cache.save_prediction(round_id, seed_index, preview.prediction.tolist())
        observed_cells = int(preview.coverage_mask.sum())
        print(f"Built seed {seed_index} prediction with {observed_cells} observed cells -> {path}")
    return 0


def parse_seed_list(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_id_list(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def command_autoquery(args: argparse.Namespace) -> int:
    client, cache = build_runtime(args)
    round_id = resolve_round_id(client, args.round_id)
    detail = cache.load_round_detail(round_id) or client.get_round_detail(round_id)
    cache.save_round_detail(detail)
    observations = cache.load_observations(round_id)
    seed_indices = parse_seed_list(args.seeds)
    budget = client.get_budget() if (client.config.access_token and args.use_remaining) else None
    count = max(0, (budget.queries_max - budget.queries_used) if budget else args.count)

    if not args.dry_run and not args.skip_preflight:
        preflight = run_training_preflight(client, cache, passes=args.preflight_passes)
        print(
            json.dumps(
                {
                    "preflight": {
                        "case_count": preflight.case_count,
                        "before_score": preflight.before_score,
                        "after_score": preflight.after_score,
                        "improved": preflight.improved,
                        "saved_params": preflight.saved_params,
                        "synced_rounds": preflight.synced_rounds,
                        "report_path": preflight.report_path,
                    }
                },
                indent=2,
            )
        )

    model_params = load_model_parameters(cache)
    historical_prior = cache.load_historical_signal_prior()

    if args.dry_run:
        stage = determine_stage(existing_queries=len(observations), requested_queries=count)
        plan = plan_query_batch(
            detail,
            observations,
            count=count,
            viewport_w=args.w,
            viewport_h=args.h,
            seed_indices=seed_indices,
            params=model_params,
            historical_prior=historical_prior,
            stage=stage,
        )
        if not plan:
            raise AstarIslandApiError("Planner did not produce any useful queries.")
        print(json.dumps({"stage": stage, "plan": [asdict(item) for item in plan]}, indent=2))
        return 0

    run = run_iterative_autopilot(
        client,
        cache,
        detail,
        round_id=round_id,
        total_queries=count,
        viewport_w=args.w,
        viewport_h=args.h,
        seed_indices=seed_indices,
        replan_every=args.replan_every,
        pause_seconds=args.pause_seconds,
        params=model_params,
        historical_prior=historical_prior,
    )
    if run.executed_queries == 0:
        raise AstarIslandApiError("Planner did not produce any useful queries.")

    print(
        json.dumps(
            {
                "requested_queries": run.requested_queries,
                "executed_queries": run.executed_queries,
                "batch_stages": run.batch_stages,
                "batches": [[asdict(item) for item in batch] for batch in run.batches],
            },
            indent=2,
        )
    )

    if args.build:
        updated_observations = cache.load_observations(round_id)
        predictions = build_all_predictions(detail, updated_observations, params=model_params)
        for seed_index, preview in predictions.items():
            path = cache.save_prediction(round_id, seed_index, preview.prediction.tolist())
            print(f"Updated prediction for seed {seed_index} -> {path}")

    if args.submit:
        detail = cache.load_round_detail(round_id) or detail
        for seed_index in range(detail.seeds_count):
            prediction = cache.load_prediction(round_id, seed_index)
            if prediction is None:
                raise AstarIslandApiError(
                    f"Missing cached prediction for seed {seed_index}. Run with --build before --submit."
                )
            response = client.submit_prediction(round_id=round_id, seed_index=seed_index, prediction=prediction)
            print(json.dumps(response, indent=2))
            if seed_index < detail.seeds_count - 1:
                time.sleep(0.6)
    return 0


def command_sync_analysis(args: argparse.Namespace) -> int:
    client, cache = build_runtime(args)
    requested_ids = parse_id_list(args.round_ids)
    synced = sync_completed_analyses(
        client,
        cache,
        round_ids=requested_ids,
        include_non_completed=args.include_non_completed,
        refresh=args.refresh,
    )
    print(json.dumps({"synced_rounds": synced}, indent=2))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    client, cache = build_runtime(args)
    round_ids = parse_id_list(args.round_ids)
    cases = load_cached_evaluation_cases(client, cache, round_ids=round_ids, completed_only=not args.include_non_completed)
    if not cases:
        raise AstarIslandApiError(
            "No evaluation cases found. Fetch analyses first and make sure you have cached observations for those rounds."
        )

    model_params = load_model_parameters(cache)
    evaluations = evaluate_cases(cases, model_params)
    report = evaluation_report(evaluations, model_params)
    path = cache.save_report("evaluation", report)
    print(json.dumps(report, indent=2))
    print(f"Saved evaluation report to {path}")
    return 0


def command_tune(args: argparse.Namespace) -> int:
    client, cache = build_runtime(args)
    round_ids = parse_id_list(args.round_ids)
    cases = load_cached_evaluation_cases(client, cache, round_ids=round_ids, completed_only=not args.include_non_completed)
    if not cases:
        raise AstarIslandApiError(
            "No training cases found. Fetch analyses first and make sure you have cached observations for those rounds."
        )

    start_params = ModelParameters() if args.reset else load_model_parameters(cache)
    best_params, best_score, history = coordinate_search(cases, start=start_params, passes=args.passes)
    evaluations = evaluate_cases(cases, best_params)
    report = evaluation_report(evaluations, best_params)
    report["tuning"] = {
        "best_mean_seed_score": best_score,
        "history": history,
    }
    report_path = cache.save_report("tuning", report)
    if args.save:
        params_path = cache.save_model_parameters(best_params)
        print(f"Saved tuned model parameters to {params_path}")
    print(json.dumps(report, indent=2))
    print(f"Saved tuning report to {report_path}")
    return 0


def command_submit(args: argparse.Namespace) -> int:
    client, cache = build_runtime(args)
    round_id = resolve_round_id(client, args.round_id)

    if args.all_seeds:
        detail = cache.load_round_detail(round_id) or client.get_round_detail(round_id)
        seed_indices = list(range(detail.seeds_count))
    elif args.seed is not None:
        seed_indices = [args.seed]
    else:
        raise AstarIslandApiError("Use --seed or --all-seeds when submitting.")

    for seed_index in seed_indices:
        prediction = cache.load_prediction(round_id, seed_index)
        if prediction is None:
            raise AstarIslandApiError(
                f"Missing cached prediction for seed {seed_index}. Run `astar-island build` first."
            )
        response = client.submit_prediction(round_id=round_id, seed_index=seed_index, prediction=prediction)
        print(json.dumps(response, indent=2))
        if seed_index != seed_indices[-1]:
            time.sleep(0.6)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Astar Island CLI")
    parser.add_argument("--base-url")
    parser.add_argument("--token")
    parser.add_argument("--data-dir")

    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="Show rounds and active budget")
    status_parser.set_defaults(func=command_status)

    pull_parser = subparsers.add_parser("pull-round", help="Fetch and cache round details")
    pull_parser.add_argument("--round-id")
    pull_parser.set_defaults(func=command_pull_round)

    sync_parser = subparsers.add_parser("sync-analysis", help="Fetch completed-round ground truth analyses into the local cache")
    sync_parser.add_argument("--round-ids", help="Comma-separated round ids. Defaults to all completed team rounds.")
    sync_parser.add_argument("--include-non-completed", action="store_true")
    sync_parser.add_argument("--refresh", action="store_true")
    sync_parser.set_defaults(func=command_sync_analysis)

    simulate_parser = subparsers.add_parser("simulate", help="Run one viewport query")
    simulate_parser.add_argument("--round-id")
    simulate_parser.add_argument("--seed", type=int, required=True)
    simulate_parser.add_argument("--x", type=int, required=True)
    simulate_parser.add_argument("--y", type=int, required=True)
    simulate_parser.add_argument("--w", type=int, default=15)
    simulate_parser.add_argument("--h", type=int, default=15)
    simulate_parser.set_defaults(func=command_simulate)

    build_parser_cmd = subparsers.add_parser("build", help="Build baseline predictions from cached observations")
    build_parser_cmd.add_argument("--round-id")
    build_parser_cmd.add_argument("--seed", type=int)
    build_parser_cmd.set_defaults(func=command_build)

    evaluate_parser = subparsers.add_parser("evaluate", help="Score the current model against cached ground truth analyses")
    evaluate_parser.add_argument("--round-ids", help="Comma-separated round ids. Defaults to all completed cached rounds.")
    evaluate_parser.add_argument("--include-non-completed", action="store_true")
    evaluate_parser.set_defaults(func=command_evaluate)

    tune_parser = subparsers.add_parser("tune", help="Tune model weights on cached completed-round analyses")
    tune_parser.add_argument("--round-ids", help="Comma-separated round ids. Defaults to all completed cached rounds.")
    tune_parser.add_argument("--include-non-completed", action="store_true")
    tune_parser.add_argument("--passes", type=int, default=2)
    tune_parser.add_argument("--save", action="store_true")
    tune_parser.add_argument("--reset", action="store_true")
    tune_parser.set_defaults(func=command_tune)

    autoquery_parser = subparsers.add_parser("autoquery", help="Plan and optionally execute automated queries")
    autoquery_parser.add_argument("--round-id")
    autoquery_parser.add_argument("--count", type=int, default=5)
    autoquery_parser.add_argument("--w", type=int, default=15)
    autoquery_parser.add_argument("--h", type=int, default=15)
    autoquery_parser.add_argument("--seeds", help="Comma-separated seed indices, for example 0,1,3")
    autoquery_parser.add_argument("--replan-every", type=int, default=5)
    autoquery_parser.add_argument("--use-remaining", action="store_true")
    autoquery_parser.add_argument("--skip-preflight", action="store_true")
    autoquery_parser.add_argument("--preflight-passes", type=int, default=1)
    autoquery_parser.add_argument("--pause-seconds", type=float, default=0.25)
    autoquery_parser.add_argument("--dry-run", action="store_true")
    autoquery_parser.add_argument("--build", action="store_true")
    autoquery_parser.add_argument("--submit", action="store_true")
    autoquery_parser.set_defaults(func=command_autoquery)

    submit_parser = subparsers.add_parser("submit", help="Submit cached predictions")
    submit_parser.add_argument("--round-id")
    submit_parser.add_argument("--seed", type=int)
    submit_parser.add_argument("--all-seeds", action="store_true")
    submit_parser.set_defaults(func=command_submit)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except AstarIslandApiError as exc:
        parser.exit(status=1, message=f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
