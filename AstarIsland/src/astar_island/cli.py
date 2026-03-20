from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .api import AstarIslandClient, AstarIslandApiError
from .baseline import build_all_predictions
from .cache import CacheStore
from .config import AppConfig


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
    predictions = build_all_predictions(detail, observations)

    for seed_index, preview in predictions.items():
        if args.seed is not None and seed_index != args.seed:
            continue
        path = cache.save_prediction(round_id, seed_index, preview.prediction.tolist())
        observed_cells = int(preview.coverage_mask.sum())
        print(f"Built seed {seed_index} prediction with {observed_cells} observed cells -> {path}")
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
