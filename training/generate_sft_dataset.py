"""Generate heuristic CyberSOC trajectories for TRL SFT training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cybersoc_openenv.client import CyberSOCEnvClient, InProcessCyberSOCEnvClient
from cybersoc_openenv.training import DEFAULT_TASKS, generate_sft_examples


def _build_env_client(env_base_url: str | None):
    if env_base_url:
        return CyberSOCEnvClient(base_url=env_base_url)

    from server.app import app

    return InProcessCyberSOCEnvClient(app)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate heuristic CyberSOC SFT data.")
    parser.add_argument(
        "--output",
        default="artifacts/training/cybersoc_sft_train.jsonl",
        help="JSONL output path.",
    )
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=6,
        help="Number of heuristic episodes to collect for each task.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=7,
        help="Starting seed used to generate deterministic rollouts.",
    )
    parser.add_argument(
        "--env-base-url",
        default=None,
        help="Optional external CyberSOC API base URL. Defaults to in-process FastAPI app.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=DEFAULT_TASKS,
        help="Task ids to include in the dataset.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.with_suffix(".summary.json")

    client = _build_env_client(args.env_base_url)
    try:
        examples = generate_sft_examples(
            env_client=client,
            tasks=args.tasks,
            episodes_per_task=args.episodes_per_task,
            seed_start=args.seed_start,
        )
    finally:
        client.close()

    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")

    summary = {
        "output_path": str(output_path),
        "tasks": args.tasks,
        "episodes_per_task": args.episodes_per_task,
        "seed_start": args.seed_start,
        "examples": len(examples),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
