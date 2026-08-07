#!/usr/bin/env python3
"""Run one or more paper experiments from a versioned JSON configuration."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
INJECTED_FLAGS = {
    "--benchmark-root",
    "--model-path",
    "--model-repository",
    "--model-revision",
    "--output-dir",
    "--run-name",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run only this id; repeat to select multiple runs.",
    )
    parser.add_argument(
        "--model-path",
        action="append",
        default=[],
        metavar="ENV=PATH",
        help="Override a model path environment variable from the config.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=REPO_ROOT / "benchmarks",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "paper",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--refresh-radius",
        type=int,
        default=None,
        help=(
            "Override --refresh-radius for selected Archer runs. The default "
            "uses the paper value stored in the configuration."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"configuration not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc
    if config.get("schema_version") != 1:
        raise SystemExit(f"{path}: unsupported or missing schema_version")
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise SystemExit(f"{path}: runs must be a non-empty list")
    ids = [run.get("id") for run in runs]
    if any(not isinstance(run_id, str) or not run_id for run_id in ids):
        raise SystemExit(f"{path}: every run needs a non-empty string id")
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{path}: run ids must be unique")
    return config


def parse_model_overrides(values: list[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise SystemExit(f"invalid --model-path {value!r}; expected ENV=PATH")
        overrides[name] = Path(path).expanduser()
    return overrides


def validate_run(run: dict[str, Any], config_path: Path) -> None:
    if not isinstance(run.get("module"), str):
        raise SystemExit(f"{config_path}: {run.get('id')}: module must be a string")
    if not isinstance(run.get("args"), list) or not all(
        isinstance(value, str) for value in run["args"]
    ):
        raise SystemExit(f"{config_path}: {run.get('id')}: args must be strings")
    collisions = INJECTED_FLAGS.intersection(run["args"])
    if collisions:
        joined = ", ".join(sorted(collisions))
        raise SystemExit(
            f"{config_path}: {run.get('id')}: injected flags must not appear in args: "
            f"{joined}"
        )
    model = run.get("model")
    required = ("repository", "revision", "path_env")
    if not isinstance(model, dict) or any(
        not isinstance(model.get(key), str) or not model[key] for key in required
    ):
        raise SystemExit(
            f"{config_path}: {run.get('id')}: model requires repository, "
            "revision, and path_env"
        )


def model_path_for(
    run: dict[str, Any],
    overrides: dict[str, Path],
    *,
    dry_run: bool,
) -> Path | str:
    path_env = run["model"]["path_env"]
    value = overrides.get(path_env)
    if value is None and os.environ.get(path_env):
        value = Path(os.environ[path_env]).expanduser()
    if value is None:
        if dry_run:
            return f"${path_env}"
        raise SystemExit(
            f"{run['id']}: set {path_env} or pass --model-path {path_env}=PATH"
        )
    if not dry_run and not value.exists():
        raise SystemExit(f"{run['id']}: model path not found: {value}")
    return value


def build_command(
    run: dict[str, Any],
    *,
    model_path: Path | str,
    benchmark_root: Path,
    output_dir: Path,
    limit: int,
    resume: bool,
    refresh_radius: int | None,
) -> list[str]:
    run_args = list(run["args"])
    if refresh_radius is not None:
        try:
            radius_index = run_args.index("--refresh-radius") + 1
        except ValueError as exc:
            raise SystemExit(
                f"{run['id']}: this run does not expose --refresh-radius"
            ) from exc
        run_args[radius_index] = str(refresh_radius)
    run_name = run["id"]
    if refresh_radius is not None:
        run_name = f"{run_name}_k{refresh_radius}"
    command = [
        sys.executable,
        "-m",
        run["module"],
        *run_args,
        "--model-path",
        str(model_path),
        "--model-revision",
        run["model"]["revision"],
        "--benchmark-root",
        str(benchmark_root),
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
    ]
    if run["module"] == "archer.benchmark":
        command.extend(["--model-repository", run["model"]["repository"]])
    if limit:
        command.extend(["--limit", str(limit)])
    if resume:
        command.append("--resume")
    return command


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.refresh_radius is not None and args.refresh_radius < 0:
        raise SystemExit("--refresh-radius must be non-negative")
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    overrides = parse_model_overrides(args.model_path)
    selected = set(args.run)
    known = {run["id"] for run in config["runs"]}
    unknown = selected.difference(known)
    if unknown:
        raise SystemExit("unknown run id(s): " + ", ".join(sorted(unknown)))

    output_dir = args.output_root.expanduser().resolve() / config_path.stem
    benchmark_root = args.benchmark_root.expanduser().resolve()
    if not args.dry_run and not benchmark_root.exists():
        raise SystemExit(f"benchmark root not found: {benchmark_root}")

    executed = 0
    for run in config["runs"]:
        validate_run(run, config_path)
        if selected and run["id"] not in selected:
            continue
        model_path = model_path_for(run, overrides, dry_run=args.dry_run)
        command = build_command(
            run,
            model_path=model_path,
            benchmark_root=benchmark_root,
            output_dir=output_dir,
            limit=args.limit,
            resume=args.resume,
            refresh_radius=args.refresh_radius,
        )
        print(f"[{run['id']}] {shlex.join(command)}", flush=True)
        executed += 1
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    if executed == 0:
        raise SystemExit("no runs selected")


if __name__ == "__main__":
    main()
