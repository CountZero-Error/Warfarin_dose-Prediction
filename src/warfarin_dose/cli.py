from __future__ import annotations

import argparse
from pathlib import Path

from .data import RAW_PATH, download_data, read_raw, sha256_file, write_audit
from .evaluation import (
    DEFAULT_SEED,
    run_ablation_frame,
    run_all_analyses,
    run_complete_case_frame,
    run_feature_selection_frame,
    run_primary_experiment,
    run_random_cv_frame,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="warfarin-dose")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-data", help="download and verify public IWPC data")
    download.add_argument("--output", type=str, default=str(RAW_PATH))
    audit = commands.add_parser("audit-data", help="build stable-dose cohort audit artifacts")
    audit.add_argument("--input", type=str, default=str(RAW_PATH))
    audit.add_argument("--output", type=str, default="artifacts/audit")
    experiment = commands.add_parser("run-experiment", help="run a prespecified research analysis")
    experiment.add_argument(
        "--analysis",
        choices=["primary", "feature-selection", "complete-case", "random-cv", "ablation", "all"],
        default="primary",
    )
    experiment.add_argument("--input", type=str, default=str(RAW_PATH))
    experiment.add_argument("--output", type=str, default="artifacts/run")
    experiment.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "download-data":
        manifest = download_data(destination=Path(args.output))
        print(f"verified {manifest['sha256']} at {manifest['path']}")
        return 0
    if args.command == "audit-data":
        summary = write_audit(Path(args.input), Path(args.output))
        print(f"source_rows: {summary['source_rows']}")
        print(f"eligible_rows: {summary['eligible_rows']}")
        print(f"sites: {summary['sites']}")
        print(f"output: {args.output}")
        return 0
    if args.command == "run-experiment":
        root = Path(args.output)
        if args.analysis == "all":
            output = run_all_analyses(Path(args.input), root, seed=args.seed)
        elif args.analysis == "primary":
            output = run_primary_experiment(Path(args.input), root / "primary", seed=args.seed)
        else:
            raw = read_raw(Path(args.input))
            raw.attrs["source_sha256"] = sha256_file(Path(args.input))
            primary = root / "primary"
            if not (primary / "manifest.json").exists():
                run_primary_experiment(Path(args.input), primary, seed=args.seed)
            runners = {
                "feature-selection": lambda: run_feature_selection_frame(
                    raw, primary, root / "feature-selection", seed=args.seed
                ),
                "complete-case": lambda: run_complete_case_frame(
                    raw, root / "complete-case", seed=args.seed
                ),
                "random-cv": lambda: run_random_cv_frame(raw, root / "random-cv", seed=args.seed),
                "ablation": lambda: run_ablation_frame(raw, root / "ablation", seed=args.seed),
            }
            output = runners[args.analysis]()
        print(f"output: {output}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")
