from __future__ import annotations

import argparse
from pathlib import Path

from .data import RAW_PATH, download_data, write_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="warfarin-dose")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-data", help="download and verify public IWPC data")
    download.add_argument("--output", type=str, default=str(RAW_PATH))
    audit = commands.add_parser("audit-data", help="build stable-dose cohort audit artifacts")
    audit.add_argument("--input", type=str, default=str(RAW_PATH))
    audit.add_argument("--output", type=str, default="artifacts/audit")
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
    raise AssertionError(f"unhandled command: {args.command}")
