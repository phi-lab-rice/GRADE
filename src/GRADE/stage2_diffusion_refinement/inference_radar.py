#!/usr/bin/env python3
"""Run the radar-only Smoke-Eval inference pass."""

from inference import parse_stage_args, run_radar


def main() -> None:
    args = parse_stage_args("Run ours_radar inference on all Smoke-Eval sequences.")
    run_radar(args.config)


if __name__ == "__main__":
    main()
