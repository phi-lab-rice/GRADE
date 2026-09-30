#!/usr/bin/env python3
"""Run full ControlNet inference with validation-matched RadarDepth conditioning."""

from inference import parse_stage_args, run_full


def main() -> None:
    args = parse_stage_args(
        "Run ours_full with the train_control-defish.validate() sampling setup."
    )
    run_full(args.config)


if __name__ == "__main__":
    main()
