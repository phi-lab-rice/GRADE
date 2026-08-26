#!/usr/bin/env python3
"""Run diffusion inference with validation-matched RadarDepth conditioning."""

from inference import parse_stage_args, run_diffusion


def main() -> None:
    args = parse_stage_args(
        "Run ours_diffusion with the train_diffusion.validate() sampling setup."
    )
    run_diffusion(args.config)


if __name__ == "__main__":
    main()
