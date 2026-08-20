"""CLI for one immutable clean-flow training run."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from interp.activations import load_activations, load_validation_report
from interp.train_flow import load_training_config, train_flow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--activation-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()

    config = load_training_config(args.config)
    dataset = load_activations(config.dataset_name, args.activation_dir)
    report = load_validation_report(
        config.dataset_name,
        args.activation_dir,
        expected_split_fingerprint=config.split_fingerprint,
        verify_hashes=True,
    )
    dataset = replace(
        dataset,
        meta={**dataset.meta, "full_validation_report": report},
    )
    train_flow(
        dataset,
        config,
        args.run_dir,
        device=args.device,
        progress=True,
        resume_checkpoint=args.resume_checkpoint,
    )


if __name__ == "__main__":
    main()
