from __future__ import annotations

import argparse
from pathlib import Path

from .manifest import build_trial_manifest, load_config
from .splits import build_group_folds, write_split_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized trial manifest.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    manifest = build_trial_manifest(config)
    manifest = build_group_folds(
        manifest,
        n_splits=config["splits"]["n_splits"],
        group_column=config["splits"]["group_column"],
        label_column=config["splits"].get("label_column", "injury_label"),
        method=config["splits"].get("method", "group_kfold"),
        random_seed=int(config["splits"].get("random_seed", 42)),
    )

    outputs_root = Path(config["data"]["outputs_root"])
    manifest_dir = outputs_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / "trials_manifest.csv"
    summary_path = manifest_dir / "split_summary.json"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    write_split_summary(manifest, summary_path)

    print(f"manifest saved: {manifest_path}")
    print(f"summary saved: {summary_path}")


if __name__ == "__main__":
    main()
