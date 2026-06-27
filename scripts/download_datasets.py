"""Dataset download helpers using Kaggle API."""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)

DATASETS = {
    "chest_xray": {
        "kaggle_id": "paultimothymooney/chest-xray-pneumonia",
        "target_dir": "datasets/chest_xray",
        "description": "Chest X-Ray Pneumonia Detection",
    },
    "brain_tumor": {
        "kaggle_id": "sartajbhuvaji/brain-tumor-classification-mri",
        "target_dir": "datasets/brain_tumor",
        "description": "Brain Tumor MRI Classification",
    },
    "skin_cancer": {
        "kaggle_id": "kmader/skin-cancer-mnist-ham10000",
        "target_dir": "datasets/skin_cancer",
        "description": "HAM10000 Skin Lesion Dataset",
    },
    "diabetic_retinopathy": {
        "kaggle_id": "competitions/aptos2019-blindness-detection",
        "target_dir": "datasets/diabetic_retinopathy",
        "description": "APTOS 2019 Diabetic Retinopathy",
    },
}


@click.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(list(DATASETS.keys()) + ["all"]),
    default="all",
    help="Dataset to download",
)
@click.option("--output-dir", "-o", default="datasets", help="Root output directory")
def main(dataset, output_dir):
    """Download Kaggle medical imaging datasets."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        click.echo("Install kaggle: pip install kaggle")
        click.echo("Then set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        sys.exit(1)

    targets = DATASETS if dataset == "all" else {dataset: DATASETS[dataset]}

    for name, info in targets.items():
        click.echo(f"\nDownloading {info['description']}...")
        target = Path(output_dir) / Path(info["target_dir"]).name
        target.mkdir(parents=True, exist_ok=True)

        is_competition = info["kaggle_id"].startswith("competitions/")
        competition_id = info["kaggle_id"].split("/")[-1] if is_competition else None

        if is_competition:
            cmd = ["kaggle", "competitions", "download", "-c", competition_id, "-p", str(target)]
        else:
            cmd = ["kaggle", "datasets", "download", "-d", info["kaggle_id"], "-p", str(target), "--unzip"]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            click.echo(f"  Error: {result.stderr}")
        else:
            click.echo(f"  Downloaded to {target}")

    click.echo("\nDataset download complete.")
    click.echo("Organize files as: datasets/<task>/<ClassName>/<image.jpg>")


if __name__ == "__main__":
    main()
