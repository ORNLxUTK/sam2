import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List

import yaml
from rich import print

BASE_CONFIGS = {
    "tiny": "/lustre/isaac24/proj/UTK0388/DomainSpecific/sam2/sam2/configs/sam2.1_training/MAZAK_finetune_tiny.yaml",
    "small": "/lustre/isaac24/proj/UTK0388/DomainSpecific/sam2/sam2/configs/sam2.1_training/MAZAK_finetune_small.yaml",
    "base_plus": "/lustre/isaac24/proj/UTK0388/DomainSpecific/sam2/sam2/configs/sam2.1_training/MAZAK_finetune_baseplus.yaml",
    "large": "/lustre/isaac24/proj/UTK0388/DomainSpecific/sam2/sam2/configs/sam2.1_training/MAZAK_finetune_large.yaml",
}

DATASETS_PATHS: Dict[str, Dict[str, str]] = {}
for dataset in Path("/lustre/isaac24/proj/UTK0388/SAM2imagescrossvalidation").iterdir():
    if dataset.is_dir():
        DATASETS_PATHS[dataset.name] = {
            "img_folder": str(dataset / "JPEGImages" / "train"),
            "gt_folder": str(dataset / "Annotations" / "train"),
        }

CHECKPOINT_PATHS: Dict[str, str] = {
    "tiny": "./checkpoints/sam2.1_hiera_tiny.pt",
    "small": "./checkpoints/sam2.1_hiera_small.pt",
    "base_plus": "./checkpoints/sam2.1_hiera_base_plus.pt",
    "large": "./checkpoints/sam2.1_hiera_large.pt",
}


@dataclass(frozen=True)
class Combination:
    checkpoint: str
    dataset: Dict[str, str]

    @property
    def baseconfig(self) -> str:
        baseconfig_str = Path(self.checkpoint).stem.split("sam2.1_hiera_")[-1]
        assert baseconfig_str in BASE_CONFIGS.keys()
        return BASE_CONFIGS[baseconfig_str]

    @property
    def modelname(self) -> str:
        return Path(self.checkpoint).stem.split("_")[-1]

    @property
    def datasetname(self) -> str:
        return Path(self.dataset["img_folder"]).parent.parent.name

    @property
    def resolution(self) -> int:
        dataset_name = self.datasetname
        if "MAZAK" in dataset_name:
            return 1024
        elif "irPOLYMER" in dataset_name:
            return 256
        elif "visPOLYMER" in dataset_name:
            return 512
        elif "TIG" in dataset_name:
            return 768
        elif "PLASMA" in dataset_name:
            return 768
        else:
            raise ValueError(f"Dataset name {dataset_name} not found")


def load_base_yaml(config_path: str) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found")
    if not config_path.suffix == ".yaml":
        raise ValueError(
            f"Config files must be in YAML format. Current file is {config_path.suffix}"
        )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def update_config(config: dict, config_combination: Combination) -> dict:
    config["scratch"]["resolution"] = config_combination.resolution
    config["scratch"]["num_epochs"] = 200
    config["dataset"]["img_folder"] = config_combination.dataset["img_folder"]
    config["dataset"]["gt_folder"] = config_combination.dataset["gt_folder"]
    config["trainer"]["checkpoint"]["model_weight_initializer"]["state_dict"][
        "checkpoint_path"
    ] = config_combination.checkpoint
    config["launcher"]["gpus_per_node"] = 1
    config["submitit"]["name"] = (
        f"FT_{config_combination.datasetname[0]}_{config_combination.modelname[0]}"
    )
    return config


def write_config(config_combination: Combination, root_dir: Path) -> None:
    assert root_dir.exists() and root_dir.is_dir(), (
        f"Root directory {root_dir} does not exist or is not a directory."
    )
    save_path = (
        root_dir
        / f"{config_combination.datasetname}_FullFinetune_{config_combination.modelname}.yaml"
    )
    config = load_base_yaml(config_combination.baseconfig)
    config = update_config(config, config_combination)
    with open(save_path, "w") as f:
        f.write("# @package _global_\n")
        yaml.dump(config, f, sort_keys=False)


def create_combinations() -> List[Combination]:
    combinations = list(product(CHECKPOINT_PATHS.keys(), DATASETS_PATHS.keys()))
    combinations = [
        Combination(CHECKPOINT_PATHS[x[0]], DATASETS_PATHS[x[1]]) for x in combinations
    ]
    return combinations


def get_user_slurm_jobs():
    """Get the number of queued/running jobs for the current user."""
    try:
        # Get current user
        username = os.getenv("USER")

        # Run squeue to get job count
        result = subprocess.run(
            ["squeue", "-u", username, "-h"], capture_output=True, text=True, check=True
        )

        # Count lines (each line is a job)
        output_lines = result.stdout.strip()
        if not output_lines:
            return 0

        job_count = len([line for line in output_lines.split("\n") if line.strip()])
        return job_count
    except subprocess.CalledProcessError:
        print("Warning: Could not check SLURM queue status")
        return 0
    except FileNotFoundError:
        print("Warning: SLURM commands not available")
        return 0


def wait_for_job_slots(max_jobs=28, check_interval=60):
    """Wait until there are fewer than max_jobs in the queue."""
    while True:
        current_jobs = get_user_slurm_jobs()
        print(f"Current jobs in queue: {current_jobs}/{max_jobs}")

        if current_jobs < max_jobs:
            print(f"Queue has space ({current_jobs}/{max_jobs}), proceeding...")
            break
        else:
            print(
                f"Queue is full ({current_jobs}/{max_jobs}), waiting {check_interval} seconds..."
            )
            print(f"Check queue status with: squeue -u {os.getenv('USER')}")
            time.sleep(check_interval)


def submit_jobs(yaml_config_dir: Path) -> None:
    for yaml_config in sorted(yaml_config_dir.glob("*.yaml")):
        config = yaml_config.relative_to(yaml_config_dir.parent.parent.parent)
        wait_for_job_slots(max_jobs=28, check_interval=60)
        cmd = ["uv", "run", "--active", "training/train.py", "-c", str(config)]
        try:
            output: subprocess.CompletedProcess = subprocess.run(
                cmd, check=True, capture_output=True
            )
            print(
                f"[green] Submitted job [/green] [blue]{yaml_config}[/blue]: [light_green]{output.stdout}[/light_green]"
            )
        except subprocess.CalledProcessError as e:
            print(f"[red] Error submitting job [/red] [blue]{yaml_config}[/blue]: {e}")
            print(f"[yellow] Output: [/yellow] {e.stdout}")
            print(f"[red] Error: [/red] {e.stderr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-configs", action="store_true", default=False)
    parser.add_argument("--submit-jobs", action="store_true", default=False)
    args = parser.parse_args()
    if args.create_configs:
        config_combinations = create_combinations()
        print(f"Number of combinations: {len(config_combinations)}")
        for config in config_combinations:
            write_config(
                config,
                Path(
                    "/lustre/isaac24/proj/UTK0388/DomainSpecific/sam2/sam2/configs/sam2.1_training/FullFinetunecrossvalidation"
                ),
            )
    if args.submit_jobs:
        submit_jobs(
            Path(
                "/lustre/isaac24/proj/UTK0388/DomainSpecific/sam2/sam2/configs/sam2.1_training/FullFinetunecrossvalidation"
            )
        )
