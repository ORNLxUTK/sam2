import yaml
import sys
from pathlib import Path
from itertools import product
from enum import Enum
from typing import Dict, List, Tuple
from rich import print
from dataclasses import dataclass

BASE_CONFIGS = {
    "tiny" : "/lustre/isaac24/proj/UTK0388/DomainSpecific/sam2/sam2/configs/sam2.1_training/MAZAK_LoRA4_tiny.yaml",
    "small" : "/lustre/isaac24/proj/UTK0388/DomainSpecific/sam2/sam2/configs/sam2.1_training/MAZAK_LoRA4_small.yaml",
    "base_plus" : "/lustre/isaac24/proj/UTK0388/DomainSpecific/sam2/sam2/configs/sam2.1_training/MAZAK_LoRA4_baseplus.yaml",
    "large" : "/lustre/isaac24/proj/UTK0388/DomainSpecific/sam2/sam2/configs/sam2.1_training/MAZAK_LoRA4_large.yaml"
}

DATASETS_PATHS: Dict[str, Dict[str, str]] = { 
    "MAZAK" : {
        "img_folder": "/lustre/isaac24/proj/UTK0388/SAM2images/MAZAK/JPEGImages/train",
        "gt_folder": "/lustre/isaac24/proj/UTK0388/SAM2images/MAZAK/Annotations/train"
    },
    "irPOLYMER" : {
        "img_folder": "/lustre/isaac24/proj/UTK0388/SAM2images/irPOLYMER/JPEGImages/train",
        "gt_folder": "/lustre/isaac24/proj/UTK0388/SAM2images/irPOLYMER/Annotations/train"
    },
    "visPOLYMER" : {
        "img_folder": "/lustre/isaac24/proj/UTK0388/SAM2images/visPOLYMER/JPEGImages/train",
        "gt_folder": "/lustre/isaac24/proj/UTK0388/SAM2images/visPOLYMER/Annotations/train"
    },
    "TIG" : {
        "img_folder": "/lustre/isaac24/proj/UTK0388/SAM2images/TIG/JPEGImages/train",
        "gt_folder": "/lustre/isaac24/proj/UTK0388/SAM2images/TIG/Annotations/train"
    }
}

CHECKPOINT_PATHS: Dict[str, str] = {
    "tiny": "./checkpoints/sam2.1_hiera_tiny.pt",
    "small": "./checkpoints/sam2.1_hiera_small.pt",
    "base_plus": "./checkpoints/sam2.1_hiera_base_plus.pt",
    "large": "./checkpoints/sam2.1_hiera_large.pt"
}

LORA_RANKS: List[int] = [2, 4, 8, 16, 32]

@dataclass(frozen=True)
class Combination():
    rank: int
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
        match dataset_name:
            case "MAZAK":
                return 1024
            case "irPOLYMER": 
                return 256
            case "visPOLYMER":
                return 512
            case "TIG":
                return 768
            case _:
                raise ValueError(f"Dataset name {dataset_name} not found")
            
def load_base_yaml(config_path: str) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found")
    if not config_path.suffix == ".yaml":
        raise ValueError(f"Config files must be in YAML format. Current file is {config_path.suffix}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def update_config(config: dict, config_combination: Combination) -> dict:
    config["scratch"]["resolution"] = config_combination.resolution
    config["dataset"]["img_folder"] = config_combination.dataset["img_folder"]
    config["dataset"]["gt_folder"] = config_combination.dataset["gt_folder"]
    config["trainer"]["LoRA"]["r"] = config_combination.rank
    config["trainer"]["LoRA"]["adapter_name"] = f"SAM2_LoRA_{config_combination.modelname}"
    config["trainer"]["checkpoint"]["model_weight_initializer"]["state_dict"]["checkpoint_path"] = config_combination.checkpoint
    config["launcher"]["gpus_per_node"] = 1
    config["submitit"]["name"] = f"L{config_combination.rank}_{config_combination.datasetname[0]}_{config_combination.modelname[0]}"
    return config

def write_config(config_combination: Combination, root_dir: Path) -> None:
    save_path = root_dir / f"{config_combination.datasetname}_LoRA{config_combination.rank}_{config_combination.modelname}.yaml"
    config = load_base_yaml(config_combination.baseconfig)
    config = update_config(config, config_combination)
    with open(save_path, "w") as f:
        f.write("# @package _global_\n")
        yaml.dump(config, f, sort_keys=False)


def create_combinations() -> List[Combination]:
    combinations = list(product(LORA_RANKS, CHECKPOINT_PATHS.keys(), DATASETS_PATHS.keys()))
    combinations = [Combination(x[0], CHECKPOINT_PATHS[x[1]], DATASETS_PATHS[x[2]]) for x in combinations]
    return combinations


if __name__ == "__main__":
    config_combinations = create_combinations()
    print(f"Number of combinations: {len(config_combinations)}")
    for config in config_combinations:
        write_config(config, Path("/lustre/isaac24/proj/UTK0388/DomainSpecific/sam2/sam2/configs/sam2.1_training/ablations"))
