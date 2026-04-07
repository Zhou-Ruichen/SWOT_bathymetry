#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型配置生成器，兼容单域与多域训练方案"""

from datetime import datetime
import copy
import hashlib
from typing import Any, Dict, List, Optional

BASE_CONFIG: Dict[str, Any] = {
    "max_epochs": 400,
    "early_stop_patience": 12,
    "num_workers": 12,
    "random_seed": 42,
    "use_test_set": True,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "use_lr_scheduler": True,
    "log_grad_hist_every": 5,
    "grad_hist_bins": 30,
    "grad_sample_cap": 200_000,
    "swan_project": "swot",
    "metrics_csv_path": None,
}

DATASET_PRESETS: Dict[str, List[str]] = {
    "single": [
        # "output/1-data/bandpass_1025_0644_global_norm.npz",
        "output/1-data/bandpass_R2_norm.npz",
    ],
    "multi": [
        # "output/1-data/bandpass_1025_0642_global_norm.npz",
        # "output/1-data/bandpass_1025_0643_global_norm.npz",
        # "output/1-data/bandpass_1025_0644_global_norm.npz",
        # "output/1-data/bandpass_1025_0645_global_norm.npz",
        # "output/1-data/bandpass_R0_norm.npz",
        # "output/1-data/bandpass_R1_norm.npz",
        "output/1-data/bandpass_R2_norm.npz",
        "output/1-data/bandpass_R3_norm.npz",
        "output/1-data/bandpass_R4_norm.npz",
        # "output/1-data/bandpass_R5_norm.npz",
        # "output/1-data/bandpass_R6_norm.npz",
        # "output/1-data/bandpass_R7_norm.npz",
    ],
}

DEFAULT_DATASET_KEY: Dict[str, str] = {
    "single": "single",
    "multi": "multi",
}

MODEL_PROFILES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "unet": {
        "single": {
            "patch_size": 256,  # UNet 可以轻松处理 256
            "batch_size": 16,  # 稍微放宽一点
            "accumulate_grad_batches": 4,  # Effective = 64
            "learning_rate": 1.5e-4,
            "weight_decay": 1e-5,
            "use_augmentation": False,
            "use_rotation": False,
            "use_noise": False,
            "use_cutout": False,
            "gradient_clip_val": 0.0,
            "loss_params": {
                "mse_weight": 1.0,
                "mae_weight": 0.0,
                "gradient_weight": 0.5,
                "ssim_weight": 0.05,
                "use_tid_weighting": True,
            },
        },
        "multi": {
            "patch_size": 256,
            "batch_size": 16,
            "accumulate_grad_batches": 4,
            "learning_rate": 1.2e-4,
            "weight_decay": 3e-5,
            "use_augmentation": False,
            "use_rotation": False,
            "use_noise": False,
            "use_cutout": False,
            "balance_domains": False,
            "gradient_clip_val": 0.0,
            "loss_params": {
                "mse_weight": 1.0,
                "mae_weight": 0.0,
                "gradient_weight": 0.6,
                "ssim_weight": 0.05,
                "use_tid_weighting": True,
            },
        },
    },
    "attention_unet": {
        "single": {
            "patch_size": 256,  # Attn UNet 计算高效，也可以 256
            "batch_size": 12,
            "accumulate_grad_batches": 6,  # Effective ~72
            "learning_rate": 9e-5,
            "weight_decay": 1e-5,
            "use_augmentation": False,
            "use_rotation": False,
            "use_noise": False,
            "use_cutout": False,
            "gradient_clip_val": 1.0,
            "loss_params": {
                "mse_weight": 1.0,
                "mae_weight": 0.0,
                "gradient_weight": 1.0,
                "ssim_weight": 0.05,
                "use_tid_weighting": True,
            },
        },
        "multi": {
            "patch_size": 256,
            "batch_size": 8,
            "accumulate_grad_batches": 8,  # Effective 64
            "learning_rate": 7e-5,
            "weight_decay": 1e-5,
            "use_augmentation": False,
            "use_rotation": False,
            "use_noise": False,
            "use_cutout": False,
            "balance_domains": False,
            "gradient_clip_val": 1.0,
            "bilinear": True,
            "loss_params": {
                "mse_weight": 1.0,
                "mae_weight": 0.0,
                "gradient_weight": 0.6,
                "ssim_weight": 0.05,
                "use_tid_weighting": True,
            },
        },
    },
    "transformer_unet": {
        "single": {
            "patch_size": 128,
            "batch_size": 16,
            "accumulate_grad_batches": 4,  # Effective 64
            "learning_rate": 4e-5,
            "weight_decay": 1e-5,
            "use_augmentation": False,
            "use_rotation": False,
            "use_noise": False,
            "use_cutout": False,
            "gradient_clip_val": 1.0,
            "loss_params": {
                "mse_weight": 1.0,
                "mae_weight": 0.0,
                "gradient_weight": 0.6,
                "ssim_weight": 0.05,
                "use_tid_weighting": True,
            },
        },
        "multi": {
            "patch_size": 64,  # 复现目标配置
            "batch_size": 16,  # 复现目标配置
            "accumulate_grad_batches": 4,  # 复现目标配置
            "learning_rate": 3e-5,
            "weight_decay": 5e-5,
            "use_augmentation": True,  # 复现目标配置
            "use_rotation": False,
            "use_noise": False,
            "use_cutout": False,
            "balance_domains": False,
            "gradient_clip_val": 1.0,
            "loss_params": {
                "mse_weight": 1.0,
                "mae_weight": 0.0,
                "gradient_weight": 1.0,
                "ssim_weight": 0.05,
                "use_tid_weighting": True,
            },
        },
    },
}

MODEL_ABBR: Dict[str, str] = {
    "unet": "unet",
    "attention_unet": "aunet",
    "transformer_unet": "tunet",
}


def _resolve_dataset(domain: str, dataset_key: Optional[str]) -> List[str]:
    key = dataset_key or DEFAULT_DATASET_KEY[domain]
    if key not in DATASET_PRESETS:
        raise ValueError(f"未知的数据集配置: {key}")
    return copy.deepcopy(DATASET_PRESETS[key])


def _build_experiment_name(
    config: Dict[str, Any], domain: str, dataset_key: str
) -> str:
    model_key = str(config["model_type"])
    abbr = MODEL_ABBR.get(model_key, model_key.replace("_", "")[:6].lower())
    lr_val = float(config.get("learning_rate", 1e-4))
    lr_sci = f"{lr_val:.0e}".replace("e-0", "e-").replace("e+0", "e+")

    hash_basis = {
        "model": model_key,
        "domain": domain,
        "dataset": dataset_key,
        "patch_size": config.get("patch_size"),
        "batch_size": config.get("batch_size"),
        "accumulate_grad_batches": config.get("accumulate_grad_batches"),
        "learning_rate": lr_val,
        "weight_decay": config.get("weight_decay"),
        "use_augmentation": config.get("use_augmentation"),
        "use_rotation": config.get("use_rotation"),
        "use_noise": config.get("use_noise"),
        "use_cutout": config.get("use_cutout"),
        "loss": config.get("loss_params"),
    }

    hash_token = hashlib.sha1(str(hash_basis).encode("utf-8")).hexdigest()[:4]
    timestamp = datetime.now().strftime("%m%d-%H%M")
    domain_tag = "multi" if domain == "multi" else "single"
    return (
        f"{abbr}-{domain_tag}-p{config.get('patch_size')}-b{config.get('batch_size')}"
        f"-lr{lr_sci}-{timestamp}-{hash_token}"
    )


def get_model_config(
    model_type: str = "transformer_unet",
    domain: str = "single",
    dataset_key: Optional[str] = None,
) -> Dict[str, Any]:
    model_key = model_type.replace("_hybrid", "")
    if model_key not in MODEL_PROFILES:
        raise ValueError(f"不支持的模型类型: {model_type}")
    if domain not in MODEL_PROFILES[model_key]:
        raise ValueError(f"模型 {model_type} 不支持域类型 {domain}")

    config = copy.deepcopy(BASE_CONFIG)
    config.update(copy.deepcopy(MODEL_PROFILES[model_key][domain]))
    config["model_type"] = model_key

    resolved_dataset_key = dataset_key or DEFAULT_DATASET_KEY[domain]
    config["data_path"] = _resolve_dataset(domain, resolved_dataset_key)
    config["experiment_name"] = _build_experiment_name(
        config, domain, resolved_dataset_key
    )

    return config
