#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SWOT海底地形超分辨率训练脚本
(重构版 - 配置与代码分离)
"""

import os
import shutil
from collections import Counter
import csv
import torch
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm
import json
import logging
from datetime import datetime
import time
import gc
import argparse
import torch.nn as nn

from models import get_model, get_model_info
from losses import get_loss_function
from data_loader import get_dataloaders
from model_configs import get_model_config
from swan_monitor import Monitor  # 轻量监控（无依赖则自动降级）

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Trainer:
    """训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = os.path.join(
            "output", "2-experiments", config["experiment_name"]
        )
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        # 将当前的全局归一化参数快照拷贝到模型目录，保证训练-推理一致可追溯
        try:
            global_norm_src = os.path.join(
                "output", "1-data", "global_normalization_params.json"
            )
            if os.path.exists(global_norm_src):
                dst = os.path.join(self.output_dir, "global_normalization_params.json")
                shutil.copyfile(global_norm_src, dst)
                logger.info(f"归一化参数快照已保存到模型目录: {dst}")
        except Exception as e:
            logger.warning(f"归一化参数快照复制失败: {e}")
        logger.info(f"实验: {config['experiment_name']}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"设备: {self.device}")
        # 初始化 SwanLab 监控（强依赖：失败将抛错并中止训练）
        self.monitor = Monitor()
        # 项目名与实验名：若未配置项目名则使用更短默认(swot)
        self.monitor.init(
            config=self.config,
            experiment_name=self.config["experiment_name"],
            project=self.config.get("swan_project", "swot"),
            tags=[self.config.get("model_type", "unknown")],
        )
        self._setup()

    def _normalize_loaders(self, loaders, include_test: bool):
        """将 get_dataloaders 的返回统一为 (train, val, test|None)。
        使用 type: ignore 避免由于不同返回元组长度导致的静态类型告警。
        """
        if include_test:
            train_loader, val_loader, test_loader = loaders  # type: ignore
        else:
            train_loader, val_loader = loaders  # type: ignore
            test_loader = None
        return train_loader, val_loader, test_loader

    def _setup(self):
        """初始化模型、数据、优化器等"""
        try:
            logger.info("加载数据...")
            loaders = get_dataloaders(
                data_path=self.config["data_path"],
                batch_size=self.config["batch_size"],
                num_workers=self.config.get("num_workers", 4),
                use_augmentation=self.config.get("use_augmentation", False),
                use_rotation=self.config.get("use_rotation", False),
                use_noise=self.config.get("use_noise", False),
                use_cutout=self.config.get("use_cutout", False),
                train_ratio=self.config.get("train_ratio", 0.8),
                val_ratio=self.config.get("val_ratio", 0.1),
                test_ratio=self.config.get("test_ratio", 0.1),
                include_test=self.config.get("use_test_set", True),
                random_seed=self.config.get("random_seed", 42),
                patch_size=self.config.get("patch_size", 64),
                balance_domains=self.config.get("balance_domains", False),
                target_domain_weight=float(
                    self.config.get("target_domain_weight", 1.0)
                ),
            )
            # 兼容返回 (train, val) 或 (train, val, test)
            self.train_loader, self.val_loader, self.test_loader = (
                self._normalize_loaders(loaders, self.config.get("use_test_set", True))
            )

            logger.info("初始化模型...")
            self.model = get_model(
                model_type=self.config["model_type"],
                n_channels=4,
                n_classes=1,
                bilinear=self.config.get("bilinear", False),
            ).to(self.device)

            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            logger.info(f"模型参数: 总计={total_params:,}, 可训练={trainable_params:,}")

            logger.info("初始化损失函数...")
            # 假设losses.py的get_loss_function也接收device参数
            loss_device = self.config["loss_params"].copy()
            loss_device["device"] = self.device
            self.criterion = get_loss_function(**loss_device)
            logger.info(f"Loss 配置: {self.config['loss_params']}")

            # 优化器：支持 Adam 与 AdamW（通过 optimizer_type 指定）
            optimizer_type = self.config.get("optimizer_type", "adam").lower()
            weight_decay = float(self.config.get("weight_decay", 0.0))
            betas_cfg = self.config.get("optimizer_betas", [0.9, 0.999])
            if isinstance(betas_cfg, (list, tuple)) and len(betas_cfg) == 2:
                betas = (float(betas_cfg[0]), float(betas_cfg[1]))
            else:
                betas = (0.9, 0.999)
            eps = float(self.config.get("optimizer_eps", 1e-8))
            if optimizer_type == "adamw":
                self.optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=self.config["learning_rate"],
                    weight_decay=weight_decay,
                    betas=betas,
                    eps=eps,
                )
                logger.info(
                    f"优化器: AdamW (wd={weight_decay}, betas={betas}, eps={eps})"
                )
            else:
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.config["learning_rate"],
                    weight_decay=weight_decay,
                    betas=betas,
                    eps=eps,
                )
                logger.info(
                    f"优化器: Adam (wd={weight_decay}, betas={betas}, eps={eps})"
                )

            # 学习率调度器：支持 Plateau 与 余弦退火 + warmup
            self.scheduler_type = self.config.get("lr_scheduler_type", "plateau")
            if self.scheduler_type == "cosine_warmup":
                total_epochs = int(self.config.get("max_epochs", 400))
                warmup_epochs = int(self.config.get("warmup_epochs", 5))

                def lr_lambda(epoch: int):
                    if epoch < warmup_epochs:
                        return float(epoch + 1) / float(max(1, warmup_epochs))
                    progress = (epoch - warmup_epochs) / float(
                        max(1, total_epochs - warmup_epochs)
                    )
                    return 0.5 * (1.0 + math.cos(math.pi * progress))

                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=lr_lambda
                )
                logger.info(
                    f"LR调度: cosine_warmup, warmup_epochs={warmup_epochs}, total_epochs={total_epochs}"
                )
            else:
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-7
                )
                self.scheduler_type = "plateau"

            self.history = {"train_loss": [], "val_loss": [], "learning_rate": []}
            self.best_val_loss = float("inf")
            self.best_model_path = None
            self.patience_counter = 0
            self.early_stop_patience = self.config.get("early_stop_patience", 15)
            self.accumulate_grad_batches = self.config.get("accumulate_grad_batches", 1)
            if self.accumulate_grad_batches > 1:
                logger.info(f"梯度累积已启用，步数: {self.accumulate_grad_batches}")

            self.loss_weights = {
                "mse": float(self.config.get("loss_params", {}).get("mse_weight", 1.0)),
                "mae": float(self.config.get("loss_params", {}).get("mae_weight", 0.0)),
                "gradient": float(
                    self.config.get("loss_params", {}).get("gradient_weight", 0.0)
                ),
                "ssim": float(
                    self.config.get("loss_params", {}).get("ssim_weight", 0.0)
                ),
                "use_tid": bool(
                    self.config.get("loss_params", {}).get("use_tid_weighting", False)
                ),
            }
            self.val_vis_samples = self._prepare_vis_samples(num_samples=4)

        except Exception as e:
            logger.error(f"设置失败: {e}")
            raise

    def train_epoch(self, capture_grad: bool = False):
        self.model.train()
        total_loss = 0.0
        epoch_loss_components = {}  # 累加各个损失项
        grad_info = None  # (counts, edges, global_norm, max_abs)
        pbar = tqdm(self.train_loader, desc="训练")
        domain_counter = (
            Counter() if self.config.get("balance_domains", False) else None
        )

        self.optimizer.zero_grad()  # 梯度累积需要在循环外先清零

        for i, batch in enumerate(pbar):
            inputs = batch["input"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)
            tid_data = batch.get("tid", None)
            if tid_data is not None:
                tid_data = tid_data.to(self.device, non_blocking=True)

            if domain_counter is not None:
                domain_ids = batch.get("domain_idx")
                if domain_ids is not None:
                    if isinstance(domain_ids, torch.Tensor):
                        domain_counter.update(
                            int(x) for x in domain_ids.detach().cpu().tolist()
                        )
                    else:
                        domain_counter.update([int(domain_ids)])

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets, tid_data)

            # 记录各个损失项
            if hasattr(self.criterion, "loss_components"):
                for k, v in self.criterion.loss_components.items():
                    # 确保是 float 类型再累加
                    val_float = v.item() if isinstance(v, torch.Tensor) else float(v)
                    epoch_loss_components[k] = (
                        epoch_loss_components.get(k, 0.0) + val_float
                    )

            if self.accumulate_grad_batches > 1:
                loss = loss / self.accumulate_grad_batches  # 标准化损失

            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()

                # --- 梯度累积的核心逻辑 ---
                if (i + 1) % self.accumulate_grad_batches == 0:
                    if self.config.get("gradient_clip_val", 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.config["gradient_clip_val"],
                        )

                    # 捕获一次梯度分布
                    if capture_grad and grad_info is None:
                        try:
                            sample_cap = int(self.config.get("grad_sample_cap", 200000))
                            bins = int(self.config.get("grad_hist_bins", 30))
                            collected = []
                            remaining = max(1, sample_cap)
                            global_sq = 0.0
                            max_abs = 0.0
                            for p in self.model.parameters():
                                if p.grad is None:
                                    continue
                                g = p.grad.detach().float()
                                global_sq += float(torch.sum(g * g).item())
                                max_abs = max(
                                    max_abs, float(torch.max(torch.abs(g)).item())
                                )
                                if remaining <= 0:
                                    continue
                                flat = torch.abs(g).reshape(-1)
                                if flat.numel() == 0:
                                    continue
                                take = min(remaining, flat.numel())
                                chunk = flat[:take].detach().cpu().numpy()
                                collected.append(chunk)
                                remaining -= take
                            if collected:
                                import numpy as _np

                                arr = _np.concatenate(collected)
                                counts, edges = _np.histogram(arr, bins=bins)
                                grad_info = (
                                    counts,
                                    edges,
                                    float(global_sq) ** 0.5,
                                    float(max_abs),
                                )
                        except Exception:
                            grad_info = None

                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulate_grad_batches  # 累加回原始损失
            pbar.set_postfix(
                {"loss": f"{loss.item() * self.accumulate_grad_batches:.6f}"}
            )

        # 处理最后一个 batch 不足以凑成一个累积步长的情况
        if (len(self.train_loader)) % self.accumulate_grad_batches != 0:
            if self.config.get("gradient_clip_val", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config["gradient_clip_val"]
                )
            self.optimizer.step()
            self.optimizer.zero_grad()

        if domain_counter:
            base_dataset = getattr(
                getattr(self.train_loader, "dataset", None), "dataset", None
            )
            label_map = {}
            if base_dataset is not None and hasattr(base_dataset, "domains"):
                for idx, info in enumerate(base_dataset.domains):
                    path = info.get("path") if isinstance(info, dict) else None
                    label_map[idx] = (
                        os.path.splitext(os.path.basename(path))[0]
                        if path
                        else str(idx)
                    )
            formatted = {
                label_map.get(idx, str(idx)): count
                for idx, count in sorted(domain_counter.items())
            }
            logger.info(f"本轮训练采样分布: {formatted}")

        avg_loss_components = {
            k: v / len(self.train_loader) for k, v in epoch_loss_components.items()
        }
        return total_loss / len(self.train_loader), avg_loss_components, grad_info

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        val_loss_components = {}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证"):
                inputs = batch["input"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)
                tid_data = batch.get("tid", None)
                if tid_data is not None:
                    tid_data = tid_data.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets, tid_data)

                # 记录各个损失项
                if hasattr(self.criterion, "loss_components"):
                    for k, v in self.criterion.loss_components.items():
                        # 确保是 float 类型再累加
                        val_float = (
                            v.item() if isinstance(v, torch.Tensor) else float(v)
                        )
                        val_loss_components[k] = (
                            val_loss_components.get(k, 0.0) + val_float
                        )

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()

        avg_loss_components = {
            k: v / len(self.val_loader) for k, v in val_loss_components.items()
        }
        return total_loss / len(self.val_loader), avg_loss_components

    def _is_rank0(self):
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return torch.distributed.get_rank() == 0
        except Exception:
            return True
        return True

    def _prepare_vis_samples(self, num_samples: int = 4):
        if self.val_loader is None:
            return None
        try:
            batch = next(iter(self.val_loader))
        except Exception:
            logger.warning("验证集为空，无法准备可视化样本")
            return None
        samples = {
            "input": batch["input"][:num_samples].cpu(),
            "target": batch["target"][:num_samples].cpu(),
        }
        if "tid" in batch:
            samples["tid"] = batch["tid"][:num_samples].cpu()
        return samples

    def _compute_ssim_map(self, pred_norm: torch.Tensor, target_norm: torch.Tensor):
        window_size = 11
        pad = window_size // 2
        device = pred_norm.device
        dtype = pred_norm.dtype
        kernel = torch.ones(
            (1, 1, window_size, window_size), device=device, dtype=dtype
        )
        kernel = kernel / (window_size * window_size)
        mu_x = torch.nn.functional.conv2d(pred_norm, kernel, padding=pad)
        mu_y = torch.nn.functional.conv2d(target_norm, kernel, padding=pad)
        sigma_x2 = (
            torch.nn.functional.conv2d(pred_norm * pred_norm, kernel, padding=pad)
            - mu_x * mu_x
        )
        sigma_y2 = (
            torch.nn.functional.conv2d(target_norm * target_norm, kernel, padding=pad)
            - mu_y * mu_y
        )
        sigma_xy = (
            torch.nn.functional.conv2d(pred_norm * target_norm, kernel, padding=pad)
            - mu_x * mu_y
        )
        c1 = 0.01**2
        c2 = 0.03**2
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x2 + sigma_y2 + c2)
        ssim_map = numerator / (denominator + 1e-8)
        return (1 - ssim_map.clamp(0, 1)).clamp(min=0)

    def _save_loss_map_images(self, base_path, name, arr):
        # 仅保存数据文件，绘图留到 Notebook/后处理
        npy_path = os.path.join(base_path, f"{name}.npy")
        txt_path = os.path.join(base_path, f"{name}.txt")
        np.save(npy_path, arr)
        np.savetxt(txt_path, arr.reshape(arr.shape[0], -1))

    def _save_val_loss_maps(self, epoch: int):
        interval = int(self.config.get("val_map_interval", 1))
        if epoch % interval != 0:
            return
        if not self._is_rank0():
            return
        if self.val_vis_samples is None:
            return

        base_dir = os.path.join(self.output_dir, "val_loss_maps", f"epoch_{epoch:04d}")
        os.makedirs(base_dir, exist_ok=True)

        inputs = self.val_vis_samples["input"].to(self.device)
        targets = self.val_vis_samples["target"].to(self.device)
        tid = self.val_vis_samples.get("tid")
        if tid is not None:
            tid = tid.to(self.device)

        mse_w = self.loss_weights.get("mse", 0.0)
        mae_w = self.loss_weights.get("mae", 0.0)
        grad_w = self.loss_weights.get("gradient", 0.0)
        ssim_w = self.loss_weights.get("ssim", 0.0)

        with torch.no_grad():
            self.model.eval()
            preds = self.model(inputs)

            weights = None
            if tid is not None and self.loss_weights.get("use_tid", False):
                try:
                    weights = self.criterion.get_tid_weights(tid)
                    weight_sum = weights.sum()
                    if weight_sum > 1e-8:
                        weights = weights / (weight_sum / weights.numel())
                except Exception:
                    weights = None

            mse_map = (preds - targets) ** 2
            if weights is not None:
                mse_map = weights * mse_map

            mae_map = torch.abs(preds - targets)
            if weights is not None:
                mae_map = weights * mae_map

            pred_grad = self.criterion.sobel(preds)
            target_grad = self.criterion.sobel(targets)
            grad_map = (pred_grad - target_grad) ** 2
            if weights is not None:
                grad_wt = torch.nn.functional.interpolate(
                    weights, size=pred_grad.shape[-2:], mode="nearest"
                )
                grad_map = grad_wt * grad_map

            pred_norm = self.criterion.normalize_for_ssim(preds)
            target_norm = self.criterion.normalize_for_ssim(targets)
            ssim_map = self._compute_ssim_map(pred_norm, target_norm)
            if weights is not None:
                ssim_map = weights * ssim_map

            total_map = torch.zeros_like(mse_map)
            if mse_w > 0:
                total_map += mse_w * mse_map
            if mae_w > 0:
                total_map += mae_w * mae_map
            if grad_w > 0:
                total_map += grad_w * grad_map
            if ssim_w > 0:
                total_map += ssim_w * ssim_map

        preds_cpu = preds.cpu()
        mse_cpu = mse_map.cpu()
        grad_cpu = grad_map.cpu()
        ssim_cpu = ssim_map.cpu()
        total_cpu = total_map.cpu()

        for idx in range(min(4, preds_cpu.shape[0])):
            sample_dir = os.path.join(base_dir, f"sample_{idx:02d}")
            os.makedirs(sample_dir, exist_ok=True)
            mse_arr = mse_cpu[idx, 0].numpy()
            grad_arr = grad_cpu[idx, 0].numpy()
            ssim_arr = ssim_cpu[idx, 0].numpy()
            total_arr = total_cpu[idx, 0].numpy()

            self._save_loss_map_images(sample_dir, "mse_map", mse_arr)
            self._save_loss_map_images(sample_dir, "grad_map", grad_arr)
            self._save_loss_map_images(sample_dir, "ssim_map", ssim_arr)
            self._save_loss_map_images(sample_dir, "total_map", total_arr)

    def test_on_test_set(self):
        if not self.test_loader:
            return None
        logger.info("🧪 开始在测试集上评估...")
        self.model.eval()
        total_loss, num_batches = 0.0, 0
        loss_accumulators = {"mse": 0.0, "mae": 0.0, "gradient": 0.0, "ssim": 0.0}
        num_test_samples = 0
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="测试集评估"):
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)
                num_test_samples += inputs.size(0)
                tid_data = batch.get("tid", None)
                if tid_data is not None:
                    tid_data = tid_data.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets, tid_data)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    if hasattr(self.criterion, "loss_components"):
                        for name, value in self.criterion.loss_components.items():
                            val_float = (
                                value.item()
                                if isinstance(value, torch.Tensor)
                                else float(value)
                            )
                            if name not in loss_accumulators:
                                loss_accumulators[name] = 0.0
                            loss_accumulators[name] += val_float
                    num_batches += 1
        if num_batches > 0:
            avg_total_loss = total_loss / num_batches
            rmse = (
                np.sqrt(loss_accumulators["mse"] / num_batches)
                if "mse" in loss_accumulators and loss_accumulators["mse"] > 0
                else 0.0
            )
            test_results = {
                "total_loss": avg_total_loss,
                "rmse": rmse,
                "num_samples": num_test_samples,
            }
            logger.info("🎯 测试集评估结果:")
            logger.info(f"总损失: {avg_total_loss:.6f}, RMSE: {rmse:.6f}")
            for name, total_value in loss_accumulators.items():
                # 只要累积值大于0，就尝试记录（不再强校验配置权重，以防是默认开启）
                if total_value > 0:
                    avg_loss = total_value / num_batches
                    test_results[f"{name}_loss"] = avg_loss
                    logger.info(f"{name.upper()}损失: {avg_loss:.6f}")
            logger.info(f"测试样本数: {test_results['num_samples']}")
            with open(os.path.join(self.output_dir, "test_results.json"), "w") as f:
                json.dump(test_results, f, indent=2)

            # 若配置指定了CSV路径，则追加写入测试结果，便于汇总
            metrics_csv_path = self.config.get("metrics_csv_path")
            if metrics_csv_path:
                try:
                    metrics_dir = os.path.dirname(metrics_csv_path)
                    if metrics_dir:
                        os.makedirs(metrics_dir, exist_ok=True)
                    fieldnames = [
                        "timestamp",
                        "experiment_name",
                        "total_loss",
                        "rmse",
                        "num_samples",
                        "mse_loss",
                        "mae_loss",
                        "gradient_loss",
                        "ssim_loss",
                    ]
                    row = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "experiment_name": self.config.get("experiment_name", ""),
                        "total_loss": test_results.get("total_loss"),
                        "rmse": test_results.get("rmse"),
                        "num_samples": test_results.get("num_samples"),
                        "mse_loss": test_results.get("mse_loss"),
                        "mae_loss": test_results.get("mae_loss"),
                        "gradient_loss": test_results.get("gradient_loss"),
                        "ssim_loss": test_results.get("ssim_loss"),
                    }
                    file_exists = os.path.exists(metrics_csv_path)
                    with open(metrics_csv_path, "a", newline="") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        if not file_exists or os.path.getsize(metrics_csv_path) == 0:
                            writer.writeheader()
                        writer.writerow(row)
                    logger.info(f"测试结果已追加到CSV: {metrics_csv_path}")
                except Exception as e:
                    logger.warning(f"写入测试结果CSV失败: {e}")
            return test_results
        return None

    def save_checkpoint(self, epoch, is_best=False):
        if is_best:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "history": self.history,
                "best_val_loss": self.best_val_loss,
            }
            self.best_model_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(checkpoint, self.best_model_path)
            logger.info(
                f"💾 保存最佳模型: epoch {epoch}, 验证损失: {self.best_val_loss:.6f}"
            )
            # 记录最佳指标与模型路径
            self.monitor.set_summary("best_val_loss", float(self.best_val_loss))

    def train(self):
        logger.info(
            f"开始训练... 目标: {self.config['max_epochs']} epochs, 早停patience={self.early_stop_patience}"
        )
        start_time = time.time()
        for epoch in range(1, self.config["max_epochs"] + 1):
            epoch_start = time.time()
            logger.info(f"Epoch {epoch}/{self.config['max_epochs']}")
            capture_grad = bool(self.config.get("log_grad_hist_every", 0)) and (
                (epoch % int(self.config.get("log_grad_hist_every", 0))) == 0
                if int(self.config.get("log_grad_hist_every", 0)) > 0
                else False
            )
            train_loss, train_loss_components, grad_info = self.train_epoch(
                capture_grad=capture_grad
            )
            val_loss, val_loss_components = self.validate()
            # 根据不同的调度器类型执行 step
            if getattr(self, "scheduler_type", "plateau") == "plateau":
                self.scheduler.step(val_loss)  # type: ignore[arg-type]
            else:
                self.scheduler.step()  # type: ignore[call-arg]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

            # 将详细的损失项也记录到 history 中，以便后续绘图
            for k, v in train_loss_components.items():
                key = f"train_loss_{k}"
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(float(v))
            for k, v in val_loss_components.items():
                key = f"val_loss_{k}"
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(float(v))

            logger.info(
                f"训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}, Epoch时间: {time.time() - epoch_start:.1f}s"
            )
            # 按 epoch 记录监控指标
            payload_epoch = {
                "train/loss": float(train_loss),
                "val/loss": float(val_loss),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "epoch/time_sec": float(time.time() - epoch_start),
            }
            # 添加各个损失项
            for k, v in train_loss_components.items():
                payload_epoch[f"train/loss_{k}"] = float(v)
            for k, v in val_loss_components.items():
                payload_epoch[f"val/loss_{k}"] = float(v)

            # 若有梯度信息，追加上报（ECharts 直方图 + 数值指标）
            if grad_info is not None:
                try:
                    counts, edges, gnorm, gmax = grad_info
                    # 数值指标
                    payload_epoch.update(
                        {
                            "grad/global_norm": float(gnorm),
                            "grad/max_abs": float(gmax),
                        }
                    )
                    # 构造直方图图表（ECharts Bar）
                    import swanlab

                    labels = [
                        f"{edges[i]:.2e}~{edges[i+1]:.2e}"
                        for i in range(len(edges) - 1)
                    ]
                    bar = (
                        swanlab.echarts.Bar()
                        .add_xaxis(labels)
                        .add_yaxis("count", counts.tolist())
                    )
                    self.monitor.log({"grad/hist": bar}, step=epoch)
                except Exception:
                    pass
            self.monitor.log(payload_epoch, step=epoch)

            self._save_val_loss_maps(epoch)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            self.save_checkpoint(epoch, is_best)
            if self.patience_counter >= self.early_stop_patience:
                logger.info("早停触发！")
                break
        logger.info(
            f"训练完成！总时间: {time.time() - start_time:.2f}s, 最佳验证损失: {self.best_val_loss:.6f}"
        )
        with open(os.path.join(self.output_dir, "training_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        # 保存最后一个epoch的权重
        last_ckpt_path = os.path.join(self.output_dir, "last_model.pth")
        try:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.config,
                    "history": self.history,
                    "best_val_loss": self.best_val_loss,
                },
                last_ckpt_path,
            )
        except Exception as e:
            logger.warning(f"保存最后权重失败: {e}")
        test_results = self.test_on_test_set()
        # 记录测试结果（若存在）
        if isinstance(test_results, dict):
            payload = {
                f"test/{k}": float(v)
                for k, v in test_results.items()
                if isinstance(v, (int, float))
            }
            self.monitor.log(payload)
        # 结束监控会话
        self.monitor.finish()
        return self.best_model_path


def main():
    """主函数 - 解析命令行参数并启动训练"""
    parser = argparse.ArgumentParser(description="SWOT海底地形超分辨率训练脚本")
    parser.add_argument(
        "model_type",
        type=str,
        choices=["unet", "attention_unet", "transformer_unet"],
        help="要训练的模型类型",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["single", "multi"],
        default="single",
        help="训练域类型",
    )
    parser.add_argument(
        "--dataset-key", type=str, default=None, help="自定义数据集Key，覆盖默认值"
    )
    args = parser.parse_args()
    config = get_model_config(
        args.model_type, domain=args.domain, dataset_key=args.dataset_key
    )

    # 临时修改：禁用早停并延长训练步长
    config["max_epochs"] = 400
    config["early_stop_patience"] = 100000  # 实际上禁用早停

    print("SWOT海底地形超分辨率训练:")
    print("=" * 60)
    model_info = get_model_info()
    info = model_info[config["model_type"]]
    print(f"模型架构: {info['name']}")
    print(f"描述: {info['description']}")
    if info.get("recommended"):
        print("推荐使用")
    print(f"参数量: {info.get('parameters', '未知')}")
    print("-" * 60)
    print(f"domain: {args.domain}")
    if args.dataset_key:
        print(f"dataset_key: {args.dataset_key}")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 60)
    try:
        trainer = Trainer(config)
        best_model_path = trainer.train()
        print(f"\n训练完成！最佳模型: {best_model_path}")
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise


if __name__ == "__main__":
    main()
