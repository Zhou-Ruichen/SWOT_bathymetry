#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
迁移学习与微调脚本。

本脚本用于加载一个已经训练好的（预训练）模型，并在一个新的、特定的
数据集上进行微调（fine-tuning），以期在该特定数据集上取得更好的性能。

--- 核心流程 ---

1.  加载一个基础模型（“巨人”）的配置和权重。
2.  准备一个新的训练环境，但使用新的目标数据集和新的实验名称。
3.  将“巨人”的权重“移植”到新的训练环境中。
4.  使用一个非常低的学习率，对模型进行短暂的、针对性的再训练。
5.  输出一个新的、经过微调的模型，可用于后续的评估。

--- 如何使用 ---

在终端中执行以下命令:

python finetune.py \
    --base-model-dir [path/to/pretrained_model_dir] \
    --target-data-path [path/to/target_data.npz] \
    --new-experiment-name [name_for_this_finetune_task] \
    --lr [learning_rate] \
    --epochs [num_epochs] \
    --batch-size [batch_size] \
    [--freeze-encoder]

参数说明:
  --base-model-dir:      一个已经训练好的模型的实验目录。
                         (例如: 'output/2-experiments/unet_..._0802_0049')
  --target-data-path:    你希望模型去适应的那个新的数据区域的文件路径。
                         (例如: 'output/1-data/wavelength_filtered_dataset_0802_D.npz')
  --new-experiment-name: 为这次微调任务起一个名字。
                         (例如: 'finetune_on_region_D')
  --lr:                  微调时使用的学习率，通常需要设得很小 (默认: 1e-5)。
  --epochs:              微调的轮数，推荐使用较少的轮数 (默认: 10)。
  --batch-size:          微调的批次大小 (默认: 8)。
  --freeze-encoder:      (可选) 如果设置此标志，将冻结模型编码器（所有下采样层）
                         的权重，只训练解码器。这是更严格意义上的“微调”。

示例 1: 低学习率继续训练 (训练所有层)
python finetune.py \
    --base-model-dir output/2-experiments/unet_..._0802_0049 \
    --target-data-path output/1-data/wavelength_filtered_dataset_..._D_global_norm.npz \
    --new-experiment-name continue_train_on_D \
    --epochs 50

示例 2: 精细微调 (只训练解码器)
python finetune.py \
    --base-model-dir output/2-experiments/unet_..._0802_0049 \
    --target-data-path output/1-data/wavelength_filtered_dataset_..._D_global_norm.npz \
    --new-experiment-name finetune_on_D \
    --epochs 10 \
    --freeze-encoder

"""

import os
import torch
import json
import argparse
import logging
from datetime import datetime

# --- 复用您项目中已有的核心组件 ---
from train import Trainer
from models import get_model

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="使用预训练模型在特定数据集上进行微调",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-model-dir",
        type=str,
        required=True,
        help="预训练模型的实验目录路径 (例如 'output/2-experiments/multi_domain_model_...')。",
    )
    parser.add_argument(
        "--target-data-path",
        type=str,
        required=True,
        help="用于微调的目标海域数据文件路径 (.npz)。",
    )
    parser.add_argument(
        "--new-experiment-name",
        type=str,
        required=True,
        help="为本次微调任务生成的新实验名称。",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="微调时使用的学习率 (通常设置得非常小)。"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,  # <-- 默认轮数设为30，一个更宽泛的微调值
        help="微调时训练的最大轮数。",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="微调时使用的批次大小。"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="数据加载的 worker 数量，设置为 0 可以在主进程中查看完整异常（用于调试）。",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",  # <-- 新增参数：冻结编码器
        help="如果设置，将冻结模型编码器（所有下采样层）的权重，只训练解码器。",
    )
    parser.add_argument(
        "--unfreeze-last-encoder-blocks",
        type=int,
        default=0,
        help="当冻结编码器时，解冻最后N个编码器块（如 down4、down3 ...），用于轻量领域适应。0 表示不解冻。",
    )
    # 进阶微调参数
    parser.add_argument(
        "--mix-base",
        action="store_true",
        help="将基础配置中的多域数据与目标域数据合并进行混合微调（需要 base config 的 data_path 为列表）。",
    )
    parser.add_argument(
        "--target-domain-weight",
        type=float,
        default=1.0,
        help="目标域在训练采样中的权重倍率（仅在多域混合时生效）。",
    )
    parser.add_argument(
        "--layered-lr",
        action="store_true",
        help="启用分层学习率：decoder/out 层使用更高 LR，其他（未冻结）使用较低 LR。",
    )
    parser.add_argument(
        "--high-lr-mult",
        type=float,
        default=3.0,
        help="高层（up/out）学习率相对基准 LR 的倍率（仅 --layered-lr 时生效）。",
    )
    parser.add_argument(
        "--low-lr-mult",
        type=float,
        default=0.3,
        help="低层（除 up/out 之外未冻结层）学习率相对基准 LR 的倍率（仅 --layered-lr 时生效）。",
    )
    parser.add_argument(
        "--encoder-lr-mult",
        type=float,
        default=0.2,
        help="被解冻的编码器层学习率相对基准 LR 的倍率（无论是否 --layered-lr 均生效）。",
    )
    args = parser.parse_args()

    logger.info("--- 开始迁移学习与微调任务 ---")

    # --- 1. 加载基础模型的配置 ---
    base_config_path = os.path.join(args.base_model_dir, "config.json")
    if not os.path.exists(base_config_path):
        logger.error(f"错误: 在基础模型目录中找不到配置文件: {base_config_path}")
        return

    with open(base_config_path, "r") as f:
        base_config = json.load(f)

    logger.info(f"成功加载基础模型配置: {base_config['experiment_name']}")

    # --- 2. 创建一个新的微调配置 ---
    # 继承大部分基础配置，但更新关键参数
    finetune_config = base_config.copy()
    finetune_config["experiment_name"] = (
        f"{args.new_experiment_name}_{datetime.now().strftime('%m%d_%H%M')}"
    )
    # 数据路径：单域或与基础多域合并
    if args.mix_base and isinstance(base_config.get("data_path"), list):
        finetune_config["data_path"] = base_config["data_path"] + [
            args.target_data_path
        ]
        finetune_config["balance_domains"] = True
        finetune_config["target_domain_weight"] = float(args.target_domain_weight)
    else:
        finetune_config["data_path"] = args.target_data_path  # 仅目标域
    finetune_config["learning_rate"] = args.lr
    finetune_config["max_epochs"] = args.epochs
    finetune_config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        finetune_config["num_workers"] = args.num_workers
    # 确保微调时也使用数据增强，这通常是好的实践
    finetune_config["use_augmentation"] = True
    finetune_config["use_rotation"] = True
    finetune_config["use_noise"] = True
    finetune_config["use_cutout"] = True

    logger.info(f"创建新的微调实验: {finetune_config['experiment_name']}")
    logger.info(f"  - 目标数据: {finetune_config['data_path']}")
    logger.info(f"  - 微调学习率: {finetune_config['learning_rate']}")
    logger.info(f"  - 微调轮数: {finetune_config['max_epochs']}")

    # --- 3. 初始化一个新的Trainer，但暂不使用它的模型 ---
    # Trainer会为我们处理好数据加载、目录创建等所有繁杂事务
    try:
        trainer = Trainer(finetune_config)
    except Exception as e:
        logger.error(f"初始化Trainer失败: {e}")
        return

    # --- 4. 加载预训练模型的权重 ---
    base_model_path = os.path.join(args.base_model_dir, "best_model.pth")
    if not os.path.exists(base_model_path):
        logger.error(f"错误: 找不到预训练模型权重文件: {base_model_path}")
        return

    logger.info(f"正在从 {base_model_path} 加载预训练权重...")
    # 加载到CPU以避免GPU内存问题，然后再移动到设备
    checkpoint = torch.load(base_model_path, map_location="cpu")

    # --- 5. 将预训练权重“移植”到新的Trainer模型中 ---
    # 使用 strict=False 来允许加载部分匹配的权重，忽略不匹配的层
    trainer.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    trainer.model.to(trainer.device)  # 确保模型移动到正确的设备
    logger.info("✅ 预训练权重已成功加载到新模型中 (使用了非严格模式)！")

    # --- 6. (可选) 冻结编码器层 ---
    if args.freeze_encoder:
        logger.info("❄️ 正在冻结编码器层的权重...")
        # 冻结 inc 和所有 down 模块
        for name, param in trainer.model.named_parameters():
            if "inc." in name or "down" in name:
                param.requires_grad = False
                # logger.info(f"  - 已冻结: {name}")
        # 轻量解冻最后 N 个编码器块
        k = max(0, int(args.unfreeze_last_encoder_blocks))
        if k > 0:
            logger.info(f"🔓 轻量解冻编码器最后 {k} 个块以适配目标域...")
            # 从 available blocks 中自后向前选择：down4, down3, down2, down1, inc
            encoder_blocks = []
            for blk_name in ["down4", "down3", "down2", "down1", "inc"]:
                if hasattr(trainer.model, blk_name):
                    encoder_blocks.append(getattr(trainer.model, blk_name))
            selected = encoder_blocks[:k]
            for module in selected:
                for p in module.parameters():
                    p.requires_grad = True
            logger.info(
                "✅ 已解冻的编码器模块: "
                + ", ".join(
                    [
                        n
                        for n in ["down4", "down3", "down2", "down1", "inc"][:k]
                        if hasattr(trainer.model, n)
                    ]
                )
            )
        else:
            logger.info("✅ 编码器已冻结，只有解码器将被训练。")

    # --- 7. 重置优化器状态和学习率 ---
    # 分层学习率或统一学习率
    # 构建优化器参数组：
    # - encoder_group: 编码器中被解冻的参数（极低 LR）
    # - high_group: up*/outc*（高 LR）
    # - mid_group: 其它可训练参数（低或基准 LR）
    base_lr = float(finetune_config["learning_rate"])
    enc_group, high_group, mid_group = [], [], []
    for name, p in trainer.model.named_parameters():
        if not p.requires_grad:
            continue
        if ("inc." in name) or ("down" in name):
            enc_group.append(p)
        elif ("up" in name) or ("outc" in name):
            high_group.append(p)
        else:
            mid_group.append(p)

    groups = []
    # 编码器组：总是使用 encoder_lr_mult（若为空则跳过）
    if enc_group:
        groups.append(
            {
                "params": enc_group,
                "lr": base_lr * float(args.encoder_lr_mult),
            }
        )

    if args.layered_lr:
        if mid_group:
            groups.append(
                {
                    "params": mid_group,
                    "lr": base_lr * float(args.low_lr_mult),
                }
            )
        if high_group:
            groups.append(
                {
                    "params": high_group,
                    "lr": base_lr * float(args.high_lr_mult),
                }
            )
        logger.info(
            f"已启用分层学习率：encoder_mult={args.encoder_lr_mult}, low_mult={args.low_lr_mult}, high_mult={args.high_lr_mult}"
        )
    else:
        # 非分层：其余参数使用基准 LR
        rest = []
        rest.extend(mid_group)
        rest.extend(high_group)
        if rest:
            groups.append(
                {
                    "params": rest,
                    "lr": base_lr,
                }
            )

    if not groups:
        # 回退：若无可训练参数（不应发生），避免创建空优化器
        trainable_params = filter(lambda p: p.requires_grad, trainer.model.parameters())
        trainer.optimizer = torch.optim.Adam(
            trainable_params, lr=base_lr, weight_decay=finetune_config["weight_decay"]
        )
    else:
        trainer.optimizer = torch.optim.Adam(
            groups, weight_decay=finetune_config["weight_decay"]
        )
    logger.info("优化器已重置，并应用了新的微调学习率。")

    # --- 8. 开始微调！---
    try:
        best_model_path = trainer.train()
        logger.info("\n" + "✨" * 20)
        logger.info("✨ 微调任务成功完成！")
        logger.info(f"✨ 新的、经过微调的模型已保存到: {best_model_path}")
        logger.info("✨ 您现在可以使用 predict.py 来评估这个新模型的性能。")
    except Exception as e:
        logger.error(f"微调过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()
