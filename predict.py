# predict.py
"""
模型预测、评估与可视化脚本 (多域评估版)。

本脚本支持对训练好的模型进行全面的评估，包括在训练时使用的“域内”数据
和全新的“跨域/泛化”数据上的表现。

--- 使用方法 ---

1. 自动评估所有“域内”数据集 + 一个“泛化”数据集:
   - 脚本会自动读取模型配置文件(config.json)中记录的训练数据路径。
   - 如果是多海域训练的模型，它会逐个在每个海域上进行独立评估。
   - 使用 --data-path 提供一个额外的泛化测试集。

   示例:
   python predict.py "path/to/your/model_dir" --data-path "path/to/generalization_data.npz"

2. 仅评估“泛化”数据集:
   - 如果你只想快速查看模型在全新数据上的表现，不关心其在域内的性能。

   示例:
   python predict.py "path/to/your/model_dir" --data-path "path/to/generalization_data.npz" --generalization-only

3. 仅评估所有“域内”数据集:
   - 如果你只想回顾模型在训练集上的表现。

   示例:
   python predict.py "path/to/your/model_dir"

--- 输出结构 ---

所有评估结果将保存在 'output/3-evaluations/YOUR_MODEL_NAME/' 下。
每个独立的评估任务（无论是域内还是泛化）都会有自己的子目录，例如:
- .../eval_on_wavelength_filtered_dataset_0714_0953/
- .../eval_on_generalization_set/
"""
import os
import torch
import numpy as np
import json
import pandas as pd
from datetime import datetime
import logging
from scipy import ndimage
from tqdm import tqdm
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.colors import LogNorm
from torch.utils.data import Subset, DataLoader
import argparse
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from models import get_model

# from data_loader import get_dataloaders # 注意：此脚本现在独立进行全局预测，不再依赖于dataloader进行样本评估

# --- 字体和符号配置 ---
plt.rcParams["font.family"] = ["sans-serif"]  # 使用无衬线字体族
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
plt.rcParams["text.usetex"] = False  # 禁用 LaTeX 渲染（避免冲突）

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    模型预测与可视化器 (多任务版)。
    """

    def __init__(self, model_path, config, tta: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.config = config
        self.metrics_results = []
        self.use_tta = bool(tta)

        logger.info(f"加载模型配置: {self.config['experiment_name']}")
        logger.info(f"使用设备: {self.device}")

        model_name = self.config["experiment_name"]
        self.base_output_dir = f"output/3-evaluations/{model_name}"
        self.output_dir = self.base_output_dir  # 默认输出目录
        os.makedirs(self.base_output_dir, exist_ok=True)

        self._load_model()
        self.gebco_coords = None

    def set_current_task_output(self, task_output_dir):
        """为当前评估任务设置独立的输出目录。"""
        self.output_dir = task_output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"当前任务结果将保存到: {self.output_dir}")

    def reset_metrics(self):
        """重置指标列表，为下一个任务做准备。"""
        self.metrics_results = []
        logger.info("评估指标已重置。")

    def _load_model(self):
        """加载训练好的模型。"""
        logger.info("加载模型...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]

        bilinear_flag = self.config.get(
            "bilinear", False
        )  # 直接从config获取，因为train.py已修复
        logger.info(f"根据配置，使用 bilinear={bilinear_flag}")

        self.model = get_model(
            model_type=self.config["model_type"],
            n_channels=4,
            n_classes=1,
            bilinear=bilinear_flag,
        ).to(self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info(f"模型加载完成。")

    def _add_metric(
        self,
        scope,
        mae,
        rmse,
        mean_residual,
        std_residual,
        correlation,
        r_squared,
        sample_id="N/A",
    ):
        """将一组指标添加到结果列表中"""
        self.metrics_results.append(
            {
                "scope": scope,
                "id": sample_id,
                "mae_m": mae,
                "rmse_m": rmse,
                "mean_residual_m": mean_residual,
                "std_residual_m": std_residual,
                "correlation": correlation,
                "r_squared": r_squared,
            }
        )

    def _denormalize_prediction(self, normalized_prediction):
        """反标准化预测结果"""
        if hasattr(self, "gebco_norm_params") and self.gebco_norm_params:
            mean = self.gebco_norm_params["mean"]
            std = self.gebco_norm_params["std"]
            return normalized_prediction * std + mean
        return normalized_prediction

    def _create_edge_weight_matrix(self, size):
        """创建二维高斯权重矩阵，以确保无缝的图块拼接。"""
        x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        d = np.sqrt(x * x + y * y)
        sigma, mu = 0.5, 0.0
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
        return g

    def predict_full_region(self, data_path, margin):
        """预测整个区域的高分辨率海底地形，并评估可靠区域。"""
        logger.info("🔮 开始全局预测...")
        try:
            data = np.load(data_path, allow_pickle=True)
        except FileNotFoundError:
            logger.error(f"数据文件未找到: {data_path}")
            return None

        self.gebco_coords = {"lat": data["gebco_lat"], "lon": data["gebco_lon"]}

        if "swot_features" in data:
            swot_data = torch.from_numpy(data["swot_features"]).permute(2, 0, 1).float()
        elif "input_features" in data:
            swot_data = (
                torch.from_numpy(data["input_features"]).permute(2, 0, 1).float()
            )
        else:
            raise KeyError(f"数据文件中找不到 'swot_features' 或 'input_features'")

        # 加载归一化参数（如果存在）
        if "normalization_params" in data:
            norm_params = json.loads(data["normalization_params"].item())
            self.gebco_norm_params = norm_params.get("gebco")
            logger.info("已加载GEBCO归一化参数。")
        else:
            self.gebco_norm_params = None
            logger.warning("数据文件中未找到归一化参数，将直接使用原始值进行计算。")

        gebco_data_normalized = (
            torch.from_numpy(data["gebco_bathymetry"]).squeeze().float()
        )
        logger.info(
            f"SWOT数据形状: {swot_data.shape}, GEBCO数据形状: {gebco_data_normalized.shape}"
        )

        patch_size = 64
        stride = 8
        C, H, W = swot_data.shape
        target_H, target_W = gebco_data_normalized.shape
        prediction = np.zeros((target_H, target_W))
        weight_map = np.zeros((target_H, target_W))

        logger.info("🚀 开始滑动窗口预测...")
        # 定义 TTA 变换（输入->输出逆变换）
        if self.use_tta:
            transforms = [
                (lambda x: x, lambda y: y),
                (lambda x: torch.flip(x, [-1]), lambda y: np.flip(y, axis=-1)),
                (lambda x: torch.flip(x, [-2]), lambda y: np.flip(y, axis=-2)),
                (
                    lambda x: torch.flip(torch.flip(x, [-1]), [-2]),
                    lambda y: np.flip(np.flip(y, axis=-1), axis=-2),
                ),
                (lambda x: torch.rot90(x, 1, dims=[-2, -1]), lambda y: np.rot90(y, 3)),
                (lambda x: torch.rot90(x, 2, dims=[-2, -1]), lambda y: np.rot90(y, 2)),
                (lambda x: torch.rot90(x, 3, dims=[-2, -1]), lambda y: np.rot90(y, 1)),
            ]
            logger.info("已启用 TTA（翻转+旋转）进行预测集成。")
        else:
            transforms = [(lambda x: x, lambda y: y)]
        with torch.no_grad():
            for i in tqdm(range(0, H - patch_size + 1, stride), desc="行进度"):
                for j in range(0, W - patch_size + 1, stride):
                    input_patch_base = (
                        swot_data[:, i : i + patch_size, j : j + patch_size]
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    # 累积 TTA 结果（在标准化空间聚合）
                    pred_acc = None
                    for fwd, inv in transforms:
                        aug_in = fwd(input_patch_base)
                        out = self.model(aug_in).detach().cpu().numpy()[0, 0]
                        out_inv = inv(out)
                        if pred_acc is None:
                            pred_acc = out_inv.astype(np.float32)
                        else:
                            pred_acc += out_inv.astype(np.float32)
                    pred_patch = pred_acc / float(len(transforms))
                    patch_weight = self._create_edge_weight_matrix(patch_size * 4)
                    target_i, target_j, target_size = i * 4, j * 4, patch_size * 4
                    if (
                        target_i + target_size <= target_H
                        and target_j + target_size <= target_W
                    ):
                        prediction[
                            target_i : target_i + target_size,
                            target_j : target_j + target_size,
                        ] += (
                            pred_patch * patch_weight
                        )
                        weight_map[
                            target_i : target_i + target_size,
                            target_j : target_j + target_size,
                        ] += patch_weight

        weight_map[weight_map == 0] = 1
        prediction_normalized = prediction / weight_map

        prediction_denormalized = self._denormalize_prediction(prediction_normalized)
        gebco_denormalized = self._denormalize_prediction(gebco_data_normalized.numpy())

        reliable_bbox = np.array([margin, target_H - margin, margin, target_W - margin])
        if reliable_bbox[0] >= reliable_bbox[1] or reliable_bbox[2] >= reliable_bbox[3]:
            logger.warning("可靠区域边距过大，评估将覆盖整个区域。")
            reliable_bbox = np.array([0, target_H, 0, target_W])

        pred_reliable = prediction_denormalized[
            reliable_bbox[0] : reliable_bbox[1], reliable_bbox[2] : reliable_bbox[3]
        ]
        truth_reliable = gebco_denormalized[
            reliable_bbox[0] : reliable_bbox[1], reliable_bbox[2] : reliable_bbox[3]
        ]
        residual_reliable = pred_reliable - truth_reliable

        mean_residual = np.mean(residual_reliable)
        std_residual = np.std(residual_reliable)
        mae = np.mean(np.abs(residual_reliable))
        rmse = np.sqrt(np.mean(residual_reliable**2))
        correlation = np.corrcoef(pred_reliable.flatten(), truth_reliable.flatten())[
            0, 1
        ]
        r_squared = correlation**2

        logger.info("=" * 40)
        logger.info(f"可靠区域评估完成 (已去除四周 {margin} 像素边距)。")
        logger.info(f"  - RMSE: {rmse:.4f} m, MAE: {mae:.4f} m, R²: {r_squared:.4f}")
        self._add_metric(
            scope="full_prediction_reliable",
            mae=mae,
            rmse=rmse,
            mean_residual=mean_residual,
            std_residual=std_residual,
            correlation=correlation,
            r_squared=r_squared,
        )
        logger.info("=" * 40)

        output_path = os.path.join(self.output_dir, "prediction.npz")
        # np.savez(output_path, prediction=prediction_denormalized, truth=gebco_denormalized, reliable_bbox=reliable_bbox)
        # 同时保存源数据路径以便后续脚本自动查找原始 bandpass 数据
        np.savez(
            output_path,
            prediction=prediction_denormalized,
            truth=gebco_denormalized,
            reliable_bbox=reliable_bbox,
            lons=self.gebco_coords["lon"],
            lats=self.gebco_coords["lat"],
            source_data_path=data_path,
        )
        logger.info(f"✅ 预测数据已保存到: {output_path}")

        return {
            "prediction": prediction_denormalized,
            "truth": gebco_denormalized,
            "reliable_bbox": reliable_bbox,
        }

    def visualize_full_prediction(self, prediction, truth, reliable_bbox):
        """使用 Matplotlib 和 Cartopy 可视化全局预测结果。"""
        logger.info("生成 Matplotlib 全局预测对比图...")

        if not self.gebco_coords:
            logger.error("坐标信息不可用，无法生成地理地图。")
            return

        lons = self.gebco_coords["lon"]
        lats = self.gebco_coords["lat"]
        extent = [lons.min(), lons.max(), lats.min(), lats.max()]

        residual = prediction - truth

        # 准备绘图
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(24, 8),
            subplot_kw={"projection": ccrs.PlateCarree()},
            constrained_layout=True,
        )
        fig.suptitle(
            f"全局地形预测结果对比: {self.config['experiment_name']}",
            fontsize=20,
            weight="bold",
        )

        # --- 1. 绘制真值图 ---
        ax1 = axes[0]
        terrain_vmin, terrain_vmax = np.percentile(truth[np.isfinite(truth)], [2, 98])
        im1 = ax1.imshow(
            truth,
            extent=extent,
            origin="lower",
            cmap="viridis",
            vmin=terrain_vmin,
            vmax=terrain_vmax,
            transform=ccrs.PlateCarree(),
        )
        ax1.set_title("a) GEBCO (真值)", fontsize=16)

        # --- 2. 绘制预测图 ---
        ax2 = axes[1]
        im2 = ax2.imshow(
            prediction,
            extent=extent,
            origin="lower",
            cmap="viridis",
            vmin=terrain_vmin,
            vmax=terrain_vmax,
            transform=ccrs.PlateCarree(),
        )
        # 从 metrics_results 获取 RMSE 和 R²
        reliable_metrics = next(
            (
                item
                for item in self.metrics_results
                if item["scope"] == "full_prediction_reliable"
            ),
            None,
        )
        if reliable_metrics:
            rmse_val = reliable_metrics["rmse_m"]
            r2_val = reliable_metrics["r_squared"]
            ax2.set_title(
                f"b) 模型预测\nRMSE: {rmse_val:.2f}m | R-squared: {r2_val:.4f}",
                fontsize=16,
            )
        else:
            ax2.set_title("b) 模型预测", fontsize=16)

        # --- 3. 绘制残差图 ---
        ax3 = axes[2]
        res_vmax = np.percentile(np.abs(residual[np.isfinite(residual)]), 99)
        im3 = ax3.imshow(
            residual,
            extent=extent,
            origin="lower",
            cmap="coolwarm_r",
            vmin=-res_vmax,
            vmax=res_vmax,
            transform=ccrs.PlateCarree(),
        )
        ax3.set_title("c) 残差 (预测 - 真值)", fontsize=16)

        # --- 统一设置地图元素和颜色条 ---
        for i, ax in enumerate(axes):
            ax.coastlines(resolution="10m", color="black", linewidth=0.7)
            gl = ax.gridlines(
                draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = i == 0  # Only on the first plot
            gl.bottom_labels = True
            gl.xlabel_style = {"size": 12}
            gl.ylabel_style = {"size": 12}

        # 为地形图创建颜色条
        cbar1 = fig.colorbar(
            im1, ax=[ax1, ax2], orientation="vertical", shrink=0.8, pad=0.03
        )
        cbar1.set_label("深度 (m)", size=14)
        cbar1.ax.tick_params(labelsize=12)

        # 为残差图创建颜色条
        cbar2 = fig.colorbar(im3, ax=ax3, orientation="vertical", shrink=0.8, pad=0.03)
        cbar2.set_label("残差 (m)", size=14)
        cbar2.ax.tick_params(labelsize=12)

        output_path = os.path.join(self.output_dir, "full_region_comparison.png")
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"✅ 新的 Matplotlib 对比图已保存: {output_path}")

    def save_report_to_csv(self):
        """将当前任务收集到的指标保存到一个CSV文件中"""
        if not self.metrics_results:
            logger.warning("没有收集到任何指标，无法生成CSV报告。")
            return

        report_df = pd.DataFrame(self.metrics_results)
        cols_order = [
            "scope",
            "id",
            "mae_m",
            "rmse_m",
            "mean_residual_m",
            "std_residual_m",
            "correlation",
            "r_squared",
        ]
        for col in cols_order:
            if col not in report_df.columns:
                report_df[col] = None
        report_df = report_df[cols_order]

        output_path = os.path.join(self.output_dir, "evaluation_report.csv")
        report_df.to_csv(
            output_path, index=False, float_format="%.6f", encoding="utf-8-sig"
        )
        logger.info(f"✅ 评估报告已成功保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="模型预测、评估与可视化脚本 (多域评估版)"
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="包含 best_model.pth 和 config.json 的模型实验目录路径",
    )
    parser.add_argument(
        "--margin", type=int, default=480, help="用于评估的可靠区域像素边距"
    )
    parser.add_argument(
        "--data-path", type=str, default=None, help="（可选）提供一个外部泛化测试集路径"
    )
    parser.add_argument(
        "--generalization-only",
        action="store_true",
        help="如果设置，将只在 --data-path 提供的泛化集上进行测试，跳过域内评估。",
    )
    parser.add_argument(
        "--srtm-path",
        type=str,
        default="/mnt/data2/00-Data/SRTM/SRTM15_V2.7.nc",
        help="SRTM 网格路径，用于地图对比（可选）",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="启用测试时增强（翻转+旋转）并做预测集成，不改变模型与损失。",
    )
    args = parser.parse_args()

    # --- 1. 加载模型和主配置 ---
    model_path = os.path.join(args.model_dir, "best_model.pth")
    config_path = os.path.join(args.model_dir, "config.json")

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        logger.error(
            f"错误: 找不到模型文件或配置文件。请确保路径 '{args.model_dir}' 正确。"
        )
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # --- 2. 准备评估任务列表 ---
    evaluation_tasks = []
    base_output_dir = f"output/3-evaluations/{config['experiment_name']}"

    # 任务1: 域内数据集评估
    if not args.generalization_only:
        in_domain_paths = config.get("data_path", [])
        if isinstance(in_domain_paths, str):
            in_domain_paths = [in_domain_paths]

        for path in in_domain_paths:
            task_name = f"eval_on_{os.path.splitext(os.path.basename(path))[0]}"
            evaluation_tasks.append({"name": task_name, "data_path": path})

    # 任务2: 泛化数据集评估
    if args.data_path:
        # 如果只测泛化，清空之前的任务列表
        if args.generalization_only:
            evaluation_tasks = []
        # 使用数据集文件名作为任务名，避免多数据集互相覆盖
        data_base = os.path.splitext(os.path.basename(args.data_path))[0]
        task_name = f"eval_on_{data_base}"
        evaluation_tasks.append({"name": task_name, "data_path": args.data_path})

    if not evaluation_tasks:
        logger.error(
            "没有定义任何评估任务。请在config.json中定义data_path或通过--data-path提供数据集。"
        )
        return

    # --- 3. 执行所有评估任务 ---
    logger.info(f"发现 {len(evaluation_tasks)} 个评估任务，开始执行...")

    # 只需要加载一次模型
    predictor = ModelPredictor(model_path=model_path, config=config, tta=args.tta)

    for task in evaluation_tasks:
        logger.info("\n" + "=" * 80)
        logger.info(f"🔬 开始评估任务: {task['name']} @ 数据: {task['data_path']}")

        # 为每个任务设置独立的输出目录
        task_output_dir = os.path.join(base_output_dir, task["name"])
        predictor.set_current_task_output(task_output_dir)

        # 执行预测并保存结果（此处不进行 SRTM/GEBCO 的详细绘图，交由 evaluate 脚本处理）
        prediction_results = predictor.predict_full_region(
            data_path=task["data_path"], margin=args.margin
        )

        if prediction_results:
            logger.info(
                "-" * 20
                + " 预测已完成并保存：请使用 scripts/2_evaluate_and_plot_region.py 进行 SRTM/GEBCO 对比与综合统计 "
                + "-" * 20
            )

    # 保存该任务的独立报告（predict.py 仍会输出基础的 evaluation_report.csv）
    predictor.save_report_to_csv()
    # 重置指标以供下一次任务使用
    predictor.reset_metrics()

    logger.info("\n" + "✨" * 20)
    logger.info("✨ 所有评估任务完成！")


if __name__ == "__main__":
    main()
