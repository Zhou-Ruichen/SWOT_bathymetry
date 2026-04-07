#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精简的数据加载模块
"""

import numpy as np
import torch
from collections import Counter
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
    Subset,
    WeightedRandomSampler,
)
import torch.nn.functional as F
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SWOTDataset(Dataset):
    """SWOT数据集，支持从单个文件或目录加载"""

    def __init__(
        self,
        data_path,
        patch_size=64,
        use_augmentation=True,
        use_rotation=False,
        use_noise=False,
        use_cutout=False,
        use_random_erasing=False,
    ):
        """
        Args:
            data_path: 数据文件路径 或 包含多个.npz文件的目录路径
            patch_size: 图像块大小
            use_augmentation: 是否使用数据增强
            use_rotation: 是否使用旋转增强
            use_noise: 是否使用随机噪声增强
            use_cutout: 是否使用Cutout增强
            use_random_erasing: 是否使用Random Erasing增强
        """
        self.patch_size = patch_size
        self.target_patch_size = self.patch_size * 4
        self.use_augmentation = use_augmentation
        self.use_rotation = use_rotation
        self.use_noise = use_noise
        self.use_cutout = use_cutout
        self.use_random_erasing = use_random_erasing

        # 加载数据
        logger.info(f"开始从路径加载数据: {data_path}")

        if os.path.isdir(data_path):
            # 如果是目录，则加载所有 .npz 文件
            file_paths = [
                os.path.join(data_path, f)
                for f in os.listdir(data_path)
                if f.endswith(".npz")
            ]
            logger.info(f"检测到目录，将加载 {len(file_paths)} 个数据文件。")
            self._load_from_multiple_files(file_paths)
        elif os.path.isfile(data_path):
            # 如果是单个文件，则沿用旧方法
            logger.info("检测到单个文件，按原方式加载。")
            self._load_from_single_file(data_path)
        else:
            raise FileNotFoundError(f"数据路径不存在: {data_path}")

        # 生成图像块索引
        self._generate_patches()

        logger.info(f"数据集加载完成:")
        logger.info(f"- SWOT形状: {self.swot_data.shape}")
        logger.info(f"- GEBCO形状: {self.gebco_data.shape}")
        logger.info(f"- 图像块数量: {len(self.patches)}")
        logger.info(f"- TID数据: 是 (形状: {self.tid_data.shape})")
        logger.info(
            f"- 数据增强: 基础翻转={self.use_augmentation}, 旋转={self.use_rotation}, 噪声={self.use_noise}, Cutout={self.use_cutout}, RandomErasing={self.use_random_erasing}"
        )

    def _load_from_single_file(self, file_path):
        """从单个 .npz 文件加载数据"""
        data = np.load(file_path)
        # 检查数据格式并使用正确的键名
        if "swot_features" in data:
            swot_features = data["swot_features"]
            self.swot_data = torch.from_numpy(swot_features).permute(2, 0, 1).float()
        elif "input_features" in data:
            input_features = data["input_features"]
            self.swot_data = torch.from_numpy(input_features).permute(2, 0, 1).float()
        else:
            raise KeyError(
                f"在文件 {file_path} 中找不到 'swot_features' 或 'input_features'"
            )

        if "gebco_bathymetry" in data:
            gebco_data = data["gebco_bathymetry"]
            self.gebco_data = torch.from_numpy(gebco_data).unsqueeze(0).float()
        elif "target_labels" in data:
            target_labels = data["target_labels"]
            self.gebco_data = torch.from_numpy(target_labels).unsqueeze(0).float()
        else:
            raise KeyError(
                f"在文件 {file_path} 中找不到 'gebco_bathymetry' 或 'target_labels'"
            )

        if "tid_data" in data:
            tid_data = data["tid_data"]
            self.tid_data = torch.from_numpy(tid_data).unsqueeze(0).float()
        else:
            h, w = self.gebco_data.shape[1], self.gebco_data.shape[2]
            self.tid_data = torch.full((1, h, w), 70.0, dtype=torch.float32)

    def _load_from_multiple_files(self, file_paths):
        """从多个 .npz 文件加载并合并数据"""
        swot_list, gebco_list, tid_list = [], [], []

        for path in file_paths:
            logger.info(f"  - 正在加载: {path}")
            data = np.load(path)

            # 加载并转换 SWOT 数据
            if "swot_features" in data:
                swot_features = data["swot_features"]
                swot_list.append(
                    torch.from_numpy(swot_features).permute(2, 0, 1).float()
                )
            elif "input_features" in data:
                input_features = data["input_features"]
                swot_list.append(
                    torch.from_numpy(input_features).permute(2, 0, 1).float()
                )
            else:
                logger.warning(
                    f"跳过文件 {path}: 找不到 'swot_features' 或 'input_features'"
                )
                continue

            # 加载并转换 GEBCO 数据
            if "gebco_bathymetry" in data:
                gebco_data = data["gebco_bathymetry"]
                gebco_list.append(torch.from_numpy(gebco_data).unsqueeze(0).float())
            elif "target_labels" in data:
                target_labels = data["target_labels"]
                gebco_list.append(torch.from_numpy(target_labels).unsqueeze(0).float())
            else:
                logger.warning(
                    f"跳过文件 {path}: 找不到 'gebco_bathymetry' 或 'target_labels'"
                )
                continue

            # 加载TID数据，如果不存在则创建
            if "tid_data" in data:
                tid_data = data["tid_data"]
                tid_list.append(torch.from_numpy(tid_data).unsqueeze(0).float())
            else:
                h, w = gebco_list[-1].shape[1], gebco_list[-1].shape[2]
                tid_list.append(torch.full((1, h, w), 70.0, dtype=torch.float32))

        # 合并所有数据 (假设所有区域的H, W都相同，如果不同需要更复杂的padding处理)
        # 注意：这里我们假设所有区域的维度都是一致的，这需要数据准备阶段来保证。
        self.swot_data = torch.cat(swot_list, dim=1)  # 沿高度方向拼接
        self.gebco_data = torch.cat(gebco_list, dim=1)
        self.tid_data = torch.cat(tid_list, dim=1)

    def _generate_patches(self):
        """生成图像块索引"""
        self.patches = []
        self.patch_coords = []

        # SWOT数据的图像块
        H_swot, W_swot = self.swot_data.shape[1:]
        target_size = self.patch_size * 4  # GEBCO是4倍分辨率

        for i in range(0, H_swot - self.patch_size + 1, self.patch_size // 2):
            for j in range(0, W_swot - self.patch_size + 1, self.patch_size // 2):
                # 检查对应的GEBCO区域是否在范围内
                i_target = i * 4
                j_target = j * 4

                if (
                    i_target + target_size <= self.gebco_data.shape[1]
                    and j_target + target_size <= self.gebco_data.shape[2]
                ):
                    self.patches.append((i, j, i_target, j_target))
                    self.patch_coords.append({"row": i_target, "col": j_target})

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        i, j, i_target, j_target = self.patches[idx]

        # 提取SWOT输入块
        swot_patch = self.swot_data[:, i : i + self.patch_size, j : j + self.patch_size]

        # 提取GEBCO目标块
        target_size = self.patch_size * 4
        gebco_patch = self.gebco_data[
            :, i_target : i_target + target_size, j_target : j_target + target_size
        ]

        # 获取对应的TID数据块（总是存在）
        tid_patch = self.tid_data[
            :, i_target : i_target + target_size, j_target : j_target + target_size
        ]

        # 移除此处的增强调用，统一移动到DatasetWrapper中
        # if self.use_augmentation:
        #     swot_patch, gebco_patch, tid_patch = self._augment(swot_patch, gebco_patch, tid_patch)

        return {"input": swot_patch, "target": gebco_patch, "tid": tid_patch}

    # _augment, _apply_cutout, _apply_random_erasing 这些方法可以从SWOTDataset中移除，
    # 因为它们的功能已经被统一到了DatasetWrapper中。
    # 为了保持简洁，您可以删除从 def _augment(...) 到 return tensor 这一整块代码。
    # 如果您想保留它们作为参考，也可以不动。


class MultiDomainSWOTDataset(Dataset):
    """
    支持从多个独立海域（.npz文件）加载数据的数据集。
    每个海域的数据和归一化参数都将被独立处理。
    """

    def __init__(self, data_paths, patch_size=64):
        """
        Args:
            data_paths (list): 包含多个海域.npz文件路径的列表
            patch_size (int): 输入图像块的大小
        """
        self.patch_size = patch_size
        self.target_patch_size = self.patch_size * 4
        self.domains = []
        self.patch_map = []

        logger.info(f"初始化多海域数据集，共 {len(data_paths)} 个海域。")

        for domain_idx, path in enumerate(data_paths):
            logger.info(f"  - 正在加载海域 {domain_idx}: {path}")
            try:
                data = np.load(path)

                # 分别提取并存储每个海域的数据
                swot_data = (
                    self._get_tensor_from_keys(
                        data, ["swot_features", "input_features"], path
                    )
                    .permute(2, 0, 1)
                    .float()
                )
                gebco_data = (
                    self._get_tensor_from_keys(
                        data, ["gebco_bathymetry", "target_labels"], path
                    )
                    .unsqueeze(0)
                    .float()
                )

                if "tid_data" in data:
                    tid_data = torch.from_numpy(data["tid_data"]).unsqueeze(0).float()
                else:  # 如果不存在，则创建占位符
                    h, w = gebco_data.shape[1], gebco_data.shape[2]
                    tid_data = torch.full((1, h, w), 70.0, dtype=torch.float32)

                domain_info = {
                    "swot": swot_data,
                    "gebco": gebco_data,
                    "tid": tid_data,
                    "path": path,
                }
                self.domains.append(domain_info)

                # 为当前海域生成样本块索引
                H_swot, W_swot = swot_data.shape[1:]
                target_size = self.patch_size * 4

                domain_patch_count = 0
                for i in range(0, H_swot - self.patch_size + 1, self.patch_size // 2):
                    for j in range(
                        0, W_swot - self.patch_size + 1, self.patch_size // 2
                    ):
                        i_target = i * 4
                        j_target = j * 4
                        if (
                            i_target + target_size <= gebco_data.shape[1]
                            and j_target + target_size <= gebco_data.shape[2]
                        ):
                            # 将 (海域索引, x, y) 添加到全局地图
                            self.patch_map.append(
                                {"domain_idx": domain_idx, "i": i, "j": j}
                            )
                            domain_patch_count += 1
                logger.info(
                    f"    ... 成功加载，发现 {domain_patch_count} 个可用样本块。"
                )

            except (FileNotFoundError, KeyError) as e:
                logger.error(f"加载海域 {path} 时出错: {e}。已跳过此海域。")

        total_patches = len(self.patch_map)
        if total_patches == 0:
            raise ValueError("未能从提供的数据路径中生成任何有效的样本块。")
        logger.info(f"多海域数据集准备就绪。总样本块数量: {total_patches}")

    def _get_tensor_from_keys(self, data, keys, file_path):
        """辅助函数，用于从多个可能的键中安全地获取数据"""
        for key in keys:
            if key in data:
                return torch.from_numpy(data[key])
        raise KeyError(f"在文件 {file_path} 中找不到任何指定的键: {keys}")

    def __len__(self):
        return len(self.patch_map)

    def __getitem__(self, idx):
        # 1. 从全局地图中找到对应的海域和局部坐标
        patch_info = self.patch_map[idx]
        domain_idx = patch_info["domain_idx"]
        domain = self.domains[domain_idx]

        i, j = patch_info["i"], patch_info["j"]
        i_target, j_target = i * 4, j * 4

        # 2. 从指定海域的数据中提取样本块
        swot_patch = domain["swot"][:, i : i + self.patch_size, j : j + self.patch_size]
        gebco_patch = domain["gebco"][
            :,
            i_target : i_target + self.target_patch_size,
            j_target : j_target + self.target_patch_size,
        ]
        tid_patch = domain["tid"][
            :,
            i_target : i_target + self.target_patch_size,
            j_target : j_target + self.target_patch_size,
        ]

        # 注意: 此处暂未包含数据增强逻辑，将在下一步集成，以保持步骤清晰。

        return {
            "input": swot_patch,
            "target": gebco_patch,
            "tid": tid_patch,
            "domain_idx": domain_idx,
            "domain_name": os.path.splitext(
                os.path.basename(domain.get("path", str(domain_idx)))
            )[0],
        }


def get_dataloaders(
    data_path,
    batch_size=16,
    num_workers=4,
    use_augmentation=True,
    use_rotation=False,
    use_noise=False,
    use_cutout=False,
    use_random_erasing=False,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    include_test=False,
    random_seed=42,
    patch_size=64,
    balance_domains=False,
    target_domain_weight: float = 1.0,
):
    """
    创建训练、验证和测试数据加载器。
    现在支持单文件、目录或多海域路径列表。
    """
    # 确保比例总和为1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("训练、验证和测试集的比例之和必须为1")

    # --- 核心修改：根据data_path类型决定使用哪个Dataset ---
    if isinstance(data_path, list):
        logger.info("检测到多海域路径列表，使用 MultiDomainSWOTDataset。")
        base_dataset = MultiDomainSWOTDataset(
            data_paths=data_path, patch_size=patch_size
        )
    else:  # data_path是字符串（单个文件或目录）
        logger.info("检测到单个数据路径，使用 SWOTDataset。")
        base_dataset = SWOTDataset(
            data_path,
            patch_size=patch_size,
            use_augmentation=False,  # 基础数据集不使用增强
            use_rotation=False,
            use_noise=False,
            use_cutout=False,
            use_random_erasing=False,
        )

    # 分割数据集
    dataset_size = len(base_dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size

    # 确保测试集有样本
    if include_test and test_size <= 0:
        # 如果测试集太小，从验证集借用一些样本
        val_size = val_size - 10
        test_size = test_size + 10
        logger.warning(f"测试集样本数调整为 {test_size}")

    logger.info(
        f"数据集划分: 训练集={train_size}, 验证集={val_size}, 测试集={test_size if include_test else 0}"
    )

    # 使用固定的随机种子来保证划分的可复现性
    generator = torch.Generator().manual_seed(random_seed)
    train_subset, val_subset, test_subset = random_split(
        base_dataset, [train_size, val_size, test_size], generator=generator
    )

    # 🔧 修复：基于索引创建不同的数据集包装器
    class DatasetWrapper(torch.utils.data.Dataset):
        def __init__(
            self,
            base_dataset,
            indices,
            use_augmentation=False,
            use_rotation=False,
            use_noise=False,
            use_cutout=False,
            use_random_erasing=False,
        ):
            self.dataset = base_dataset
            # 确保 indices 在多进程中可序列化且可以按整数索引
            try:
                self.indices = list(indices)
            except Exception:
                # 兜底：如果无法直接转换，保留原样（尽量避免），但记录警告
                logger.warning("无法将传入的 indices 转为 list，保留原始对象。")
                self.indices = indices
            self.use_augmentation = use_augmentation
            self.use_rotation = use_rotation
            self.use_noise = use_noise
            self.use_cutout = use_cutout
            self.use_random_erasing = use_random_erasing

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            # 获取基础数据（无增强）
            try:
                base_idx = self.indices[idx]
                # 改进：直接调用基础数据集的__getitem__，避免重复逻辑
                data_dict = self.dataset[base_idx]
            except Exception as e:
                # 捕获并记录更明确的错误信息，便于在 worker 日志中排查
                logger.exception(
                    f"DatasetWrapper 在获取索引时出错: idx={idx}, base_idx_obj={getattr(self, 'indices', None)}"
                )
                raise
            swot_patch = data_dict["input"]
            gebco_patch = data_dict["target"]
            tid_patch = data_dict["tid"]
            domain_idx = data_dict.get("domain_idx")
            domain_name = data_dict.get("domain_name")

            # 根据配置应用增强
            if (
                self.use_augmentation
                or self.use_rotation
                or self.use_noise
                or self.use_cutout
                or self.use_random_erasing
            ):
                swot_patch, gebco_patch, tid_patch = self._augment(
                    swot_patch, gebco_patch, tid_patch
                )

            # 仅在多海域场景下返回 domain 字段；单文件数据集避免返回 None
            result = {"input": swot_patch, "target": gebco_patch, "tid": tid_patch}
            if domain_idx is not None and domain_name is not None:
                result["domain_idx"] = domain_idx
                result["domain_name"] = domain_name
            return result

        def _augment(self, swot_patch, gebco_patch, tid_patch):
            """应用数据增强（从原SWOTDataset复制）"""
            import numpy as np

            # 1. 基础翻转增强
            if self.use_augmentation:
                # 水平翻转
                if np.random.random() > 0.5:
                    swot_patch = torch.flip(swot_patch, [-1])
                    gebco_patch = torch.flip(gebco_patch, [-1])
                    tid_patch = torch.flip(tid_patch, [-1])

                # 垂直翻转
                if np.random.random() > 0.5:
                    swot_patch = torch.flip(swot_patch, [-2])
                    gebco_patch = torch.flip(gebco_patch, [-2])
                    tid_patch = torch.flip(tid_patch, [-2])

            # 2. 旋转增强
            if self.use_rotation and np.random.random() > 0.5:
                # 随机选择90度的倍数进行旋转
                k = np.random.choice([1, 2, 3])  # 90°, 180°, 270°
                swot_patch = torch.rot90(swot_patch, k, dims=[-2, -1])
                gebco_patch = torch.rot90(gebco_patch, k, dims=[-2, -1])
                tid_patch = torch.rot90(tid_patch, k, dims=[-2, -1])

            # 3. 随机噪声增强（只对输入数据）
            if self.use_noise and np.random.random() > 0.3:
                # 给SWOT输入数据添加小量高斯噪声
                noise_std = 0.01  # 噪声标准差
                noise = torch.randn_like(swot_patch) * noise_std
                swot_patch = swot_patch + noise

            # 4. Cutout增强（只对输入数据）
            if self.use_cutout and np.random.random() > 0.5:
                swot_patch = self._apply_cutout(swot_patch)

            # 5. Random Erasing增强（只对输入数据）
            if self.use_random_erasing and np.random.random() > 0.5:
                swot_patch = self._apply_random_erasing(swot_patch)

            return swot_patch, gebco_patch, tid_patch

        def _apply_cutout(self, tensor, cutout_ratio=0.25):
            """应用Cutout增强"""
            import numpy as np

            C, H, W = tensor.shape
            cutout_area = int(H * W * cutout_ratio)
            cutout_size = int(np.sqrt(cutout_area))
            cutout_size = min(cutout_size, H // 2, W // 2)

            if cutout_size > 0:
                x = np.random.randint(0, H - cutout_size + 1)
                y = np.random.randint(0, W - cutout_size + 1)
                tensor = tensor.clone()
                tensor[:, x : x + cutout_size, y : y + cutout_size] = 0

            return tensor

        def _apply_random_erasing(
            self,
            tensor,
            erasing_prob=0.5,
            area_ratio_range=(0.02, 0.4),
            aspect_ratio_range=(0.3, 3.3),
            value="random",
        ):
            """应用Random Erasing增强"""
            import numpy as np

            if np.random.random() > erasing_prob:
                return tensor

            C, H, W = tensor.shape
            area = H * W

            for _ in range(100):
                target_area = np.random.uniform(*area_ratio_range) * area
                aspect_ratio = np.random.uniform(*aspect_ratio_range)
                h = int(round(np.sqrt(target_area * aspect_ratio)))
                w = int(round(np.sqrt(target_area / aspect_ratio)))

                if w < W and h < H:
                    x1 = np.random.randint(0, H - h + 1)
                    y1 = np.random.randint(0, W - w + 1)
                    tensor = tensor.clone()

                    if value == "random":
                        tensor[:, x1 : x1 + h, y1 : y1 + w] = torch.randn(C, h, w) * 0.1
                    elif value == "zero":
                        tensor[:, x1 : x1 + h, y1 : y1 + w] = 0
                    else:
                        tensor[:, x1 : x1 + h, y1 : y1 + w] = value
                    break

            return tensor

    # 创建数据集包装器
    train_dataset = DatasetWrapper(
        base_dataset,
        train_subset.indices,
        use_augmentation=use_augmentation,
        use_rotation=use_rotation,
        use_noise=use_noise,
        use_cutout=use_cutout,
        use_random_erasing=use_random_erasing,
    )

    train_sampler = None
    if balance_domains and isinstance(base_dataset, MultiDomainSWOTDataset):
        train_indices = list(train_subset.indices)
        domain_counts = Counter(
            base_dataset.patch_map[idx]["domain_idx"] for idx in train_indices
        )
        if domain_counts:
            base_weights = {
                domain: 1.0 / count for domain, count in domain_counts.items()
            }
            target_idx = len(base_dataset.domains) - 1
            sample_weights = []
            for idx in train_indices:
                d = base_dataset.patch_map[idx]["domain_idx"]
                w = base_weights.get(d, 1.0)
                if d == target_idx:
                    w *= float(target_domain_weight)
                sample_weights.append(w)
            train_sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(train_indices), replacement=True
            )
            logger.info(
                f"已启用多海域加权采样：目标域=最后一个域(idx={target_idx}), 权重倍率={target_domain_weight}"
            )

    val_dataset = DatasetWrapper(
        base_dataset,
        val_subset.indices,
        use_augmentation=False,  # 验证集不使用增强
        use_rotation=False,
        use_noise=False,
        use_cutout=False,
        use_random_erasing=False,
    )

    # 创建数据加载器
    train_loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if train_sampler is not None:
        train_loader_kwargs["sampler"] = train_sampler
    else:
        train_loader_kwargs["shuffle"] = True

    train_loader = DataLoader(**train_loader_kwargs)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"数据加载器创建完成:")
    logger.info(f"- 训练集: {len(train_dataset)} 样本")
    logger.info(f"- 验证集: {len(val_dataset)} 样本")

    if include_test:
        test_dataset = DatasetWrapper(
            base_dataset,
            test_subset.indices,
            use_augmentation=False,  # 测试集不使用增强
            use_rotation=False,
            use_noise=False,
            use_cutout=False,
            use_random_erasing=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        logger.info(f"- 测试集: {len(test_dataset)} 样本")
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def get_test_dataloader(
    data_path, batch_size=16, num_workers=4, train_ratio=0.7, val_ratio=0.15
):
    """
    单独创建测试数据加载器（从完整数据集中提取测试部分）

    Args:
        data_path: 数据文件路径
        batch_size: 批次大小
        num_workers: 工作进程数
        train_ratio: 训练集比例（用于确定划分）
        val_ratio: 验证集比例（用于确定划分）

    Returns:
        test_loader
    """
    test_ratio = 1.0 - train_ratio - val_ratio

    # 创建测试数据集（不使用增强）
    full_dataset = SWOTDataset(
        data_path,
        use_augmentation=False,
        use_rotation=False,
        use_noise=False,
        use_cutout=False,
        use_random_erasing=False,
    )

    # 分割数据集
    dataset_size = len(full_dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size

    # 获取测试集部分
    _, _, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),  # 与训练时使用相同随机种子
    )

    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"测试数据加载器创建完成:")
    logger.info(f"- 测试集: {len(test_dataset)} 样本")

    return test_loader
