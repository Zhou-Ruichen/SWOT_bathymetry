#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的损失函数模块
修复了SSIM归一化和其他潜在问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM


class SobelFilter(nn.Module):
    """Sobel滤波器用于计算梯度"""

    def __init__(self):
        super(SobelFilter, self).__init__()
        # 注册为buffer而不是参数，避免梯度更新
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # 核心修复：在卷积前，将Sobel核的数据类型动态转换为与输入张量x一致
        grad_x = F.conv2d(x, self.sobel_x.to(x.dtype), padding=1)
        grad_y = F.conv2d(x, self.sobel_y.to(x.dtype), padding=1)
        gradient_magnitude = torch.sqrt(
            grad_x**2 + grad_y**2 + 1e-8
        )  # 添加小常数避免梯度消失
        return gradient_magnitude


class ComprehensiveLoss(nn.Module):
    """
    改进的综合损失函数
    支持MSE、MAE、梯度损失、SSIM损失的任意组合
    支持基于TID的自适应权重
    """

    def __init__(
        self,
        mse_weight=1.0,
        mae_weight=0.0,
        gradient_weight=0.0,
        ssim_weight=0.0,
        use_tid_weighting=False,
        **kwargs,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.gradient_weight = gradient_weight
        self.ssim_weight = ssim_weight
        self.use_tid_weighting = use_tid_weighting

        # 从 kwargs 安全地获取 device，并提供一个合理的默认值
        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 工具模块 - 并将其移动到正确的设备
        self.sobel = SobelFilter().to(self.device)
        # 修改SSIM配置以更好地处理海底地形数据
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=1, win_size=11)

        # TID权重配置
        self.tid_weights = {
            # 直接测量：高质量数据，权重最高
            10: 3.0,
            11: 3.0,
            12: 3.0,
            13: 3.0,
            14: 3.0,
            15: 3.0,
            16: 3.0,
            17: 3.0,
            # 间接测量：中等质量数据，中等权重
            40: 2.0,
            41: 2.0,
            42: 2.0,
            43: 2.0,
            44: 2.0,
            45: 2.0,
            46: 2.0,
            # 未知来源：低质量数据，基础权重
            70: 1.0,
            71: 1.0,
            72: 1.0,
            # 陆地/无效：权重为0
            0: 0.0,
        }

        # 用于存储各损失分量的字典
        self.loss_components = {}

    def normalize_for_ssim(self, tensor):
        """改进的SSIM归一化方法"""
        # 按batch逐个归一化，避免全局极值影响
        batch_size = tensor.shape[0]
        normalized = torch.zeros_like(tensor)

        for i in range(batch_size):
            img = tensor[i]
            img_min = img.min()
            img_max = img.max()

            # 避免除零错误
            if img_max - img_min > 1e-8:
                normalized[i] = (img - img_min) / (img_max - img_min)
            else:
                normalized[i] = torch.zeros_like(img)

        return normalized

    def get_tid_weights(self, tid_data):
        """根据TID数据生成权重掩膜"""
        if not self.use_tid_weighting:
            return torch.ones_like(tid_data, dtype=torch.float32)

        weight_mask = torch.zeros_like(tid_data, dtype=torch.float32)

        for tid_value, weight in self.tid_weights.items():
            mask = tid_data == tid_value
            weight_mask[mask] = weight

        return weight_mask

    def forward(self, pred, target, tid_data=None):
        """
        计算综合损失

        Args:
            pred: 预测值 [B, 1, H, W]
            target: 目标值 [B, 1, H, W]
            tid_data: TID数据 [B, 1, H, W]

        Returns:
            total_loss: 总损失值
        """
        total_loss = 0.0
        self.loss_components.clear()  # 每次前向传播时清空

        # 获取TID权重（如果启用）
        weights = None
        if tid_data is not None and self.use_tid_weighting:
            weights = self.get_tid_weights(tid_data)
            # 归一化权重，保持平均权重为1，避免除零
            weight_sum = weights.sum()
            if weight_sum > 1e-8:
                weights = weights / (weight_sum / weights.numel())

        # 1. MSE损失
        if self.mse_weight > 0:
            if weights is not None:
                mse_loss = torch.mean(weights * (pred - target) ** 2)
            else:
                mse_loss = F.mse_loss(pred, target)
            self.loss_components["mse"] = mse_loss.item()
            total_loss += self.mse_weight * mse_loss

        # 2. MAE损失
        if self.mae_weight > 0:
            if weights is not None:
                mae_loss = torch.mean(weights * torch.abs(pred - target))
            else:
                mae_loss = F.l1_loss(pred, target)
            self.loss_components["mae"] = mae_loss.item()
            total_loss += self.mae_weight * mae_loss

        # 3. 梯度损失
        if self.gradient_weight > 0:
            pred_grad = self.sobel(pred)
            target_grad = self.sobel(target)
            if weights is not None:
                # 梯度权重需要与梯度图像尺寸匹配
                grad_weights = F.interpolate(
                    weights, size=pred_grad.shape[-2:], mode="nearest"
                )
                grad_loss = torch.mean(grad_weights * (pred_grad - target_grad) ** 2)
            else:
                grad_loss = F.mse_loss(pred_grad, target_grad)
            self.loss_components["gradient"] = grad_loss.item()
            total_loss += self.gradient_weight * grad_loss

        # 4. SSIM损失（改进版本）
        if self.ssim_weight > 0:
            try:
                # 使用改进的归一化方法
                pred_norm = self.normalize_for_ssim(pred)
                target_norm = self.normalize_for_ssim(target)

                # 确保输入有效
                if pred_norm.sum() > 0 and target_norm.sum() > 0:
                    ssim_value = self.ssim(pred_norm, target_norm)
                    ssim_loss = 1 - ssim_value
                else:
                    # 如果归一化后全为0，使用MSE作为备选
                    ssim_loss = F.mse_loss(pred, target)

                self.loss_components["ssim"] = ssim_loss.item()
                total_loss += self.ssim_weight * ssim_loss
            except Exception as e:
                # 如果SSIM计算失败，使用MSE作为备选
                print(f"SSIM计算失败，使用MSE作为备选: {e}")
                ssim_loss = F.mse_loss(pred, target)
                self.loss_components["ssim"] = ssim_loss.item()
                total_loss += self.ssim_weight * ssim_loss

        return total_loss


def get_loss_function(
    mse_weight=1.0,
    mae_weight=0.0,
    gradient_weight=0.0,
    ssim_weight=0.0,
    use_tid_weighting=False,
    **kwargs,
):
    """
    创建损失函数 (通用版本)

    Args:
        mse_weight: MSE损失权重
        mae_weight: MAE损失权重
        gradient_weight: 梯度损失权重
        ssim_weight: SSIM损失权重
        use_tid_weighting: 是否使用TID自适应权重
        **kwargs: 任何其他参数，如device, perceptual_weight等

    Returns:
        loss_function: 损失函数实例
    """
    return ComprehensiveLoss(
        mse_weight=mse_weight,
        mae_weight=mae_weight,
        gradient_weight=gradient_weight,
        ssim_weight=ssim_weight,
        use_tid_weighting=use_tid_weighting,
        **kwargs,  # 将所有其他参数（如device, perceptual_weight）直接传递下去
    )
