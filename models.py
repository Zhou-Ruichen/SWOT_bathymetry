#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
现代化模型定义文件
最终优化版，采用全局平滑上采样策略以获得最稳健的基线结果
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InterpolateConvUp(nn.Module):
    """
    使用插值+卷积的上采样块，以避免棋盘格伪影。
    被验证为在此任务中表现最佳的模块。
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = ResConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class DoubleConv(nn.Module):
    """双重卷积块"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ResConvBlock(nn.Module):
    """带有残差连接的双重卷积块"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.double_conv(x)
        x += residual
        return self.final_relu(x)


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.GroupNorm(1, embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, mlp_hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(mlp_hidden_dim, embed_dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """跨层注意力块 - 用于skip connection"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_query, x_key_value):
        B, C, H_q, W_q = x_query.shape
        B, C, H_kv, W_kv = x_key_value.shape
        q = x_query.flatten(2).transpose(1, 2)
        kv = x_key_value.flatten(2).transpose(1, 2)
        q = self.q_proj(q).reshape(B, H_q*W_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(kv).reshape(B, H_kv*W_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, H_q*W_q, C)
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(B, C, H_q, W_q)
        return x


class Down(nn.Module):
    """下采样块 (使用ResConvBlock + Transformer)"""
    def __init__(self, in_channels, out_channels, use_transformer=True):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = ResConvBlock(in_channels, out_channels)
        self.use_transformer = use_transformer
        if use_transformer and out_channels >= 256:
            self.transformer = TransformerBlock(out_channels, num_heads=8)
        else:
            self.transformer = None

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        if self.transformer is not None:
            x = self.transformer(x)
        return x


class Up(nn.Module):
    """上采样块 (使用ResConvBlock)"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_reduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1_up = self.up(x1)
        if self.bilinear:
            x1_up = self.conv_reduce(x1_up)
        diffY = x2.size()[2] - x1_up.size()[2]
        diffX = x2.size()[3] - x1_up.size()[3]
        x1_padded = F.pad(x1_up, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1_padded], dim=1)
        return self.conv(x)


class TransformerUp(nn.Module):
    """现代化Transformer上采样块"""
    def __init__(self, in_channels, out_channels, bilinear=True, use_cross_attention=True):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.bilinear = bilinear
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_reduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        if use_cross_attention:
            self.cross_attn = CrossAttentionBlock(out_channels, num_heads=8)
            self.norm = nn.GroupNorm(1, out_channels)
        self.conv = ResConvBlock(in_channels, out_channels)
        if out_channels >= 128:
            self.transformer = TransformerBlock(out_channels, num_heads=8)
        else:
            self.transformer = None

    def forward(self, x1, x2):
        x1_up = self.up(x1)
        if self.bilinear:
            x1_up = self.conv_reduce(x1_up)
        diffY = x2.size()[2] - x1_up.size()[2]
        diffX = x2.size()[3] - x1_up.size()[3]
        x1_padded = F.pad(x1_up, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        if self.use_cross_attention:
            x2_att = self.cross_attn(x1_padded, x2)
            x2 = x2 + x2_att
            x2 = self.norm(x2)
        x = torch.cat([x2, x1_padded], dim=1)
        x = self.conv(x)
        if self.transformer is not None:
            x = self.transformer(x)
        return x


class OutConv(nn.Module):
    """输出卷积块"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """改进且通道数正确校对的标准U-Net模型"""
    def __init__(self, n_channels=4, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = ResConvBlock(n_channels, 64)
        self.down1 = Down(64, 128, use_transformer=False)
        self.down2 = Down(128, 256, use_transformer=False)
        self.down3 = Down(256, 512, use_transformer=False)
        self.down4 = Down(512, 1024, use_transformer=False)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.final_up = nn.Sequential(
            InterpolateConvUp(64, 32, scale_factor=2),
            InterpolateConvUp(32, 16, scale_factor=2)
        )
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.final_up(x)
        x = self.outc(x)
        return x


class TransformerUNet(nn.Module):
    """现代化的Transformer U-Net模型"""
    def __init__(self, n_channels=4, n_classes=1, bilinear=True):
        super(TransformerUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = ResConvBlock(n_channels, 64)
        self.down1 = Down(64, 128, use_transformer=False)
        self.down2 = Down(128, 256, use_transformer=True)
        self.down3 = Down(256, 512, use_transformer=True)
        self.down4 = Down(512, 1024, use_transformer=True)
        self.bottleneck = nn.Sequential(
            TransformerBlock(1024, num_heads=16, mlp_ratio=4.0),
            TransformerBlock(1024, num_heads=16, mlp_ratio=4.0),
            TransformerBlock(1024, num_heads=16, mlp_ratio=4.0),
            TransformerBlock(1024, num_heads=16, mlp_ratio=4.0),
        )
        self.up1 = TransformerUp(1024, 512, bilinear, use_cross_attention=True)
        self.up2 = TransformerUp(512, 256, bilinear, use_cross_attention=True)
        self.up3 = TransformerUp(256, 128, bilinear, use_cross_attention=True)
        self.up4 = TransformerUp(128, 64, bilinear, use_cross_attention=False)
        self.final_up = nn.Sequential(
            InterpolateConvUp(64, 32, scale_factor=2),
            InterpolateConvUp(32, 16, scale_factor=2)
        )
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.bottleneck(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.final_up(x)
        x = self.outc(x)
        return x


class AttentionGate(nn.Module):
    """注意力门机制"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.size() != x1.size():
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUp(nn.Module):
    """注意力上采样块"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(AttentionUp, self).__init__()
        self.bilinear = bilinear
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_reduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.attention = AttentionGate(
            F_g=in_channels // 2,
            F_l=out_channels,
            F_int=out_channels // 2
        )
        self.conv = ResConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1_up = self.up(x1)
        if self.bilinear:
            x1_up = self.conv_reduce(x1_up)
        x2_att = self.attention(g=x1_up, x=x2)
        diffY = x2.size()[2] - x1_up.size()[2]
        diffX = x2.size()[3] - x1_up.size()[3]
        x1_padded = F.pad(x1_up, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2_att, x1_padded], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """现代化的注意力U-Net模型"""
    def __init__(self, n_channels=4, n_classes=1, bilinear=True):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = ResConvBlock(n_channels, 64)
        self.down1 = Down(64, 128, use_transformer=False)
        self.down2 = Down(128, 256, use_transformer=False)
        self.down3 = Down(256, 512, use_transformer=False)
        self.down4 = Down(512, 1024, use_transformer=False)
        self.up1 = AttentionUp(1024, 512, bilinear)
        self.up2 = AttentionUp(512, 256, bilinear)
        self.up3 = AttentionUp(256, 128, bilinear)
        self.up4 = AttentionUp(128, 64, bilinear)
        self.final_up = nn.Sequential(
            InterpolateConvUp(64, 32, scale_factor=2),
            InterpolateConvUp(32, 16, scale_factor=2)
        )
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.final_up(x)
        x = self.outc(x)
        return x


def get_model(model_type='transformer_unet', n_channels=4, n_classes=1, bilinear=True):
    """获取模型"""
    if model_type == 'attention_unet':
        return AttentionUNet(n_channels, n_classes, bilinear)
    elif model_type == 'transformer_unet':
        return TransformerUNet(n_channels, n_classes, bilinear)
    elif model_type == 'unet':
        return UNet(n_channels, n_classes, bilinear)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def get_model_info():
    """返回可用模型的信息"""
    return {
        'transformer_unet': {
            'name': 'Transformer U-Net',
            'description': '现代化的Transformer U-Net，结合CNN和Transformer优势，具有全局建模能力',
            'recommended': True,
            'parameters': '~80M'
        },
        'attention_unet': {
            'name': 'Attention U-Net',
            'description': '使用注意力门机制的U-Net，专注于重要特征区域',
            'recommended': True,
            'parameters': '~35M'
        },
        'unet': {
            'name': 'Standard U-Net',
            'description': '经典U-Net架构，轻量级基线模型',
            'recommended': False,
            'parameters': '~31M'
        }
    }


if __name__ == '__main__':
    import os
    import torch

    # --- 配置 ---
    # 选择要可视化的模型: 'transformer_unet', 'attention_unet', 'unet'
    MODEL_TO_VISUALIZE = 'unet'

    # 设为 True 则导出为 ONNX 格式供 Netron 查看，否则使用 torchview 生成静态图
    EXPORT_FOR_NETRON = False

    # 模型输入参数
    N_CHANNELS = 4
    N_CLASSES = 1

    # 模型输入尺寸 (B, C, H, W)
    # U-Net 类模型通常要求输入尺寸的高和宽是 2 的 N 次方，N 为下采样层数
    INPUT_SIZE = (1, N_CHANNELS, 256, 256)

    # 输出配置
    OUTPUT_DIR = 'model_visualizations'

    # --- 执行 ---
    print(f"准备为 '{MODEL_TO_VISUALIZE}' 生成模型可视化文件...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = get_model(
        model_type=MODEL_TO_VISUALIZE,
        n_channels=N_CHANNELS,
        n_classes=N_CLASSES
    )
    model.eval()

    dummy_input = torch.randn(INPUT_SIZE, requires_grad=False)

    if EXPORT_FOR_NETRON:
        try:
            import torch.onnx
            print("正在导出模型到 ONNX 格式...")
            onnx_path = os.path.join(OUTPUT_DIR, f'{MODEL_TO_VISUALIZE}.onnx')

            torch.onnx.export(model,
                              dummy_input,
                              onnx_path,
                              export_params=True,
                              opset_version=11,
                              do_constant_folding=True,
                              input_names=['input'],
                              output_names=['output'],
                              dynamic_axes={'input': {0: 'batch_size'},
                                            'output': {0: 'batch_size'}})

            print(f"模型已成功导出到: {onnx_path}")
            print("请使用 Netron 打开此文件查看 (如访问 https://netron.app)")

        except ImportError:
            print("错误: 需要安装 onnx 和 onnxruntime。请运行: pip install onnx onnxruntime")
    else:
        try:
            from torchview import draw_graph
            print("正在使用 torchview 生成结构图...")
            model_graph = draw_graph(
                model,
                input_size=INPUT_SIZE,
                # 注意：文件名直接决定了输出格式 (如 .eps, .svg, .pdf)
                filename=f'{MODEL_TO_VISUALIZE}.pdf',
                save_graph=True,
                directory=OUTPUT_DIR,
                expand_nested=True,
                depth=1, # 保持 torchview 的视图简洁
                graph_dir='LR' # 'LR' 表示从左到右，可以更好地展示U型结构
            )
            print(f"模型 '{MODEL_TO_VISUALIZE}' 的结构图已保存到: {model_graph.visual_graph.filepath}")

        except ImportError:
            print("错误: 需要安装 torchview。请运行: pip install torchview")

    print("完成!")
