#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据训练域数据计算全局归一化参数，并将其应用到训练/泛化区域。
保持旧版脚本的数值流程，但默认按照最新输出文件一键运行。
"""

import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


DATA_DIR = 'output/1-data'
TRAIN_REGION_CODES = ('R2', 'R3', 'R4')
GENERALIZATION_CODES = ('T1', 'T2', 'T3')
OUTPUT_SUFFIX = '_norm.npz'


def _latest_region_file(code: str) -> Optional[str]:
    pattern = os.path.join(DATA_DIR, f'bandpass_{code}_*.npz')
    candidates = [path for path in glob.glob(pattern) if '_norm' not in os.path.basename(path)]

    plain_path = os.path.join(DATA_DIR, f'bandpass_{code}.npz')
    if os.path.exists(plain_path) and '_norm' not in os.path.basename(plain_path):
        candidates.append(plain_path)

    if not candidates:
        return None

    candidates = list(dict.fromkeys(candidates))
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _collect_region_files(codes) -> List[str]:
    files: List[str] = []
    for code in codes:
        latest = _latest_region_file(code)
        if latest:
            files.append(latest)
        else:
            print(f"⚠️ 警告: 未找到区域 {code} 的数据文件，已跳过")
    return files


def get_shortwave_data(file_path: str) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[np.ndarray], Optional[List[str]]]:
    """从 .npz 内提取用于统计的短波（带通信号）数据。"""
    try:
        data = np.load(file_path, allow_pickle=True)
        feature_names = data['feature_names'].tolist() if 'feature_names' in data else None

        swot_shortwave: Dict[str, np.ndarray] = {}
        if feature_names:
            for name in feature_names:
                key = f'swot_shortwave_{name}'
                if key in data:
                    swot_shortwave[name] = data[key]

        if not swot_shortwave and feature_names:
            # 兼容早期仅保存 swot_bandpass_* 的情况
            for name in feature_names:
                key = f'swot_bandpass_{name}'
                if key in data:
                    swot_shortwave[name] = data[key]

        gebco_shortwave = None
        if 'gebco_shortwave' in data:
            gebco_shortwave = data['gebco_shortwave']
        elif 'gebco_bandpass' in data:
            gebco_shortwave = data['gebco_bandpass']

        if not swot_shortwave or gebco_shortwave is None:
            return None, None, None
        return swot_shortwave, gebco_shortwave, feature_names
    except Exception as exc:  # noqa: BLE001
        print(f"❌ 警告: 加载 {os.path.basename(file_path)} 时出错: {exc}")
        return None, None, None


def calculate_global_norm_params(all_swot_data, all_gebco_data, feature_names):
    """计算全局归一化参数。"""
    print("🌍 正在计算全局归一化参数...")
    norm_params = {'swot': {}, 'gebco': {}}

    for name in feature_names:
        stacked = [domain[name].ravel() for domain in all_swot_data if name in domain]
        if not stacked:
            continue
        combined = np.concatenate(stacked)
        valid = combined[np.isfinite(combined)]
        if valid.size == 0:
            continue
        mean_val = float(np.mean(valid))
        std_val = float(np.std(valid))
        if std_val < 1e-8:
            std_val = 1.0
        norm_params['swot'][name] = {'mean': mean_val, 'std': std_val}
        print(f"  SWOT {name}: mean={mean_val:.4f}, std={std_val:.4f}")

    gebco_combined = np.concatenate([domain.ravel() for domain in all_gebco_data])
    gebco_valid = gebco_combined[np.isfinite(gebco_combined)]
    if gebco_valid.size == 0:
        raise ValueError("GEBCO 数据为空或无有效值，无法计算全局参数")

    mean_val = float(np.mean(gebco_valid))
    std_val = float(np.std(gebco_valid))
    if std_val < 1e-8:
        std_val = 1.0
    norm_params['gebco'] = {'mean': mean_val, 'std': std_val}
    print(f"  GEBCO: mean={mean_val:.4f}, std={std_val:.4f}")

    return norm_params


def apply_normalization(data: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    return (data - params['mean']) / params['std']


def rewrite_dataset(file_path: str, feature_names: List[str], global_norm_params: Dict[str, dict]) -> Optional[str]:
    print(f"  处理中: {os.path.basename(file_path)}")
    original = dict(np.load(file_path, allow_pickle=True))
    updated = original.copy()

    swot_features = []
    for name in feature_names:
        key = f'swot_shortwave_{name}'
        if key not in updated:
            # 兼容仅存储 bandpass 的文件
            key = f'swot_bandpass_{name}'
            if key not in updated:
                print(f"    ⚠️ 缺少通道 {name}，已跳过")
                continue
        norm_params = global_norm_params['swot'].get(name)
        if norm_params is None:
            print(f"    ⚠️ 全局参数缺少通道 {name}，已跳过")
            continue

        normalized = apply_normalization(updated[key], norm_params).astype(np.float32)
        updated[f'swot_shortwave_norm_{name}'] = normalized
        updated[f'swot_bandpass_norm_{name}'] = normalized
        swot_features.append(normalized)

    if not swot_features:
        print("    ⚠️ 未能生成任何 SWOT 通道，已跳过")
        return None

    updated['swot_features'] = np.stack(swot_features, axis=-1).astype(np.float32)

    if 'gebco_shortwave' in updated:
        gebco_source = updated['gebco_shortwave']
    else:
        gebco_source = updated.get('gebco_bandpass')

    if gebco_source is None:
        print("    ⚠️ 缺少 GEBCO 数据，已跳过")
        return None

    gebco_norm = apply_normalization(gebco_source, global_norm_params['gebco']).astype(np.float32)
    updated['gebco_shortwave_norm'] = gebco_norm
    updated['gebco_bandpass_norm'] = gebco_norm
    updated['gebco_bathymetry'] = gebco_norm

    updated['normalization_params'] = json.dumps(global_norm_params, ensure_ascii=False)

    output_path = file_path.replace('.npz', OUTPUT_SUFFIX)
    np.savez_compressed(output_path, **{k: v for k, v in updated.items() if v is not None})
    print(f"    -> 已保存: {os.path.basename(output_path)}")
    return output_path


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    train_files = _collect_region_files(TRAIN_REGION_CODES)
    if not train_files:
        print("❌ 错误: 未找到任何训练域文件，请先运行 1_prepare_single_region_data.py 或新的 1_prepare_data.py")
        return

    print("🔍 用于统计的训练域文件:")
    for path in train_files:
        print(f"  - {path}")

    all_swot_shortwave, all_gebco_shortwave = [], []
    feature_names: Optional[List[str]] = None

    for file_path in train_files:
        swot, gebco, names = get_shortwave_data(file_path)
        if swot and gebco is not None:
            all_swot_shortwave.append(swot)
            all_gebco_shortwave.append(gebco)
            if feature_names is None and names:
                feature_names = list(names)

    if not all_swot_shortwave or not feature_names:
        print("❌ 错误: 无法从训练域文件中解析有效短波数据")
        return

    global_norm_params = calculate_global_norm_params(
        all_swot_shortwave, all_gebco_shortwave, feature_names
    )

    apply_files = train_files + _collect_region_files(GENERALIZATION_CODES)
    apply_files = list(dict.fromkeys(apply_files))

    if not apply_files:
        print("❌ 错误: 未找到需要写回的文件")
        return

    print("\n🚀 正在应用全局归一化并生成新文件...")
    written_paths = []
    for file_path in apply_files:
        output_path = rewrite_dataset(file_path, feature_names, global_norm_params)
        if output_path:
            written_paths.append(output_path)

    if written_paths:
        print("\n" + "=" * 50)
        print("🎉 全部处理完成，请在训练配置中使用以下文件:")
        for path in written_paths:
            print(f"  - '{path}'")
        print("=" * 50)
    else:
        print("⚠️ 未生成任何新文件，请检查原始数据是否完整")


if __name__ == '__main__':
    main()
