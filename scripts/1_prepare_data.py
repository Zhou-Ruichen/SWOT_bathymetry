#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
带通滤波数据集创建器 (高斯实现)
核心功能：读取数据 -> 应用高斯带通滤波 -> 网格对齐修复 -> 保存结果
"""
import os
import json
import numpy as np
import xarray as xr
from scipy import interpolate
from scipy.ndimage import gaussian_filter

class GaussianBandpassFilter:
    """
    使用高斯滤波器实现带通滤波，以避免振铃效应。
    通过两个不同尺度的高斯低通滤波结果相减实现。
    """
    def __init__(self, highpass_cutoff_km=150.0, lowpass_cutoff_km=8.0):
        self.highpass_cutoff_km = highpass_cutoff_km
        self.lowpass_cutoff_km = lowpass_cutoff_km
        print(f"🔧 高斯带通滤波器初始化: 通带 ~{lowpass_cutoff_km}km - ~{highpass_cutoff_km}km")

    def _wavelength_to_sigma(self, wavelength_km, dx_km):
        """将物理波长转换为高斯滤波器的sigma值"""
        # 高斯滤波器的sigma与截止频率/波长的关系约为 F_cutoff = 1 / (2 * pi * sigma)
        # 因此 sigma = 1 / (2 * pi * F_cutoff) = wavelength / (2 * pi)
        # 需要将物理单位(km)转换为像素单位
        sigma_pixels = (wavelength_km / (2 * np.pi)) / dx_km
        return sigma_pixels

    def apply_filter(self, data, dx_km, dy_km):
        """应用高斯带通滤波"""
        # 确保数据为浮点类型以支持NaN
        data = data.astype(np.float32)

        # 注意：dx_km 和 dy_km 应该相似，我们取平均值
        avg_dx_km = (dx_km + dy_km) / 2.0

        # 计算对应高通和低通的sigma值
        sigma_highpass = self._wavelength_to_sigma(self.highpass_cutoff_km, avg_dx_km)
        sigma_lowpass = self._wavelength_to_sigma(self.lowpass_cutoff_km, avg_dx_km)

        # 处理NaN: 使用插值或均值填充
        valid_mask = np.isfinite(data)
        data_filled = data.copy()
        # 一个简单的填充策略
        if np.any(~valid_mask):
            data_filled[~valid_mask] = np.nanmean(data)

        # 应用高斯滤波
        # 1. 滤除超长波 (> highpass_cutoff_km), 得到 < highpass_cutoff_km 的信号
        low_pass_filtered_1 = gaussian_filter(data_filled, sigma=sigma_highpass)
        signal_below_highpass = data_filled - low_pass_filtered_1

        # 2. 从中滤除超短波 (< lowpass_cutoff_km)
        low_pass_filtered_2 = gaussian_filter(signal_below_highpass, sigma=sigma_lowpass)
        bandpass_signal = low_pass_filtered_2

        # 恢复NaN
        bandpass_signal[~valid_mask] = np.nan

        print(f"📊 高斯滤波完成: "
              f"sigma_high={sigma_highpass:.2f}px, sigma_low={sigma_lowpass:.2f}px. "
              f"带通信号 std={np.nanstd(bandpass_signal):.2f}")
        return bandpass_signal

def validate_grid_alignment(swot_lat, swot_lon, gebco_lat, gebco_lon):
    """验证SWOT和GEBCO网格对齐情况"""
    print("🔍 验证网格对齐情况...")

    # 检查分辨率比例
    swot_lat_step = swot_lat[1] - swot_lat[0] if len(swot_lat) > 1 else 0
    swot_lon_step = swot_lon[1] - swot_lon[0] if len(swot_lon) > 1 else 0
    gebco_lat_step = gebco_lat[1] - gebco_lat[0] if len(gebco_lat) > 1 else 0
    gebco_lon_step = gebco_lon[1] - gebco_lon[0] if len(gebco_lon) > 1 else 0

    lat_ratio = swot_lat_step / gebco_lat_step if gebco_lat_step != 0 else 0
    lon_ratio = swot_lon_step / gebco_lon_step if gebco_lon_step != 0 else 0

    print(f"   分辨率比例: 纬度 {lat_ratio:.2f}, 经度 {lon_ratio:.2f}")

    # 检查格点对齐
    lat_offsets = []
    lon_offsets = []

    # 检查部分SWOT点是否在GEBCO网格上
    for i in range(0, len(swot_lat), max(1, len(swot_lat)//10)):
        swot_lat_val = swot_lat[i]
        closest_idx = np.argmin(np.abs(gebco_lat - swot_lat_val))
        closest_val = gebco_lat[closest_idx]
        offset = abs(swot_lat_val - closest_val)
        lat_offsets.append(offset)

    for i in range(0, len(swot_lon), max(1, len(swot_lon)//10)):
        swot_lon_val = swot_lon[i]
        closest_idx = np.argmin(np.abs(gebco_lon - swot_lon_val))
        closest_val = gebco_lon[closest_idx]
        offset = abs(swot_lon_val - closest_val)
        lon_offsets.append(offset)

    lat_max_offset = max(lat_offsets) if lat_offsets else 0
    lon_max_offset = max(lon_offsets) if lon_offsets else 0

    print(f"   格点偏移: 纬度最大 {lat_max_offset:.6f}° ({lat_max_offset*111:.1f}m)")
    print(f"             经度最大 {lon_max_offset:.6f}° ({lon_max_offset*111:.1f}m)")

    # 判断是否需要对齐修复
    need_alignment = (lat_max_offset > gebco_lat_step/2) or (lon_max_offset > gebco_lon_step/2)

    if need_alignment:
        print("   ⚠️ 检测到网格不对齐，建议使用对齐修复")
    else:
        print("   ✅ 网格对齐良好")

    return need_alignment, {
        'lat_max_offset': lat_max_offset,
        'lon_max_offset': lon_max_offset,
        'lat_ratio': lat_ratio,
        'lon_ratio': lon_ratio
    }

def fix_swot_grid_alignment(swot_data, swot_lat, swot_lon, gebco_lat, gebco_lon):
    """修复SWOT网格与GEBCO的对齐问题"""
    print("🔧 修复SWOT网格对齐...")

    # 创建精确对齐的SWOT坐标（每4个GEBCO点取1个）
    aligned_swot_lat = gebco_lat[::4]
    aligned_swot_lon = gebco_lon[::4]

    # 确保数量匹配
    if len(aligned_swot_lat) != len(swot_lat):
        target_count = len(swot_lat)
        # 尝试不同的起始点和步长来匹配
        for start_idx in [0, 1, 2, 3]:
            test_lat = gebco_lat[start_idx::4]
            if len(test_lat) >= target_count:
                aligned_swot_lat = test_lat[:target_count]
                break

        for start_idx in [0, 1, 2, 3]:
            test_lon = gebco_lon[start_idx::4]
            if len(test_lon) >= target_count:
                aligned_swot_lon = test_lon[:target_count]
                break

    print(f"   修正后SWOT坐标数量: {len(aligned_swot_lat)} × {len(aligned_swot_lon)}")

    # 计算坐标偏移
    lat_shift = aligned_swot_lat - swot_lat
    lon_shift = aligned_swot_lon - swot_lon

    max_lat_shift = np.abs(lat_shift).max()
    max_lon_shift = np.abs(lon_shift).max()

    print(f"   坐标偏移: 纬度最大{max_lat_shift:.6f}°, 经度最大{max_lon_shift:.6f}°")
    print(f"   偏移距离: 纬度最大{max_lat_shift*111:.1f}m, 经度最大{max_lon_shift*111:.1f}m")

    # 重新插值SWOT数据到对齐的坐标
    print("   🔄 重新插值SWOT数据...")

    aligned_swot_data = {}
    for key, data in swot_data.items():
        if key in ['lat', 'lon']:
            continue

        if isinstance(data, np.ndarray) and data.ndim == 2:
            print(f"     插值变量: {key}")

            # 创建插值函数
            interp_func = interpolate.RectBivariateSpline(
                swot_lat, swot_lon, data, kx=1, ky=1  # 双线性插值
            )

            # 在新坐标上插值
            data_interpolated = interp_func(aligned_swot_lat, aligned_swot_lon)
            aligned_swot_data[key] = data_interpolated
        else:
            # 非2D数据直接复制
            aligned_swot_data[key] = data

    # 更新坐标
    aligned_swot_data['lat'] = aligned_swot_lat
    aligned_swot_data['lon'] = aligned_swot_lon

    print("   ✅ SWOT网格对齐修复完成")

    return aligned_swot_data, {
        'max_lat_shift_degrees': float(max_lat_shift),
        'max_lon_shift_degrees': float(max_lon_shift),
        'max_shift_meters': float(max(max_lat_shift, max_lon_shift) * 111),
        'method': 'bilinear_interpolation_to_4x_gebco_grid'
    }

def load_data(target_lat=None, target_lon=None):
    """加载SWOT数据、GEBCO海底地形和TID数据

    Args:
        target_lat (tuple|None): (lat_min, lat_max)。若为None，则使用脚本内部默认设置。
        target_lon (tuple|None): (lon_min, lon_max)。若为None，则使用脚本内部默认设置。
    """
    print("📂 加载数据...")
    # 参数有效性检查：必须同时提供或同时省略
    if (target_lat is None) ^ (target_lon is None):
        raise ValueError("target_lat 与 target_lon 需要同时提供或同时省略")

    # 数据路径
    swot_paths = {
        'DOV_EW': '/mnt/data2/00-Data/nc/east_SWOT_02.nc',   # 垂线偏差东西分量
        'DOV_NS': '/mnt/data2/00-Data/nc/north_SWOT_02.nc',  # 垂线偏差南北分量
        'GA': '/mnt/data2/00-Data/nc/grav_SWOT_02.nc',       # 重力异常
        'VGG': '/mnt/data2/00-Data/nc/curv_SWOT_02.nc'       # 垂直重力异常梯度
    }

    gebco_path = '/mnt/data2/00-Data/nc/GEBCO_2024.nc'       # GEBCO 2024海底地形
    tid_path = '/mnt/data2/00-Data/nc/GEBCO_2024_TID.nc'     # TID数据类型

    # 参考区域说明与示例坐标可在 docs/ 内查阅

    if target_lat is None:
        target_lat = (-27.0, -8.0)
    if target_lon is None:
        target_lon = (-22.0, -3.0)

    # 1. 加载SWOT数据
    print("  📡 加载SWOT数据...")
    swot_data = {}
    for name, path in swot_paths.items():
        if os.path.exists(path):
            with xr.open_dataset(path) as ds:
                var_name = list(ds.data_vars.keys())[0]
                data = ds[var_name].sel(lat=slice(target_lat[0], target_lat[1]),
                                       lon=slice(target_lon[0], target_lon[1]))
                swot_data[name] = data.values
                if 'lat' not in swot_data:
                    swot_data['lat'] = data.lat.values
                    swot_data['lon'] = data.lon.values
                print(f"    {name}: {data.shape}")

    # 2. 加载GEBCO海底地形数据
    print("  🌊 加载GEBCO 2024海底地形...")
    gebco_data = None
    if os.path.exists(gebco_path):
        with xr.open_dataset(gebco_path) as ds:
            # 查找海底地形变量
            var_name = None
            for var in ds.data_vars:
                var_str = str(var)
                if 'elevation' in var_str.lower() or 'bathymetry' in var_str.lower():
                    var_name = var
                    break

            if var_name is None:
                var_name = list(ds.data_vars.keys())[0]  # 使用第一个变量

            data = ds[var_name].sel(lat=slice(target_lat[0], target_lat[1]),
                                   lon=slice(target_lon[0], target_lon[1]))
            gebco_data = {
                'elevation': data.values,
                'lat': data.lat.values,
                'lon': data.lon.values
            }
            print(f"    海底地形 ({var_name}): {data.shape}")

    # 3. 加载TID数据
    print("  🏷️ 加载TID数据...")
    tid_data = None
    if os.path.exists(tid_path):
        with xr.open_dataset(tid_path) as ds:
            print(f"    TID文件维度: {list(ds.dims.keys())}")
            print(f"    TID文件坐标: {list(ds.coords.keys())}")

            # 查找TID变量
            tid_var = None
            for var in ds.data_vars:
                var_str = str(var)
                if 'tid' in var_str.lower():
                    tid_var = var
                    break

            if tid_var is None:
                # 使用第一个数据变量（排除crs）
                data_vars = [v for v in ds.data_vars if 'crs' not in str(v).lower()]
                tid_var = data_vars[0] if data_vars else list(ds.data_vars.keys())[0]

            data = ds[tid_var].sel(lat=slice(target_lat[0], target_lat[1]),
                                   lon=slice(target_lon[0], target_lon[1]))
            tid_data = {
                'tid': data.values,
                'lat': data.lat.values,
                'lon': data.lon.values
            }
            print(f"    TID ({tid_var}): {data.shape}")
            unique_values = np.unique(data.values)
            print(f"    TID值范围: {unique_values[:10]}..." if len(unique_values) > 10 else f"    TID值范围: {unique_values}")
    else:
        print("    ⚠️ TID文件不存在，跳过")

    # 验证数据完整性
    if not swot_data:
        print("❌ SWOT数据加载失败")
        return None, None, None
    if not gebco_data:
        print("❌ GEBCO数据加载失败")
        return None, None, None

    # 打印数据范围验证
    print(f"📍 SWOT范围: 纬度 {swot_data['lat'].min():.2f}-{swot_data['lat'].max():.2f}, "
          f"经度 {swot_data['lon'].min():.2f}-{swot_data['lon'].max():.2f}")
    print(f"📍 GEBCO范围: 纬度 {gebco_data['lat'].min():.2f}-{gebco_data['lat'].max():.2f}, "
          f"经度 {gebco_data['lon'].min():.2f}-{gebco_data['lon'].max():.2f}")

    if tid_data:
        print(f"📍 TID范围: 纬度 {tid_data['lat'].min():.2f}-{tid_data['lat'].max():.2f}, "
              f"经度 {tid_data['lon'].min():.2f}-{tid_data['lon'].max():.2f}")

    return swot_data, gebco_data, tid_data

def apply_wavelength_filter_to_all(swot_data, gebco_data, highpass_cutoff_km=150.0, lowpass_cutoff_km=8.0):
    """对所有数据应用高斯带通滤波"""

    print(f"🔧 应用高斯带通滤波 (通带: {lowpass_cutoff_km}-{highpass_cutoff_km}km)...")

    # 使用GEBCO数据的分辨率计算
    lat = gebco_data['lat']
    lon = gebco_data['lon']
    dy_km = (lat[1] - lat[0]) * 111.32
    dx_km = (lon[1] - lon[0]) * 111.32 * np.cos(np.radians(np.mean(lat)))
    print(f"📏 GEBCO 分辨率: dx={dx_km:.3f} km, dy={dy_km:.3f} km")

    # SWOT 分辨率（可能与GEBCO略有不同）
    swot_lat = swot_data['lat']
    swot_lon = swot_data['lon']
    swot_dy_km = (swot_lat[1] - swot_lat[0]) * 111.32
    swot_dx_km = (swot_lon[1] - swot_lon[0]) * 111.32 * np.cos(np.radians(np.mean(swot_lat)))
    print(f"📏 SWOT 分辨率: dx={swot_dx_km:.3f} km, dy={swot_dy_km:.3f} km")

    filter_obj = GaussianBandpassFilter(highpass_cutoff_km, lowpass_cutoff_km)

    gebco_processed = filter_obj.apply_filter(gebco_data['elevation'], dx_km, dy_km)

    swot_processed = {}
    for name in ['DOV_EW', 'DOV_NS', 'GA', 'VGG']:
        if name in swot_data:
            filtered = filter_obj.apply_filter(swot_data[name], swot_dx_km, swot_dy_km)
            swot_processed[f'{name}_bandpass'] = filtered

    return {
        'gebco_bandpass': gebco_processed,
        'swot_bandpass': swot_processed,
        'resolution_km': {
            'gebco_dx': dx_km,
            'gebco_dy': dy_km,
            'swot_dx': swot_dx_km,
            'swot_dy': swot_dy_km
        }
    }

def normalize_bandpass_data(filtered_data, feature_names):
    """对带通数据进行Z标准化"""
    print("🔄 对带通数据应用Z标准化...")

    normalized_data = {}
    norm_params = {}

    # 1. 标准化SWOT带通数据
    swot_norm_params = {}
    for name in feature_names:
        bandpass_key = f'{name}_bandpass'
        if bandpass_key in filtered_data['swot_bandpass']:
            data = filtered_data['swot_bandpass'][bandpass_key]
            valid_mask = np.isfinite(data)
            valid_data = data[valid_mask]

            if len(valid_data) > 0:
                mean_val = np.mean(valid_data)
                std_val = np.std(valid_data)

                if std_val > 1e-8:
                    normalized = (data - mean_val) / std_val
                    print(f"  SWOT {name}: mean={mean_val:.3f}, std={std_val:.3f}")
                else:
                    normalized = data - mean_val
                    print(f"  SWOT {name}: 标准差过小，只减均值")

                normalized_data[f'{name}_bandpass_norm'] = normalized
                swot_norm_params[name] = {
                    'mean': float(mean_val),
                    'std': float(std_val) if std_val > 1e-8 else 1.0
                }

    # 2. 标准化GEBCO带通数据
    gebco_bandpass = filtered_data['gebco_bandpass']
    valid_mask = np.isfinite(gebco_bandpass)
    valid_data = gebco_bandpass[valid_mask]

    if len(valid_data) > 0:
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)

        if std_val > 1e-8:
            gebco_normalized = (gebco_bandpass - mean_val) / std_val
            print(f"  GEBCO: mean={mean_val:.3f}, std={std_val:.3f}")
        else:
            gebco_normalized = gebco_bandpass - mean_val
            print("  GEBCO: 标准差过小，只减均值")

        normalized_data['gebco_bandpass_norm'] = gebco_normalized
        norm_params['gebco'] = {
            'mean': float(mean_val),
            'std': float(std_val) if std_val > 1e-8 else 1.0
        }

    norm_params['swot'] = swot_norm_params
    return normalized_data, norm_params


def create_dataset(fix_grid_alignment=True, region_code=None, target_lat=None, target_lon=None,
                   base_output_dir="output/1-data"):
    """创建完整的带通滤波数据集，核心逻辑与旧版保持一致"""
    print("🌊 SWOT海底地形带通滤波数据集创建器")
    print("="*60)

    # 定义滤波参数
    HIGHPASS_CUTOFF_KM = 150.0
    LOWPASS_CUTOFF_KM = 8.0

    # 1. 加载数据
    swot_data, gebco_data, tid_data = load_data(target_lat=target_lat, target_lon=target_lon)

    if swot_data is None or gebco_data is None:
        print("❌ 数据加载失败")
        return None

    # 2. 检查网格对齐情况
    need_alignment, alignment_info = validate_grid_alignment(
        swot_data['lat'], swot_data['lon'],
        gebco_data['lat'], gebco_data['lon']
    )

    # 3. 可选的网格对齐修复
    alignment_correction_info = None
    if fix_grid_alignment and need_alignment:
        print("🔧 执行网格对齐修复...")
        swot_data, alignment_correction_info = fix_swot_grid_alignment(
            swot_data, swot_data['lat'], swot_data['lon'],
            gebco_data['lat'], gebco_data['lon']
        )
    elif need_alignment:
        print("⚠️ 检测到网格不对齐，但已跳过修复 (fix_grid_alignment=False)")

    # 4. 对所有数据应用带通滤波
    filtered_results = apply_wavelength_filter_to_all(
        swot_data,
        gebco_data,
        highpass_cutoff_km=HIGHPASS_CUTOFF_KM,
        lowpass_cutoff_km=LOWPASS_CUTOFF_KM
    )

    # 5. 准备特征列表
    feature_names = ['DOV_EW', 'DOV_NS', 'GA', 'VGG']
    available_features = [name for name in feature_names if name in swot_data]
    print(f"🎯 可用特征: {available_features}")

    # 6. 对带通数据进行Z标准化
    normalized_data, norm_params = normalize_bandpass_data(filtered_results, available_features)

    # 7. 构建完整数据集
    print("📦 构建数据集...")
    dataset = {
        # ==== 原始数据 (用于评估) ====
        'swot_raw': {name: swot_data[name].astype(np.float32) for name in available_features},
        'gebco_raw': gebco_data['elevation'].astype(np.float32),

        # ==== 滤波结果 (未标准化) ====
        'swot_bandpass': {name: filtered_results['swot_bandpass'][f'{name}_bandpass'].astype(np.float32)
                         for name in available_features},
        'gebco_bandpass': filtered_results['gebco_bandpass'].astype(np.float32),

        # ==== 兼容旧流程的“短波”数据表示 ====
        'swot_shortwave': {name: filtered_results['swot_bandpass'][f'{name}_bandpass'].astype(np.float32)
                           for name in available_features},
        'gebco_shortwave': filtered_results['gebco_bandpass'].astype(np.float32),

        # ==== 标准化数据 (用于训练) ====
        'swot_bandpass_norm': {name: normalized_data[f'{name}_bandpass_norm'].astype(np.float32)
                               for name in available_features if f'{name}_bandpass_norm' in normalized_data},
        'gebco_bandpass_norm': normalized_data['gebco_bandpass_norm'].astype(np.float32),

        # ==== 坐标和辅助信息 ====
        'swot_lat': swot_data['lat'].astype(np.float32),
        'swot_lon': swot_data['lon'].astype(np.float32),
        'gebco_lat': gebco_data['lat'].astype(np.float32),
        'gebco_lon': gebco_data['lon'].astype(np.float32),
        'tid_data': tid_data['tid'].astype(np.float32) if tid_data else None,
        'feature_names': np.array(available_features, dtype='U10'),
        'normalization_params': json.dumps(norm_params),

        # ==== 元数据 ====
        'metadata': json.dumps({
            'task_type': 'bandpass_filtering',
            'filter_type': 'gaussian',
            'highpass_cutoff_km': HIGHPASS_CUTOFF_KM,
            'lowpass_cutoff_km': LOWPASS_CUTOFF_KM,
            'swot_features': available_features,
            'swot_resolution': f"{swot_data[available_features[0]].shape}",
            'gebco_resolution': f"{gebco_data['elevation'].shape}",
            'has_tid': tid_data is not None,
            'normalization': 'z_score_bandpass_only',
            'resolution_km': filtered_results['resolution_km'],
            'grid_alignment': {
                'was_needed': bool(need_alignment),
                'was_applied': bool(alignment_correction_info is not None),
                'alignment_info': alignment_info,
                'correction_info': alignment_correction_info
            }
        })
    }

    # 8. 保存数据集
    # 根据模式区分文件名前缀，保持内部键名不变
    region_label = region_code or 'default'
    filename = f"bandpass_{region_label}.npz"
    output_path = os.path.join(base_output_dir, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 扁平化嵌套字典以便保存
    save_dict = {}
    for key, value in dataset.items():
        if isinstance(value, dict) and value is not None:
            for subkey, subvalue in value.items():
                save_dict[f'{key}_{subkey}'] = subvalue
        elif value is not None:
            save_dict[key] = value

    # === 添加与data_loader.py兼容的训练格式 ===
    print("🔄 生成兼容训练格式...")

    # 组合SWOT标准化带通特征 [H, W, C] 格式
    swot_features_list = []
    for name in available_features:
        key = f'swot_bandpass_norm_{name}'
        if key in save_dict:
            swot_features_list.append(save_dict[key])

    if swot_features_list:
        save_dict['swot_features'] = np.stack(swot_features_list, axis=-1).astype(np.float32)
        print(f"  ✓ swot_features (for training): {save_dict['swot_features'].shape}")

    # GEBCO标准化带通作为目标 [H, W] 格式
    if 'gebco_bandpass_norm' in save_dict:
        save_dict['gebco_bathymetry'] = save_dict['gebco_bandpass_norm'].astype(np.float32)
        print(f"  ✓ gebco_bathymetry (for training): {save_dict['gebco_bathymetry'].shape}")

    # === 添加方便评估的原始数据组合格式 ===
    # 组合SWOT原始特征用于评估 [H, W, C] 格式
    swot_raw_list = []
    for name in available_features:
        key = f'swot_raw_{name}'
        if key in save_dict:
            swot_raw_list.append(save_dict[key])

    if swot_raw_list:
        save_dict['input_features'] = np.stack(swot_raw_list, axis=-1).astype(np.float32)
        print(f"  ✓ input_features (for evaluation): {save_dict['input_features'].shape}")

    # GEBCO原始数据作为参考目标 [H, W] 格式
    if 'gebco_raw' in save_dict:
        save_dict['target_labels'] = save_dict['gebco_raw'].astype(np.float32)
        print(f"  ✓ target_labels (for evaluation): {save_dict['target_labels'].shape}")

    np.savez_compressed(output_path, **save_dict)

    print("✅ 数据集创建完成!")
    print(f"📁 保存路径: {output_path}")
    print(f"📊 SWOT数据形状: {swot_data[available_features[0]].shape}")
    print(f"📊 GEBCO数据形状: {gebco_data['elevation'].shape}")
    print(f"🎯 SWOT特征: {available_features}")
    print(f"🌊 滤波通带: {LOWPASS_CUTOFF_KM} - {HIGHPASS_CUTOFF_KM} km (高斯)")
    print(f"🏷️ TID数据: {'包含' if tid_data else '未包含'}")
    print(f"💾 包含数据: 原始数据, 带通分量, 标准化带通分量")
    print(f"🔧 网格对齐: {'需要且已修复' if (need_alignment and alignment_correction_info) else '不需要修复' if not need_alignment else '需要但未修复'}")

    return dataset

def create_dataset_with_alignment_check(**kwargs):
    """创建带网格对齐检查的数据集（推荐使用）"""
    return create_dataset(fix_grid_alignment=True, **kwargs)


def create_dataset_without_alignment(**kwargs):
    """创建不进行网格对齐修复的数据集"""
    return create_dataset(fix_grid_alignment=False, **kwargs)


def generate_all_regions(fix_alignment=True, output_dir="output/1-data"):
    """批量生成预设训练/测试区域的数据文件，免参数一键运行"""
    regions = [
        # 训练区域 格式是纬度+经度
        # ('R0', (-45.0, -35.0), (105.0, 120.0)),
        # ('R1', (-25.0, -15.0), (-115.0, -105.0)),
        ('R2', (43.0, 62.0), (-42.0, -23.0)),
        ('R3', (-27.0, -8.0), (-22.0, -3.0)),
        ('R4', (-37.0, -18.0), (48.0, 67.0)),
        #('R5', (-30.0, -23.0), (-120.0, -114.0)),
        #('R6', (-20.0, -15.0), (-112.0, -108.0)),
        #('R7', (-13.0, -8.0), (-112.0, -107.0)),

        # 测试/泛化区域
        ('T1', (-45.0, -35.0), (105.0, 120.0)),
        ('T2', (-22.0, -14.0), (-114.0, -108.0)),
        ('T3', (-20.0, -16.0), (-113.0, -109.0)),
    ]

    for code, lat_rng, lon_rng in regions:
        print(f"\n=== 处理区域 {code}: lat={lat_rng}, lon={lon_rng} ===")
        try:
            create_dataset(
                fix_grid_alignment=fix_alignment,
                region_code=code,
                target_lat=lat_rng,
                target_lon=lon_rng,
                base_output_dir=output_dir
            )
        except Exception as e:
            print(f"❌ 区域 {code} 处理失败: {e}")

    print("\n✅ 批量生成完成。")

if __name__ == '__main__':
    generate_all_regions()
