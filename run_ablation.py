"""
======================================================================
                     自动化消融实验运行器
======================================================================

脚本目的:
    本脚本旨在全自动化地执行一系列预定义的消融实验。它通过动态修改模型
    配置文件来创建不同的实验变体，并依次完成训练、评估、结果汇总的全过程，
    极大地简化了模型比较和调优的工作。

核心流程:
    1. 读取 `ablation_config.json`: 这是所有实验的中央控制文件。
    2. 循环处理实验: 对每个在配置文件中被启用的实验：
        a. 创建临时环境: 动态生成一个临时的模型配置文件 (`.py`)，
           该文件包含了基础配置和当前实验特定的覆盖项。
        b. 执行训练: 调用 `train.py` 脚本，使用新生成的配置进行模型训练
           和标准的域内验证。
        c. 执行泛化评估: 训练完成后，自动调用 `predict.py` 脚本，
           在指定的“域外”或“泛化”数据集上进行评估。
    3. 汇总结果: 所有实验完成后，脚本会收集每次运行的关键指标（如域内
       和跨域的RMSE, MAE等），并生成一个详细的 `ablation_summary.json`
       和一个易于分析的 `ablation_summary.csv` 报告。

======================================================================
                      如何运行一组消融实验
======================================================================

请遵循以下步骤来配置并运行您的实验:

步骤 1: 创建 `ablation_config.json` 文件
-------------------------------------------------
在项目根目录下创建一个名为 `ablation_config.json` 的文件。这是您
定义所有实验的地方。文件结构如下：

{
    "global_settings": {
        "generalization_data_path": "path/to/your/generalization_data.npz",
        "evaluation_margin": 100
    },
    "experiments": [
        {
            "name": "baseline_model",
            "group": "baseline",
            "enabled": true,
            "model_type": "unet",
            "config_overrides": {
                "loss_weights": { "mse": 1.0, "gradient": 1.0, "ssim": 0.2 }
            }
        },
        {
            "name": "no_data_augmentation",
            "group": "augmentation",
            "enabled": true,
            "model_type": "unet",
            "config_overrides": {
                "data_augmentation": {
                    "apply": false
                }
            }
        },
        {
            "name": "loss_without_ssim",
            "group": "loss_function",
            "enabled": false,
            "model_type": "unet",
            "config_overrides": {
                "loss_weights": { "mse": 1.0, "gradient": 1.0, "ssim": 0.0 }
            }
        }
    ]
}

- `global_settings`:
    - `generalization_data_path`: 指定用于所有实验的、统一的泛化（跨域）
      测试数据集路径。
    - `evaluation_margin`: 在泛化评估时，从数据边缘裁剪的像素数。
- `experiments` (列表):
    - `name`: 实验的唯一标识符，将用于生成结果目录名。
    - `group`: 用于对相关实验进行分组，方便在最终报告中进行比较。
    - `enabled` (true/false): 是否执行此实验。设为 `false` 可以暂时禁用。
    - `model_type`: 要使用的基础模型类型，必须与 `model_configs.py` 中
      的 `get_model_config` 函数所支持的类型匹配。
    - `config_overrides`: **核心部分**。一个字典，其中的键值对将深度
      覆盖 `model_configs.py` 返回的基础配置。例如，您可以修改学习率、
      损失函数权重、数据增强选项等。

步骤 2: 检查基础模型配置
-------------------------------------------------
确保您的 `model_configs.py` 文件中定义了清晰的、可供覆盖的基础配置。
本脚本正是通过修改这些基础配置来创建实验变体的。

步骤 3: 运行脚本
-------------------------------------------------
一切准备就绪后，在终端中直接运行此脚本：

$ python run_ablation.py

脚本将开始按顺序执行 `ablation_config.json` 中所有 `enabled: true`
的实验。您将在控制台中看到详细的训练和评估日志。

步骤 4: 分析结果
-------------------------------------------------
所有实验完成后，结果会自动汇总到一个以时间戳命名的目录中，例如：
`output/ablation_runs/run_20231027_153000/`

在该目录中，您会找到两个关键文件：
- `ablation_summary.json`: 包含每次实验所有细节的完整JSON报告。
- `ablation_summary.csv`: **推荐查看此文件**。这是一个扁平化的CSV表格，
  清晰地列出了每个实验的配置和最终的关键性能指标（如域内RMSE, 跨域
  RMSE等），非常便于在Excel或Pandas中进行横向对比分析。

"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
import argparse
import csv
from datetime import datetime
import pandas as pd
import numpy as np


# --- 常量定义 ---
ABLATION_CONFIG_FILE = "ablation_config.json"
BASE_MODEL_CONFIG_FILE = "model_configs.py"
BASE_TRAIN_SCRIPT = "train.py"
BASE_PREDICT_SCRIPT = "predict.py"

# 临时文件的名称
TEMP_CONFIG_MODULE = "temp_ablation_config"
TEMP_TRAIN_SCRIPT = "temp_ablation_train.py"


def calculate_metrics(pred, truth):
    """计算全面的评估指标: Mean, STD, MAE, RMSE, Correlation"""
    valid_mask = np.isfinite(pred) & np.isfinite(truth)
    if not np.any(valid_mask):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    residual = pred[valid_mask] - truth[valid_mask]

    mean_residual = np.mean(residual)
    std_residual = np.std(residual)
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual**2))

    # 计算相关系数
    pred_flat = pred[valid_mask]
    truth_flat = truth[valid_mask]
    correlation = np.corrcoef(pred_flat, truth_flat)[0, 1]

    return mean_residual, std_residual, mae, rmse, correlation


def calculate_denormalized_metrics(
    prediction_npz_path, generalization_data_path, norm_params_source_path, margin
):
    """
    加载预测结果和真值，进行反归一化，并计算以真实单位（米）为标准的性能指标。
    """
    # 1. 加载全局归一化参数
    with np.load(norm_params_source_path) as f:
        params_str = f["normalization_params"].item()
        norm_params = json.loads(params_str)
        gebco_params = norm_params["gebco"]
        mean, std = gebco_params["mean"], gebco_params["std"]

    # 2. 加载模型预测结果 (归一化的)
    with np.load(prediction_npz_path) as f:
        prediction_norm = f["prediction"]

    # 3. 加载真值 (未经归一化的，单位是米)
    with np.load(generalization_data_path) as f:
        truth_meters = f["gebco_shortwave"]

    # 4. **核心步骤: 反归一化**
    prediction_meters = prediction_norm * std + mean
    print(f"  反归一化完成: 使用 mean={mean:.4f}, std={std:.4f}")

    # 5. 对真值应用边距裁剪，以匹配预测结果的范围
    if margin > 0:
        h, w = truth_meters.shape
        if margin * 2 < h and margin * 2 < w:
            truth_meters = truth_meters[margin:-margin, margin:-margin]
        else:
            print(
                f"⚠️ 警告: 边距 {margin} 过大，无法在真值数据 (shape: {(h,w)}) 上应用。"
            )

    # 6. 检查形状是否匹配
    if prediction_meters.shape != truth_meters.shape:
        print(
            f"❌ 错误: 形状不匹配! 预测结果: {prediction_meters.shape}, 真值: {truth_meters.shape}",
            file=sys.stderr,
        )
        print("  请检查 predict.py 中的边距处理逻辑是否与此处一致。", file=sys.stderr)
        return None

    # 7. 计算指标
    metrics = calculate_metrics(prediction_meters, truth_meters)

    return {
        "mean_residual": metrics[0],
        "std_residual": metrics[1],
        "mae": metrics[2],
        "rmse": metrics[3],
        "correlation": metrics[4],
    }


def run_command(command, cwd=None):
    """执行一个shell命令并实时打印输出"""
    print(f"🚀 Executing: {' '.join(command)}")

    # 解决 ModuleNotFoundError 的问题 (例如 'models' 模块)
    # 我们需要将当前工作目录添加到子进程的 PYTHONPATH 中，
    # 这样 Python 才能找到项目本地的模块。
    env = os.environ.copy()
    if cwd:
        pythonpath = env.get("PYTHONPATH", "")
        paths = [p for p in pythonpath.split(os.pathsep) if p]
        if cwd not in paths:
            paths.insert(0, cwd)
        env["PYTHONPATH"] = os.pathsep.join(paths)

    # 使用 Popen 以便实时捕获输出
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        cwd=cwd,
        env=env,
    )
    if process.stdout is None:
        raise RuntimeError("Failed to capture subprocess stdout")

    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            # 直接打印，让日志看起来像是原始脚本输出的
            sys.stdout.write(output)
            sys.stdout.flush()

    return_code = process.poll()
    if return_code != 0:
        print(f"❌ Command failed with exit code {return_code}", file=sys.stderr)
        return False
    print(f"✅ Command executed successfully.")
    return True


def create_temp_config_file(base_config_path, overrides, temp_dir):
    """
    读取基础配置文件，应用覆盖项，并写入临时.py文件。
    """
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_config_content = f.read()

    # 我们需要动态地将 overrides 字典插入到 get_model_config 函数中
    # 这是一种比较hacky但有效的方法，避免了复杂的AST解析

    # --- START OF CHANGE ---
    # 找到 get_model_config 函数的定义 (更健壮的方式)
    import re

    match = re.search(
        r"def\s+get_model_config\s*\(.*?\)\s*->.*?:", base_config_content, re.S
    )
    if not match:
        raise ValueError(
            f"Could not find a suitable 'def get_model_config(...)' function definition in {base_config_path}"
        )

    func_def_str = match.group(0)
    # --- END OF CHANGE ---

    # 准备要插入的代码
    # 使用 repr() 来获取字典的Python代码表示
    overrides_str = repr(overrides)

    # 插入的代码逻辑：加载基础配置，然后用我们的覆盖项更新它
    injection_code = f"""
    # --- Injected by run_ablation.py ---
    overrides = {overrides_str}
    # 获取原始配置
    config = get_original_model_config(model_type)
    # 深度合并字典
    import collections.abc
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = deep_update(d.get(k, {{}}), v)
            else:
                d[k] = v
        return d
    config = deep_update(config, overrides)
    # --- End of injection ---
    """

    # 将原始函数重命名，并用我们的包装器替换它
    modified_content = base_config_content.replace(
        func_def_str,
        "def get_original_model_config(model_type, domain='single', dataset_key=None):",
    )
    modified_content += f"\n\n{func_def_str}\n{injection_code}\n    return config\n"

    temp_config_path = os.path.join(temp_dir, f"{TEMP_CONFIG_MODULE}.py")
    with open(temp_config_path, "w", encoding="utf-8") as f:
        f.write(modified_content)

    return temp_config_path


def create_temp_train_script(base_script_path, temp_dir, experiment_name):
    """
    修改原始训练脚本，使其从临时配置文件导入，并使用新的实验名称。
    """
    with open(base_script_path, "r", encoding="utf-8") as f:
        train_script_content = f.read()

    # 1. 修改导入语句
    modified_script = train_script_content.replace(
        "from model_configs import get_model_config",
        f"from {TEMP_CONFIG_MODULE} import get_model_config",
    )

    # 2. 修改实验名称生成逻辑
    # 我们直接覆盖 config['experiment_name']
    # 兼容当前 train.py 的参数签名 (包含 domain 与 dataset_key)
    target_line = (
        "config = get_model_config(\n"
        "        args.model_type, domain=args.domain, dataset_key=args.dataset_key\n"
        "    )"
    )
    replacement = (
        f"{target_line}\n" f"    config['experiment_name'] = '{experiment_name}'"
    )
    if target_line not in modified_script:
        raise ValueError(
            "未能在 train.py 中找到 get_model_config 调用，请检查函数签名是否变化"
        )
    modified_script = modified_script.replace(target_line, replacement)

    temp_script_path = os.path.join(temp_dir, TEMP_TRAIN_SCRIPT)
    with open(temp_script_path, "w", encoding="utf-8") as f:
        f.write(modified_script)

    return temp_script_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments with optional filtering"
    )
    parser.add_argument(
        "--config", default=ABLATION_CONFIG_FILE, help="Path to ablation_config.json"
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Case-insensitive substring to filter experiments by name. Example: multi_aug_on",
    )
    parser.add_argument(
        "--filter-fields",
        choices=["name", "group", "both"],
        default="name",
        help="Which fields to apply filter on",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List matched experiments and exit without running",
    )
    return parser.parse_args()


def main():
    """主函数，循环执行所有定义的消融实验"""
    args = parse_args()

    config_path = args.config if args.config else ABLATION_CONFIG_FILE

    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件 '{config_path}'", file=sys.stderr)
        sys.exit(1)

    # --- 1. 创建独立的消融实验结果目录 ---
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ablation_results_dir = os.path.join(
        "output", "ablation_runs", f"run_{run_timestamp}"
    )
    os.makedirs(ablation_results_dir, exist_ok=True)
    print(f"📁 本次消融实验所有总结报告将保存到: {ablation_results_dir}")

    with open(config_path, "r") as f:
        ablation_plan = json.load(f)

    global_settings = ablation_plan["global_settings"]
    experiments = [
        exp for exp in ablation_plan["experiments"] if exp.get("enabled", False)
    ]

    # 可选过滤：按名称/分组的子串（不区分大小写）筛选
    if args.filter:
        key = args.filter.lower()

        def match(exp):
            name = exp.get("name", "")
            group = exp.get("group", "")
            if args.filter_fields == "name":
                return key in name.lower()
            elif args.filter_fields == "group":
                return key in group.lower()
            else:
                return (key in name.lower()) or (key in group.lower())

        experiments = [exp for exp in experiments if match(exp)]

    if args.list:
        print("匹配到的实验（仅列出，不执行）：")
        for exp in experiments:
            print(f" - [{exp.get('group')}] {exp.get('name')}")
        print(f"总计: {len(experiments)} 个实验")
        return

    all_results = []  # <-- 用于收集所有实验结果

    total_experiments = len(experiments)
    if args.filter:
        print(
            f"发现 {total_experiments} 个匹配筛选条件的已启用实验 (filter='{args.filter}', fields='{args.filter_fields}')."
        )
    else:
        print(f"发现 {total_experiments} 个已启用的消融实验。")

    # 为本轮消融运行创建一个前缀，确保训练/评估输出目录与历史隔离
    ablation_run_prefix = f"abl_{run_timestamp}"

    for i, exp_config in enumerate(experiments):
        exp_name_with_timestamp = f"{ablation_run_prefix}__{exp_config['name']}_{datetime.now().strftime('%m%d_%H%M')}"

        result_entry = {
            "experiment_name": exp_name_with_timestamp,
            "base_name": exp_config["name"],
            "group": exp_config["group"],
            "status": "Not Started",
            "in_domain_metrics": None,
            "cross_domain_metrics": None,
            "config_overrides": exp_config["config_overrides"],
        }

        print("\n" + "=" * 80)
        print(f"🔬 开始消融实验 {i+1}/{total_experiments}: {exp_name_with_timestamp}")
        print(f"🔬 配置详情: {json.dumps(exp_config)}")
        print("=" * 80 + "\n")

        # 创建一个临时目录来存放本次实验的临时文件
        with tempfile.TemporaryDirectory() as temp_dir:
            # 将临时目录添加到Python路径，以便导入
            sys.path.insert(0, temp_dir)

            try:
                # --- 准备临时文件 ---
                print("🔧 正在生成临时配置文件和训练脚本...")
                create_temp_config_file(
                    BASE_MODEL_CONFIG_FILE, exp_config["config_overrides"], temp_dir
                )
                temp_train_script_path = create_temp_train_script(
                    BASE_TRAIN_SCRIPT, temp_dir, exp_name_with_timestamp
                )

                # --- 训练 ---
                print("\n--- STAGE 1: TRAINING & IN-DOMAIN EVALUATION ---\n")
                train_cmd = [
                    sys.executable,
                    temp_train_script_path,
                    exp_config["model_type"],
                ]
                model_dir = os.path.join(
                    "output", "2-experiments", exp_name_with_timestamp
                )

                if not run_command(train_cmd, cwd=os.getcwd()):
                    print(
                        f"😭 训练失败: {exp_name_with_timestamp}. 跳过此实验。",
                        file=sys.stderr,
                    )
                    result_entry["status"] = "Training Failed"
                    all_results.append(result_entry)
                    continue

                # --- 收集域内评估结果 ---
                in_domain_results_path = os.path.join(model_dir, "test_results.json")
                if os.path.exists(in_domain_results_path):
                    with open(in_domain_results_path, "r") as f:
                        result_entry["in_domain_metrics"] = json.load(f)
                    print(f"📊 已收集域内评估结果: {in_domain_results_path}")

                # --- 泛化评估 ---
                print("\n--- STAGE 2: PREDICTION ON CROSS-DOMAIN DATA ---\n")

                if not os.path.exists(model_dir):
                    print(
                        f"😭 找不到模型目录: {model_dir}. 无法进行评估。",
                        file=sys.stderr,
                    )
                    result_entry["status"] = "Prediction Failed (Model Dir Not Found)"
                    all_results.append(result_entry)
                    continue

                predict_cmd = [
                    sys.executable,
                    BASE_PREDICT_SCRIPT,
                    model_dir,
                    "--data-path",
                    global_settings["generalization_data_path"],
                    "--margin",
                    str(global_settings["evaluation_margin"]),
                ]
                if not run_command(predict_cmd, cwd=os.getcwd()):
                    print(
                        f"😭 泛化预测失败: {exp_name_with_timestamp}.", file=sys.stderr
                    )
                    result_entry["status"] = "Prediction Failed"
                    all_results.append(result_entry)
                    continue

                # --- STAGE 2.5: COLLECTING CROSS-DOMAIN RESULTS ---
                print(
                    "\n--- STAGE 2.5: Collecting Cross-Domain Metrics from Report ---\n"
                )

                # 修正路径: 直接使用 predict.py 中硬编码的目录名
                eval_sub_dir_name = "eval_on_generalization_set"
                report_path = os.path.join(
                    "output",
                    "3-evaluations",
                    exp_name_with_timestamp,
                    eval_sub_dir_name,
                    "evaluation_report.csv",
                )

                if os.path.exists(report_path):
                    print(f"  读取 predict.py 生成的评估报告: {report_path}")
                    try:
                        df = pd.read_csv(report_path)
                        # 筛选出我们关心的、已去除边缘的可靠区域指标
                        reliable_metrics = df[df["scope"] == "full_prediction_reliable"]
                        if not reliable_metrics.empty:
                            # 将指标行转换为字典并存储
                            cross_domain_metrics_dict = reliable_metrics.iloc[
                                0
                            ].to_dict()
                            result_entry["cross_domain_metrics"] = (
                                cross_domain_metrics_dict
                            )
                            print("\n📊 已收集跨域泛化评估结果 (单位: 米):")
                            print(json.dumps(cross_domain_metrics_dict, indent=4))
                            # 若实验覆盖中指定了 metrics_csv_path，则将泛化指标也写入同一CSV，便于汇总
                            metrics_csv_path = exp_config.get(
                                "config_overrides", {}
                            ).get("metrics_csv_path")
                            if metrics_csv_path:
                                try:
                                    metrics_dir = os.path.dirname(metrics_csv_path)
                                    if metrics_dir:
                                        os.makedirs(metrics_dir, exist_ok=True)
                                    # 追加写入，保持列覆盖域内/跨域主要指标
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
                                        "cross_mean_residual",
                                        "cross_std_residual",
                                        "cross_mae",
                                        "cross_rmse",
                                        "cross_correlation",
                                    ]
                                    file_exists = os.path.exists(metrics_csv_path)
                                    row = {
                                        "timestamp": datetime.now().isoformat(
                                            timespec="seconds"
                                        ),
                                        "experiment_name": exp_name_with_timestamp,
                                        "total_loss": result_entry.get(
                                            "in_domain_metrics", {}
                                        ).get("total_loss"),
                                        "rmse": result_entry.get(
                                            "in_domain_metrics", {}
                                        ).get("rmse"),
                                        "num_samples": result_entry.get(
                                            "in_domain_metrics", {}
                                        ).get("num_samples"),
                                        "mse_loss": result_entry.get(
                                            "in_domain_metrics", {}
                                        ).get("mse_loss"),
                                        "mae_loss": result_entry.get(
                                            "in_domain_metrics", {}
                                        ).get("mae_loss"),
                                        "gradient_loss": result_entry.get(
                                            "in_domain_metrics", {}
                                        ).get("gradient_loss"),
                                        "ssim_loss": result_entry.get(
                                            "in_domain_metrics", {}
                                        ).get("ssim_loss"),
                                        "cross_mean_residual": cross_domain_metrics_dict.get(
                                            "mean_residual"
                                        ),
                                        "cross_std_residual": cross_domain_metrics_dict.get(
                                            "std_residual"
                                        ),
                                        "cross_mae": cross_domain_metrics_dict.get(
                                            "mae"
                                        ),
                                        "cross_rmse": cross_domain_metrics_dict.get(
                                            "rmse"
                                        ),
                                        "cross_correlation": cross_domain_metrics_dict.get(
                                            "correlation"
                                        ),
                                    }
                                    with open(
                                        metrics_csv_path, "a", newline=""
                                    ) as csvfile:
                                        writer = csv.DictWriter(
                                            csvfile, fieldnames=fieldnames
                                        )
                                        if (
                                            not file_exists
                                            or os.path.getsize(metrics_csv_path) == 0
                                        ):
                                            writer.writeheader()
                                        writer.writerow(row)
                                    print(
                                        f"📑 已将泛化指标追加到CSV: {metrics_csv_path}"
                                    )
                                except Exception as e:
                                    print(f"⚠️ 泛化指标写入CSV失败: {e}")
                            result_entry["status"] = "Success"
                        else:
                            print(
                                f"😭 报告中未找到 'full_prediction_reliable' 范围的指标。",
                                file=sys.stderr,
                            )
                            result_entry["status"] = "Metrics Scope Not Found"

                    except Exception as e:
                        print(f"😭 读取或处理评估报告时发生错误: {e}", file=sys.stderr)
                        result_entry["status"] = f"Report Processing Crash: {e}"
                else:
                    print(
                        f"😭 找不到 predict.py 生成的评估报告: {report_path}",
                        file=sys.stderr,
                    )
                    result_entry["status"] = "Evaluation Report Not Found"

            except Exception as e:
                print(f"😭 执行实验时发生严重错误: {e}", file=sys.stderr)
                result_entry["status"] = f"Crashed: {e}"

            finally:
                # 从Python路径中移除临时目录
                sys.path.pop(0)
                all_results.append(result_entry)

        print("\n" + "=" * 80)
        print(f"🎉 完成消融实验: {exp_name_with_timestamp}")
        print("=" * 80 + "\n")

    # --- 3. 保存最终的总结报告 ---
    print("\n" + "✨" * 20)
    print("✨ 所有实验执行完毕，正在生成总结报告...")

    # 保存详细的JSON报告
    summary_json_path = os.path.join(ablation_results_dir, "ablation_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✅ 详细JSON报告已保存: {summary_json_path}")

    # 创建并保存扁平化的CSV报告以便于比较
    flat_results_for_csv = []
    for res in all_results:
        flat_row = {
            "group": res["group"],
            "name": res["base_name"],
            "status": res["status"],
            "experiment_id": res["experiment_name"],
        }
        # 添加域内指标
        if res.get("in_domain_metrics"):
            for k, v in res["in_domain_metrics"].items():
                flat_row[f"in_domain_{k}"] = v
        # 添加跨域指标
        if res.get("cross_domain_metrics"):
            for k, v in res["cross_domain_metrics"].items():
                flat_row[f"cross_domain_{k}"] = v
        flat_results_for_csv.append(flat_row)

    summary_df = pd.DataFrame(flat_results_for_csv)
    summary_csv_path = os.path.join(ablation_results_dir, "ablation_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 简明CSV报告已保存: {summary_csv_path}")
    print(f"✨ 请检查 {ablation_results_dir} 目录以查看所有结果。")


if __name__ == "__main__":
    main()
