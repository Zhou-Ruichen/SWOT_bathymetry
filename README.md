# SWOT Bathymetry (Core Code)

This repository provides the core training and inference pipeline for bathymetry estimation experiments based on SWOT-related data processing.

## Scope

This public repository contains only core code and basic usage instructions for paper reproducibility.

Included files:
- `train.py`
- `predict.py`
- `model_configs.py`
- `models.py`
- `data_loader.py`
- `losses.py`
- `finetune.py`
- `run_ablation.py`
- `requirements.txt`

Not included:
- Model checkpoints / weights
- Large experiment logs and figures
- Backup and internal documentation materials

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

## Inference

```bash
python predict.py
```

## Fine-tuning (Optional)

```bash
python finetune.py
```

## Ablation Runner (Optional)

```bash
python run_ablation.py
```

## Data

The dataset is not distributed in this repository.
Please use publicly available data sources and configure local paths in scripts/configs as needed.

## Citation

If you use this code in academic work, please cite the corresponding paper.
