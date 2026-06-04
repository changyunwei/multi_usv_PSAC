# PSAC-MATD3

Ssource code for the PSAC-MATD3 pursuit-encirclement experiments. The package includes the training and evaluation entry points, pursuit-evasion environment, MATD3 agents, reward/CBF/HPER/ARM components, and default configuration files.

## Requirements

Use Python 3.6 or 3.7 with TensorFlow 1.x. TensorFlow 2.x is not supported because the implementation depends on TF1 graph/session APIs and `tensorflow.contrib`.

```powershell
conda create -n psac_matd3_tf1 python=3.7
conda activate psac_matd3_tf1
pip install -r requirements.txt
```

## Usage

Run a smoke test:

```powershell
python scripts/train.py --episode_num 1 --episode_len 5 --display False --save_dir results/smoke
```

Run training with the default configuration:

```powershell
python scripts/train.py --config configs/train_default.json --display False --save_dir results
```

Training creates a numbered run directory under `--save_dir` and prints it as `Resolved run dir`, for example `results/run_1`.

Evaluate a trained checkpoint:

```powershell
python scripts/evaluate.py --load_dir <resolved-run-dir>/latest --display False
```

## Configuration

Default training settings are defined in `configs/train_default.json`. `configs/eval_default.json` enables evaluation mode and inherits training defaults through `base_config`. Command-line arguments override JSON values.


## Outputs

Each run directory contains:

- `latest/`: latest TensorFlow checkpoint and `reward_curve_latest.csv`.
- `checkpoints/`: periodic checkpoints.
- `logs/`: merged configuration, command, and seed metadata.

Reward-curve PNGs and trajectory PDFs are disabled by default. Enable them with `--save_plots True` or `--trajectory_save_mode`.
