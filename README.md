# MIE424 Project: Reproducing Lookahead Optimizer

This repository reproduces and studies the paper **Lookahead Optimizer: k steps forward, 1 step back** using PyTorch.

The project centers on comparing standard first-order optimizers against their Lookahead-wrapped variants on image classification benchmarks. The main experimental pipeline supports:

- `CIFAR-10`
- `MNIST`
- `Fashion-MNIST`

The primary model used in the repo is `ResNet-18`, trained from scratch for 10-class classification.

## Project Goal
The goal of the project is to reproduce the core empirical story of the Lookahead paper:

- wrap Lookahead around baseline optimizers such as SGD and Adam
- compare convergence and validation/test performance
- study sensitivity to Lookahead hyperparameters `k` and `alpha`
- evaluate whether Lookahead improves stability and final performance

## Implemented Features
This repo currently includes:

- dataset loaders for `CIFAR-10`, `MNIST`, and `Fashion-MNIST`
- deterministic train/validation split from the official training set
- baseline optimizers:
  - `sgd`
  - `sgd_momentum`
  - `adam`
- Lookahead optimizers:
  - `lookahead_sgd`
  - `lookahead_sgd_momentum`
  - `lookahead_adam`
- training and evaluation entrypoints
- TensorBoard logging
- experiment suite runner for reproducible sweeps
- synthetic quadratic-bowl analysis for visualizing Lookahead behavior
- animation script for presentation/demo use

## Repository Structure
```text
.
├── README.md
├── requirements.txt
├── data/
├── results/
├── scripts/
│   ├── animation.py
│   ├── evaluate.py
│   ├── quadratic_analysis.py
│   ├── run_experiments.py
│   ├── run_quadratic_analysis.py
│   ├── smoke_dataloader.py
│   └── train.py
└── src/
    ├── analysis/
    ├── data/
    ├── engine/
    ├── models/
    ├── optim/
    └── utils/
```

## Setup
Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets
### CIFAR-10
The CIFAR-10 dataset is available at:

- [Dataset page](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Direct download: cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

If you download it manually, extract it so the folder structure is:

```text
data/
└── cifar-10-batches-py/
```

The archive contains:

- `data_batch_1` through `data_batch_5`
- `test_batch`
- `batches.meta`

The loader uses `torchvision.datasets.CIFAR10`, which reads that extracted format directly.

### MNIST and Fashion-MNIST
MNIST and Fashion-MNIST can be downloaded automatically through `torchvision` if they are missing locally.

If you want manual local copies, the expected roots are:

- `data/MNIST/`
- `data/FashionMNIST/`

## Data Loading Behavior
All dataset loaders:

- create a deterministic train/validation split from the training set using a seed
- keep the official test set untouched for final evaluation
- apply augmentation only to the training split
- use evaluation transforms for validation and test

The loader entrypoints are implemented in [dataloaders.py](/Users/yusufmoola/Desktop/UofT%20Y4/MIE424/MIE424-Project/src/data/dataloaders.py).

## Quick Smoke Test
To verify the dataloaders and deterministic validation split:

```bash
python scripts/smoke_dataloader.py --dataset cifar10 --data-root data
```

You can replace `cifar10` with `mnist` or `fashion_mnist`.

## Training a Single Run
The main training entrypoint is [train.py](/Users/yusufmoola/Desktop/UofT%20Y4/MIE424/MIE424-Project/scripts/train.py).

Example: train `Lookahead + Adam` on CIFAR-10

```bash
python scripts/train.py \
  --dataset cifar10 \
  --data-root data \
  --model resnet18 \
  --optimizer lookahead_adam \
  --lr 0.001 \
  --lookahead-k 5 \
  --lookahead-alpha 0.5 \
  --epochs 100 \
  --batch-size 128 \
  --device auto \
  --no-download
```

Example: baseline `SGD + momentum`

```bash
python scripts/train.py \
  --dataset cifar10 \
  --data-root data \
  --model resnet18 \
  --optimizer sgd_momentum \
  --lr 0.03 \
  --momentum 0.9 \
  --epochs 100 \
  --batch-size 128 \
  --device auto \
  --no-download
```

### Training Outputs
Each training run writes:

- metrics CSV
- TensorBoard logs
- checkpoints including `latest.pt` and `best.pt`

By default, outputs are written under:

```text
results/<dataset>/
├── logs/
├── tensorboard/
└── checkpoints/
```

## Evaluating a Checkpoint
Use [evaluate.py](/Users/yusufmoola/Desktop/UofT%20Y4/MIE424/MIE424-Project/scripts/evaluate.py) to evaluate a checkpoint on the validation or test split.

Example:

```bash
python scripts/evaluate.py \
  --checkpoint results/cifar10/checkpoints/final_repeats/<run_name>/best.pt \
  --dataset cifar10 \
  --data-root data \
  --model resnet18 \
  --split test \
  --device auto \
  --no-download
```

To also save the evaluation as a CSV row:

```bash
python scripts/evaluate.py \
  --checkpoint results/cifar10/checkpoints/final_repeats/<run_name>/best.pt \
  --dataset cifar10 \
  --data-root data \
  --model resnet18 \
  --split test \
  --device auto \
  --no-download \
  --output-csv results/cifar10/experiments/manual_eval.csv
```

## TensorBoard
Launch TensorBoard from the project root:

```bash
python -m tensorboard.main --logdir results --port 6006
```

Then open:

```text
http://localhost:6006
```

Useful scalars logged during training include:

- `Loss/train`
- `Loss/validation`
- `Accuracy/train`
- `Accuracy/validation`
- `Optimizer/learning_rate`
- `Optimizer/grad_norm`
- `Diagnostics/generalization_gap_acc`
- `Diagnostics/generalization_gap_loss`
- `Diagnostics/epoch_time_sec`
- `Lookahead/fast_slow_distance` for Lookahead runs

## Reproducing the Main Experiments
The full experiment pipeline is handled by [run_experiments.py](/Users/yusufmoola/Desktop/UofT%20Y4/MIE424/MIE424-Project/scripts/run_experiments.py).

### Supported Suites
- `pilot_lr`
- `core_comparison`
- `lookahead_sensitivity`
- `final_repeats`
- `final_test`
- `all`

### Recommended Run Order
If running suites separately, use this order:

1. `pilot_lr`
2. `core_comparison`
3. `lookahead_sensitivity`
4. `final_repeats`
5. `final_test`

This order matters because later suites depend on CSV summaries produced by earlier ones.

### Run All Suites for CIFAR-10
```bash
python scripts/run_experiments.py --dataset cifar10 --suite all --device auto
```

Equivalent shorthand:

```bash
python scripts/run_experiments.py --dataset cifar10 --device auto
```

### Run Individual Suites for CIFAR-10
```bash
python scripts/run_experiments.py --dataset cifar10 --suite pilot_lr --device auto
python scripts/run_experiments.py --dataset cifar10 --suite core_comparison --device auto
python scripts/run_experiments.py --dataset cifar10 --suite lookahead_sensitivity --device auto
python scripts/run_experiments.py --dataset cifar10 --suite final_repeats --device auto
python scripts/run_experiments.py --dataset cifar10 --suite final_test --device auto
```

### Run Experiments for MNIST
```bash
python scripts/run_experiments.py --dataset mnist --suite all --device auto
```

### Run Experiments for Fashion-MNIST
```bash
python scripts/run_experiments.py --dataset fashion_mnist --suite all --device auto
```

### Download Behavior
If the dataset is already present locally, keep downloads disabled:

```bash
python scripts/run_experiments.py --dataset cifar10 --suite all --device auto
```

If you want to allow `torchvision` to download missing data:

```bash
python scripts/run_experiments.py --dataset cifar10 --suite all --device auto --allow-download
```

## Experiment Design in the Runner
The experiment runner is dataset-aware and uses different learning-rate grids and epoch counts per dataset.

For `cifar10`, the default study is:

- `pilot_lr`
  - 10 epochs
  - LR sweep for SGD-family and Adam-family
- `core_comparison`
  - 100 epochs
  - compares `sgd`, `sgd_momentum`, `adam`, `lookahead_sgd_momentum`, `lookahead_adam`
- `lookahead_sensitivity`
  - 30 epochs
  - sweeps:
    - `(k=5, alpha=0.2)`
    - `(k=5, alpha=0.5)`
    - `(k=5, alpha=0.8)`
    - `(k=10, alpha=0.5)`
    - `(k=20, alpha=0.5)`
- `final_repeats`
  - 100 epochs
  - seeds `42`, `52`, `62`
- `final_test`
  - evaluates `best.pt` from final repeated runs on the official test set

## Results Layout
Results are separated by dataset:

```text
results/
└── <dataset>/
    ├── checkpoints/
    │   └── <suite>/
    │       └── <run_name>/
    ├── logs/
    │   └── <suite>/
    │       └── <run_name>.csv
    ├── tensorboard/
    │   └── <suite>/
    │       └── <run_name>/
    └── experiments/
        ├── manifest.csv
        ├── pilot_lr_selected.csv
        ├── lookahead_sensitivity_selected.csv
        ├── final_repeats_summary.csv
        ├── final_test_summary.csv
        └── <suite>/
```

This keeps CIFAR-10, MNIST, and Fashion-MNIST artifacts separate and reproducible.

## Reproducing the CIFAR-10 Results End-to-End
Assuming the dataset is already present at `data/cifar-10-batches-py`:

```bash
source .venv/bin/activate
python scripts/smoke_dataloader.py --dataset cifar10 --data-root data --no-download
python scripts/run_experiments.py --dataset cifar10 --suite all --device auto
```

After completion, the key summary files are:

- `results/cifar10/experiments/pilot_lr_selected.csv`
- `results/cifar10/experiments/lookahead_sensitivity_selected.csv`
- `results/cifar10/experiments/final_repeats_summary.csv`
- `results/cifar10/experiments/final_test_summary.csv`

## Quadratic Analysis
In addition to the image-classification experiments, the repo includes a synthetic quadratic-bowl analysis that helps visualize the smoothing/stability effect of Lookahead.

Run:

```bash
python scripts/run_quadratic_analysis.py
```

Outputs are written under:

```text
results/quadratic/
```

Generated artifacts include:

- `trajectories.png`
- `distance_to_optimum.png`
- `variance_across_runs.png`
- `lookahead_fast_vs_slow_variance.png`
- `metrics.csv`

## Notes on Compute
- `--device auto` selects `cuda`, then `mps`, then `cpu`
- on Apple Silicon, training uses `mps` when available
- heavy suites should generally be run one at a time rather than in parallel, because parallel training jobs will compete for the same accelerator

## Citation
Paper reproduced in this project:

> Michael R. Zhang, James Lucas, Geoffrey Hinton, and Jimmy Ba. 2019. *Lookahead Optimizer: k steps forward, 1 step back*.
