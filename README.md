# MIE424 Project

Reproducing the paper **LookAhead Optimizer: k steps forward, 1 step back**.

## Setup (venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## The CIFAR-10 dataset: download and pre-processing

The CIFAR-10 dataset can be downloaded at [www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html), and it is a single compressed file containing serialized NumPy arrays.

You can also download the dataset directly from:

- [cifar-10-python.tar.gz: test and training sets (163 MB)](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

The archive contains the files `data_batch_1`, `data_batch_2`, ..., `data_batch_5`, as well as `test_batch`.

## CIFAR-10 data location

- Recommended local path: `data/cifar-10-batches-py`
- Loader behavior:
  - Uses local `cifar-10-batches-py` if found.
  - Downloads via torchvision only when missing and download is enabled.

## Quick dataloader smoke test

```bash
python scripts/smoke_dataloader.py --data-root data
```
