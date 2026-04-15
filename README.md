# dafne-torch-trainer

[![PyPI version](https://img.shields.io/pypi/v/dafne-torch-trainer)](https://pypi.org/project/dafne-torch-trainer/)
[![Python](https://img.shields.io/pypi/pyversions/dafne-torch-trainer)](https://pypi.org/project/dafne-torch-trainer/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

PyTorch-based model trainer for the [Dafne](https://github.com/dafne-imaging) segmentation framework. Trains 2D and 3D U-Net-style models on medical images and serializes them into the `.model` format used by `dafne-dl`.

## Installation

```bash
pip install dafne-torch-trainer
```

Requires Python >= 3.9. A CUDA-capable GPU is strongly recommended for training.

## Entry points

| Command | Description |
|---|---|
| `dafne_trainer` | Launch the PyQt5 GUI trainer |
| `dafne_train` | Command-line training interface |

## Input data format

Training data must be `.npz` files, each containing:

- `data`: the image volume (numpy array)
- `mask_<label>`: one binary mask per anatomical structure (e.g. `mask_muscle`, `mask_femur`)
- `resolution`: voxel spacing array

The data folder is scanned recursively. All `.npz` files found are split into train and validation sets automatically.

## Output

All files produced by a training run are saved inside a dedicated folder named after the model, created automatically under the output directory. For example, given `--output /models/mymodel.model`, the following structure is created:

```
/models/mymodel/
    mymodel.model          # final serialized model (DynamicTorchModel format)
    mymodel_best_model.pth # best checkpoint by validation Dice (removed after packaging)
    mymodel.csv            # per-epoch metrics log
    _ewc.pt                # EWC snapshot (Fisher matrix + parameter snapshot, always saved)
    logs/
        train/             # TensorBoard training logs
        val/               # TensorBoard validation logs
```

The `.model` file embeds:

- model weights
- network architecture metadata (model name, spatial dims, patch size, spacing, etc.)
- training metadata
- a dependency hint pointing to `dafne-monai-inference` for inference-time use

The `_ewc.pt` file is saved at the end of every training run regardless of mode. It is required when running a subsequent continual learning session on the same output directory.

## CLI usage

### Training from scratch

```bash
dafne_train --data <data_dir> --output <output_path> [options]
```

| Argument | Short | Default | Description |
|---|---|---|---|
| `--data` | `-d` | required | Path to the folder containing training data |
| `--output` | `-o` | required | Output path for the `.model` file |
| `--epochs` | | 50 | Number of training epochs |
| `--batch-size` | | 2 | Batch size |
| `--lr` | | 0.001 | Learning rate |
| `--3d` | | off | Train a 3D model (default: 2D) |
| `--dynunet` | | off | Use Dynamic U-Net with auto-computed parameters |
| `--levels` | | 5 | Number of U-Net encoder/decoder levels |
| `--kernel-size` | | 3 | Convolution kernel size |
| `--conv-layers` | | 2 | Number of convolutional layers per level |
| `--early-stopping` | | off | Stop training when validation loss stops improving |
| `--mixed-precision` | | off | Enable AMP (automatic mixed precision) |
| `--scheduler` | | off | Enable learning rate scheduler |

Example:
```bash
dafne_train -d /data/training_set -o /models/my_model.model --epochs 100 --lr 0.0005 --early-stopping
```

### Fine-tuning an existing model

Pass `--pretrained` with the path to an existing `.model` file, and set `--mode` to `finetune`, `lora`, or `continual`.

```bash
dafne_train --data <data_dir> --output <output_path> --pretrained <model_path> --mode finetune [options]
```

| Argument | Default | Description |
|---|---|---|
| `--pretrained` | none | Path to a pretrained `.model` file |
| `--mode` | `scratch` | Training mode: `scratch`, `finetune`, `lora`, or `continual` |
| `--freeze-degree` | 0.5 | Fraction of layers to freeze (used with `--mode finetune`) |
| `--gradual-unfreeze` | off | Gradually unfreeze frozen layers during training |
| `--lora-rank` | 8 | LoRA rank (used with `--mode lora`) |
| `--lora-alpha` | 16 | LoRA alpha scaling factor (used with `--mode lora`) |
| `--lambda-reg` | 1.0 | EWC regularization weight (used with `--mode continual`) |

Example — fine-tuning with 70% of layers frozen:
```bash
dafne_train -d /data/new_data -o /models/finetuned.model --pretrained /models/base.model \
    --mode finetune --freeze-degree 0.7 --gradual-unfreeze --epochs 30
```

Example — LoRA adaptation:
```bash
dafne_train -d /data/new_data -o /models/lora.model --pretrained /models/base.model \
    --mode lora --lora-rank 8 --lora-alpha 16 --epochs 30
```

Example — continual learning with EWC:
```bash
dafne_train -d /data/task_b -o /models/mymodel/mymodel.model --pretrained /models/mymodel/mymodel.model \
    --mode continual --lambda-reg 1.0 --epochs 30
```

> The output directory must already contain a `_ewc.pt` file produced by a prior training run on the same path.

## Training modes

- **From scratch** (`--mode scratch`): network architecture and preprocessing are derived automatically from dataset statistics (median spacing, median shape, label count).
- **Fine-tuning** (`--mode finetune`): loads an existing `.model` file and resumes training, preserving the original architecture. Supports partial freezing and gradual unfreezing.
- **LoRA** (`--mode lora`): injects low-rank adapter layers into the frozen base model. Only adapter weights are trained. Useful for adaptation with very little data.
- **Continual learning** (`--mode continual`): fine-tunes on a new task while penalizing changes to weights that were important for the previous task, using Elastic Weight Consolidation (EWC). The penalty is `λ * Σ F_i * (θ_i - θ*_i)²`, where `F` is the diagonal Fisher Information Matrix and `θ*` are the weights from the prior training run. Both are loaded from `_ewc.pt`.
