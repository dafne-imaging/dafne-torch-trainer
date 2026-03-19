# dafne-torch-trainer

PyTorch-based model trainer for the Dafne segmentation framework. Trains 2D and 3D U-Net-style models on medical images (NIfTI format) and serializes them into the `.model` format used by `dafne-dl`.

## Dependencies

- `dafne-dl` (from `dafne-imaging/dafne-dl`, branch `master`)
- `dafne-monai-inference` (from `dafne-imaging/dafne-monai-inference`, branch `main`)
- PyTorch >= 2.0, MONAI >= 1.3, PyQt5 >= 5.15

See `requirements.txt` for the full list.

## Installation

```
pip install -e .
```

Requires Python >= 3.9. A CUDA-capable GPU is strongly recommended for training.

## Entry points

| Command | Description |
|---|---|
| `dafne_trainer` | Launch the PyQt5 GUI trainer |
| `dafne_train` | Command-line training interface |

## Input data format

Training data must be NIfTI files (`.nii` or `.nii.gz`) organized as image/mask pairs. Masks are binary volumes where each file represents one region of interest. The data folder structure expected by the data loader is configured via the GUI or the CLI config.

## Output

Training produces a `.model` file (serialized via `dafne-dl`'s `DynamicTorchModel`). The file embeds:

- model weights
- network architecture metadata (model name, spatial dims, patch size, spacing, etc.)
- training metadata
- a dependency hint pointing to `dafne-monai-inference` for inference-time use

A `_best_model.pth` checkpoint is saved during training and removed after the final `.model` is packaged.

## Project structure

```
src_torch/dafne_models/
    bin/                    # CLI entry points and model serialization
        train_cli.py        # CLI trainer
        create_torch_model.py  # DynamicTorchModel creation and serialization
    config/
        config_params.py    # Dataclasses for model, dataset, training, and metrics config
    core/
        data_manager.py     # Dataset split, CacheDataset, DataLoader construction
        train.py            # Main training loop (called by GUI and CLI)
        training_worker_engine.py  # PyQt5 QThread worker that wraps train.py
        transform/
            transforms_builder.py   # MONAI transform pipelines for training and fine-tuning
            custom_transforms.py    # Project-specific custom transforms
        engine/
            trainer_engine.py   # Custom training engine (trainer + evaluator loop)
            factory.py          # Engine factory (assembles trainer with callbacks)
            state.py            # Engine state dataclass
            events.py           # Engine event enum
            tasks/
                supervised_task.py  # Forward pass, loss, optimizer step
            callbacks/
                callbacks.py        # MetricsCallback, CheckpointCallback, EarlyStoppingCallback,
                                    # VisualizationCallback, GradualUnfreezeCallback, ClearGPUMemory
                save_metrics_callbacks.py  # TensorBoard and CSV logging
    models/
        dafne_networks.py   # Network architecture definitions
        factory.py          # ModelFactory: instantiation, LoRA wrapping, layer freezing
        wrapper.py          # DafneModelWrapper: load/save weights and metadata
        lora/
            layers.py       # LoRA linear layers
            lora_models.py  # LoRA model wrapping utilities
    ui/
        ModelTrainerSplit.py    # Main PyQt5 GUI window
        training_controller.py # GUI-side training control logic
        FineTuningDialog.py     # Fine-tuning options dialog
        AugmentationDialog_Ui.py  # Augmentation settings dialog (generated UI)
    utils/
        data_fingerprint.py # Dataset statistics: spacing, shape, label count
        optimizer.py        # Optimizer utilities (discriminative LR helpers)
```

## Training modes

- **From scratch**: network architecture and preprocessing are derived automatically from dataset statistics (median spacing, median shape, label count).
- **Fine-tuning**: loads an existing `.model` file and resumes training, preserving the original architecture. Supports partial freezing, gradual unfreezing, and LoRA adaptation.

## Notes

- Spurious/legacy files still present but unused: `core/pytorch_loop.py`, `core/training_worker.py`, `core/utils.py`, `ui/ModelTrainer.py`, `ui/ModelTrainer_Ui.py`. These can be removed.
- The `build/` directory at the repo root can also be removed.
