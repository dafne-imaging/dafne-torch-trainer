import argparse
import torch
from ..ui.training_controller import build_training_configs
from ..core.train import run_training

def main():
    parser = argparse.ArgumentParser(description='Dafne model trainer CLI')

    parser.add_argument('--data', '-d', required=True, help='Input data path for training or fine-tuning model')
    parser.add_argument('--output', '-o', required=True, help='Output path for .model saving')
    parser.add_argument('--pretrained', help='Path of pretrained model')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('--batch-size', dest='batch_size', default=2, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--mode', default='scratch', choices=['scratch', 'finetune', 'lora'], help='Training mode')
    parser.add_argument('--3d', dest='is_3d', action='store_true', help='Train a 3D model')
    parser.add_argument('--dynunet', action='store_true', help='Use Dynamic U-Net (auto params)')
    parser.add_argument('--early-stopping', dest='early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--mixed-precision', dest='mixed_precision', action='store_true', help='Enable mixed precision')
    parser.add_argument('--scheduler', action='store_true', help='Enable LR scheduler')
    parser.add_argument('--levels', dest='n_levels', default=5, type=int, help='Number of U-Net levels')
    parser.add_argument('--kernel-size', dest='kernel_size', default=3, type=int, help='Convolution kernel size')
    parser.add_argument('--conv-layers', dest='conv_layers', default=2, type=int, help='Number of convolutional layers per level')
    parser.add_argument('--lora-rank', dest='lora_rank', default=8, type=int, help='LoRA rank (used with --mode lora)')
    parser.add_argument('--lora-alpha', dest='lora_alpha', default=16, type=int, help='LoRA alpha (used with --mode lora)')
    parser.add_argument('--freeze-degree', dest='freeze_degree', default=0.5, type=float, help='Fraction of layers to freeze (used with --mode finetune)')
    parser.add_argument('--gradual-unfreeze', dest='gradual_unfreeze', action='store_true', help='Enable gradual unfreezing (used with --mode finetune)')

    args = parser.parse_args()

    augm_params = {
        'rotate': False, 'flip_x': False, 'flip_y': False,
        'zoom': False, 'noise': False,
    }

    adaptation_params = {
        'mode': args.mode,
        'freeze_degree': args.freeze_degree,
        'gradual_unfreeze': args.gradual_unfreeze,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
    }

    configs = build_training_configs(
        data_path=args.data,
        pretrained_path=args.pretrained,
        n_levels=args.n_levels,
        kernel_size=args.kernel_size,
        conv_layers=args.conv_layers,
        is_3d=args.is_3d,
        use_dynunet=args.dynunet,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stopping=args.early_stopping,
        mixed_precision=args.mixed_precision,
        scheduler=args.scheduler,
        augm_params=augm_params,
        adaptation_params=adaptation_params,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_training(
        model_config=configs['model_config'],
        dataset_config=configs['dataset_config'],
        inference_metrics=configs['inference_config'],
        train_config=configs['train_config'],
        save_path=args.output,
        device=device,
        sig_status=print,
        sig_progress=lambda p: print(f"Progress: {p}%"),
        sig_error=lambda e: print(f"Error: {e}"),
        sig_stopped=lambda: print("Training stopped"),
        sig_finished=lambda: print("Training completed"),
        sig_update_plot=None,
        _callback_log=print,
        _callback_check_stop=lambda: False,
    )


if __name__ == '__main__':
    main()
