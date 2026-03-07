import torch
from .events import EngineEvents
from .trainer_engine import Engine
from .tasks.supervised_task import SupervisedModelTask
from .callbacks.callbacks import (
    MetricsCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    VisualizationCallback,
    GradualUnfreezeCallback,
    ClearGPUMemory
)
from .callbacks.save_metrics_callbacks import (
    CSVLoggingCallback,
    TensorboardCallback)

def create_supervised_trainer(
                        model,
                        criterion,
                        optimizer,
                        device,
                        spatial_dims,
                        val_roi_size,
                        mixed_precision,
                        val_loader,
                        params,
                        labels_name,
                        save_path,
                        model_name,
                        scheduler,
                        sig_update_plot,
                        unfreeze_fn=None,
                        early_stopping:bool=True,
                        initial_freeze_degree=0.0,
                        on_log=None
                    ):

    task = SupervisedModelTask(
        model,
        criterion,
        optimizer,
        device,
        spatial_dims,
        val_roi_size,
        mixed_precision
    )

    trainer = Engine(task.train_step)
    evaluator = Engine(task.validation_step)
    
    metrics_cb = MetricsCallback(params, 
                        n_classes = len(labels_name) + 1,
                        labels_name = labels_name,
                        )
    
    vis_cb = VisualizationCallback(sig_update_plot, val_loader)
    
    checkpoint_cb = CheckpointCallback(save_path, model_name, monitor='avg_dice', on_log=on_log)
    csv_cb = CSVLoggingCallback(save_path, labels_name=labels_name)
    tb_cb = TensorboardCallback(save_path=save_path)
    grad_unfreeze = GradualUnfreezeCallback(initial_freeze_degree=initial_freeze_degree, unfreeze_fn=unfreeze_fn, task=task, on_log=on_log)
    gpu_cleanup_cb = ClearGPUMemory()

    if early_stopping:
        early_stop_cb = EarlyStoppingCallback(patience=20, monitor='avg_dice', on_log=on_log)

    def run_evaluator(engine, max_epochs: int = 1):
        evaluator.state.metadata = []
        eval_state = evaluator.run(val_loader, max_epochs=max_epochs, check_stop=engine.state.check_stop)
        if 'avg_loss' in eval_state.metrics:
            eval_state.metrics['avg_val_loss'] = eval_state.metrics.pop('avg_loss')
        engine.state.metrics.update(eval_state.metrics)
        engine.state.metadata = evaluator.state.metadata

    evaluator.add_event_handler(EngineEvents.ITERATION_COMPLETED, metrics_cb.on_iteration_completed)
    evaluator.add_event_handler(EngineEvents.ITERATION_COMPLETED, gpu_cleanup_cb.on_iteration_completed)
    evaluator.add_event_handler(EngineEvents.EPOCH_COMPLETED, metrics_cb.on_epoch_completed)
    evaluator.add_event_handler(EngineEvents.EPOCH_COMPLETED, gpu_cleanup_cb.on_epoch_completed)

    if scheduler is not None:
        def step_scheduler(engine):
            scheduler.step()
        trainer.add_event_handler(EngineEvents.EPOCH_COMPLETED, step_scheduler)

    # Gradual unfreezing — EPOCH_STARTED so unfreeze applies before training that epoch
    if unfreeze_fn is not None:
        trainer.add_event_handler(EngineEvents.EPOCH_STARTED, grad_unfreeze.on_epoch_started)

    trainer.add_event_handler(EngineEvents.EPOCH_COMPLETED, run_evaluator)
    trainer.add_event_handler(EngineEvents.EPOCH_COMPLETED, checkpoint_cb.on_epoch_completed)
    trainer.add_event_handler(EngineEvents.EPOCH_COMPLETED, csv_cb.on_epoch_completed)
    trainer.add_event_handler(EngineEvents.EPOCH_COMPLETED, vis_cb.on_epoch_completed)
    trainer.add_event_handler(EngineEvents.EPOCH_COMPLETED, tb_cb.on_epoch_completed)
    trainer.add_event_handler(EngineEvents.COMPLETED, tb_cb.on_completed)
    if early_stopping:
        trainer.add_event_handler(EngineEvents.EPOCH_COMPLETED, early_stop_cb.on_epoch_completed)

    # on_log on start
    def on_started(engine):
        if on_log: on_log(f"Engine Starting on device {device}. {engine.state.max_epochs} epochs")
    trainer.add_event_handler(EngineEvents.STARTED, on_started)

    # GPU cleanup + final log on training completed
    def on_completed(engine):
        import gc
        task.model.to('cpu')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        best = engine.state.best_metric
        if on_log: on_log(f"Training engine finished. Best Dice {best:.4f}" if best is not None else "Training engine finished.")
    trainer.add_event_handler(EngineEvents.COMPLETED, on_completed)

    return trainer

