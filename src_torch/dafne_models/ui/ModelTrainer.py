import os
import sys

from ..core.training_worker import TrainingWorker
from .ModelTrainer_Ui import Ui_ModelTrainerUI
from .AugmentationDialog_Ui import AugmentationDialog
from .FineTuningDialog import FineTuningDialog

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFileDialog, 
    QMessageBox, QDialog
)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ..config.config_params import ModelConfig, \
                                        DatasetConfig, \
                                        TrainingConfig, \
                                        AugmentationConfig, \
                                        LoraConfig, \
                                        InferenceMetricsConfig

import numpy as np

PATIENCE = 10
MIN_EPOCHS = 20

class ModelTrainer(QWidget, Ui_ModelTrainerUI):

    def __init__(self, parent=None):
        super(ModelTrainer, self).__init__(parent)
        self.setupUi(self) #load widget in UI system
        monitor_dim = QtWidgets.QDesktopWidget().availableGeometry()
        width = monitor_dim.width() * 0.7
        height = monitor_dim.height() * 0.7
        self.resize(int(width), int(height))
        x = (monitor_dim.width() - width) // 2
        y = (monitor_dim.height() - height) // 2
        self.move(int(x), int(y))
        self.setWindowTitle('Dafne Model Trainer (PyTorch Backend)')
        
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)

        self.save_path = None
        self.worker = None #training thread
        self.pretrained_path = None

        # augmentation params
        self.augm_params = {'rotate':False,
                            'flip_x':False,
                            'flip_y':False,
                            'zoom':False,
                            'noise':False}
        
        # fine-tuning and adaptation params
        self.adaptation_params = {
            'mode': 'scratch',
            'freeze_degree': 0.5,
            'lora_rank': 8,
            'lora_alpha': 16
        }

        # loss caches
        self.loss_history = []

        self.fit_output_box.setVisible(False)
        self.advanced_widget.setVisible(False)
        self.fit_Button.setEnabled(False)
        self.save_choose_Button.setEnabled(True)
        self.preprocess_Button.setEnabled(False)
        self.train_settings_widget.setVisible(False)
        
        # Set path fields as permanently read-only and disabled
        self.location_Text.setReadOnly(True)
        self.location_Text.setEnabled(False)
        self.model_location_Text.setReadOnly(True)
        self.model_location_Text.setEnabled(False)
        self.pretrained_location_Text.setReadOnly(True)
        self.pretrained_location_Text.setEnabled(False)

        self._init_matplotlib_canvas()

        self.choose_Button.clicked.connect(self.select_data)
        self.advanced_button.clicked.connect(self.toggle_advanced_options)
        self.train_settings_btn.clicked.connect(self.toggle_advanced_options_training)
        self.save_choose_Button.clicked.connect(self.select_save_path)
        self.augmentation_button.clicked.connect(self.open_augm_settings)
        self.finetuning_settings_btn.clicked.connect(self.open_finetuning_settings)
        self.stop_btn.clicked.connect(self.stop_training)

        self.pretrain_choose_Button.clicked.connect(self.select_pretrained_path)
        self._update_adaptation_ui_state()

        self.model_3d_check.toggled.connect(self._on_3d_mode_changed)
        self._on_3d_mode_changed(self.model_3d_check.isChecked())

        self.check_auto_params.toggled.connect(self._on_dynunet_changed)
        self._on_dynunet_changed(self.check_auto_params.isChecked())

        self.fit_Button.clicked.connect(self.start_training)
    
    def _init_matplotlib_canvas(self):      
        self.pyplot_layout = QVBoxLayout(self.fit_output_box)
        self.fig = plt.figure(figsize=(2, 2)) 
        
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Ignored, 
                                  QtWidgets.QSizePolicy.Ignored)
        self.canvas.setMinimumSize(10, 10) 
        
        self.canvas.updateGeometry()

        self.pyplot_layout.addWidget(self.canvas)

        self.ax_loss = self.fig.add_subplot(121)
        self.ax_loss.set_title("Loss")
        self.ax_loss.grid(True, linestyle='--', alpha=0.6)

        self.ax_preview = self.fig.add_subplot(122)
        self.ax_preview.set_title("Live Preview")
        self.ax_preview.axis('off')
    
    def open_augm_settings(self):
        dialog = AugmentationDialog(self, self.augm_params)
        if dialog.exec_() == QDialog.Accepted:
            self.augm_params = dialog.get_settings()
            print(f"Augmentation updated: {self.augm_params}")
    
    def open_finetuning_settings(self):
        dialog = FineTuningDialog(self, self.adaptation_params)
        if dialog.exec_() == QDialog.Accepted:
            self.adaptation_params = dialog.get_settings()
            print(f"Adaptation updated: {self.adaptation_params}")
            self._update_adaptation_ui_state()
    
    def _update_adaptation_ui_state(self):
        mode = self.adaptation_params.get('mode', 'scratch')
        is_pretrained_mode = (mode != 'scratch')

        # The text field remains disabled (read-only), only the button is toggled
        self.pretrain_choose_Button.setEnabled(is_pretrained_mode)
        
        # update the actual path from the text field if mode is scratch
        if not is_pretrained_mode:
            self.pretrained_path = None
            self.pretrained_location_Text.clear()

        # Change button text or color to show it's configured
        if mode == 'lora':
            self.finetuning_settings_btn.setText("Fine-tuning (LoRA)")
            self.lr_spin.setValue(1e-4)
        elif mode == 'finetune':
            self.finetuning_settings_btn.setText("Fine-tuning (Classic)")
            self.lr_spin.setValue(1e-5)
        else:
            self.finetuning_settings_btn.setText("Fine-tuning Settings")
            self.lr_spin.setValue(1e-3)
        
        self._update_advanced_ui_state()
    
    def _update_advanced_ui_state(self):
        mode = self.adaptation_params.get('mode', 'scratch')
        if mode == 'finetune' or mode == 'lora':
            self.advanced_widget.setEnabled(False)
            self.advanced_button.setEnabled(False)
            self.check_auto_params.setEnabled(False)
            self.model_3d_check.setEnabled(False)
        else:
            self.advanced_widget.setEnabled(True)
            self.advanced_button.setEnabled(True)
            self.check_auto_params.setEnabled(True)
            self.model_3d_check.setEnabled(True)
        
        self.train_settings_btn.setEnabled(True)

    def select_pretrained_path(self) -> None:
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Choose pretrained dafne model",
            "",
            "Dafne Model (*.dafne);;All files (*)",
            options=options
        )

        if filename:
            self.pretrained_path = filename
            self.pretrained_location_Text.setText(filename)

    def _on_3d_mode_changed(self, is_3d_active):
        self.check_auto_params.setEnabled(is_3d_active)

        if not is_3d_active:
            self.check_auto_params.setChecked(False)

    def _on_dynunet_changed(self, is_dynunet_active):
        self.convlayers_spin.setEnabled(not is_dynunet_active)
        self.kernsize_spin.setEnabled(not is_dynunet_active)
        self.levels_spin.setEnabled(not is_dynunet_active)
        self.batch_spin.setEnabled(not is_dynunet_active)

        if not is_dynunet_active:
            self.convlayers_spin.setEnabled(True)
            self.kernsize_spin.setEnabled(True)
            self.levels_spin.setEnabled(True)
            self.batch_spin.setEnabled(True)
    
    def select_save_path(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Choose save model path",
            "",
            "Dafne Model (*.dafne);;Pytorch Model (*.pth);;All files (*)",
            options=options
        )

        if filename:
            # Add .dafne extension if no extension is provided
            if not '.' in os.path.basename(filename):
                filename += '.dafne'
        
        self.save_path = filename
        self.model_location_Text.setText(filename)
     
    def select_data(self):      
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select folder data (Images/Masks or NPZ)"
        )
        if not folder_path:
            return
        
        has_data = any(f.endswith('.npz') for root, _, files in os.walk(folder_path) for f in files)

        if not has_data:
            QMessageBox.warning(self, "No data found", "Folder selected does not contain valid .npz files!")
            self.fit_Button.setEnabled(False)
            return

        self.location_Text.setText(folder_path)
        self.fit_Button.setEnabled(True)
        QMessageBox.information(self, "Data Selected", "Dataset folder linked successfully. Ready to train.")


    def update_progress_bar(self, percent):
        self.progressBar.setValue(percent)
    
    def start_training(self):
        #check if data are loaded
        if not self.location_Text.text():
            QMessageBox.warning(self, 'Input Error', "No data loaded. Please select data first")
            return

        # Update paths from text fields in case user edited them manually
        self.save_path = self.model_location_Text.text()
        self.pretrained_path = self.pretrained_location_Text.text()

        # check if save path is provided
        if not self.save_path:
            QMessageBox.warning(self, 'Input Error', "Please select a save path for the model")
            return

        #check if pretrained model is selected for finetuning or lora
        if (self.adaptation_params.get('mode', 'scratch') == 'finetune' \
            or self.adaptation_params.get('mode', 'scratch') == 'lora') \
                and not self.pretrained_path:
            QMessageBox.warning(self, 'Input Error', "No pretrained model selected. Please select a \
                pretrained model")
            return

        self._status_btn(True)
        self.fit_output_box.setVisible(True)

        self.loss_history = []
        self.val_loss_history = []
        self.ax_loss.clear()
        self.ax_preview.clear()
        
        self.ax_loss.set_title("Loss")
        self.ax_loss.grid(True, linestyle='--', alpha=0.6)
        self.ax_preview.axis('off')
        self.canvas.draw()

        self.progressBar.setValue(0)
        self.progress_Label.setText("Inizialization...")

        n_levels = self.levels_spin.value()
        kernel_size = self.kernsize_spin.value()
        # conv_layers = self.convlayers_spin.value()

        batch_size = self.batch_spin.value()
        lr = self.lr_spin.value()
        epochs = self.epochs_spin.value()

        early_stopping = self.early_stopping_check.isChecked()
        spatial_dims = self.model_3d_check.isChecked()

        dyn_unet = self.check_auto_params.isChecked()

        dataset_config = DatasetConfig(
            root_dir=self.location_Text.text(),
            val_split=0.2,
            random_seed=42
        )

        self.model_config = ModelConfig(
            model_name='dynunet' if dyn_unet else 'unet',
            spatial_dims=3 if spatial_dims else 2,
            out_channels=2,
            in_channels=1,
            use_dynamic=dyn_unet,
            extra_params={
                'n_levels': n_levels,
                'kernel_size': kernel_size,
                'kernels': None,
                'strides': None
            }
        )

        augm_config = AugmentationConfig(
            rotate=self.augm_params['rotate'],
            flip_x=self.augm_params['flip_x'],
            flip_y=self.augm_params['flip_y'],
            zoom=self.augm_params['zoom'],
            noise=self.augm_params['noise'],
        )

        train_config = TrainingConfig(
            epochs=epochs,
            learning_rate=lr,
            batch_size=batch_size,
            augmentation=augm_config,
            pretrained_model_path=self.pretrained_path,
            mixed_precision=self.mixed_precision_check.isChecked(),
            early_stopping=early_stopping,
            scheduler=self.scheduler_check.isChecked()
        )

        inference_config = InferenceMetricsConfig(
            include_background=False,
            reduction='mean_batch'
        )


        # Add adaptation parameters
        mode = self.adaptation_params.get('mode', 'scratch')
        if mode == 'finetune':
            self.model_config.fine_tuning = True
            self.model_config.percent_to_freeze = self.adaptation_params.get('freeze_degree', 0.5)
        elif mode == 'lora':
            self.model_config.fine_tuning = True
            self.model_config.percent_to_freeze = None
            # Default to all layers for now
            self.model_config.lora_config = LoraConfig(
                r=self.adaptation_params.get('lora_rank', 8),
                lora_alpha=self.adaptation_params.get('lora_alpha', 16),
                lora_dropout=0.1,
                rank_for='channels',
                target_modules=['.*']
            )
        else: # scratch
            self.model_config.fine_tuning = False
            self.model_config.percent_to_freeze = None
            self.model_config.lora_config = None

        self.worker = TrainingWorker(
            dataset_config = dataset_config,
            model_config = self.model_config,
            train_config = train_config,
            inference_metrics = inference_config,
            save_path = self.save_path,
        )

        self.worker.sig_update_plot.connect(self.update_plots)
        self.worker.sig_status.connect(self.update_status_label)
        self.worker.sig_progress.connect(self.update_progress_bar)
        self.worker.sig_error.connect(self.handle_error)
        self.worker.sig_finished.connect(self.on_training_finished)
        self.worker.sig_stopped.connect(self.on_training_stopped)

        self.worker.start()
    
    def stop_training(self):
        if self.worker is not None and self.worker.isRunning():
            self.stop_btn.setEnabled(False)
            self.update_status_label("Stopping training. Please wait...")
            self.worker.stop()

    def _status_btn(self, active:bool):
        widgets_to_toggle = [
            self.choose_Button, self.save_choose_Button, 
            self.advanced_button, self.train_settings_btn, 
            self.augmentation_button, self.fit_Button, 
            self.preprocess_Button, self.levels_spin,
            self.epochs_spin, self.lr_spin, self.batch_spin,
            self.finetuning_settings_btn, self.pretrain_choose_Button
        ]

        for btn in widgets_to_toggle:
            btn.setEnabled(not active)
        
        # If we are stopping training, we need to respect the adaptation mode for pretrained widgets
        if not active:
            self._update_adaptation_ui_state()

        self.stop_btn.setEnabled(active)
    
    def update_status_label(self, message): 
        self.progress_Label.setText(message)

    def restore_image_contrast(self, img):
        # Normalize non-zero pixels to [0, 1] to restore the visual color range
        img_vis = img.copy()
        mask_nonzero = (img_vis != 0)
        if np.any(mask_nonzero):
            min_val = img_vis[mask_nonzero].min()
            max_val = img_vis[mask_nonzero].max()
            img_vis[mask_nonzero] = (img_vis[mask_nonzero] - min_val) / (max_val - min_val + 1e-8)
        return img_vis
    
    @QtCore.pyqtSlot(float, object, object, float, object, float, dict)
    def update_plots(self, loss, img, mask, val_loss, spacing, best_dice, per_mask_dice):
        self.loss_history.append(loss)
        self.val_loss_history.append(val_loss)
        self.ax_loss.clear()
        self.ax_loss.plot(self.loss_history, 'r-', label='Training Loss')
        self.ax_loss.plot(self.val_loss_history, 'b-', label='Validation Loss')
        self.ax_loss.legend(loc='upper right')
        self.ax_loss.set_title(f'Loss: {loss:.4f} | Best Dice: {best_dice:.4f}')
        self.ax_loss.grid(True, alpha=0.5)

        self.ax_preview.clear()
        
        if img.ndim == 3: img = img[0, :, :] # only the first channel
        img = self.restore_image_contrast(img)
        
        pixel_aspect = 1.0 # isotropic 
        if spacing is not None and len(spacing) >= 2:
            try:
                sp_y = spacing[-2] 
                sp_x = spacing[-1]
                pixel_aspect = float(sp_x / sp_y)
            except Exception as e:
                print(f"Error pixel aspect calcultation: {e}")
                pixel_aspect = 1.0

        self.ax_preview.imshow(img, cmap='gray', aspect=pixel_aspect)

        if mask is not None: 
            masked = np.ma.masked_where(mask == 0, mask)
            vmin = 0
            vmax = self.model_config.out_channels
            im = self.ax_preview.imshow(masked, cmap='tab20', alpha=0.5, aspect=pixel_aspect, vmin=vmin, vmax=vmax)
            if per_mask_dice:
                legend_elements = []
                cmap = plt.get_cmap('tab20')
                for i, (name, score) in enumerate(per_mask_dice.items()):
                    color_idx = (i + 1) / vmax if vmax > 0 else 0
                    color = cmap(color_idx)
                    legend_elements.append(Patch(facecolor=color, 
                                                label=f"{name}: {score:.3f}", 
                                                alpha=0.6))
                self.ax_preview.legend(handles=legend_elements, 
                                     loc='upper left', 
                                     bbox_to_anchor=(1.05, 1), 
                                     fontsize='small',
                                     title="Dice per Organ")
        self.ax_preview.axis('off')
        self.canvas.draw()

        QtWidgets.QApplication.processEvents()
    
    def handle_error(self, err):
        self._status_btn(False)
        QMessageBox.critical(self, "Error", f"Training is stopping:\n{err}")

        self.progressBar.setValue(0)
        self.progress_Label.setText("Error occurred.")
        self.fit_output_box.setVisible(False)

        self._reset_ui_state()
    
    def on_training_stopped(self):
        self._status_btn(False)
        self.progressBar.setValue(0)
        self.progress_Label.setText("")
        
        self.fit_output_box.setVisible(False)
        self.loss_history = []
        self.val_loss_history = []
        self.ax_loss.clear()
        self.ax_preview.clear()

        # Clear paths and data on stop
        self._clear_paths_and_data()

        QMessageBox.information(self, "Stopped", "Training stopped correctly.")

    def _clear_paths_and_data(self):
        """Reset all paths, internal data lists and UI mode switches"""
        self.save_path = None
        self.pretrained_path = None
        
        self.location_Text.clear()
        self.model_location_Text.clear()
        self.pretrained_location_Text.clear()
        
        # Ensure they stay disabled
        self.location_Text.setEnabled(False)
        self.model_location_Text.setEnabled(False)
        self.pretrained_location_Text.setEnabled(False)

        # Reset mode switches to avoid "inheritance" from previous run
        self.model_3d_check.setChecked(False)
        self.check_auto_params.setChecked(False)
        
        self.fit_Button.setEnabled(False)
        
    def on_training_finished(self):
        self._status_btn(False)
        self.progressBar.setValue(100)
        self.progress_Label.setText('Training completed!')
        
        # Clear paths and data on finish
        self._clear_paths_and_data()
        
        QMessageBox.information(self, 'Finished!', "The model was trained successfully")
    
    def _reset_ui_state(self):
        self.fit_Button.setEnabled(True)
        self.choose_Button.setEnabled(True)
    
    def toggle_advanced_options_training(self):
        is_visible = self.train_settings_widget.isVisible()
        new_state = not is_visible
        self.train_settings_widget.setVisible(new_state)

        if new_state:
            self.train_settings_btn.setText("Hide Training Settings")
        else: 
            self.train_settings_btn.setText("Show Training Settings")
    
    def toggle_advanced_options(self):
        is_visible = self.advanced_widget.isVisible()
        new_state = not is_visible
        self.advanced_widget.setVisible(new_state)

        if new_state:
            self.advanced_button.setText("Hide Advanced Settings")
        else:
            self.advanced_button.setText('Show Advanced Settings')

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ModelTrainer()
    window.show()
    sys.exit(app.exec_())
