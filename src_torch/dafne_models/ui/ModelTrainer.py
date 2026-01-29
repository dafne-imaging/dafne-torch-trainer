import os
import sys
import glob

from ..core.training_worker import TrainingWorker
from .ModelTrainer_Ui import Ui_ModelTrainerUI

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QObject, QVariant
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QSpinBox, QDoubleSpinBox, 
    QTextEdit, QMessageBox, QGroupBox, QFormLayout
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter

import numpy as np

PATIENCE = 10
MIN_EPOCHS = 20

class ModelTrainer(QWidget, Ui_ModelTrainerUI):

    def __init__(self, parent=None):
        super(ModelTrainer, self).__init__(parent)
        self.setupUi(self) #load widget in UI system

        self.setWindowTitle('Dafne Model Trainer (PyTorch Backend)')
        
        self.worker = None #training thread
        self.image_paths = [] #path to images
        self.mask_paths = [] #path to masks
        self.train_params = {} #training params

        # loss caches
        self.loss_history = []

        self.fit_output_box.setVisible(False)
        self.advanced_widget.setVisible(False)
        self.fit_Button.setEnabled(False)
        self.preprocess_Button.setEnabled(False)

        self._init_matplotlib_canvas()

        self.choose_Button.clicked.connect(self.select_data)
        self.advanced_button.clicked.connect(self.toggle_adavanced_options)

        self.fit_Button.clicked.connect(self.start_training)
    
    def _init_matplotlib_canvas(self):      
        self.pyplot_layout = QVBoxLayout(self.fit_output_box)
        self.fig = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        self.pyplot_layout.addWidget(self.canvas)

        self.ax_loss = self.fig.add_subplot(121)
        self.ax_loss.set_title("Loss")
        self.ax_loss.grid(True, linestyle='--', alpha=0.6)

        self.ax_preview = self.fig.add_subplot(122)
        self.ax_preview.set_title("Live Preview")
        self.ax_preview.axis('off')

        self.canvas.draw()
    
    def toggle_advanced_option(self):
        is_visible = self.advanced_widget.isVisible()
        self.advanced_widget.setVisible(not is_visible)
     
    def select_data(self):      
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select folder data (Images/Masks or NPZ)"
        )
        if not folder_path:
            return
        
        self.location_Text.setText(folder_path)
        extensions = ['.npz', '.nii', '.nii.gz', '.dcm']
        found_files = self._scan_directory(folder_path, extensions)

        if not found_files:
            QMessageBox.warning(self, "No data found", f'Folder selected does not contain valid extension {extensions}')
            self.image_paths = []
            self.fit_Button.setEnabled(False)
            return

        self.image_paths = found_files
        self.mask_paths = None

        self.fit_Button.setEnabled(True)

        QMessageBox.information(self, "Data loaded successfully", f"Founded {len(found_files)}")

    def _scan_directory(self, folder_path, extensions):
        found_files = []
        try:
            for file in os.listdir(folder_path):
                if any(file.endswith(ext) for ext in extensions):
                    full_path = os.path.join(folder_path, file)
                    found_files.append(full_path)
            found_files.sort()
            return found_files
            
        except Exception as e:
            print(f"Error scanning directory {folder_path}: {e}")
            return []
    
    def start_training(self):
        if not self.image_paths:
            QMessageBox.warning(self, 'Input Error', "No data loaded. Please select data first")
            return

        self.fit_Button.setEnabled(False)
        self.choose_Button.setEnabled(False)
        self.fit_output_box.setVisible(True)

        self.loss_history = []
        self.ax_loss.clear()
        self.ax_preview.clear()
        self.progressBar.setValue(0)
        self.progress_Label.setText("Inizialization...")

        n_levels = self.levels_spin.value()
        kernel_size = self.kernsize_spin.value()
        # conv_layers = self.convlayers_spin.value()

        model_params = {
            'spatial_dims': 2,
            'n_levels': n_levels,
            'kernel_size': kernel_size,
            'n_classes': 2,
            'in_channels': 1
        }

        # default values: to be change by the user
        train_params = {
            'epochs': 50,
            'learning_rate': 1e-3,
            'batch_size': 2
        }

        self.worker = TrainingWorker(
            file_list=self.image_paths,
            mask_list=self.mask_paths,
            model_params=model_params,
            train_params=train_params
        )

        self.worker.sig_update_plot.connect(self.update_plots)
        self.worker.sig_status.connect(self.update_status_label)
        self.worker.sig_error.connect(self.handle_error)
        self.worker.sig_finished.connect(self.on_training_finished)

        self.worker.start()
    
    def update_status_label(self, message): 
        self.progress_Label.setText(message)
        if self.progressBar.value() < 99:
            self.progressBar.setValue(self.progressBar.value() + 1)
        else: 
            self.progressBar.setValue(0)
    
    @QtCore.pyqtSlot(float, object, object)
    def update_plots(self, loss, img, mask):
        self.loss_history.append(loss)
        self.ax_loss.clear()
        self.ax_loss.plot(self.loss_history, 'r-', label='Training Loss')
        self.ax_loss.set_title(f'Loss: {loss:.4f}')
        self.ax_loss.grid(True, alpha=0.5)

        self.ax_preview.clear()
        
        if img.ndim == 3: img = img[0, :, :] # only the first channel

        self.ax_preview.imshow(img, cmap='gray')

        if mask is not None: 
            masked = np.ma.masked_where(mask == 0, mask)
            self.ax_preview.imshow(masked, cmap='autumn', alpha=0.5)
        
        self.ax_preview.axis('off')
        self.canvas.draw()

        QtWidgets.QApplication.processEvents()
    
    def handle_error(self, err):
        QMessageBox.critical(self, "Error", f"Training is stopping:\n{err}")
        self._reset_ui_state()
    
    def on_training_finished(self):
        self.progressBar.setValue(100)
        self.progress_Label.setText('Training completed!')
        QMessageBox.information(self, 'Finished!', "The model was trained successfully")

    def _reset_ui_state(self):
        self.fit_Button.setEnabled(True)
        self.choose_Button.setEnabled(True)
    
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
    print("GUI avviata in modalità Test.")
    sys.exit(app.exec_())
