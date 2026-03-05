from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QComboBox, QDoubleSpinBox, QSpinBox, QLabel, 
    QDialogButtonBox, QWidget, QStackedWidget, QCheckBox
)
from PyQt5.QtCore import Qt

class FineTuningDialog(QDialog):
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Fine-tuning & Adaptation Settings")
        self.resize(450, 250)

        # Default settings if none provided
        if not current_settings:
            current_settings = {
                'mode': 'scratch',
                'freeze_degree': 0.5,
                'gradual_unfreeze': False,
                'lora_rank': 8,
                'lora_alpha': 16
            }

        self.layout = QVBoxLayout(self)

        # 1. Mode Selection Header
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Train from Scratch (Standard)", "scratch")
        self.mode_combo.addItem("Classic Fine-tuning (Freezing)", "finetune")
        self.mode_combo.addItem("LoRA (Low-Rank Adaptation)", "lora")
        
        mode_header = QHBoxLayout()
        label = QLabel("Select Strategy:")
        label.setStyleSheet("font-weight: bold;")
        mode_header.addWidget(label)
        mode_header.addWidget(self.mode_combo)
        self.layout.addLayout(mode_header)

        # Separator line
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #cccccc;")
        self.layout.addWidget(line)

        # 2. Dynamic Stacked Content
        self.stacked_widget = QStackedWidget()
        
        # --- Page 0: Scratch ---
        self.page_scratch = QWidget()
        scratch_layout = QVBoxLayout(self.page_scratch)
        scratch_info = QLabel("The model will be trained from scratch using the current architecture settings.")
        scratch_info.setWordWrap(True)
        scratch_info.setAlignment(Qt.AlignCenter)
        scratch_layout.addWidget(scratch_info)
        self.stacked_widget.addWidget(self.page_scratch)

        # --- Page 1: Fine-tuning ---
        self.page_finetune = QWidget()
        finetune_layout = QFormLayout(self.page_finetune)
        finetune_layout.setContentsMargins(20, 20, 20, 20)
        
        self.freeze_spin = QDoubleSpinBox()
        self.freeze_spin.setRange(0.0, 1.0)
        self.freeze_spin.setSingleStep(0.05)
        self.freeze_spin.setValue(current_settings.get('freeze_degree', 0.5))
        self.freeze_spin.setToolTip("0.0 = all filters trainable, 1.0 = all layers frozen")
        
        finetune_layout.addRow("Freeze Layers degree (0-1):", self.freeze_spin)

        self.gradual_unfreeze_check = QCheckBox()
        self.gradual_unfreeze_check.setChecked(current_settings.get('gradual_unfreeze', False))
        self.gradual_unfreeze_check.setToolTip("Unfreeze layers progressively during training")
        finetune_layout.addRow("Gradual Unfreezing:", self.gradual_unfreeze_check)

        self.stacked_widget.addWidget(self.page_finetune)

        # --- Page 2: LoRA ---
        self.page_lora = QWidget()
        lora_layout = QFormLayout(self.page_lora)
        lora_layout.setContentsMargins(20, 10, 20, 10)
        
        self.rank_spin = QSpinBox()
        self.rank_spin.setRange(1, 256)
        self.rank_spin.setValue(current_settings.get('lora_rank', 8))
        
        self.alpha_spin = QSpinBox()
        self.alpha_spin.setRange(1, 512)
        self.alpha_spin.setValue(current_settings.get('lora_alpha', 16))
        
        lora_layout.addRow("LoRA Rank (r):", self.rank_spin)
        lora_layout.addRow("LoRA Alpha:", self.alpha_spin)
        
        lora_info = QLabel("<i>Note: LoRA will be applied to all compatible Conv and Linear layers.</i>")
        lora_info.setStyleSheet("font-size: 10px; color: #666666;")
        lora_layout.addRow("", lora_info)
        
        self.stacked_widget.addWidget(self.page_lora)

        self.layout.addWidget(self.stacked_widget)

        # 3. Dialog Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

        # Connections
        self.mode_combo.currentIndexChanged.connect(self.stacked_widget.setCurrentIndex)

        # Sync UI with current_settings
        mode = current_settings.get('mode', 'scratch')
        index = self.mode_combo.findData(mode)
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)
            self.stacked_widget.setCurrentIndex(index)

    def get_settings(self):
        """Returns a dictionary with all the settings from the dialog."""
        return {
            'mode': self.mode_combo.currentData(),
            'freeze_degree': self.freeze_spin.value(),
            'gradual_unfreeze': self.gradual_unfreeze_check.isChecked(),
            'lora_rank': self.rank_spin.value(),
            'lora_alpha': self.alpha_spin.value()
        }
