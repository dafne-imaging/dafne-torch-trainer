from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QDoubleSpinBox, QSpinBox, QLabel,
    QDialogButtonBox, QWidget, QStackedWidget, QCheckBox,
    QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class FineTuningDialog(QDialog):
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Training Strategy")
        self.setFixedSize(420, 240)

        if not current_settings:
            current_settings = {
                'mode': 'scratch',
                'freeze_degree': 0.5,
                'gradual_unfreeze': False,
                'lora_rank': 8,
                'lora_alpha': 16
            }

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 14, 16, 12)

        # ── Strategy selector ──────────────────────────────────────────
        selector_row = QHBoxLayout()
        selector_row.setSpacing(10)

        lbl = QLabel("Strategy:")
        lbl.setFixedWidth(62)
        f = QFont()
        f.setBold(True)
        lbl.setFont(f)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Train from scratch", "scratch")
        self.mode_combo.addItem("Classic fine-tuning (layer freezing)", "finetune")
        self.mode_combo.addItem("LoRA (low-rank adaptation)", "lora")

        selector_row.addWidget(lbl)
        selector_row.addWidget(self.mode_combo)
        root.addLayout(selector_row)

        # ── Divider ───────────────────────────────────────────────────
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        root.addWidget(divider)

        # ── Stacked content ───────────────────────────────────────────
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setFixedHeight(110)

        # Page 0: Scratch
        page_scratch = QWidget()
        sl = QVBoxLayout(page_scratch)
        sl.setContentsMargins(8, 14, 8, 4)
        sl.setAlignment(Qt.AlignCenter)
        info = QLabel(
            "The model will be trained from scratch\n"
            "using the current architecture settings."
        )
        info.setAlignment(Qt.AlignCenter)
        sl.addWidget(info)
        self.stacked_widget.addWidget(page_scratch)

        # Page 1: Fine-tuning
        page_finetune = QWidget()
        fl = QFormLayout(page_finetune)
        fl.setContentsMargins(8, 10, 8, 4)
        fl.setSpacing(10)
        fl.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.freeze_spin = QDoubleSpinBox()
        self.freeze_spin.setRange(0.0, 1.0)
        self.freeze_spin.setSingleStep(0.1)
        self.freeze_spin.setDecimals(2)
        self.freeze_spin.setValue(current_settings.get('freeze_degree', 0.5))
        self.freeze_spin.setToolTip("0 = all layers trainable · 1 = all layers frozen")
        self.freeze_spin.setFixedWidth(80)

        self.gradual_unfreeze_check = QCheckBox("Enable gradual unfreezing")
        self.gradual_unfreeze_check.setChecked(current_settings.get('gradual_unfreeze', False))
        self.gradual_unfreeze_check.setToolTip("Progressively unfreeze layers during training")

        fl.addRow("Freeze degree (0–1):", self.freeze_spin)
        fl.addRow("", self.gradual_unfreeze_check)
        self.stacked_widget.addWidget(page_finetune)

        # Page 2: LoRA
        page_lora = QWidget()
        ll = QFormLayout(page_lora)
        ll.setContentsMargins(8, 10, 8, 4)
        ll.setSpacing(10)
        ll.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.rank_spin = QSpinBox()
        self.rank_spin.setRange(1, 256)
        self.rank_spin.setValue(current_settings.get('lora_rank', 8))
        self.rank_spin.setFixedWidth(70)

        self.alpha_spin = QSpinBox()
        self.alpha_spin.setRange(1, 512)
        self.alpha_spin.setValue(current_settings.get('lora_alpha', 16))
        self.alpha_spin.setFixedWidth(70)

        lora_note = QLabel("Applied to all compatible Conv and Linear layers.")
        lora_note.setWordWrap(True)

        ll.addRow("Rank (r):", self.rank_spin)
        ll.addRow("Alpha:", self.alpha_spin)
        ll.addRow("", lora_note)
        self.stacked_widget.addWidget(page_lora)

        root.addWidget(self.stacked_widget)

        # ── OK / Cancel ───────────────────────────────────────────────
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        # ── Sync with current settings ────────────────────────────────
        self.mode_combo.currentIndexChanged.connect(self.stacked_widget.setCurrentIndex)
        mode = current_settings.get('mode', 'scratch')
        idx = self.mode_combo.findData(mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
            self.stacked_widget.setCurrentIndex(idx)

    def get_settings(self):
        return {
            'mode': self.mode_combo.currentData(),
            'freeze_degree': self.freeze_spin.value(),
            'gradual_unfreeze': self.gradual_unfreeze_check.isChecked(),
            'lora_rank': self.rank_spin.value(),
            'lora_alpha': self.alpha_spin.value()
        }
