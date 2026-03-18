import os
import sys

from ..core.training_worker_engine import TrainingWorker
#from ..core.training_worker import TrainingWorker
from .AugmentationDialog_Ui import AugmentationDialog
from .FineTuningDialog import FineTuningDialog
from .training_controller import build_training_configs

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QScrollArea, QGroupBox, QLabel, QLineEdit,
    QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QSpacerItem, QSizePolicy,
    QFileDialog, QMessageBox, QDialog, QStyleFactory
)
from PyQt5.QtGui import QPalette, QColor, QIcon, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import numpy as np


def _apply_fusion_style(app):
    """Force Fusion style + explicit light palette for cross-platform consistency."""
    app.setStyle(QStyleFactory.create('Fusion'))
    p = QPalette()
    p.setColor(QPalette.Window,          QColor('#f0f0f0'))
    p.setColor(QPalette.WindowText,      QColor('#1a1a1a'))
    p.setColor(QPalette.Base,            QColor('#ffffff'))
    p.setColor(QPalette.AlternateBase,   QColor('#f5f5f5'))
    p.setColor(QPalette.Text,            QColor('#1a1a1a'))
    p.setColor(QPalette.Button,          QColor('#e0e0e0'))
    p.setColor(QPalette.ButtonText,      QColor('#1a1a1a'))
    p.setColor(QPalette.Highlight,       QColor('#4a90d9'))
    p.setColor(QPalette.HighlightedText, QColor('#ffffff'))
    p.setColor(QPalette.ToolTipBase,     QColor('#ffffdc'))
    p.setColor(QPalette.ToolTipText,     QColor('#1a1a1a'))
    p.setColor(QPalette.Link,            QColor('#2a70b9'))
    p.setColor(QPalette.Disabled, QPalette.WindowText, QColor('#909090'))
    p.setColor(QPalette.Disabled, QPalette.Text,       QColor('#909090'))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor('#909090'))
    p.setColor(QPalette.Disabled, QPalette.Base,       QColor('#e8e8e8'))
    app.setPalette(p)

PATIENCE = 10
MIN_EPOCHS = 20

# ── Canvas colour palette ──────────────────────────────────────────────────
_FIG_BG      = '#f0f0f0'   # matches Qt Fusion Window colour
_AX_BG       = '#fafafa'
_TRAIN_COLOR = '#1565C0'   # deep blue
_VAL_COLOR   = '#E65100'   # deep orange
_GRID_COLOR  = '#cccccc'
_SPINE_COLOR = '#cccccc'
_TICK_COLOR  = '#555555'

_ICON_PATH = os.path.join(os.path.dirname(__file__), '..', 'icons', 'icon_dafne_trainer.png')


def _make_app_icon() -> QIcon:
    """Build a multi-resolution QIcon so the OS picks the sharpest size."""
    icon = QIcon()
    src = QPixmap(_ICON_PATH)
    if not src.isNull():
        for size in (16, 24, 32, 48, 64, 128, 256, 512):
            icon.addPixmap(
                src.scaled(size, size,
                           QtCore.Qt.KeepAspectRatio,
                           QtCore.Qt.SmoothTransformation)
            )
    return icon


class ModelTrainerSplit(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        monitor_dim = QtWidgets.QDesktopWidget().availableGeometry()
        width = monitor_dim.width() * 0.70
        height = monitor_dim.height() * 0.78
        self.resize(int(width), int(height))
        self.setMinimumSize(780, 480)
        x = (monitor_dim.width() - width) // 2
        y = (monitor_dim.height() - height) // 2
        self.move(int(x), int(y))
        self.setWindowTitle('Dafne Model Trainer')
        self.setWindowIcon(_make_app_icon())

        self.save_path = None
        self.worker = None
        self.pretrained_path = None

        self.augm_params = {
            'rotate': False, 'flip_x': False, 'flip_y': False,
            'zoom': False, 'noise': False
        }
        self.adaptation_params = {
            'mode': 'scratch', 'freeze_degree': 0.5,
            'gradual_unfreeze': False,
            'lora_rank': 8, 'lora_alpha': 16
        }
        self.loss_history = []
        self.val_loss_history = []

        self._build_ui()
        self._connect_signals()
        self._init_matplotlib_canvas()

        # Initial UI state
        self._update_adaptation_ui_state()
        self._on_3d_mode_changed(self.model_3d_check.isChecked())
        self._on_dynunet_changed(self.check_auto_params.isChecked())

    # ─────────────────────────────────────────────
    # UI CONSTRUCTION
    # ─────────────────────────────────────────────

    def _build_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(0)

        # ── App header ──────────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(40)
        header.setStyleSheet(
            "background-color: #e8e8e8; border-bottom: 1px solid #ccc;"
        )
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(8, 0, 8, 0)
        h_layout.setSpacing(8)

        icon_lbl = QLabel()
        src_px = QPixmap(_ICON_PATH)
        if not src_px.isNull():
            icon_lbl.setPixmap(
                src_px.scaled(26, 26,
                              QtCore.Qt.KeepAspectRatio,
                              QtCore.Qt.SmoothTransformation)
            )
        h_layout.addWidget(icon_lbl)

        title_lbl = QLabel(
            "Dafne Model Trainer"
            "  <span style='color:#777; font-size:9pt;'>— PyTorch backend</span>"
        )
        title_lbl.setTextFormat(QtCore.Qt.RichText)
        title_lbl.setStyleSheet("font-size: 11pt; font-weight: bold; color: #1a1a1a;")
        h_layout.addWidget(title_lbl)
        h_layout.addStretch()

        root_layout.addWidget(header)
        root_layout.addSpacing(4)
        # ────────────────────────────────────────────────────────────────────

        self.splitter = QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        root_layout.addWidget(self.splitter)

        self.splitter.addWidget(self._build_left_panel())
        self.splitter.addWidget(self._build_right_panel())

        # Left panel gets ~380px; right panel takes the rest
        left_w = 380
        self.splitter.setSizes([left_w, max(400, self.width() - left_w)])
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

    def _build_left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(480)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)
        layout.setContentsMargins(8, 10, 8, 10)

        # ── Paths ──
        layout.addLayout(self._path_row("Data folder:", 'location_Text', 'choose_Button'))
        layout.addLayout(self._path_row("Output model:", 'model_location_Text', 'save_choose_Button'))
        layout.addLayout(self._path_row("Pretrained model:", 'pretrained_location_Text', 'pretrain_choose_Button'))

        layout.addSpacing(6)

        # ── Collapsible sections ──
        layout.addWidget(self._advanced_section())
        layout.addWidget(self._training_settings_section())

        layout.addSpacing(4)

        # ── Augmentation + Fine-tuning ──
        self.augmentation_button = QPushButton("Augmentation settings...")
        self.augmentation_button.setMinimumHeight(30)
        layout.addWidget(self.augmentation_button)

        self.finetuning_settings_btn = QPushButton("Fine-tuning settings...")
        self.finetuning_settings_btn.setMinimumHeight(30)
        layout.addWidget(self.finetuning_settings_btn)

        layout.addSpacing(8)

        # ── Progress ──
        self.progressBar = QProgressBar()
        self.progressBar.setValue(0)
        self.progressBar.setMinimumHeight(18)
        layout.addWidget(self.progressBar)

        self.progress_Label = QLabel("")
        layout.addWidget(self.progress_Label)

        # ── Spacer ──
        layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # ── Action buttons ──
        layout.addLayout(self._action_buttons_row())

        scroll.setWidget(container)
        return scroll

    def _build_right_panel(self):
        self.fit_output_box = QGroupBox("Training Monitor")
        self.fit_output_box.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        return self.fit_output_box

    # ── Helper builders ──

    def _path_row(self, label_text, lineedit_name, button_name):
        layout = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setFixedWidth(110)
        layout.addWidget(lbl)
        le = QLineEdit()
        le.setReadOnly(True)
        le.setEnabled(False)
        setattr(self, lineedit_name, le)
        layout.addWidget(le)
        btn = QPushButton("Browse...")
        btn.setFixedWidth(80)
        setattr(self, button_name, btn)
        layout.addWidget(btn)
        return layout

    def _advanced_section(self):
        """Model architecture settings — collapsed by default."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.advanced_button = QPushButton("▶  Model Architecture")
        self.advanced_button.setMinimumHeight(28)
        self.advanced_button.setStyleSheet(
            "QPushButton { text-align: left; padding-left: 6px; "
            "background-color: #e8e8e8; border: 1px solid #ccc; border-radius: 3px; }"
            "QPushButton:hover { background-color: #d8d8d8; }"
        )
        layout.addWidget(self.advanced_button)

        self.advanced_widget = QWidget()
        inner = QVBoxLayout(self.advanced_widget)
        inner.setContentsMargins(12, 6, 4, 8)
        inner.setSpacing(8)

        # Row 1: spinboxes
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.addWidget(QLabel("Levels:"))
        self.levels_spin = QSpinBox()
        self.levels_spin.setMinimum(1)
        self.levels_spin.setValue(5)
        self.levels_spin.setMinimumWidth(48)
        row1.addWidget(self.levels_spin)

        row1.addSpacing(4)
        row1.addWidget(QLabel("Conv:"))
        self.convlayers_spin = QSpinBox()
        self.convlayers_spin.setValue(2)
        self.convlayers_spin.setMinimumWidth(48)
        row1.addWidget(self.convlayers_spin)

        row1.addSpacing(4)
        row1.addWidget(QLabel("Kernel:"))
        self.kernsize_spin = QSpinBox()
        self.kernsize_spin.setMinimum(1)
        self.kernsize_spin.setValue(3)
        self.kernsize_spin.setMinimumWidth(48)
        row1.addWidget(self.kernsize_spin)
        row1.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        inner.addLayout(row1)

        # Row 2: architecture checkboxes
        row2 = QHBoxLayout()
        self.model_3d_check = QCheckBox("3D Model")
        self.check_auto_params = QCheckBox("Dynamic U-Net (Auto)")
        self.check_auto_params.setEnabled(False)
        row2.addWidget(self.model_3d_check)
        row2.addWidget(self.check_auto_params)
        row2.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        inner.addLayout(row2)

        self.advanced_widget.setVisible(False)
        layout.addWidget(self.advanced_widget)

        return container

    def _training_settings_section(self):
        """Training hyperparameters — collapsed by default."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.train_settings_btn = QPushButton("▶  Training Parameters")
        self.train_settings_btn.setMinimumHeight(28)
        self.train_settings_btn.setStyleSheet(
            "QPushButton { text-align: left; padding-left: 6px; "
            "background-color: #e8e8e8; border: 1px solid #ccc; border-radius: 3px; }"
            "QPushButton:hover { background-color: #d8d8d8; }"
        )
        layout.addWidget(self.train_settings_btn)

        self.train_settings_widget = QWidget()
        inner = QVBoxLayout(self.train_settings_widget)
        inner.setContentsMargins(12, 6, 4, 8)
        inner.setSpacing(8)

        # Row 1: Epochs, Batch, LR
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(50)
        self.epochs_spin.setMinimumWidth(52)
        row1.addWidget(self.epochs_spin)

        row1.addSpacing(4)
        row1.addWidget(QLabel("Batch:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(2)
        self.batch_spin.setMinimumWidth(48)
        row1.addWidget(self.batch_spin)

        row1.addSpacing(4)
        row1.addWidget(QLabel("LR:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setMinimumWidth(72)
        row1.addWidget(self.lr_spin)
        row1.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        inner.addLayout(row1)

        # Row 2: training checkboxes
        row2 = QHBoxLayout()
        row2.setSpacing(10)
        self.early_stopping_check = QCheckBox("Early stopping")
        self.early_stopping_check.setChecked(True)
        self.mixed_precision_check = QCheckBox("Mixed precision")
        self.scheduler_check = QCheckBox("LR Scheduler")
        row2.addWidget(self.early_stopping_check)
        row2.addWidget(self.mixed_precision_check)
        row2.addWidget(self.scheduler_check)
        row2.addItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        inner.addLayout(row2)

        self.train_settings_widget.setVisible(False)
        layout.addWidget(self.train_settings_widget)

        return container

    def _action_buttons_row(self):
        row = QVBoxLayout()
        row.setSpacing(6)

        self.preprocess_Button = QPushButton()   # kept for signal compatibility, hidden
        self.preprocess_Button.setVisible(False)

        self.fit_Button = QPushButton("Start training")
        self.fit_Button.setEnabled(False)
        self.fit_Button.setMinimumHeight(36)
        self.fit_Button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.stop_btn = QPushButton("Stop training")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(36)
        self.stop_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #d9534f; color: white; border-radius: 3px; }"
            "QPushButton:disabled { background-color: #c0c0c0; color: #888; }"
        )

        row.addWidget(self.fit_Button)
        row.addWidget(self.stop_btn)

        return row

    # ─────────────────────────────────────────────
    # SIGNALS
    # ─────────────────────────────────────────────

    def _connect_signals(self):
        self.choose_Button.clicked.connect(self.select_data)
        self.save_choose_Button.clicked.connect(self.select_save_path)
        self.pretrain_choose_Button.clicked.connect(self.select_pretrained_path)
        self.advanced_button.clicked.connect(self.toggle_advanced_options)
        self.train_settings_btn.clicked.connect(self.toggle_advanced_options_training)
        self.augmentation_button.clicked.connect(self.open_augm_settings)
        self.finetuning_settings_btn.clicked.connect(self.open_finetuning_settings)
        self.fit_Button.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)
        self.model_3d_check.toggled.connect(self._on_3d_mode_changed)
        self.check_auto_params.toggled.connect(self._on_dynunet_changed)

    # ─────────────────────────────────────────────
    # MATPLOTLIB
    # ─────────────────────────────────────────────

    def _init_matplotlib_canvas(self):
        box_layout = QVBoxLayout(self.fit_output_box)
        box_layout.setContentsMargins(4, 8, 4, 4)
        box_layout.setSpacing(6)

        # Style shared by monitor groupboxes
        group_style = """
            QGroupBox {
                font-size: 10pt; font-weight: bold;
                border: 1px solid #d0d0d0;
                border-radius: 5px;
                margin-top: 10px;
                background-color: #fafafa;
                color: #555;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 3px;
            }
        """

        # ── Row 1: Loss Chart ───────────────────────────────────────────────
        self.fig_loss = plt.figure(constrained_layout=True, facecolor=_FIG_BG)
        self.canvas_loss = FigureCanvas(self.fig_loss)
        self.canvas_loss.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_loss.setMinimumSize(10, 10)
        box_layout.addWidget(self.canvas_loss, stretch=14)

        self.ax_loss = self.fig_loss.add_subplot(111)
        self._style_loss_ax()

        # ── Row 2: Live Preview + Legend ────────────────────────────────────
        preview_row_layout = QHBoxLayout()
        preview_row_layout.setSpacing(8)

        # Live Preview Group
        self.preview_group = QGroupBox("Live Preview")
        self.preview_group.setStyleSheet(group_style)
        self.preview_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        pg_layout = QVBoxLayout(self.preview_group)
        pg_layout.setContentsMargins(0, 12, 0, 0)

        self.fig_prev = plt.figure(facecolor=_FIG_BG)
        self.canvas_prev = FigureCanvas(self.fig_prev)
        self.canvas_prev.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_prev.setMinimumSize(10, 10)
        pg_layout.addWidget(self.canvas_prev)

        # Add initial stretch to center the content
        preview_row_layout.addStretch(1)

        # We add the group without stretch, and we'll manage its width in resizeEvent
        preview_row_layout.addWidget(self.preview_group)

        self.ax_preview = self.fig_prev.add_axes([0, 0, 1, 1])
        self.ax_preview.set_facecolor('#111111')
        self.ax_preview.axis('off')

        # ── Qt legend panel ─────────────────
        legend_group = QGroupBox("Segmentation Legend")
        legend_group.setMinimumWidth(220)
        legend_group.setMaximumWidth(340)
        legend_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        legend_group.setStyleSheet(group_style)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.legend_container = QWidget()
        self.legend_layout = QVBoxLayout(self.legend_container)
        self.legend_layout.setContentsMargins(2, 4, 2, 4)
        self.legend_layout.setSpacing(5)
        self.legend_layout.addStretch()
        scroll.setWidget(self.legend_container)

        lg_box = QVBoxLayout(legend_group)
        lg_box.setContentsMargins(4, 12, 4, 4)
        lg_box.addWidget(scroll)

        preview_row_layout.addWidget(legend_group)

        # Add stretch at the end to push both boxes to the left, keeping them together
        preview_row_layout.addStretch(1)

        box_layout.addLayout(preview_row_layout, stretch=10)

    def _style_loss_ax(self):
        """Apply consistent visual style to ax_loss (called after init and after clear)."""
        self.ax_loss.set_facecolor(_AX_BG)
        self.ax_loss.set_title("Loss", fontsize=10, fontweight='bold', pad=6, color=_TICK_COLOR)
        self.ax_loss.grid(True, linestyle='--', linewidth=0.6, alpha=0.7, color=_GRID_COLOR)
        self.ax_loss.spines['top'].set_visible(False)
        self.ax_loss.spines['right'].set_visible(False)
        self.ax_loss.spines['left'].set_color(_SPINE_COLOR)
        self.ax_loss.spines['bottom'].set_color(_SPINE_COLOR)
        self.ax_loss.tick_params(colors=_TICK_COLOR, labelsize=8)
        self.ax_loss.yaxis.label.set_color(_TICK_COLOR)
        self.ax_loss.xaxis.label.set_color(_TICK_COLOR)

    def _reset_monitor(self):
        """Restore the Training Monitor to its initial empty state."""
        self.loss_history = []
        self.val_loss_history = []
        self.ax_loss.clear()
        self.ax_preview.clear()
        self._style_loss_ax()
        self.ax_preview.set_facecolor('#111111')
        self.ax_preview.set_axis_off()
        self._clear_legend_widget()
        self.canvas_loss.draw()
        self.canvas_prev.draw()

    def _clear_legend_widget(self):
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _update_legend_widget(self, per_mask_dice, out_channels):
        self._clear_legend_widget()
        if not per_mask_dice:
            return

        cmap = plt.get_cmap('tab20')
        vmax = max(out_channels, 1)

        for i, (name, score) in enumerate(per_mask_dice.items()):
            color = cmap((i + 1) / vmax)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

            row_w = QWidget()
            row_w.setStyleSheet("background-color: #f0f0f0; border-radius: 4px;")
            rl = QHBoxLayout(row_w)
            rl.setContentsMargins(6, 2, 6, 2)
            rl.setSpacing(8)

            patch = QLabel()
            patch.setFixedSize(13, 13)
            patch.setStyleSheet(
                f"background-color: {hex_color}; border: 1px solid rgba(0,0,0,0.15); border-radius: 3px;")

            name_lbl = QLabel(f"<b>{name}</b>")
            name_lbl.setTextFormat(QtCore.Qt.RichText)
            name_lbl.setWordWrap(True)
            name_lbl.setStyleSheet("font-size: 10pt; color: #222;")

            score_lbl = QLabel(f": {score:.3f}")
            score_lbl.setStyleSheet("font-size: 10pt; color: #555; font-weight: bold;")
            score_lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

            rl.addWidget(patch)
            rl.addWidget(name_lbl)
            rl.addWidget(score_lbl)
            rl.addStretch()

            self.legend_layout.addWidget(row_w)

        self.legend_layout.addStretch()

    def showEvent(self, event):
        super().showEvent(event)
        # Ensure the preview is squared after the first layout pass
        QtCore.QTimer.singleShot(50, self._force_square_preview)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._force_square_preview()

    def _force_square_preview(self):
        """Force the Live Preview box to be square based on its current height,
        but cap it to available width to prevent overlapping the legend."""
        if hasattr(self, 'preview_group'):
            self.preview_group.blockSignals(True)
            h = self.preview_group.height()
            
            # Parent/Container check to avoid overlap
            # Total width of monitor - (legend width + margins + spacing)
            container_w = self.fit_output_box.width()
            legend_w = self.legend_container.width() if hasattr(self, 'legend_container') else 240
            spacing_margins = 40
            max_w = max(100, container_w - legend_w - spacing_margins)
            
            target_w = min(h, max_w)
            
            if target_w > 20:
                self.preview_group.setFixedWidth(int(target_w))
            self.preview_group.blockSignals(False)

        if hasattr(self, 'canvas_loss'):
            self.canvas_loss.draw_idle()
        if hasattr(self, 'canvas_prev'):
            self.canvas_prev.draw_idle()

    def open_augm_settings(self):
        dialog = AugmentationDialog(self, self.augm_params)
        if dialog.exec_() == QDialog.Accepted:
            self.augm_params = dialog.get_settings()

    def open_finetuning_settings(self):
        dialog = FineTuningDialog(self, self.adaptation_params)
        if dialog.exec_() == QDialog.Accepted:
            self.adaptation_params = dialog.get_settings()
            self._update_adaptation_ui_state()

    def _update_adaptation_ui_state(self):
        mode = self.adaptation_params.get('mode', 'scratch')
        is_pretrained_mode = (mode != 'scratch')
        self.pretrain_choose_Button.setEnabled(is_pretrained_mode)

        if not is_pretrained_mode:
            self.pretrained_path = None
            self.pretrained_location_Text.clear()

        if mode == 'lora':
            self.finetuning_settings_btn.setText("Fine-tuning settings... (LoRA)")
            self.lr_spin.setValue(1e-4)
        elif mode == 'finetune':
            self.finetuning_settings_btn.setText("Fine-tuning settings... (Classic)")
            self.lr_spin.setValue(1e-4)
        else:
            self.finetuning_settings_btn.setText("Fine-tuning settings...")
            self.lr_spin.setValue(1e-3)

        self._update_advanced_ui_state()

    def _update_advanced_ui_state(self):
        mode = self.adaptation_params.get('mode', 'scratch')
        locked = mode in ('finetune', 'lora')
        self.advanced_widget.setEnabled(not locked)
        self.advanced_button.setEnabled(not locked)
        self.check_auto_params.setEnabled(not locked)
        self.model_3d_check.setEnabled(not locked)
        self.train_settings_btn.setEnabled(True)

    def select_pretrained_path(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Choose pretrained dafne model", "",
            "Dafne Model (*.model);;All files (*)"
        )
        if filename:
            self.pretrained_path = filename
            self.pretrained_location_Text.setText(filename)

    def _on_3d_mode_changed(self, is_3d_active):
        self.check_auto_params.setEnabled(is_3d_active)
        if not is_3d_active:
            self.check_auto_params.setChecked(False)

    def _on_dynunet_changed(self, is_dynunet_active):
        for w in [self.convlayers_spin, self.kernsize_spin,
                  self.levels_spin, self.batch_spin]:
            w.setEnabled(not is_dynunet_active)

    def select_save_path(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Choose save model path", "",
            "Dafne Model (*.model);;Pytorch Model (*.pth);;All files (*)"
        )
        if filename:
            if '.' not in os.path.basename(filename):
                filename += '.model'
            self.save_path = filename
            self.model_location_Text.setText(filename)

    def select_data(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select folder data (Images/Masks or NPZ)"
        )
        if not folder_path:
            return

        has_data = any(
            f.endswith('.npz')
            for _, _, files in os.walk(folder_path)
            for f in files
        )
        if not has_data:
            QMessageBox.warning(self, "No data found",
                                "Folder selected does not contain valid .npz files!")
            self.fit_Button.setEnabled(False)
            return

        self.location_Text.setText(folder_path)
        self.fit_Button.setEnabled(True)
        QMessageBox.information(self, "Data Selected",
                                "Dataset folder linked successfully. Ready to train.")

    def update_progress_bar(self, percent):
        self.progressBar.setValue(percent)

    def start_training(self):
        if not self.location_Text.text():
            QMessageBox.warning(self, 'Input Error', "No data loaded. Please select data first")
            return

        self.save_path = self.model_location_Text.text()
        self.pretrained_path = self.pretrained_location_Text.text()

        if not self.save_path:
            QMessageBox.warning(self, 'Input Error', "Please select a save path for the model")
            return

        mode = self.adaptation_params.get('mode', 'scratch')
        if mode in ('finetune', 'lora') and not self.pretrained_path:
            QMessageBox.warning(self, 'Input Error',
                                "No pretrained model selected. Please select a pretrained model")
            return

        self._status_btn(True)
        self._collapse_all_sections()
        self._reset_monitor()

        self.progressBar.setValue(0)
        self.progress_Label.setText("Initialization...")

        configs = build_training_configs(
            data_path=self.location_Text.text(),
            pretrained_path=self.pretrained_path,
            n_levels=self.levels_spin.value(),
            kernel_size=self.kernsize_spin.value(),
            conv_layers=self.convlayers_spin.value(),
            is_3d=self.model_3d_check.isChecked(),
            use_dynunet=self.check_auto_params.isChecked(),
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            lr=self.lr_spin.value(),
            early_stopping=self.early_stopping_check.isChecked(),
            mixed_precision=self.mixed_precision_check.isChecked(),
            scheduler=self.scheduler_check.isChecked(),
            augm_params=self.augm_params,
            adaptation_params=self.adaptation_params,
        )

        self.model_config = configs['model_config']

        self.worker = TrainingWorker(
            dataset_config=configs['dataset_config'],
            model_config=configs['model_config'],
            train_config=configs['train_config'],
            inference_metrics=configs['inference_config'],
            save_path=self.save_path,
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

    def _status_btn(self, active: bool):
        for w in [self.choose_Button, self.save_choose_Button,
                  self.advanced_button, self.train_settings_btn,
                  self.augmentation_button, self.fit_Button,
                  self.levels_spin, self.epochs_spin,
                  self.lr_spin, self.batch_spin,
                  self.finetuning_settings_btn, self.pretrain_choose_Button]:
            w.setEnabled(not active)

        if not active:
            self._update_adaptation_ui_state()

        self.stop_btn.setEnabled(active)

    def update_status_label(self, message):
        self.progress_Label.setText(message)

    def restore_image_contrast(self, img):
        img_vis = img.copy()
        mask_nonzero = (img_vis != 0)
        if np.any(mask_nonzero):
            mn = img_vis[mask_nonzero].min()
            mx = img_vis[mask_nonzero].max()
            img_vis[mask_nonzero] = (img_vis[mask_nonzero] - mn) / (mx - mn + 1e-8)
        return img_vis

    @staticmethod
    def _pad_to_square(arr, fill_value=0):
        """Pad a 2D array to square by centering it, preserving dtype."""
        h, w = arr.shape
        size = max(h, w)
        ph = (size - h) // 2
        pw = (size - w) // 2
        return np.pad(arr,
                      ((ph, size - h - ph), (pw, size - w - pw)),
                      mode='constant', constant_values=fill_value)

    @QtCore.pyqtSlot(float, object, object, float, object, float, dict)
    def update_plots(self, loss, img, mask, val_loss, spacing, best_dice, per_mask_dice):
        self.loss_history.append(loss)
        self.val_loss_history.append(val_loss)

        self.ax_loss.clear()
        self._style_loss_ax()

        x = range(len(self.loss_history))
        self.ax_loss.plot(self.loss_history, color=_TRAIN_COLOR, linewidth=1.8,
                          label='Train Loss', solid_capstyle='round')
        self.ax_loss.fill_between(x, self.loss_history, alpha=0.10, color=_TRAIN_COLOR)

        x2 = range(len(self.val_loss_history))
        self.ax_loss.plot(self.val_loss_history, color=_VAL_COLOR, linewidth=1.8,
                          label='Val Loss', solid_capstyle='round')
        self.ax_loss.fill_between(x2, self.val_loss_history, alpha=0.10, color=_VAL_COLOR)

        self.ax_loss.legend(loc='upper right', fontsize=8, framealpha=0.8,
                            facecolor=_AX_BG, edgecolor=_SPINE_COLOR)
        self.ax_loss.set_title(
            f'Train: {loss:.4f}  |  Val: {val_loss:.4f}  |  Best Dice: {best_dice:.4f}',
            fontsize=10, fontweight='bold', pad=7, color=_TICK_COLOR
        )

        self.ax_preview.clear()
        self.ax_preview.set_facecolor('#111111')

        if img.ndim == 3:
            img = img[0, :, :]
        img = self.restore_image_contrast(img)

        pixel_aspect = 1.0
        if spacing is not None and len(spacing) >= 2:
            try:
                pixel_aspect = float(spacing[-1] / spacing[-2])
            except Exception:
                pixel_aspect = 1.0

        # Pad image and mask to square so ax_preview never changes size.
        if mask is not None:
            mask = self._pad_to_square(mask, fill_value=0)
        img = self._pad_to_square(img, fill_value=0)

        self.ax_preview.imshow(img, cmap='gray', aspect=pixel_aspect,
                               vmin=0.0, vmax=1.0)

        if mask is not None:
            masked = np.ma.masked_where(mask == 0, mask)
            vmin, vmax = 0, self.model_config.out_channels
            self.ax_preview.imshow(masked, cmap='tab20', alpha=0.5,
                                   aspect=pixel_aspect, vmin=vmin, vmax=vmax)

        self._update_legend_widget(per_mask_dice, self.model_config.out_channels)

        self.ax_preview.set_axis_off()
        self.canvas_loss.draw()
        self.canvas_prev.draw()
        QtWidgets.QApplication.processEvents()

    def handle_error(self, err):
        self._status_btn(False)
        QMessageBox.critical(self, "Error", f"Training is stopping:\n{err}")
        self.progressBar.setValue(0)
        self.progress_Label.setText("Error occurred.")
        self._reset_monitor()
        self._reset_ui_state()

    def on_training_stopped(self):
        self._status_btn(False)
        self.progressBar.setValue(0)
        self.progress_Label.setText("")
        self._reset_monitor()
        #self._clear_paths_and_data()
        QMessageBox.information(self, "Stopped", "Training stopped correctly.")

    def _clear_paths_and_data(self):
        self.save_path = None
        self.pretrained_path = None
        self.location_Text.clear()
        self.model_location_Text.clear()
        self.pretrained_location_Text.clear()
        self.model_3d_check.setChecked(False)
        self.check_auto_params.setChecked(False)
        self.fit_Button.setEnabled(False)

    def on_training_finished(self):
        self._status_btn(False)
        self.progressBar.setValue(100)
        self.progress_Label.setText('Training completed!')
        self._clear_paths_and_data()
        QMessageBox.information(self, 'Finished!', "The model was trained successfully")

    def _reset_ui_state(self):
        self.fit_Button.setEnabled(True)
        self.choose_Button.setEnabled(True)

    def toggle_advanced_options_training(self):
        visible = not self.train_settings_widget.isVisible()
        self.train_settings_widget.setVisible(visible)
        self.train_settings_btn.setText(
            "▼  Training Parameters" if visible else "▶  Training Parameters"
        )

    def toggle_advanced_options(self):
        visible = not self.advanced_widget.isVisible()
        self.advanced_widget.setVisible(visible)
        self.advanced_button.setText(
            "▼  Model Architecture" if visible else "▶  Model Architecture"
        )

    def _collapse_all_sections(self):
        """Force-collapse all expandable UI sections."""
        self.advanced_widget.setVisible(False)
        self.advanced_button.setText("▶  Model Architecture")
        self.train_settings_widget.setVisible(False)
        self.train_settings_btn.setText("▶  Training Parameters")


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Model Trainer")
    app.setApplicationDisplayName("Model Trainer")
    _apply_fusion_style(app)
    app.setWindowIcon(_make_app_icon())
    window = ModelTrainerSplit()
    window.show()
    sys.exit(app.exec_())
