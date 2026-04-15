from PyQt5.QtWidgets import QDialog, QVBoxLayout, QCheckBox, QDialogButtonBox

class AugmentationDialog(QDialog):
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Data Augmentation Settings")
        self.resize(300, 200)

        self.layout = QVBoxLayout(self)

        # save checkbox choises
        self.checkboxes = {}

        options = [
            ('rotate', 'Random Rotation (90°)'),
            ('flip_x', 'Random Flip (Horizontal)'),
            ('flip_y', 'Random Flip (Vertical)'),
            ('zoom', "Random Zoom"),
            ('noise', "Random Gaussian Noise")
        ]

        for key, label in options:
            cb = QCheckBox(label)
            if current_settings and current_settings.get(key, False):
                cb.setChecked(True)
            self.layout.addWidget(cb)
            self.checkboxes[key] = cb
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.layout.addWidget(self.buttons)
    
    def get_settings(self):
        settings = {}
        for key, cb in self.checkboxes.items():
            settings[key] = cb.isChecked()
        return settings
    

