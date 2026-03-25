# _image_import_widget.py
import os

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QLineEdit,
)
from qtpy.QtCore import Signal

from ._helpers import (
    pt_meta_dict,
    layer_name,
    vispy_colormap_from_rgb,
    image_root_from_path,
    extract_nd2_channel_info,
)

try:
    import nd2
except ImportError:
    nd2 = None


class ImageImportWidget(QWidget):
    layers_loaded = Signal()

    def __init__(self, viewer=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer

        self.nd2_path = None
        self._cached_array = None
        self._channel_axis = None
        self._n_channels = 0
        self._channel_info = []

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(1)

        title = QLabel("ND2 import")
        f = title.font()
        f.setBold(True)
        f.setPointSize(f.pointSize() + 1)
        title.setFont(f)
        layout.addWidget(title)

        file_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Select a .nd2 file...")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse)

        file_row.addWidget(self.path_edit)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        load_btn = QPushButton("Load channels into viewer")
        load_btn.clicked.connect(self._on_load_channels)
        layout.addWidget(load_btn)

        layout.addStretch(1)

    def _on_browse(self):
        from napari.utils.notifications import show_error

        if nd2 is None:
            show_error("The 'nd2' package is not installed. Please `pip install nd2`.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ND2 file",
            "",
            "ND2 files (*.nd2);;All files (*.*)",
        )
        if not path:
            return

        self.nd2_path = path
        self.path_edit.setText(path)

        try:
            arr = nd2.imread(path)
        except Exception as e:
            show_error(f"Failed to read ND2 file:\n{e}")
            self.info_label.setText("Failed to read ND2 file.")
            self._cached_array = None
            self._n_channels = 0
            self._channel_axis = None
            self._channel_info = []
            return

        self._cached_array = arr
        shape = arr.shape
        ndim = arr.ndim

        self._channel_axis = None
        self._n_channels = 1

        if ndim == 4:
            _, c, _, _ = shape
            self._channel_axis = 1
            self._n_channels = c
        elif ndim == 3:
            c = shape[0]
            if c <= 8:
                self._channel_axis = 0
                self._n_channels = c

        self._channel_info = extract_nd2_channel_info(
            path,
            nd2,
            fallback_n_channels=self._n_channels,
        )

        info = f"Loaded {os.path.basename(path)}\nshape: {shape}\n"
        if self._channel_axis is None:
            info += "Detected single channel (or no explicit channel axis).\n"
        else:
            info += f"Detected {self._n_channels} channel(s) on axis {self._channel_axis}.\n"

        if self._channel_info:
            info += "\nChannel labels:\n" + "\n".join(
                f"  Ch {i}: {ch['label']}" for i, ch in enumerate(self._channel_info)
            )

        self.info_label.setText(info)

    def _iter_channel_images(self, arr):
        if self._channel_axis is None:
            yield 0, arr
            return

        for ch in range(self._n_channels):
            sl = [slice(None)] * arr.ndim
            sl[self._channel_axis] = ch
            yield ch, arr[tuple(sl)]

    def _on_load_channels(self):
        from napari.utils.notifications import show_error, show_warning

        if self.viewer is None:
            show_error("No napari viewer attached.")
            return

        if self._cached_array is None:
            show_warning("No ND2 file loaded yet.")
            return

        image_root = image_root_from_path(self.nd2_path or "image.nd2")

        for ch, img in self._iter_channel_images(self._cached_array):
            info = self._channel_info[ch] if ch < len(self._channel_info) else {
                "index": ch,
                "label": f"Ch {ch}",
                "rgb": None,
                "name": f"Ch {ch}",
            }

            name = layer_name(image_root, info["label"], "raw")
            kwargs = {
                "name": name,
                "metadata": pt_meta_dict(
                    role="raw_image",
                    image_path=self.nd2_path,
                    image_root=image_root,
                    channel_index=info["index"],
                    channel_label=info["label"],
                    channel_name=info["name"],
                    channel_rgb=info["rgb"],
                ),
            }

            if info["rgb"] is not None:
                kwargs["colormap"] = vispy_colormap_from_rgb(info["rgb"], info["label"])

            self.viewer.add_image(img, **kwargs)

        self.layers_loaded.emit()