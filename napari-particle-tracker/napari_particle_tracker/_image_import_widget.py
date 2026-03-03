# _image_import_widget.py

import os

from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QFormLayout,
    QLineEdit,
)
from qtpy.QtCore import Qt

try:
    import nd2
except ImportError:
    nd2 = None


class ImageImportWidget(QWidget):
    """
    ND2 import widget.

    - Choose an ND2 file
    - Read the array with nd2.imread
    - Infer channel axis and number of channels (1 .. N)
    - For each channel:
        * Slice out that channel
        * Name the layer based on ND2 metadata (e.g. "488 nm (GFP)")
          or fall back to "Ch 0", "Ch 1", ...
    """

    def __init__(self, viewer=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer

        self.nd2_path = None
        self._cached_array = None
        self._channel_axis = None
        self._n_channels = 0
        self._channel_labels = []

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Title
        title = QLabel("ND2 import")
        f = title.font()
        f.setBold(True)
        f.setPointSize(f.pointSize() + 1)
        title.setFont(f)
        layout.addWidget(title)

        # File row
        file_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Select a .nd2 file...")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse)

        file_row.addWidget(self.path_edit)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        # Info label
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # Load button
        load_btn = QPushButton("Load channels into viewer")
        load_btn.clicked.connect(self._on_load_channels)
        layout.addWidget(load_btn)

        layout.addStretch(1)

    # ------------------------------------------------------------------
    # File loading + metadata
    # ------------------------------------------------------------------
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

        # Read image data
        try:
            arr = nd2.imread(path)
        except Exception as e:
            show_error(f"Failed to read ND2 file:\n{e}")
            self.info_label.setText("Failed to read ND2 file.")
            self._cached_array = None
            self._n_channels = 0
            self._channel_axis = None
            self._channel_labels = []
            return

        self._cached_array = arr
        shape = arr.shape
        ndim = arr.ndim

        # Infer channel axis and number of channels
        # Common ND2 patterns:
        #   (t, c, y, x)
        #   (c, y, x)
        #   (t, y, x)   -> single channel, no explicit c
        #   (y, x)      -> single channel image
        self._channel_axis = None
        self._n_channels = 1  # default single-channel

        if ndim == 4:
            # Heuristic: (t, c, y, x) is very common
            # shape: (T, C, Y, X)
            _, c, _, _ = shape
            self._channel_axis = 1
            self._n_channels = c
        elif ndim == 3:
            # Could be (c, y, x) or (t, y, x)
            # We assume multi-channel if the first dim is "small-ish"
            c = shape[0]
            if c <= 8:  # heuristic: up to 8 channels is common
                self._channel_axis = 0
                self._n_channels = c
            else:
                # Treat as time series single-channel (t, y, x)
                self._channel_axis = None
                self._n_channels = 1
        else:
            # 2D or other shapes: treat as single-channel
            self._channel_axis = None
            self._n_channels = 1

        # Extract labels from metadata, if possible
        self._channel_labels = self._extract_channel_labels(path, self._n_channels)

        # Info text
        info = f"Loaded {os.path.basename(path)}\nshape: {shape}\n"
        if self._channel_axis is None:
            info += f"Detected single channel (or no explicit channel axis).\n"
        else:
            info += f"Detected {self._n_channels} channel(s) on axis {self._channel_axis}.\n"

        if self._channel_labels:
            info += "\nChannel labels:\n" + "\n".join(
                f"  Ch {i}: {lab}" for i, lab in enumerate(self._channel_labels)
            )

        self.info_label.setText(info)

    def _extract_channel_labels(self, path: str, n_channels: int):
        """
        Build human-readable labels per channel from ND2 metadata.

        Prefer excitation wavelength (e.g. "488 nm"), then emission, then name.
        Fallback: "Ch 0", "Ch 1", ...
        """
        labels = [f"Ch {i}" for i in range(n_channels)]

        if nd2 is None:
            return labels

        try:
            with nd2.ND2File(path) as f:
                md = f.metadata
                ch_meta = getattr(md, "channels", None)
                if not ch_meta:
                    return labels

                for i in range(min(n_channels, len(ch_meta))):
                    ch = ch_meta[i]

                    def get(key, default=None):
                        if isinstance(ch, dict):
                            return ch.get(key, default)
                        return getattr(ch, key, default)

                    ex = get("excitationLambda", None)
                    em = get("emissionLambda", None)
                    name = get("name", None) or get("channelName", None)

                    if ex:
                        lbl = f"{int(round(ex))} nm"
                        if name:
                            lbl += f" ({name})"
                    elif em:
                        lbl = f"{int(round(em))} nm (em)"
                        if name:
                            lbl += f" ({name})"
                    elif name:
                        lbl = str(name)
                    else:
                        lbl = f"Ch {i}"

                    labels[i] = lbl
        except Exception:
            # If metadata parsing fails, keep default labels
            pass

        return labels

    # ------------------------------------------------------------------
    # Add channels to viewer
    # ------------------------------------------------------------------
    def _on_load_channels(self):
        from napari.utils.notifications import show_error, show_warning

        if self.viewer is None:
            show_error("No napari viewer attached.")
            return

        if self._cached_array is None:
            show_warning("No ND2 file loaded yet.")
            return

        arr = self._cached_array
        ndim = arr.ndim

        # Helper to slice out individual channel images
        def _get_channel_image(a, ch_idx):
            # No explicit channel axis -> just return the full array once
            if self._channel_axis is None:
                return a

            # Build a slice with this channel index on the channel axis
            sl = [slice(None)] * a.ndim
            sl[self._channel_axis] = ch_idx
            return a[tuple(sl)]

        # Single-channel case: just one layer
        if self._n_channels == 1:
            label = self._channel_labels[0] if self._channel_labels else "Ch 0"
            self.viewer.add_image(arr, name=label)
            return

        # Multi-channel: one layer per channel
        for ch in range(self._n_channels):
            img = _get_channel_image(arr, ch)
            label = (
                self._channel_labels[ch]
                if self._channel_labels and ch < len(self._channel_labels)
                else f"Ch {ch}"
            )
            self.viewer.add_image(img, name=label)