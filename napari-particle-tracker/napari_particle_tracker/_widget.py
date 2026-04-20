# _widget.py
from __future__ import annotations

import numpy as np
import pandas as pd

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QCheckBox, QDoubleSpinBox, QSpinBox, QComboBox,
    QFormLayout,
)
from qtpy.QtCore import Qt

from napari.utils.notifications import show_info, show_warning, show_error
from napari.qt.threading import thread_worker

from ._processing import (
    preprocess_stack_single_channel,
    detect_puncta_dask,
    filter_dense_blobs,
)
from ._helpers import pt_meta_dict, layer_name, image_root_from_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_layers(viewer):
    """Return list of (name, layer) for all Image layers in the viewer."""
    return [
        ly for ly in viewer.layers
        if ly.__class__.__name__ == "Image"
    ]


def _array_from_layer(layer):
    """Extract a (t, y, x) numpy array from a napari Image layer."""
    import dask.array as da
    arr = layer.data
    if isinstance(arr, da.Array):
        arr = arr.compute()
    arr = np.asarray(arr)

    if arr.ndim == 2:
        return arr[np.newaxis]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        # (t, c, y, x) — take channel 0
        show_warning(f"Layer '{layer.name}' is 4D; using channel 0.")
        return arr[:, 0]
    raise ValueError(f"Cannot use layer with shape {arr.shape} for detection.")


def _pt_meta_from_layer(layer):
    meta = getattr(layer, "metadata", {}) or {}
    return meta.get("particle_tracking", {}) or {}


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class DetectionWidget(QWidget):
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self._build_ui()
        self._connect_layer_events()
        self._refresh_layer_combos()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(self._build_preprocess_group())
        layout.addWidget(self._build_detection_group())
        layout.addStretch(1)

    # --- Preprocessing group -------------------------------------------

    def _build_preprocess_group(self):
        group = QGroupBox("Preprocessing (optional)")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setSpacing(4)

        self.pre_layer_combo = QComboBox()
        form.addRow("Input layer", self.pre_layer_combo)

        self.median_filter_size = QSpinBox()
        self.median_filter_size.setRange(1, 101)
        self.median_filter_size.setSingleStep(1)
        self.median_filter_size.setValue(10)
        form.addRow("Median filter size", self.median_filter_size)

        run_pre_btn = QPushButton("Run preprocessing")
        run_pre_btn.clicked.connect(self._on_run_preprocessing)
        form.addRow(run_pre_btn)

        return group

    # --- Detection group -----------------------------------------------

    def _build_detection_group(self):
        group = QGroupBox("Spot detection")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setSpacing(4)

        self.det_layer_combo = QComboBox()
        form.addRow("Input layer", self.det_layer_combo)

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 0.01)
        self.threshold.setSingleStep(0.00001)
        self.threshold.setDecimals(6)
        self.threshold.setValue(0.00012)
        form.addRow("Threshold", self.threshold)

        self.min_sigma = QDoubleSpinBox()
        self.min_sigma.setRange(0.1, 20.0)
        self.min_sigma.setSingleStep(0.1)
        self.min_sigma.setDecimals(1)
        self.min_sigma.setValue(1.0)
        form.addRow("Min sigma", self.min_sigma)

        self.max_sigma = QDoubleSpinBox()
        self.max_sigma.setRange(0.1, 40.0)
        self.max_sigma.setSingleStep(0.1)
        self.max_sigma.setDecimals(1)
        self.max_sigma.setValue(3.0)
        form.addRow("Max sigma", self.max_sigma)

        # Density filter
        self.enable_density_filter = QCheckBox("Enable density filter")
        self.enable_density_filter.setChecked(False)
        form.addRow(self.enable_density_filter)

        self.bin_size_label = QLabel("Bin size (px)")
        self.bin_size = QSpinBox()
        self.bin_size.setRange(1, 256)
        self.bin_size.setValue(5)
        form.addRow(self.bin_size_label, self.bin_size)

        self.blob_filter_label = QLabel("Max blobs per bin")
        self.blob_filter = QSpinBox()
        self.blob_filter.setRange(1, 10000)
        self.blob_filter.setValue(10)
        form.addRow(self.blob_filter_label, self.blob_filter)

        self._set_density_filter_visible(False)
        self.enable_density_filter.toggled.connect(self._set_density_filter_visible)

        run_det_btn = QPushButton("Run detection")
        run_det_btn.clicked.connect(self._on_run_detection)
        form.addRow(run_det_btn)

        return group

    def _set_density_filter_visible(self, visible):
        self.bin_size_label.setVisible(visible)
        self.bin_size.setVisible(visible)
        self.blob_filter_label.setVisible(visible)
        self.blob_filter.setVisible(visible)

    # ------------------------------------------------------------------
    # Layer combo management
    # ------------------------------------------------------------------

    def _connect_layer_events(self):
        self.viewer.layers.events.inserted.connect(self._refresh_layer_combos)
        self.viewer.layers.events.removed.connect(self._refresh_layer_combos)
        self.viewer.layers.events.reordered.connect(self._refresh_layer_combos)

    def _refresh_layer_combos(self, event=None):
        names = [ly.name for ly in _image_layers(self.viewer)]

        for combo in (self.pre_layer_combo, self.det_layer_combo):
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            # restore previous selection if still present
            idx = combo.findText(prev)
            combo.setCurrentIndex(max(idx, 0))
            combo.blockSignals(False)

    def _get_layer(self, combo):
        """Return the napari layer currently selected in a combo, or None."""
        name = combo.currentText()
        if not name:
            return None
        try:
            return self.viewer.layers[name]
        except KeyError:
            return None

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _on_run_preprocessing(self):
        layer = self._get_layer(self.pre_layer_combo)
        if layer is None:
            show_warning("No image layer selected for preprocessing.")
            return

        try:
            arr = _array_from_layer(layer)
        except ValueError as e:
            show_error(str(e))
            return

        mf_size = self.median_filter_size.value()

        try:
            pre = preprocess_stack_single_channel(arr, median_filter_size=mf_size)
        except Exception as e:
            show_error(f"Preprocessing failed: {e}")
            return

        pt_meta = _pt_meta_from_layer(layer)
        image_root = pt_meta.get("image_root") or image_root_from_path(
            pt_meta.get("image_path", layer.name)
        )
        channel_label = pt_meta.get("channel_label")
        name = layer_name(image_root, channel_label, "preprocessed")

        self.viewer.add_image(
            pre,
            name=name,
            rgb=False,
            metadata=pt_meta_dict(
                role="preprocessed",
                source_layer=layer.name,
                image_root=image_root,
                channel_label=channel_label,
            ),
        )
        show_info(f"Preprocessing complete → '{name}'")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _on_run_detection(self):
        layer = self._get_layer(self.det_layer_combo)
        if layer is None:
            show_warning("No image layer selected for detection.")
            return

        try:
            arr = _array_from_layer(layer)
        except ValueError as e:
            show_error(str(e))
            return

        # snapshot params
        threshold  = self.threshold.value()
        min_sigma  = self.min_sigma.value()
        max_sigma  = self.max_sigma.value()
        use_filter = self.enable_density_filter.isChecked()
        bin_size   = self.bin_size.value()
        blob_filter = self.blob_filter.value()

        pt_meta = _pt_meta_from_layer(layer)
        image_root = pt_meta.get("image_root") or image_root_from_path(
            pt_meta.get("image_path", layer.name)
        )
        channel_label = pt_meta.get("channel_label")
        source_name = layer.name

        @thread_worker
        def _run(stack, thr, smin, smax):
            return detect_puncta_dask(stack, threshold=thr, min_sigma=smin, max_sigma=smax)

        worker = _run(arr, threshold, min_sigma, max_sigma)

        def _on_error(err):
            show_error(f"Detection failed: {err}")

        def _on_done(det_df: pd.DataFrame):
            if det_df is None or len(det_df) == 0:
                show_info("No puncta detected with current parameters.")
                return

            n_before = len(det_df)
            if use_filter:
                try:
                    det_df = filter_dense_blobs(
                        det_df, bin_size=bin_size, blob_filter=blob_filter
                    )
                except Exception as e:
                    show_warning(f"Density filtering skipped: {e}")
            n_after = len(det_df)

            if n_after == 0:
                show_info("All puncta filtered out by density settings.")
                return

            points_data = det_df[["frame", "y", "x"]].to_numpy()
            props = {}
            if "size" in det_df.columns:
                props["size"] = det_df["size"].to_numpy()

            run_params = {
                "median_filter_size":       self.median_filter_size.value(),
                "threshold":                float(threshold),
                "min_sigma":                float(min_sigma),
                "max_sigma":                float(max_sigma),
                "enable_density_filter":    bool(use_filter),
                "bin_size":                 int(bin_size),
                "blob_filter":              int(blob_filter),
                "n_detected_before_filter": int(n_before),
                "n_detected_after_filter":  int(n_after),
            }

            puncta_name = layer_name(image_root, channel_label, "puncta")
            pts_layer = self.viewer.add_points(
                points_data,
                name=puncta_name,
                size=30,
                face_color="transparent",
                properties=props,
                border_color="red",
                border_width=0.1,
            )
            pts_layer.metadata["run_params"] = run_params
            pts_layer.metadata["particle_tracking"] = {
                **pt_meta,
                "role":          "puncta",
                "run_params":    run_params,
                "image_root":    image_root,
                "channel_label": channel_label,
                "source_layer":  source_name,
            }

            msg = (
                f"Detection complete: {n_after} puncta"
                + (f" (filtered from {n_before})" if use_filter else "")
                + f" → '{puncta_name}'"
            )
            show_info(msg)

        worker.errored.connect(_on_error)
        worker.returned.connect(_on_done)
        worker.start()