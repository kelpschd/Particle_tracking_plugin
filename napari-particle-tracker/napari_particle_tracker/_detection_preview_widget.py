# _detection_preview_widget.py
import numpy as np
import pandas as pd
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QCheckBox, QGroupBox,
    QComboBox, QLineEdit,
)
from qtpy.QtCore import Qt, QTimer, Signal

from ._processing import filter_dense_blobs, preprocess_stack_single_channel
from ._helpers import (
    image_layers, array_from_layer, pt_meta_from_layer,
    pt_meta_dict, layer_name, image_root_from_path,
)
from skimage.feature import blob_log


class DetectionPreviewWidget(QWidget):
    run_full_detection = Signal(dict)

    _DEFAULTS = dict(threshold=0.00012, min_sigma=1.0, max_sigma=3.0,
                     bin_size=5, blob_filter=10)
    _THRESH_RANGE = (1e-5, 1e-3)
    _SIGMA_RANGE  = (0.5, 8.0)
    _BIN_RANGE    = (2, 30)
    _BLOBF_RANGE  = (2, 50)

    def __init__(self, viewer=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self._stack = None
        self._preview_layer = None
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._run_preview)
        self._adv_visible = False
        self._build_ui()
        if viewer is not None:
            self._connect_viewer_events()
            self._refresh_layer_combos()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self._build_preprocessing_group())
        layout.addWidget(self._build_detection_group())
        run_btn = QPushButton("Run full detection on all frames")
        run_btn.clicked.connect(self._on_run_full)
        layout.addWidget(run_btn)
        layout.addStretch(1)

    def _build_preprocessing_group(self):
        grp = QGroupBox("Preprocessing")
        gl = QVBoxLayout(grp)
        gl.setSpacing(6)

        pre_layer_row = QHBoxLayout()
        pre_layer_row.addWidget(QLabel("Input layer"))
        self._pre_layer_combo = QComboBox()
        pre_layer_row.addWidget(self._pre_layer_combo)
        gl.addLayout(pre_layer_row)

        mf_row = QHBoxLayout()
        mf_row.addWidget(QLabel("Median filter size"))
        self._mf_slider = QSlider(Qt.Horizontal)
        self._mf_slider.setRange(1, 51)
        self._mf_slider.setSingleStep(2)
        self._mf_slider.setValue(10)
        self._mf_lbl = QLabel("10")
        self._mf_slider.valueChanged.connect(lambda v: self._mf_lbl.setText(str(v)))
        mf_row.addWidget(self._mf_slider)
        mf_row.addWidget(self._mf_lbl)
        gl.addLayout(mf_row)

        run_pre_btn = QPushButton("Run preprocessing")
        run_pre_btn.clicked.connect(self._on_run_preprocessing)
        gl.addWidget(run_pre_btn)

        return grp

    def _build_detection_group(self):
        grp = QGroupBox("Detection preview")
        gl = QVBoxLayout(grp)
        gl.setSpacing(6)

        # layer selector
        layer_row = QHBoxLayout()
        layer_row.addWidget(QLabel("Input layer"))
        self._layer_combo = QComboBox()
        self._layer_combo.currentTextChanged.connect(self._on_layer_selected)
        layer_row.addWidget(self._layer_combo)
        gl.addLayout(layer_row)

        # live checkbox + frame badge
        top_row = QHBoxLayout()
        self._live_cb = QCheckBox("Preview current frame live")
        self._live_cb.setChecked(True)
        self._live_cb.stateChanged.connect(self._on_live_toggled)
        self._frame_lbl = QLabel("frame –")
        self._frame_lbl.setStyleSheet("color: gray; font-size: 11px;")
        top_row.addWidget(self._live_cb)
        top_row.addStretch()
        top_row.addWidget(self._frame_lbl)
        gl.addLayout(top_row)

        # threshold
        gl.addWidget(QLabel("Threshold"))
        thresh_row = QHBoxLayout()
        self._thresh_slider = QSlider(Qt.Horizontal)
        self._thresh_slider.setRange(0, 1000)
        self._thresh_slider.setValue(self._thresh_to_slider(self._DEFAULTS["threshold"]))
        self._thresh_slider.valueChanged.connect(self._on_thresh_slider)
        self._thresh_edit = QLineEdit(f"{self._DEFAULTS['threshold']:.6f}")
        self._thresh_edit.setFixedWidth(80)
        self._thresh_edit.editingFinished.connect(self._on_thresh_edit)
        thresh_row.addWidget(self._thresh_slider)
        thresh_row.addWidget(self._thresh_edit)
        gl.addLayout(thresh_row)

        # sigma range
        gl.addWidget(QLabel("Spot size σ  (min – max)"))
        sigma_row = QHBoxLayout()
        self._sig_min_slider = QSlider(Qt.Horizontal)
        self._sig_min_slider.setRange(0, 1000)
        self._sig_min_slider.setValue(self._sigma_to_slider(self._DEFAULTS["min_sigma"]))
        self._sig_min_slider.valueChanged.connect(self._on_sigma_changed)
        self._sig_max_slider = QSlider(Qt.Horizontal)
        self._sig_max_slider.setRange(0, 1000)
        self._sig_max_slider.setValue(self._sigma_to_slider(self._DEFAULTS["max_sigma"]))
        self._sig_max_slider.valueChanged.connect(self._on_sigma_changed)
        self._sigma_lbl = QLabel(
            f"{self._DEFAULTS['min_sigma']:.1f} – {self._DEFAULTS['max_sigma']:.1f}"
        )
        self._sigma_lbl.setFixedWidth(70)
        sigma_row.addWidget(self._sig_min_slider)
        sigma_row.addWidget(self._sig_max_slider)
        sigma_row.addWidget(self._sigma_lbl)
        gl.addLayout(sigma_row)

        # stat
        self._lbl_detected = self._make_stat("Detected (this frame)", "–")
        gl.addWidget(self._lbl_detected)

        # advanced
        self._adv_btn = QPushButton("▶  Advanced filters")
        self._adv_btn.setFlat(True)
        self._adv_btn.clicked.connect(self._toggle_adv)
        gl.addWidget(self._adv_btn)

        self._adv_box = QWidget()
        adv_layout = QVBoxLayout(self._adv_box)
        adv_layout.setContentsMargins(0, 0, 0, 0)

        bin_row = QHBoxLayout()
        self._bin_slider = QSlider(Qt.Horizontal)
        self._bin_slider.setRange(*self._BIN_RANGE)
        self._bin_slider.setValue(self._DEFAULTS["bin_size"])
        self._bin_lbl = QLabel(str(self._DEFAULTS["bin_size"]))
        self._bin_slider.valueChanged.connect(
            lambda v: (self._bin_lbl.setText(str(v)), self._debounce.start())
        )
        bin_row.addWidget(QLabel("Bin size (px)"))
        bin_row.addWidget(self._bin_slider)
        bin_row.addWidget(self._bin_lbl)
        adv_layout.addLayout(bin_row)

        blobf_row = QHBoxLayout()
        self._blobf_slider = QSlider(Qt.Horizontal)
        self._blobf_slider.setRange(*self._BLOBF_RANGE)
        self._blobf_slider.setValue(self._DEFAULTS["blob_filter"])
        self._blobf_lbl = QLabel(str(self._DEFAULTS["blob_filter"]))
        self._blobf_slider.valueChanged.connect(
            lambda v: (self._blobf_lbl.setText(str(v)), self._debounce.start())
        )
        blobf_row.addWidget(QLabel("Max per bin"))
        blobf_row.addWidget(self._blobf_slider)
        blobf_row.addWidget(self._blobf_lbl)
        adv_layout.addLayout(blobf_row)

        self._adv_box.setVisible(False)
        gl.addWidget(self._adv_box)

        return grp

    def _make_stat(self, label_text, value_text):
        w = QWidget()
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(6, 4, 6, 4)
        vbox.setSpacing(1)
        lbl = QLabel(label_text)
        lbl.setStyleSheet("font-size: 11px; color: gray;")
        val = QLabel(value_text)
        val.setStyleSheet("font-size: 18px; font-weight: 500;")
        vbox.addWidget(lbl)
        vbox.addWidget(val)
        w.setStyleSheet("background: palette(midlight); border-radius: 4px;")
        return w

    def _set_stats(self, detected):
        self._lbl_detected.findChildren(QLabel)[1].setText(str(detected))

    # ------------------------------------------------------------------
    # Layer combos
    # ------------------------------------------------------------------

    def _connect_viewer_events(self):
        self.viewer.layers.events.inserted.connect(self._refresh_layer_combos)
        self.viewer.layers.events.removed.connect(self._refresh_layer_combos)
        self.viewer.layers.events.reordered.connect(self._refresh_layer_combos)

    def _refresh_layer_combos(self, event=None):
        names = [ly.name for ly in image_layers(self.viewer)]
        for combo in (self._pre_layer_combo, self._layer_combo):
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            idx = combo.findText(prev)
            combo.setCurrentIndex(max(idx, 0))
            combo.blockSignals(False)

    def _selected_layer(self):
        name = self._layer_combo.currentText()
        if not name:
            return None
        try:
            return self.viewer.layers[name]
        except KeyError:
            return None

    def _on_layer_selected(self, name):
        layer = self._selected_layer()
        if layer is None:
            self._stack = None
            return
        try:
            arr = array_from_layer(layer)
        except ValueError:
            self._stack = None
            return
        self._stack = arr
        self._debounce.start()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _on_run_preprocessing(self):
        from napari.utils.notifications import show_error, show_info, show_warning

        name = self._pre_layer_combo.currentText()
        if not name:
            show_warning("No layer selected for preprocessing.")
            return
        try:
            layer = self.viewer.layers[name]
        except KeyError:
            show_warning(f"Layer '{name}' not found.")
            return
        try:
            arr = array_from_layer(layer)
        except ValueError as e:
            show_error(str(e))
            return

        mf_size = self._mf_slider.value()
        try:
            pre = preprocess_stack_single_channel(arr, median_filter_size=mf_size)
        except Exception as e:
            show_error(f"Preprocessing failed: {e}")
            return

        pt_meta = pt_meta_from_layer(layer)
        image_root = pt_meta.get("image_root") or image_root_from_path(
            pt_meta.get("image_path", layer.name)
        )
        channel_label = pt_meta.get("channel_label")
        out_name = layer_name(image_root, channel_label, "preprocessed")

        self.viewer.add_image(
            pre,
            name=out_name,
            rgb=False,
            metadata=pt_meta_dict(
                role="preprocessed",
                source_layer=layer.name,
                image_root=image_root,
                channel_label=channel_label,
                median_filter_size=mf_size,
            ),
        )

        idx = self._layer_combo.findText(out_name)
        if idx >= 0:
            self._layer_combo.setCurrentIndex(idx)

        show_info(f"Preprocessing complete → '{out_name}'")

    # ------------------------------------------------------------------
    # Slider <-> float helpers
    # ------------------------------------------------------------------

    def _thresh_to_slider(self, v):
        lo, hi = np.log10(self._THRESH_RANGE[0]), np.log10(self._THRESH_RANGE[1])
        return int((np.log10(max(v, 1e-10)) - lo) / (hi - lo) * 1000)

    def _slider_to_thresh(self, s):
        lo, hi = np.log10(self._THRESH_RANGE[0]), np.log10(self._THRESH_RANGE[1])
        return 10 ** (lo + s / 1000 * (hi - lo))

    def _sigma_to_slider(self, v):
        lo, hi = self._SIGMA_RANGE
        return int((v - lo) / (hi - lo) * 1000)

    def _slider_to_sigma(self, s):
        lo, hi = self._SIGMA_RANGE
        return round(lo + s / 1000 * (hi - lo), 2)

    @property
    def _threshold(self):
        try:
            return float(self._thresh_edit.text())
        except ValueError:
            return self._slider_to_thresh(self._thresh_slider.value())

    @property
    def _min_sigma(self):
        return self._slider_to_sigma(self._sig_min_slider.value())

    @property
    def _max_sigma(self):
        return self._slider_to_sigma(self._sig_max_slider.value())

    def _current_params(self):
        return dict(
            threshold=self._threshold,
            min_sigma=self._min_sigma,
            max_sigma=self._max_sigma,
            bin_size=self._bin_slider.value(),
            blob_filter=self._blobf_slider.value(),
            layer_name=self._layer_combo.currentText(),
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_live_toggled(self, state):
        if self.viewer is None:
            return
        if state:
            self.viewer.dims.events.current_step.connect(self._on_frame_changed)
            self._debounce.start()
        else:
            try:
                self.viewer.dims.events.current_step.disconnect(self._on_frame_changed)
            except Exception:
                pass

    def _on_frame_changed(self, event=None):
        if self._live_cb.isChecked():
            self._debounce.start()

    def _on_thresh_slider(self, v):
        t = self._slider_to_thresh(v)
        self._thresh_edit.setText(f"{t:.6f}")
        self._debounce.start()

    def _on_thresh_edit(self):
        try:
            t = float(self._thresh_edit.text())
            if t <= 0:
                raise ValueError
        except ValueError:
            self._thresh_edit.setText(f"{self._threshold:.6f}")
            return
        self._thresh_slider.blockSignals(True)
        self._thresh_slider.setValue(self._thresh_to_slider(t))
        self._thresh_slider.blockSignals(False)
        self._debounce.start()

    def _on_sigma_changed(self):
        mn, mx = self._min_sigma, self._max_sigma
        self._sigma_lbl.setText(f"{mn:.1f} – {mx:.1f}")
        self._debounce.start()

    def _toggle_adv(self):
        self._adv_visible = not self._adv_visible
        self._adv_box.setVisible(self._adv_visible)
        self._adv_btn.setText(
            ("▼" if self._adv_visible else "▶") + "  Advanced filters"
        )

    def _on_run_full(self):
        self.run_full_detection.emit(self._current_params())

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _current_frame_index(self):
        if self.viewer is None or self._stack is None:
            return 0
        step = self.viewer.dims.current_step
        t = step[0] if len(step) > 0 else 0
        return int(np.clip(t, 0, len(self._stack) - 1))

    def _run_preview(self):
        if self._stack is None or self.viewer is None:
            return

        t = self._current_frame_index()
        frame = self._stack[t]
        self._frame_lbl.setText(f"frame {t}")
        params = self._current_params()

        blobs = blob_log(
            frame,
            min_sigma=params["min_sigma"],
            max_sigma=params["max_sigma"],
            threshold=params["threshold"],
        )

        if blobs.size:
            df = pd.DataFrame(blobs[:, :2], columns=["y", "x"])
            df["frame"] = t
            df_filt = filter_dense_blobs(df, params["bin_size"], params["blob_filter"])
        else:
            df_filt = pd.DataFrame(columns=["y", "x", "frame"])

        n_kept = len(df_filt)
        self._set_stats(n_kept)

        coords = np.column_stack([
            np.full(n_kept, t),
            df_filt["y"].to_numpy(),
            df_filt["x"].to_numpy(),
        ]) if n_kept else np.empty((0, 3))

        if self._preview_layer is not None and self._preview_layer in self.viewer.layers:
            self._preview_layer.data = coords
        else:
            self._preview_layer = self.viewer.add_points(
                coords,
                name="_preview_spots",
                size=28,
                face_color="transparent",
                border_color="yellow",
                border_width=0.1,
            )