# _tabs.py
import numpy as np
import pandas as pd

from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QPushButton, QLabel
from napari import current_viewer
from napari.utils.notifications import show_info, show_warning, show_error
from napari.qt.threading import thread_worker

from ._detection_preview_widget import DetectionPreviewWidget
from ._tracking_widget import tracking_widget, TracksListWidget
from ._image_import_widget import ImageImportWidget
from .tracks_table_widget import TracksTableWidget
from ._helpers import (
    tracks_layer_to_dataframe,
    dataframe_to_tracks_layer_data,
    image_layers,
    array_from_layer,
    pt_meta_from_layer,
    pt_meta_dict,
    layer_name,
    image_root_from_path,
)
from ._processing import detect_puncta_dask, filter_dense_blobs
from ._export import ExportWidget
from ._validation_state import get_or_create_validation_state, init_validation_from_tracks

_TRACKS_LIST_WIDGETS: dict[int, TracksListWidget] = {}


def make_plugin_gui(viewer=None, **_):
    viewer = viewer or current_viewer()
    get_or_create_validation_state(viewer)
    tracking_widget.viewer.value = viewer

    tabs = QTabWidget()
    tabs.setTabPosition(QTabWidget.North)

    # --- Import tab ---
    import_page = QWidget()
    imp_layout = QVBoxLayout(import_page)
    imp_layout.setContentsMargins(0, 0, 0, 0)
    imp_layout.setSpacing(0)
    import_widget = ImageImportWidget(viewer=viewer)
    imp_layout.addWidget(import_widget)
    tabs.addTab(import_page, "Import")

    # --- Detection tab ---
    detection_page = QWidget()
    det_layout = QVBoxLayout(detection_page)
    det_layout.setContentsMargins(0, 0, 0, 0)
    det_layout.setSpacing(0)
    detection_widget = DetectionPreviewWidget(viewer=viewer)
    det_layout.addWidget(detection_widget)
    tabs.addTab(detection_page, "Detection")

    # --- Tracking tab ---
    tracking_page = QWidget()
    trk_layout = QVBoxLayout(tracking_page)
    trk_layout.setContentsMargins(0, 0, 0, 0)
    trk_layout.setSpacing(0)
    trk_layout.addWidget(tracking_widget.native)

    open_tracks_list_btn = QPushButton("Open Tracks List")
    open_tracks_list_btn.setToolTip(
        "Open the validation queue (pending tracks) as a docked panel."
    )
    trk_layout.addWidget(open_tracks_list_btn)
    trk_layout.addStretch(1)
    tabs.addTab(tracking_page, "Tracking")

    # --- Validated tracks tab ---
    validated_page = QWidget()
    val_layout = QVBoxLayout(validated_page)
    val_layout.setContentsMargins(0, 0, 0, 0)
    val_layout.setSpacing(0)
    val_layout.addWidget(QLabel("Validated (kept) tracks"))
    state = get_or_create_validation_state(viewer)
    validated_table = TracksTableWidget(
        viewer=viewer,
        tracks_layer=None,
        tracks_df=state.kept_df,
    )
    val_layout.addWidget(validated_table)
    tabs.addTab(validated_page, "Validated Tracks")

    # --- Export tab ---
    export_page = QWidget()
    exp_layout = QVBoxLayout(export_page)
    exp_layout.setContentsMargins(0, 0, 0, 0)
    exp_layout.setSpacing(0)
    export_widget = ExportWidget(viewer=viewer, image_import_widget=import_widget)
    exp_layout.addWidget(export_widget)
    tabs.addTab(export_page, "Export")

    # ------------------------------------------------------------------
    # Full detection — wired to DetectionPreviewWidget.run_full_detection
    # ------------------------------------------------------------------

    def _on_run_full(params: dict):
        layer_nm = params.get("layer_name", "")
        try:
            layer = viewer.layers[layer_nm]
        except KeyError:
            show_warning(f"Layer '{layer_nm}' not found. Select a layer in the Detection tab.")
            return

        try:
            arr = array_from_layer(layer)
        except ValueError as e:
            show_error(str(e))
            return

        threshold   = params["threshold"]
        min_sigma   = params["min_sigma"]
        max_sigma   = params["max_sigma"]
        bin_size    = params["bin_size"]
        blob_filter = params["blob_filter"]

        pt_meta    = pt_meta_from_layer(layer)
        image_root = pt_meta.get("image_root") or image_root_from_path(
            pt_meta.get("image_path", layer.name)
        )
        channel_label = pt_meta.get("channel_label")
        source_name   = layer.name

        @thread_worker
        def _run(stack):
            return detect_puncta_dask(
                stack, threshold=threshold, min_sigma=min_sigma, max_sigma=max_sigma
            )

        worker = _run(arr)

        def _on_error(err):
            show_error(f"Detection failed: {err}")

        def _on_done(det_df: pd.DataFrame):
            if det_df is None or len(det_df) == 0:
                show_info("No puncta detected with current parameters.")
                return

            n_before = len(det_df)
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
            props = {"size": det_df["size"].to_numpy()} if "size" in det_df.columns else {}

            run_params = {
                "threshold":                float(threshold),
                "min_sigma":                float(min_sigma),
                "max_sigma":                float(max_sigma),
                "bin_size":                 int(bin_size),
                "blob_filter":              int(blob_filter),
                "n_detected_before_filter": int(n_before),
                "n_detected_after_filter":  int(n_after),
            }

            puncta_name = layer_name(image_root, channel_label, "puncta")

            # remove stale preview layer before adding the real one
            if (detection_widget._preview_layer is not None
                    and detection_widget._preview_layer in viewer.layers):
                viewer.layers.remove(detection_widget._preview_layer)
                detection_widget._preview_layer = None

            pts_layer = viewer.add_points(
                points_data,
                name=puncta_name,
                size=30,
                face_color="transparent",
                properties=props,
                border_color="red",
                border_width=0.1,
                metadata=pt_meta_dict(
                    role="detected_spots",
                    image_root=image_root,
                    channel_label=channel_label,
                    source_layer=source_name,
                    run_params=run_params,
                ),
            )

            show_info(
                f"Detection complete: {n_after} puncta"
                f" (filtered from {n_before}) → '{puncta_name}'"
            )

        worker.errored.connect(_on_error)
        worker.returned.connect(_on_done)
        worker.start()

    detection_widget.run_full_detection.connect(_on_run_full)

    # ------------------------------------------------------------------
    # Tracks list singleton
    # ------------------------------------------------------------------

    def _open_tracks_list_singleton():
        state = get_or_create_validation_state(viewer)
        if state.is_empty:
            tracks_layers = [
                ly for ly in viewer.layers if ly.__class__.__name__ == "Tracks"
            ]
            if not tracks_layers:
                show_warning("Run tracking first to populate the validation queue.")
                return
            df = tracks_layer_to_dataframe(tracks_layers[0])
            init_validation_from_tracks(viewer, df)
        vid = id(viewer)
        existing = _TRACKS_LIST_WIDGETS.get(vid)
        widget_alive = (
            existing is not None
            and hasattr(existing, "isVisible")
            and existing.isVisible()
        )
        if widget_alive:
            existing.refresh_from_state()
            return
        widget = TracksListWidget(viewer=viewer)
        viewer.window.add_dock_widget(widget, name="Tracks List", area="right")
        _TRACKS_LIST_WIDGETS[vid] = widget
        widget.track_kept.connect(
            lambda: validated_table.__setattr__(
                "dataframe", get_or_create_validation_state(viewer).kept_df
            )
        )

    open_tracks_list_btn.clicked.connect(_open_tracks_list_singleton)

    # ------------------------------------------------------------------
    # Layer event hooks
    # ------------------------------------------------------------------

    def _on_layer_inserted(event):
        layer = event.value
        if layer.__class__.__name__ == "Tracks":
            state = get_or_create_validation_state(viewer)
            if state.is_empty:
                df = tracks_layer_to_dataframe(layer)
                init_validation_from_tracks(viewer, df)
            existing = _TRACKS_LIST_WIDGETS.get(id(viewer))
            if existing is not None and hasattr(existing, "isVisible") and existing.isVisible():
                existing.refresh_from_state()

    def _refresh_tracking_choices(*_):
        try:
            tracking_widget.reset_choices()
        except Exception:
            pass

    viewer.layers.events.inserted.connect(_on_layer_inserted)
    viewer.layers.events.inserted.connect(_refresh_tracking_choices)
    viewer.layers.events.removed.connect(_refresh_tracking_choices)
    viewer.layers.events.reordered.connect(_refresh_tracking_choices)
    viewer.layers.events.changed.connect(_refresh_tracking_choices)

    return tabs