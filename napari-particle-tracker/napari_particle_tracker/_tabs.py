# _tabs.py
import pandas as pd
from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QPushButton, QLabel
from magicgui.widgets import Container, Label
from napari import current_viewer
from napari.utils.notifications import show_warning

from ._widget import DetectionWidget
from ._tracking_widget import tracking_widget, TracksListWidget
from ._image_import_widget import ImageImportWidget
from .tracks_table_widget import TracksTableWidget
from ._helpers import tracks_layer_to_dataframe, dataframe_to_tracks_layer_data
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
    detection_widget = DetectionWidget(viewer=viewer)
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
    export_widget = ExportWidget(
        viewer=viewer,
        image_import_widget=import_widget,
    )
    exp_layout.addWidget(export_widget)
    tabs.addTab(export_page, "Export")

    # --- Tracks list singleton ---
    def _open_tracks_list_singleton():
        state = get_or_create_validation_state(viewer)
        if state.is_empty:
            tracks_layers = [
                ly for ly in viewer.layers
                if ly.__class__.__name__ == "Tracks"
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

    # --- Layer event hooks ---
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