# _tabs.py
import pandas as pd
from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QPushButton, QLabel
from magicgui.widgets import Container, Label
from napari import current_viewer
from napari.utils.notifications import show_warning

from ._widget import particle_detection_widget
from ._tracking_widget import tracking_widget, TracksListWidget
from ._image_import_widget import ImageImportWidget
from .tracks_table_widget import TracksTableWidget
from ._helpers import tracks_layer_to_dataframe, dataframe_to_tracks_layer_data
from ._export import ExportWidget
from ._validation_state import get_or_create_validation_state, init_validation_from_tracks

# keep the layer dropdowns fresh
def _refresh_layer_choices(*_):
    try:
        particle_detection_widget.reset_choices()
    except Exception:
        pass
    try:
        tracking_widget.reset_choices()
    except Exception:
        pass

def make_plugin_gui(viewer=None, **_):
    viewer = viewer or current_viewer()
    state = get_or_create_validation_state(viewer)

    particle_detection_widget.viewer.value = viewer
    tracking_widget.viewer.value = viewer 

    tabs = QTabWidget()
    tabs.setTabPosition(QTabWidget.North)

    # Import page
    import_page = QWidget()
    imp_layout = QVBoxLayout(import_page)
    imp_layout.setContentsMargins(0, 0, 0, 0)
    imp_layout.setSpacing(0)
    import_widget = ImageImportWidget(viewer=viewer) 
    import_widget.layers_loaded.connect(_refresh_layer_choices)
    imp_layout.addWidget(import_widget)
    tabs.addTab(import_page, "Import")

    # Detection page
    detection_page = QWidget()
    det_layout = QVBoxLayout(detection_page)
    det_layout.setContentsMargins(0, 0, 0, 0)
    det_layout.setSpacing(0)
    det_layout.addWidget(particle_detection_widget.native)
    det_layout.addStretch(1)
    tabs.addTab(detection_page, "Detection")

    # Tracking page
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

    # Validated tracks page
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

    # Export tab
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
        try:
            qt_win = viewer.window._qt_window
        except Exception:
            qt_win = viewer.window
        existing = getattr(qt_win, "_tracks_list_widget", None)
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
        setattr(qt_win, "_tracks_list_widget", widget)
        widget.track_kept.connect(
            lambda: validated_table.__setattr__(
                "dataframe", get_or_create_validation_state(viewer).kept_df
            )
        )

    open_tracks_list_btn.clicked.connect(_open_tracks_list_singleton)

    def _on_layer_inserted(event):
        layer = event.value
        if layer.__class__.__name__ == "Tracks":
            state = get_or_create_validation_state(viewer)
            if state.is_empty:
                df = tracks_layer_to_dataframe(layer)
                init_validation_from_tracks(viewer, df)
            try:
                existing = viewer.window._qt_window._tracks_list_widget
            except Exception:
                existing = getattr(getattr(viewer.window, "_qt_window", viewer.window),
                                   "_tracks_list_widget", None)
            if existing is not None and hasattr(existing, "isVisible") and existing.isVisible():
                existing.refresh_from_state()

    viewer.layers.events.inserted.connect(_on_layer_inserted)
    viewer.layers.events.inserted.connect(_refresh_layer_choices)
    viewer.layers.events.removed.connect(_refresh_layer_choices)
    viewer.layers.events.reordered.connect(_refresh_layer_choices)
    viewer.layers.events.changed.connect(_refresh_layer_choices)

    _refresh_layer_choices()

    return tabs