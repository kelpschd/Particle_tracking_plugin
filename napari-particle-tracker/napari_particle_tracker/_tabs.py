# _tabs.py
import pandas as pd
from qtpy.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QTabWidget, QPushButton, QSizePolicy
from magicgui.widgets import Container, Label
from napari import current_viewer
from napari.utils.notifications import show_warning

from ._widget import particle_detection_widget
from ._tracking_widget import tracking_widget, TracksListWidget
from ._image_import_widget import ImageImportWidget
from .tracks_table_widget import TracksTableWidget
from ._helpers import tracks_layer_to_dataframe, dataframe_to_tracks_layer_data
from ._export import ExportWidget

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

    # --- Controls to open the dockable TracksListWidget ---
    open_tracks_list_btn = QPushButton("Open Tracks List")
    open_tracks_list_btn.setToolTip("Open a docked list of tracks (select a track to zoom/preview).")
    trk_layout.addWidget(open_tracks_list_btn)

    trk_layout.addStretch(1)
    tabs.addTab(tracking_page, "Tracking")

    # Find first Tracks layer and return (layer, dataframe) !move to helpers
    def _get_first_tracks_layer_and_df():
        tracks_layers = [ly for ly in viewer.layers if ly.__class__.__name__ == "Tracks"]
        if not tracks_layers:
            return None, None
        layer = tracks_layers[0]
        df = tracks_layer_to_dataframe(layer)
        return layer, df

    # keep a single persistent tracks list on viewer.window !move to helpers
    def _open_tracks_list_singleton():
        # reuse if exists
        existing = getattr(viewer.window, "_tracks_list_widget", None)
        if existing is not None:
            # refresh its contents
            layer, df = _get_first_tracks_layer_and_df()
            if df is None or df.empty:
                show_warning("No Tracks layer found to populate the list.")
                return
            existing.set_tracks(df)
            return

        layer, df = _get_first_tracks_layer_and_df()
        if layer is None or df is None or df.empty:
            show_warning("No Tracks layer found to populate the list.")
            return

        widget = TracksListWidget(viewer=viewer, tracks_df=df)
        viewer.window.add_dock_widget(widget, name="Tracks List", area="right")
        viewer.window._tracks_list_widget = widget

    open_tracks_list_btn.clicked.connect(_open_tracks_list_singleton)

    # refresh when layers are added/removed/renamed
    viewer.layers.events.inserted.connect(_refresh_layer_choices)
    viewer.layers.events.removed.connect(_refresh_layer_choices)
    viewer.layers.events.reordered.connect(_refresh_layer_choices)
    viewer.layers.events.changed.connect(_refresh_layer_choices)

    # also refresh once on startup
    _refresh_layer_choices()

    # ------------------------------------------------------------------
    # Tracks Table page
    # ------------------------------------------------------------------
    tracks_table_page = QWidget()
    ttable_layout = QVBoxLayout(tracks_table_page)
    ttable_layout.setContentsMargins(0, 0, 0, 0)
    ttable_layout.setSpacing(0)

    # Button to link to current Tracks layer
    link_btn = QPushButton("Load table from active Tracks layer")
    ttable_layout.addWidget(link_btn)

    # Start with an empty dataframe and no layer
    empty_df = pd.DataFrame()
    dummy_layer = None

    tracks_table_widget = TracksTableWidget(
        viewer=viewer,
        tracks_layer=dummy_layer,
        tracks_df=empty_df,
    )
    ttable_layout.addWidget(tracks_table_widget)

    def _link_tracks_table():
        """Bind the table to the first Tracks layer, if one exists."""
        tracks_layers = [ly for ly in viewer.layers if ly.__class__.__name__ == "Tracks"]
        if not tracks_layers:
            # You can comment this out if the warning gets annoying on startup
            show_warning("No Tracks layer found.")
            return

        layer = tracks_layers[0]
        df = tracks_layer_to_dataframe(layer)

        tracks_table_widget.tracks_layer = layer
        # depending on your TracksTableWidget API you might need to call a setter or method;
        # you currently set .dataframe property in other code, so keep that usage:
        tracks_table_widget.dataframe = df

    # Button triggers reload from active Tracks layer
    link_btn.clicked.connect(_link_tracks_table)

    # Optional: auto-load if there's already a Tracks layer when the plugin opens
    _link_tracks_table()

    tabs.addTab(tracks_table_page, "Tracks Table")

    # Export tab
    export_page = QWidget()
    exp_layout = QVBoxLayout(export_page)
    exp_layout.setContentsMargins(0, 0, 0, 0)
    exp_layout.setSpacing(0)

    export_widget = ExportWidget(
        viewer=viewer,
        image_import_widget=import_widget,
        tracks_table_widget=tracks_table_widget,
    )
    exp_layout.addWidget(export_widget)
    tabs.addTab(export_page, "Export")

    return tabs