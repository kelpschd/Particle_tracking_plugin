# _tabs.py

import pandas as pd
from qtpy.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QTabWidget, QPushButton, QSizePolicy
from magicgui.widgets import Container, Label
from napari import current_viewer
from napari.utils.notifications import show_warning

from ._widget import particle_detection_widget, tracking_widget
from ._image_import_widget import ImageImportWidget
from .tracks_table_widget import TracksTableWidget, tracks_layer_to_dataframe

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
    trk_layout.addStretch(1)
    tabs.addTab(tracking_page, "Tracking")

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
        tracks_table_widget.dataframe = df

    # Button triggers reload from active Tracks layer
    link_btn.clicked.connect(_link_tracks_table)

    # Optional: auto-load if there's already a Tracks layer when the plugin opens
    _link_tracks_table()

    tabs.addTab(tracks_table_page, "Tracks Table")

    # Visualization placeholder
    viz_page = QWidget()
    viz_layout = QVBoxLayout(viz_page)
    viz_layout.setContentsMargins(0, 0, 0, 0)
    viz_layout.setSpacing(0)
    viz_layout.addWidget(Container(widgets=[Label(value="Visualization page coming soon")]).native)
    tabs.addTab(viz_page, "Visualization")

    return tabs
