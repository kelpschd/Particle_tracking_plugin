# tracks_table_widget.py

from qtpy import QtWidgets, QtCore
import numpy as np
import pandas as pd

# -------------------------
# Tracks table helpers
# -------------------------
def show_tracks_table(viewer):
    try:
        tracks_layer = viewer.layers['Tracks']
    except Exception:
        show_warning("No 'Tracks' layer found.")
        return
    tracks_df = tracks_layer_to_dataframe(tracks_layer)
    widget = TracksTableWidget(viewer, tracks_layer, tracks_df)
    viewer.window.add_dock_widget(widget, name="Tracks Table", area="right")

def _open_tracks_table(viewer):
    tracks_layers = [ly for ly in viewer.layers if ly.__class__.__name__ == "Tracks"]
    if not tracks_layers:
        show_warning("No Tracks layer found.")
        return
    tracks_layer = tracks_layers[0]
    tracks_df = tracks_layer_to_dataframe(tracks_layer)
    widget = TracksTableWidget(viewer, tracks_layer, tracks_df)
    viewer.window.add_dock_widget(widget, name="Tracks Table", area="right")

class TracksTableWidget(QtWidgets.QWidget):
    """
    Dockable widget showing a tracks DataFrame, with basic editing and
    sync to a napari Tracks layer.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    tracks_layer : napari.layers.Tracks
        The Tracks layer to link to.
    tracks_df : pandas.DataFrame
        Initial DataFrame of track data. This is what will be shown/edited.
        You can keep whatever columns you want (track_id, frame, x, y, etc.).
    """

    def __init__(self, viewer, tracks_layer, tracks_df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.tracks_layer = tracks_layer

        if tracks_df is None:
            tracks_df = pd.DataFrame()

        self._df = tracks_df.copy()

        self._building_table = False  # guard to avoid recursive signals

        self._init_ui()
        self._populate_table_from_df()

        # Optional: when the tracks layer changes, you could reconnect here
        # self.tracks_layer.events.data.connect(self._on_layer_data_changed)

    # ------------------------------------------------------------------
    # UI SETUP
    # ------------------------------------------------------------------
    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(1)

        # --- Table ---
        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked |
                                   QtWidgets.QAbstractItemView.SelectedClicked |
                                   QtWidgets.QAbstractItemView.EditKeyPressed)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Interactive
        )
        self.table.itemChanged.connect(self._on_item_changed)

        layout.addWidget(self.table, stretch=1)

        # --- Buttons ---
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(4)

        self.btn_reload = QtWidgets.QPushButton("Reload from layer")
        self.btn_apply = QtWidgets.QPushButton("Apply to layer")
        self.btn_delete = QtWidgets.QPushButton("Delete selected rows")

        self.btn_reload.clicked.connect(self._reload_from_layer)
        self.btn_apply.clicked.connect(self._apply_to_layer)
        self.btn_delete.clicked.connect(self._delete_selected_rows)

        btn_layout.addWidget(self.btn_reload)
        btn_layout.addWidget(self.btn_apply)
        btn_layout.addWidget(self.btn_delete)

        layout.addLayout(btn_layout)

        self.setLayout(layout)

    # ------------------------------------------------------------------
    # DATA <-> TABLE
    # ------------------------------------------------------------------
    def _populate_table_from_df(self):
        """Fill the QTableWidget from self._df."""
        self._building_table = True
        self.table.clear()

        n_rows, n_cols = self._df.shape
        self.table.setRowCount(n_rows)
        self.table.setColumnCount(n_cols)

        # Headers
        self.table.setHorizontalHeaderLabels(list(self._df.columns))

        # Cell values
        for i in range(n_rows):
            for j, col in enumerate(self._df.columns):
                val = self._df.iloc[i, j]
                item = QtWidgets.QTableWidgetItem(str(val))
                # Optional: align numeric columns
                if pd.api.types.is_numeric_dtype(self._df[col]):
                    item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.table.setItem(i, j, item)

        self._building_table = False

    def _update_df_from_table(self):
        """Update self._df from the current table contents."""
        cols = list(self._df.columns)
        n_rows = self.table.rowCount()

        data = {}
        for j, col in enumerate(cols):
            col_vals = []
            for i in range(n_rows):
                item = self.table.item(i, j)
                text = item.text() if item is not None else ""
                col_vals.append(text)
            data[col] = col_vals

        new_df = pd.DataFrame(data)

        # Try to cast numeric columns back to numeric types
        for col in cols:
            if pd.api.types.is_numeric_dtype(self._df[col]):
                new_df[col] = pd.to_numeric(new_df[col], errors="coerce")

        self._df = new_df

    # ------------------------------------------------------------------
    # BUTTON ACTIONS
    # ------------------------------------------------------------------
    def _reload_from_layer(self):
        """Reload table from the current Tracks layer data."""
        df = tracks_layer_to_dataframe(self.tracks_layer)
        self._df = df
        self._populate_table_from_df()

    def _apply_to_layer(self):
        """Push table edits back into the Tracks layer."""
        self._update_df_from_table()
        data, properties = dataframe_to_tracks_layer_data(self._df)
        self.tracks_layer.data = data
        self.tracks_layer.properties = properties

    def _delete_selected_rows(self):
        """Delete selected rows from the table + df (apply button will push to layer)."""
        selected_rows = sorted(
            {idx.row() for idx in self.table.selectedIndexes()},
            reverse=True,
        )
        if not selected_rows:
            return

        for row in selected_rows:
            self.table.removeRow(row)

        # Update df to match table after deletion
        self._update_df_from_table()

    # ------------------------------------------------------------------
    # TABLE CALLBACKS
    # ------------------------------------------------------------------
    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem):
        """Update DataFrame on single-cell edit."""
        if self._building_table:
            return  # ignore signals during bulk updates

        row = item.row()
        col = item.column()
        col_name = self._df.columns[col]

        text = item.text()
        val = text

        # try casting to numeric if original column was numeric
        if pd.api.types.is_numeric_dtype(self._df[col_name]):
            try:
                val = float(text)
            except ValueError:
                val = np.nan

        self._df.iloc[row, col] = val

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df.copy()

    @dataframe.setter
    def dataframe(self, df: pd.DataFrame):
        self._df = df.copy()
        self._populate_table_from_df()