# tracks_table_widget.py

from qtpy import QtWidgets, QtCore
import numpy as np
import pandas as pd


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


# ======================================================================
# Helper functions to convert between DataFrame <-> napari Tracks layer
# ======================================================================

def tracks_layer_to_dataframe(tracks_layer) -> pd.DataFrame:
    """
    Convert a napari Tracks layer to a DataFrame.

    Assumes:
        - tracks_layer.data is (N, D+1), where first column is time/frame
        - tracks_layer.properties contains 'track_id'

    Returns
    -------
    df : pandas.DataFrame
        Columns: ['track_id', 'frame', 'y', 'x'] or ['track_id', 'frame', 'z', 'y', 'x']
        depending on dimensionality. Any extra properties are added as columns.
    """
    data = np.asarray(tracks_layer.data)
    props = dict(tracks_layer.properties)

    n_coords = data.shape[1] - 1  # excluding frame/time column
    frame = data[:, 0]

    if n_coords == 2:
        y, x = data[:, 1], data[:, 2]
        base = {"frame": frame, "y": y, "x": x}
    elif n_coords == 3:
        z, y, x = data[:, 1], data[:, 2], data[:, 3]
        base = {"frame": frame, "z": z, "y": y, "x": x}
    else:
        # Fallback: create generic coord columns
        base = {"frame": frame}
        for i in range(n_coords):
            base[f"coord_{i}"] = data[:, i + 1]

    # Add properties (must be length N)
    for key, values in props.items():
        base[key] = values

    df = pd.DataFrame(base)

    # If there is no explicit track_id in properties, synthesize one
    if "track_id" not in df.columns:
        df["track_id"] = np.zeros(len(df), dtype=int)

    return df


def dataframe_to_tracks_layer_data(df: pd.DataFrame):
    """
    Convert a DataFrame back to (data, properties) for a napari Tracks layer.

    Looks for these columns (if present):
        - 'frame' or 't'
        - 'z' (optional)
        - 'y'
        - 'x'
        - 'track_id' (as a property)
    Any other columns are returned as extra properties.

    Returns
    -------
    data : (N, D+1) ndarray
    properties : dict of {name: 1D array}
    """
    df = df.copy()

    # time/frame column
    if "frame" in df.columns:
        t = df["frame"].to_numpy()
    elif "t" in df.columns:
        t = df["t"].to_numpy()
    else:
        # if no time, just zeros
        t = np.zeros(len(df), dtype=float)

    coords = []

    # determine order: (z), y, x if present
    coord_cols = []
    if "z" in df.columns:
        coord_cols.append("z")
    if "y" in df.columns:
        coord_cols.append("y")
    if "x" in df.columns:
        coord_cols.append("x")

    if coord_cols:
        for c in coord_cols:
            coords.append(df[c].to_numpy())
    else:
        # fallback: use any 'coord_*' columns
        coord_cols = [c for c in df.columns if c.startswith("coord_")]
        coord_cols = sorted(coord_cols)
        for c in coord_cols:
            coords.append(df[c].to_numpy())

    if coords:
        coords_arr = np.vstack(coords).T  # shape (N, D)
    else:
        coords_arr = np.zeros((len(df), 0))

    data = np.column_stack([t, coords_arr])

    # properties: everything that is not a coordinate or frame
    exclude_cols = set(["frame", "t", "z", "y", "x"])
    exclude_cols.update(c for c in df.columns if c.startswith("coord_"))

    properties = {}
    for col in df.columns:
        if col in exclude_cols:
            continue
        properties[col] = df[col].to_numpy()

    return data, properties
