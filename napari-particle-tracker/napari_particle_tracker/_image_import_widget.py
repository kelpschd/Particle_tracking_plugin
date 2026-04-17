# _image_import_widget.py
import os

import numpy as np
import pandas as pd
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QGroupBox,
)
from qtpy.QtCore import Signal

from ._helpers import (
    pt_meta_dict,
    layer_name,
    vispy_colormap_from_rgb,
    image_root_from_path,
    extract_nd2_channel_info,
)
from ._validation_state import init_validation_from_tracks

try:
    import nd2
except ImportError:
    nd2 = None

# --- CSV column specs -----------------------------------------------------------
SPOTS_REQUIRED    = {"x", "y", "frame"}
SPOTS_RECOMMENDED = {"spot_id", "channel", "intensity", "z"}

TRACKS_REQUIRED    = {"x", "y", "frame", "track_id"}
TRACKS_RECOMMENDED = {"spot_id", "channel", "intensity", "z"}

# Columns kept as Points layer properties; everything else goes to metadata only.
TRACKS_CORE_COLS = {"particle", "frame", "y", "x", "track_id", "z",
                    "intensity", "channel", "spot_id"}
SPOTS_CORE_COLS  = {"particle", "frame", "y", "x", "z",
                    "intensity", "channel", "spot_id"}
# -------------------------------------------------------------------------------


def _load_csv_flexible(path):
    """Read a CSV, stripping whitespace from column names."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _check_columns(df, required, recommended, label):
    cols = set(df.columns)
    return required - cols, recommended - cols


class _CsvImportRow(QWidget):
    """A compact browse-row for a single CSV type."""

    def __init__(self, label_text, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self.label = QLabel(label_text)
        self.label.setFixedWidth(52)
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText(f"Select a {label_text.lower()} CSV…")
        self.browse_btn = QPushButton("Browse…")

        row.addWidget(self.label)
        row.addWidget(self.path_edit)
        row.addWidget(self.browse_btn)

    @property
    def path(self):
        p = self.path_edit.text().strip()
        return p if p else None

    def set_path(self, p):
        self.path_edit.setText(p)


class ImageImportWidget(QWidget):
    layers_loaded = Signal()

    def __init__(self, viewer=None, parent=None):
        super().__init__(parent)
        self.viewer = viewer

        self.nd2_path = None
        self._cached_array = None
        self._channel_axis = None
        self._n_channels = 0
        self._channel_info = []

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # --- ND2 group ---
        nd2_group = QGroupBox("ND2 Import")
        nd2_layout = QVBoxLayout(nd2_group)

        nd2_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Select a .nd2 file...")
        browse_nd2_btn = QPushButton("Browse…")
        browse_nd2_btn.clicked.connect(self._on_browse_nd2)
        nd2_row.addWidget(self.path_edit)
        nd2_row.addWidget(browse_nd2_btn)
        nd2_layout.addLayout(nd2_row)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        nd2_layout.addWidget(self.info_label)

        load_nd2_btn = QPushButton("Load channels into viewer")
        load_nd2_btn.clicked.connect(self._on_load_channels)
        nd2_layout.addWidget(load_nd2_btn)

        layout.addWidget(nd2_group)

        # --- CSV group ---
        csv_group = QGroupBox("CSV Import (Spots / Tracks)")
        csv_layout = QVBoxLayout(csv_group)

        csv_row = QHBoxLayout()
        self._csv_path_edit = QLineEdit()
        self._csv_path_edit.setReadOnly(True)
        self._csv_path_edit.setPlaceholderText("Select a spots or tracks CSV…")
        browse_csv_btn = QPushButton("Browse…")
        browse_csv_btn.clicked.connect(self._on_browse_csv)
        csv_row.addWidget(self._csv_path_edit)
        csv_row.addWidget(browse_csv_btn)
        csv_layout.addLayout(csv_row)

        self.csv_info_label = QLabel("")
        self.csv_info_label.setWordWrap(True)
        csv_layout.addWidget(self.csv_info_label)

        load_csv_btn = QPushButton("Load CSV into viewer")
        load_csv_btn.clicked.connect(self._on_load_csvs)
        csv_layout.addWidget(load_csv_btn)

        layout.addWidget(csv_group)

        layout.addStretch(1)

    # ------------------------------------------------------------------
    # ND2 helpers
    # ------------------------------------------------------------------

    def _on_browse_nd2(self):
        from napari.utils.notifications import show_error

        if nd2 is None:
            show_error("The 'nd2' package is not installed. Please `pip install nd2`.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select ND2 file", "", "ND2 files (*.nd2);;All files (*.*)",
        )
        if not path:
            return

        self.nd2_path = path
        self.path_edit.setText(path)

        try:
            arr = nd2.imread(path)
        except Exception as e:
            show_error(f"Failed to read ND2 file:\n{e}")
            self.info_label.setText("Failed to read ND2 file.")
            self._cached_array = None
            self._n_channels = 0
            self._channel_axis = None
            self._channel_info = []
            return

        self._cached_array = arr
        shape = arr.shape
        ndim = arr.ndim

        self._channel_axis = None
        self._n_channels = 1

        if ndim == 4:
            _, c, _, _ = shape
            self._channel_axis = 1
            self._n_channels = c
        elif ndim == 3:
            c = shape[0]
            if c <= 8:
                self._channel_axis = 0
                self._n_channels = c

        self._channel_info = extract_nd2_channel_info(
            path, nd2, fallback_n_channels=self._n_channels,
        )

        info = f"Loaded {os.path.basename(path)}\nshape: {shape}\n"
        if self._channel_axis is None:
            info += "Detected single channel (or no explicit channel axis).\n"
        else:
            info += f"Detected {self._n_channels} channel(s) on axis {self._channel_axis}.\n"

        if self._channel_info:
            info += "\nChannel labels:\n" + "\n".join(
                f"  Ch {i}: {ch['label']}" for i, ch in enumerate(self._channel_info)
            )

        self.info_label.setText(info)

    def _iter_channel_images(self, arr):
        if self._channel_axis is None:
            yield 0, arr
            return
        for ch in range(self._n_channels):
            sl = [slice(None)] * arr.ndim
            sl[self._channel_axis] = ch
            yield ch, arr[tuple(sl)]

    def _on_load_channels(self):
        from napari.utils.notifications import show_error, show_warning

        if self.viewer is None:
            show_error("No napari viewer attached.")
            return
        if self._cached_array is None:
            show_warning("No ND2 file loaded yet.")
            return

        image_root = image_root_from_path(self.nd2_path or "image.nd2")

        for ch, img in self._iter_channel_images(self._cached_array):
            info = self._channel_info[ch] if ch < len(self._channel_info) else {
                "index": ch, "label": f"Ch {ch}", "rgb": None, "name": f"Ch {ch}",
            }

            name = layer_name(image_root, info["label"], "raw")
            kwargs = {
                "name": name,
                "metadata": pt_meta_dict(
                    role="raw_image",
                    image_path=self.nd2_path,
                    image_root=image_root,
                    channel_index=info["index"],
                    channel_label=info["label"],
                    channel_name=info["name"],
                    channel_rgb=info["rgb"],
                ),
            }

            if info["rgb"] is not None:
                kwargs["colormap"] = vispy_colormap_from_rgb(info["rgb"], info["label"])

            self.viewer.add_image(img, **kwargs)

        self.layers_loaded.emit()

    # ------------------------------------------------------------------
    # CSV browse slots
    # ------------------------------------------------------------------

    def _on_browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV file", "", "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return
        self._csv_path_edit.setText(path)
        # Preview columns and auto-detect type
        try:
            df = _load_csv_flexible(path)
            has_tracks = "track_id" in df.columns
            detected = "spots + tracks" if has_tracks else "spots only"
            self.csv_info_label.setText(
                f"Detected: {detected}\n"
                f"{len(df)} rows | cols: {', '.join(df.columns)}"
            )
        except Exception as e:
            self.csv_info_label.setText(f"Failed to read CSV: {e}")

    # ------------------------------------------------------------------
    # CSV load
    # ------------------------------------------------------------------

    def _on_load_csvs(self):
        from napari.utils.notifications import show_error, show_warning

        if self.viewer is None:
            show_error("No napari viewer attached.")
            return

        csv_path = self._csv_path_edit.text().strip() or None

        if csv_path is None:
            show_warning("No CSV file selected.")
            return

        image_root = image_root_from_path(self.nd2_path or "image.nd2")
        loaded_any = False

        try:
            df = _load_csv_flexible(csv_path)
            has_tracks = "track_id" in df.columns

            if has_tracks:
                missing_req, _ = _check_columns(df, TRACKS_REQUIRED, TRACKS_RECOMMENDED, "Tracks")
            else:
                missing_req, _ = _check_columns(df, SPOTS_REQUIRED, SPOTS_RECOMMENDED, "Spots")

            if missing_req:
                show_warning(
                    f"CSV is missing required columns: "
                    f"{', '.join(sorted(missing_req))}. "
                    "Attempting to load with available columns."
                )

            core_cols = TRACKS_CORE_COLS if has_tracks else SPOTS_CORE_COLS
            core_df, meta_df = self._split_core_meta(df, core_cols)

            role = "track_data" if has_tracks else "detected_spots"
            ok = self._add_puncta_layer(core_df, meta_df, image_root, csv_path, role)
            loaded_any = loaded_any or ok

            if has_tracks:
                track_data = self._build_track_data(core_df)
                if track_data is not None:
                    pipeline_params = meta_df.iloc[0].to_dict() if len(meta_df.columns) else {}
                    track_name = layer_name(image_root, None, "tracks")
                    self.viewer.add_tracks(
                        track_data,
                        name=track_name,
                        tail_length=10,
                        head_length=0,
                        blending="translucent",
                        metadata=pt_meta_dict(
                            role="tracks",
                            image_root=image_root,
                            csv_path=csv_path,
                            dataframe=df.to_dict(orient="list"),
                            pipeline_params=pipeline_params,
                        ),
                    )
                    tracks_df = core_df[
                        [c for c in ["particle", "frame", "y", "x"] if c in core_df.columns]
                    ].copy()
                    if "particle" not in tracks_df.columns and "track_id" in tracks_df.columns:
                        tracks_df = tracks_df.rename(columns={"track_id": "particle"})
                    try:
                        init_validation_from_tracks(self.viewer, tracks_df)
                    except Exception as e:
                        show_warning(f"Could not initialise validation state: {e}")
                    loaded_any = True

        except Exception as e:
            show_error(f"Failed to load CSV:\n{e}")

        if loaded_any:
            self.layers_loaded.emit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_puncta_layer(self, core_df, meta_df, image_root, csv_path, role):
        """
        Add a Points layer styled identically to the puncta layer produced by
        particle_detection_widget: transparent fill, red border, size 30.
        Returns True if added, False otherwise.
        """
        coords = self._build_coords(core_df)
        if coords is None:
            from napari.utils.notifications import show_error
            show_error("CSV has no usable positional columns (need at least x and y).")
            return False

        props = self._layer_properties(core_df)
        pipeline_params = meta_df.iloc[0].to_dict() if len(meta_df.columns) else {}
        puncta_name = layer_name(image_root, None, "puncta")

        self.viewer.add_points(
            coords,
            name=puncta_name,
            size=30,
            face_color="transparent",
            border_color="red",
            border_width=0.1,
            properties=props if props else None,
            metadata=pt_meta_dict(
                role=role,
                image_root=image_root,
                csv_path=csv_path,
                dataframe=core_df.to_dict(orient="list"),
                pipeline_params=pipeline_params,
            ),
        )
        return True

    @staticmethod
    def _build_track_data(core_df):
        """
        Build the (N, 4) array napari's add_tracks() expects: (track_id, frame, y, x).
        Returns None if required columns are absent.
        """
        needed = {"track_id", "frame", "y", "x"}
        if not needed.issubset(core_df.columns):
            return None

        df = core_df.dropna(subset=list(needed)).copy()
        df["track_id"] = df["track_id"].astype(int)
        df["frame"]    = df["frame"].astype(int)
        df = df.sort_values(["track_id", "frame"])
        return df[["track_id", "frame", "y", "x"]].to_numpy(dtype=float)

    @staticmethod
    def _split_core_meta(df, core_cols):
        """
        Split df into (core_df, meta_df).
        core_df  — columns in core_cols (coords + layer properties).
        meta_df  — remaining columns (pipeline params, settings, etc.).
        """
        present_core = [c for c in df.columns if c in core_cols]
        present_meta = [c for c in df.columns if c not in core_cols]
        return df[present_core], df[present_meta]

    @staticmethod
    def _layer_properties(core_df):
        """
        Return a dict of non-positional core columns for napari layer properties.
        Skips x, y, z, frame since those are encoded in the coords array.
        """
        skip = {"x", "y", "z", "frame"}
        return {
            col: core_df[col].to_numpy()
            for col in core_df.columns
            if col not in skip
        }

    @staticmethod
    def _build_coords(df):
        """
        Build an (N, ndim) napari-ordered coordinate array: (frame, [z,] y, x).
        Returns None if x or y is absent.
        """
        if "x" not in df.columns or "y" not in df.columns:
            return None

        parts = []
        if "frame" in df.columns:
            parts.append(df["frame"].to_numpy(dtype=float))
        if "z" in df.columns:
            parts.append(df["z"].to_numpy(dtype=float))
        parts.append(df["y"].to_numpy(dtype=float))
        parts.append(df["x"].to_numpy(dtype=float))

        return np.stack(parts, axis=1)