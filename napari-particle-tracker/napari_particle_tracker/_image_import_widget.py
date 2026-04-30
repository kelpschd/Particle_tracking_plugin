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

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import czifile
except ImportError:
    czifile = None

try:
    import readlif
    from readlif.reader import LifFile
except ImportError:
    readlif = None
    LifFile = None

try:
    import oiffile
except ImportError:
    oiffile = None

# --- CSV column specs -----------------------------------------------------------
SPOTS_REQUIRED    = {"x", "y", "frame"}
SPOTS_RECOMMENDED = {"spot_id", "channel", "intensity", "z"}

TRACKS_REQUIRED    = {"x", "y", "frame", "track_id"}
TRACKS_RECOMMENDED = {"spot_id", "channel", "intensity", "z"}

TRACKS_CORE_COLS = {"particle", "frame", "y", "x", "track_id", "z",
                    "intensity", "channel", "spot_id"}
SPOTS_CORE_COLS  = {"particle", "frame", "y", "x", "z",
                    "intensity", "channel", "spot_id"}
# -------------------------------------------------------------------------------

# File-type filter string used in the browse dialog
_IMAGE_FILTER = (
    "All supported images (*.nd2 *.tif *.tiff *.czi *.lif *.oib *.oif);;"
    "ND2 files (*.nd2);;"
    "TIFF files (*.tif *.tiff);;"
    "Zeiss CZI (*.czi);;"
    "Leica LIF (*.lif);;"
    "Olympus OIB/OIF (*.oib *.oif);;"
    "All files (*.*)"
)


# ==============================================================================
# Per-format readers
# Each reader returns:
#   channels : list of dicts  {"index", "label", "name", "rgb", "array"}
#   shape    : tuple of the raw array shape
# ==============================================================================

def _read_nd2(path):
    if nd2 is None:
        raise ImportError("nd2 package not installed. Run: pip install nd2")
    arr = nd2.imread(path)
    shape = arr.shape
    ndim = arr.ndim
    channel_axis, n_channels = None, 1
    if ndim == 4:
        channel_axis, n_channels = 1, shape[1]
    elif ndim == 3 and shape[0] <= 8:
        channel_axis, n_channels = 0, shape[0]

    raw_info = extract_nd2_channel_info(path, nd2, fallback_n_channels=n_channels)
    channels = []
    for ch in range(n_channels):
        sl = [slice(None)] * ndim
        if channel_axis is not None:
            sl[channel_axis] = ch
        info = raw_info[ch] if ch < len(raw_info) else {
            "index": ch, "label": f"Ch {ch}", "rgb": None, "name": f"Ch {ch}",
        }
        channels.append({**info, "array": arr[tuple(sl)]})
    return channels, shape


def _ome_packed_int_to_rgb(packed):
    """
    Convert an OME/ImageJ packed ARGB integer (may be signed 32-bit) to (r, g, b)
    each in [0, 1]. Returns None if parsing fails.
    """
    try:
        v = int(packed) & 0xFFFFFFFF   # treat as unsigned 32-bit
        r = ((v >> 16) & 0xFF) / 255.0
        g = ((v >>  8) & 0xFF) / 255.0
        b = (  v        & 0xFF) / 255.0
        return (r, g, b)
    except (TypeError, ValueError):
        return None


# Rough emission-wavelength → RGB mapping (nm → approximate display color).
_WAVELENGTH_RGB = [
    (380, (0.5,  0.0,  1.0 )),   # deep violet
    (440, (0.0,  0.0,  1.0 )),   # blue
    (490, (0.0,  0.8,  1.0 )),   # cyan
    (530, (0.0,  1.0,  0.0 )),   # green
    (580, (1.0,  1.0,  0.0 )),   # yellow
    (620, (1.0,  0.5,  0.0 )),   # orange
    (700, (1.0,  0.0,  0.0 )),   # red
    (800, (0.6,  0.0,  0.0 )),   # deep red / NIR
]

def _wavelength_to_rgb(nm):
    """Linearly interpolate emission wavelength to an RGB tuple, or None."""
    try:
        nm = float(nm)
    except (TypeError, ValueError):
        return None
    if nm <= _WAVELENGTH_RGB[0][0]:
        return _WAVELENGTH_RGB[0][1]
    if nm >= _WAVELENGTH_RGB[-1][0]:
        return _WAVELENGTH_RGB[-1][1]
    for (w0, c0), (w1, c1) in zip(_WAVELENGTH_RGB, _WAVELENGTH_RGB[1:]):
        if w0 <= nm <= w1:
            t = (nm - w0) / (w1 - w0)
            return tuple(c0[i] + t * (c1[i] - c0[i]) for i in range(3))
    return None


def _read_tiff(path):
    if tifffile is None:
        raise ImportError("tifffile not installed. Run: pip install tifffile")
    arr = tifffile.imread(path)

    channels = []
    try:
        with tifffile.TiffFile(path) as tif:
            axes = tif.series[0].axes.upper() if tif.series else ""
            c_idx = axes.index("C") if "C" in axes else None

            ome_names, ome_colors, ome_waves = [], [], []
            if tif.ome_metadata:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                ns = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
                prefix = f"{{{ns}}}" if ns else ""
                for img in root.iter(f"{prefix}Image"):
                    for px in img.iter(f"{prefix}Pixels"):
                        for ch in px.iter(f"{prefix}Channel"):
                            ome_names.append(ch.attrib.get("Name", ""))
                            ome_colors.append(ch.attrib.get("Color"))
                            ome_waves.append(ch.attrib.get("EmissionWavelength"))

            if c_idx is not None:
                n_ch = arr.shape[c_idx]
                for i in range(n_ch):
                    sl = [slice(None)] * arr.ndim
                    sl[c_idx] = i
                    label = (ome_names[i] if i < len(ome_names) else "") or f"Ch {i}"
                    # RGB priority: explicit Color → EmissionWavelength → None
                    rgb = None
                    if i < len(ome_colors):
                        rgb = _ome_packed_int_to_rgb(ome_colors[i])
                    if rgb is None and i < len(ome_waves):
                        rgb = _wavelength_to_rgb(ome_waves[i])
                    channels.append({
                        "index": i, "label": label, "name": label,
                        "rgb": rgb, "array": arr[tuple(sl)],
                    })
    except Exception:
        pass

    if not channels:
        channels = [{
            "index": 0, "label": "Ch 0", "name": "Ch 0",
            "rgb": None, "array": arr,
        }]

    return channels, arr.shape


def _read_czi(path):
    if czifile is None:
        raise ImportError("czifile not installed. Run: pip install czifile")
    arr = czifile.imread(path)           # shape: (1, C, Z, Y, X, 1) typically
    arr = arr.squeeze()                  # remove size-1 dims

    # Parse channel names and colors from the CZI XML metadata
    ch_names, ch_colors = [], []
    try:
        import czifile as _cz
        import xml.etree.ElementTree as ET
        with _cz.CziFile(path) as czi:
            root = ET.fromstring(czi.metadata())
            for ch in root.iter("Channel"):
                ch_names.append(ch.attrib.get("Name", ""))
                # Color is a packed ARGB int stored as a signed decimal string
                ch_colors.append(ch.attrib.get("Color"))
    except Exception:
        pass

    # Heuristic: first axis ≤ 8 → channel axis
    channel_axis = None
    if arr.ndim >= 3 and arr.shape[0] <= 8:
        channel_axis = 0
    n_ch = arr.shape[channel_axis] if channel_axis is not None else 1

    channels = []
    for i in range(n_ch):
        sl = [slice(None)] * arr.ndim
        if channel_axis is not None:
            sl[channel_axis] = i
        label = (ch_names[i] if i < len(ch_names) else "") or f"Ch {i}"
        rgb = _ome_packed_int_to_rgb(ch_colors[i]) if i < len(ch_colors) else None
        channels.append({
            "index": i, "label": label, "name": label,
            "rgb": rgb, "array": arr[tuple(sl)],
        })
    return channels, arr.shape


def _read_lif(path):
    if readlif is None:
        raise ImportError("readlif not installed. Run: pip install readlif")
    lif = LifFile(path)
    # Use the first image series in the file
    img_obj = lif.get_image(0)
    n_ch = img_obj.channels
    n_z  = img_obj.nz
    # Build numpy array: iterate frames/z-planes per channel
    frames = []
    for ch in range(n_ch):
        planes = [np.array(img_obj.get_frame(z=z, t=0, c=ch)) for z in range(max(n_z, 1))]
        frames.append(np.stack(planes, axis=0).squeeze())
    arr = np.stack(frames, axis=0)       # (C, [Z,] Y, X)

    ch_names, ch_rgbs = getattr(img_obj, "channel_as_list", None) or [], []
    # readlif ≥ 0.6.5 exposes channel LUT info via img_obj.settings
    try:
        settings = img_obj.settings  # dict of channel metadata dicts
        for i in range(n_ch):
            ch_meta = settings.get(i, {})
            # LUT color is stored as an RGBA hex string e.g. "#FF00FF00"
            hex_color = ch_meta.get("LUTName") or ch_meta.get("Color") or ""
            hex_color = hex_color.lstrip("#")
            if len(hex_color) == 8:   # AARRGGBB
                hex_color = hex_color[2:]
            if len(hex_color) == 6:
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0
                b = int(hex_color[4:6], 16) / 255.0
                ch_rgbs.append((r, g, b))
            else:
                ch_rgbs.append(None)
    except Exception:
        ch_rgbs = [None] * n_ch

    channels = []
    for i in range(n_ch):
        raw_label = ch_names[i] if i < len(ch_names) else f"Ch {i}"
        label = str(raw_label) if raw_label else f"Ch {i}"
        rgb = ch_rgbs[i] if i < len(ch_rgbs) else None
        channels.append({
            "index": i, "label": label, "name": label,
            "rgb": rgb, "array": arr[i],
        })
    return channels, arr.shape


def _read_oif(path):
    if oiffile is None:
        raise ImportError("oiffile not installed. Run: pip install oiffile")
    arr = oiffile.imread(path)           # axes order varies; typically (C, Z, Y, X)
    arr = arr.squeeze()

    channel_axis = None
    if arr.ndim >= 3 and arr.shape[0] <= 8:
        channel_axis = 0
    n_ch = arr.shape[channel_axis] if channel_axis is not None else 1

    # Parse channel names and colors from the OIF settings file
    ch_names, ch_rgbs = [], []
    try:
        with oiffile.OifFile(path) as oif:
            settings = oif.settings  # ConfigParser-like object
            for section in sorted(settings.sections()):
                if not section.lower().startswith("channel"):
                    continue
                name = settings.get(section, "DyeName", fallback=f"Ch {len(ch_names)}")
                ch_names.append(name)
                # ExcitationWavelength or EmissionWavelength → approximate RGB
                wave = (
                    settings.get(section, "EmissionWavelength", fallback=None)
                    or settings.get(section, "ExcitationWavelength", fallback=None)
                )
                ch_rgbs.append(_wavelength_to_rgb(wave))
    except Exception:
        pass

    channels = []
    for i in range(n_ch):
        sl = [slice(None)] * arr.ndim
        if channel_axis is not None:
            sl[channel_axis] = i
        label = (ch_names[i] if i < len(ch_names) else "") or f"Ch {i}"
        rgb = ch_rgbs[i] if i < len(ch_rgbs) else None
        channels.append({
            "index": i, "label": label, "name": label,
            "rgb": rgb, "array": arr[tuple(sl)],
        })
    return channels, arr.shape


# Map extension → reader function
_READERS = {
    ".nd2":  _read_nd2,
    ".tif":  _read_tiff,
    ".tiff": _read_tiff,
    ".czi":  _read_czi,
    ".lif":  _read_lif,
    ".oib":  _read_oif,
    ".oif":  _read_oif,
}


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

        self.image_path = None
        self._channels = []   # list of channel dicts produced by the active reader
        self._raw_shape = ()

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # --- Image group ---
        img_group = QGroupBox("Image Import")
        img_layout = QVBoxLayout(img_group)

        img_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText(
            "Select an image file (.nd2 .tif .tiff .czi .lif .oib .oif)…"
        )
        browse_img_btn = QPushButton("Browse…")
        browse_img_btn.clicked.connect(self._on_browse_image)
        img_row.addWidget(self.path_edit)
        img_row.addWidget(browse_img_btn)
        img_layout.addLayout(img_row)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        img_layout.addWidget(self.info_label)

        load_img_btn = QPushButton("Load channels into viewer")
        load_img_btn.clicked.connect(self._on_load_channels)
        img_layout.addWidget(load_img_btn)

        layout.addWidget(img_group)

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
    # Image browse / load
    # ------------------------------------------------------------------

    def _on_browse_image(self):
        from napari.utils.notifications import show_error

        path, _ = QFileDialog.getOpenFileName(
            self, "Select image file", "", _IMAGE_FILTER,
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        reader = _READERS.get(ext)
        if reader is None:
            show_error(f"Unsupported file extension: {ext}")
            return

        self.image_path = path
        self.path_edit.setText(path)

        try:
            channels, shape = reader(path)
        except ImportError as e:
            show_error(str(e))
            self.info_label.setText(str(e))
            self._channels = []
            return
        except Exception as e:
            show_error(f"Failed to read file:\n{e}")
            self.info_label.setText(f"Failed to read file: {e}")
            self._channels = []
            return

        self._channels = channels
        self._raw_shape = shape

        info = (
            f"Loaded {os.path.basename(path)}\n"
            f"shape: {shape}\n"
            f"{len(channels)} channel(s) detected\n"
        )
        if channels:
            info += "\nChannel labels:\n" + "\n".join(
                f"  Ch {ch['index']}: {ch['label']}" for ch in channels
            )
        self.info_label.setText(info)

    def _on_load_channels(self):
        from napari.utils.notifications import show_error, show_warning

        if self.viewer is None:
            show_error("No napari viewer attached.")
            return
        if not self._channels:
            show_warning("No image file loaded yet.")
            return

        image_root = image_root_from_path(self.image_path or "image")

        for info in self._channels:
            name = layer_name(image_root, info["label"], "raw")
            kwargs = {
                "name": name,
                "metadata": pt_meta_dict(
                    role="raw_image",
                    image_path=self.image_path,
                    image_root=image_root,
                    channel_index=info["index"],
                    channel_label=info["label"],
                    channel_name=info["name"],
                    channel_rgb=info["rgb"],
                ),
            }
            if info.get("rgb") is not None:
                kwargs["colormap"] = vispy_colormap_from_rgb(info["rgb"], info["label"])

            self.viewer.add_image(info["array"], **kwargs)

        self.layers_loaded.emit()

    # ------------------------------------------------------------------
    # CSV browse / load  (unchanged from original)
    # ------------------------------------------------------------------

    def _on_browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV file", "", "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return
        self._csv_path_edit.setText(path)
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

    def _on_load_csvs(self):
        from napari.utils.notifications import show_error, show_warning

        if self.viewer is None:
            show_error("No napari viewer attached.")
            return

        csv_path = self._csv_path_edit.text().strip() or None
        if csv_path is None:
            show_warning("No CSV file selected.")
            return

        image_root = image_root_from_path(self.image_path or "image")
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
    # Helpers (unchanged)
    # ------------------------------------------------------------------

    def _add_puncta_layer(self, core_df, meta_df, image_root, csv_path, role):
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
        present_core = [c for c in df.columns if c in core_cols]
        present_meta = [c for c in df.columns if c not in core_cols]
        return df[present_core], df[present_meta]

    @staticmethod
    def _layer_properties(core_df):
        skip = {"x", "y", "z", "frame"}
        return {
            col: core_df[col].to_numpy()
            for col in core_df.columns
            if col not in skip
        }

    @staticmethod
    def _build_coords(df):
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