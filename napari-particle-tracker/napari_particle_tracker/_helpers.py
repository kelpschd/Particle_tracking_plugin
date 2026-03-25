# helpers.py
import numpy as np
import pandas as pd
from vispy.color import Colormap
import os

PT_META_KEY = "particle_tracking"

def pt_meta_dict(**kwargs):
    return {PT_META_KEY: kwargs}


def image_root_from_path(path: str | None) -> str | None:
    if not path:
        return None
    return os.path.splitext(os.path.basename(str(path)))[0]


def layer_name(image_root: str | None, channel_label: str | None, role: str) -> str:
    parts = [image_root, channel_label, role]
    return " | ".join(str(p) for p in parts if p)


def vispy_colormap_from_rgb(rgb, name="channel"):
    r, g, b = [v / 255.0 for v in rgb]
    return Colormap([[0, 0, 0, 1], [r, g, b, 1]])


def normalize_rgb(color):
    if color is None:
        return None

    if hasattr(color, "r") and hasattr(color, "g") and hasattr(color, "b"):
        vals = [float(color.r), float(color.g), float(color.b)]
    else:
        arr = np.asarray(color).reshape(-1)
        if arr.size < 3:
            return None
        vals = [float(arr[0]), float(arr[1]), float(arr[2])]

    if max(vals) <= 1.0:
        vals = [255.0 * v for v in vals]

    return tuple(int(np.clip(round(v), 0, 255)) for v in vals)

def extract_nd2_channel_info(path: str, nd2_module, fallback_n_channels: int = 1):
    info = [
        {
            "index": i,
            "name": f"Ch {i}",
            "label": f"Ch {i}",
            "rgb": None,
            "emission_nm": None,
            "excitation_nm": None,
        }
        for i in range(int(fallback_n_channels))
    ]

    if nd2_module is None:
        return info

    try:
        with nd2_module.ND2File(path) as f:
            channels = list(getattr(getattr(f, "metadata", None), "channels", []) or [])
            n_channels = max(int(fallback_n_channels), len(channels))

            # grow list if needed
            if len(info) < n_channels:
                for i in range(len(info), n_channels):
                    info.append(
                        {
                            "index": i,
                            "name": f"Ch {i}",
                            "label": f"Ch {i}",
                            "rgb": None,
                            "emission_nm": None,
                            "excitation_nm": None,
                        }
                    )

            for i, ch in enumerate(channels[:n_channels]):
                meta = getattr(ch, "channel", ch)

                name = getattr(meta, "name", None) or f"Ch {i}"
                rgb = normalize_rgb(getattr(meta, "color", None))

                info[i].update(
                    {
                        "index": i,
                        "name": str(name),
                        "label": str(name),
                        "rgb": rgb,
                        "emission_nm": getattr(meta, "emissionLambdaNm", None),
                        "excitation_nm": getattr(meta, "excitationLambdaNm", None),
                    }
                )
    except Exception:
        pass

    return info

# ======================================================================
# Helper functions to convert between DataFrame <-> napari Tracks layer
# ======================================================================

def tracks_layer_to_dataframe(tracks_layer) -> pd.DataFrame:
    """
    Convert a napari Tracks layer to a DataFrame with canonical column names.
    Returns columns: ['particle','frame','y','x'] or ['particle','frame','z','y','x'].
    """

    data = np.asarray(tracks_layer.data)
    props = dict(tracks_layer.properties) if hasattr(tracks_layer, "properties") else {}

    if data.size == 0:
        return pd.DataFrame(columns=['particle','frame','y','x'])

    # Napari Tracks convention: first column = particle id, second = frame/time
    particle_col = data[:, 0].astype(int)
    frame = data[:, 1].astype(float)

    n_coords = data.shape[1] - 2  # subtract particle and frame columns
    base = {'particle': particle_col, 'frame': frame}

    if n_coords == 2:
        # data columns: [particle, frame, y, x]
        base['y'] = data[:, 2]
        base['x'] = data[:, 3]
    elif n_coords == 3:
        # data columns: [particle, frame, z, y, x]
        base['z'] = data[:, 2]
        base['y'] = data[:, 3]
        base['x'] = data[:, 4]
    else:
        # generic fallback for unusual dimensionalities
        for i in range(n_coords):
            base[f'coord_{i}'] = data[:, i + 2]

    # Add properties into base (ensure length matches rows). If properties are
    # per-track (length == number of unique particles), expand them to per-row.
    n_rows = data.shape[0]
    unique_particles = np.unique(particle_col)
    for key, vals in props.items():
        vals_arr = np.asarray(vals)
        if vals_arr.shape[0] == n_rows:
            base[key] = vals_arr
        elif vals_arr.shape[0] == unique_particles.shape[0]:
            # property is likely per-particle: expand by mapping particle->value
            mapping = dict(zip(unique_particles, vals_arr))
            base[key] = np.array([mapping.get(pid, np.nan) for pid in particle_col])
        elif vals_arr.size == 1:
            base[key] = np.repeat(vals_arr.item(), n_rows)
        else:
            # fallback: broadcast NaNs
            base[key] = np.repeat(np.nan, n_rows)

    df = pd.DataFrame(base)

    # Ensure 'particle' exists and is int
    if 'particle' not in df.columns:
        df['particle'] = 0
    df['particle'] = df['particle'].astype(int)

    # return canonical column ordering: particle, frame, (z), y, x, then other props
    core_cols = ['particle', 'frame'] + [c for c in ['z','y','x'] if c in df.columns]
    other_cols = [c for c in df.columns if c not in core_cols]
    return df[core_cols + other_cols]


def dataframe_to_tracks_layer_data(df: pd.DataFrame):
    """
    Convert DataFrame -> (data, properties) for napari Tracks layer.
    Expects 'frame' and coordinate columns ('z','y','x' or 'y','x') and optional 'particle' property.
    Returns (data, properties) where data first column is particle id.
    """

    df = df.copy()

    # particle id (preferred)
    if 'particle' in df.columns:
        p = df['particle'].to_numpy().astype(int)
    else:
        # try track_id fallback or synthesize zeros
        if 'track_id' in df.columns:
            p = df['track_id'].to_numpy().astype(int)
        else:
            p = np.zeros(len(df), dtype=int)

    # time/frame
    if 'frame' in df.columns:
        t = df['frame'].to_numpy()
    elif 't' in df.columns:
        t = df['t'].to_numpy()
    else:
        t = np.zeros(len(df), dtype=float)

    coord_cols = []
    if 'z' in df.columns:
        coord_cols.append('z')
    if 'y' in df.columns:
        coord_cols.append('y')
    if 'x' in df.columns:
        coord_cols.append('x')

    if coord_cols:
        coords_arr = np.vstack([df[c].to_numpy() for c in coord_cols]).T
    else:
        coords_arr = np.zeros((len(df), 0), dtype=float)

    # build data with napari ordering: particle, frame, coords...
    data = np.column_stack([p, t, coords_arr]).astype(float)

    # properties: everything else except frame, coords, particle
    exclude = set(['frame', 't', 'z', 'y', 'x', 'particle'])
    properties = {}
    for col in df.columns:
        if col in exclude:
            continue
        properties[col] = df[col].to_numpy()

    # ensure properties arrays are 1D and same length as data
    for k, v in list(properties.items()):
        arr = np.asarray(v)
        if arr.ndim != 1 or arr.shape[0] != data.shape[0]:
            if arr.size == 1:
                properties[k] = np.repeat(arr.item(), data.shape[0])
            else:
                properties[k] = np.repeat(np.nan, data.shape[0])

    return data, properties
