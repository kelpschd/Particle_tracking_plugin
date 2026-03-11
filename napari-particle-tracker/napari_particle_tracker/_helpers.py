# helpers.py
import numpy as np
import pandas as pd

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
