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

    frame = data[:, 0].astype(float)

    n_coords = data.shape[1] - 1
    base = {'frame': frame}

    if n_coords == 2:
        # data columns: [t, y, x]
        base['y'] = data[:, 1]
        base['x'] = data[:, 2]
    elif n_coords == 3:
        # data columns: [t, z, y, x]
        base['z'] = data[:, 1]
        base['y'] = data[:, 2]
        base['x'] = data[:, 3]
    else:
        # generic fallback
        for i in range(n_coords):
            base[f'coord_{i}'] = data[:, i + 1]

    # Add properties into base (ensure length matches)
    for key, vals in props.items():
        vals_arr = np.asarray(vals)
        if vals_arr.shape[0] == data.shape[0]:
            base[key] = vals_arr
        else:
            # try to broadcast single value to length N
            if vals_arr.size == 1:
                base[key] = np.repeat(vals_arr.item(), data.shape[0])
            else:
                # skip mismatched property lengths with a warning
                base[key] = np.repeat(np.nan, data.shape[0])

    df = pd.DataFrame(base)

    # Normalize property/track id name -> ensure 'particle' exists
    # Common property names: 'particle', 'track_id', 'track', 'id'
    if 'particle' not in df.columns:
        for candidate in ('track_id', 'track', 'id'):
            if candidate in df.columns:
                df = df.rename(columns={candidate: 'particle'})
                break

    if 'particle' not in df.columns:
        # synthesize sequential particle ids per-track
        # when no property provided, try to reconstruct from data ordering
        df['particle'] = 0  # fallback single-track
        # if we can infer multiple particles via grouping of track index in properties, skip

    # ensure particle is int
    df['particle'] = df['particle'].astype(int)

    return df[['particle', 'frame'] + [c for c in ['z','y','x'] if c in df.columns] + 
              [c for c in df.columns if (c not in ('particle','frame','z','y','x'))]]


def dataframe_to_tracks_layer_data(df: pd.DataFrame):
    """
    Convert DataFrame -> (data, properties) for napari Tracks layer.
    Expects 'frame' and coordinate columns ('z','y','x' or 'y','x') and optional 'particle' property.
    """

    df = df.copy()

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

    data = np.column_stack([t, coords_arr]).astype(float)

    # properties: everything else except frame & coords
    exclude = set(['frame', 't', 'z', 'y', 'x'])
    properties = {}
    for col in df.columns:
        if col in exclude:
            continue
        properties[col] = df[col].to_numpy()

    # ensure there is a 'particle' property (some UI code expects it)
    if 'particle' not in properties:
        # try track_id fallback
        if 'track_id' in properties:
            properties['particle'] = properties.pop('track_id').astype(int)
        else:
            # synthesize zeros
            properties['particle'] = np.zeros(len(df), dtype=int)

    # ensure property arrays are 1D and same length as data
    for k, v in list(properties.items()):
        arr = np.asarray(v)
        if arr.ndim != 1 or arr.shape[0] != data.shape[0]:
            # try to coerce / broadcast single value
            if arr.size == 1:
                properties[k] = np.repeat(arr.item(), data.shape[0])
            else:
                properties[k] = np.repeat(np.nan, data.shape[0])

    return data, properties
