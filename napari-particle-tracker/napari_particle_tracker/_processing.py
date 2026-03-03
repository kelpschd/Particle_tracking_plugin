# _processing.py
import numpy as np
import pandas as pd
import trackpy as tp

from scipy.ndimage import median_filter
from scipy.spatial import cKDTree
from skimage.feature import blob_log
from dask import delayed, compute
from shapely.geometry import LineString

def preprocess_stack_single_channel(
    img_stack: np.ndarray,
    median_filter_size: int = 10,
) -> np.ndarray:
    """Background = median-filtered mean(t). Return uint16 (t, y, x)."""
    arr = np.asarray(img_stack)
    if arr.ndim != 3:
        raise ValueError(f"Expected (t,y,x); got {arr.shape}")
    mean_proj = arr.mean(axis=0)
    background = median_filter(mean_proj, size=median_filter_size)
    pre = np.clip(arr - background, 0, None).astype(np.uint16)
    return pre

def detect_puncta_dask(
    img_stack,
    threshold: float = 0.00012,
    min_sigma: float = 1.0,
    max_sigma: float = 3.0,
) -> pd.DataFrame:

    @delayed
    def process_frame(t, frame):
        blobs = blob_log(frame, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
        return [{'frame': t, 'x': x, 'y': y, 'size': sigma * np.sqrt(2)} for y, x, sigma in blobs]

    tasks = [process_frame(t, frame) for t, frame in enumerate(img_stack)]
    results = compute(*tasks)
    positions = [entry for sublist in results for entry in sublist]

    if not positions:
        return pd.DataFrame(columns=["frame", "y", "x", "size"])

    return pd.DataFrame(positions, columns=["frame", "y", "x", "size"])

def filter_dense_blobs(blob_df: pd.DataFrame, bin_size: int = 5, blob_filter: int = 10) -> pd.DataFrame:
    """
    Filters out blobs that appear too frequently within a very small local neighborhood.

    Parameters
    ----------
    blob_df : DataFrame with at least ['x','y'] columns.
    bin_size : int, pixel binning size for rounding positions.
    blob_filter : int, max allowed blobs per bin; bins with >= blob_filter are removed.

    Returns
    -------
    pd.DataFrame
        Filtered rows preserving all original columns.
    """
    if blob_df is None or len(blob_df) == 0:
        return blob_df

    rounded = blob_df.copy()
    rounded["x_rounded"] = (rounded["x"] // bin_size * bin_size).astype(int)
    rounded["y_rounded"] = (rounded["y"] // bin_size * bin_size).astype(int)

    counts = (
        rounded.groupby(["x_rounded", "y_rounded"])
        .size()
        .reset_index(name="count")
    )
    merged = rounded.merge(counts, on=["x_rounded", "y_rounded"], how="left")
    filtered = merged[merged["count"] < blob_filter].reset_index(drop=True)

    # drop helper cols if you don't want them in the final table
    return filtered.drop(columns=["x_rounded", "y_rounded", "count"], errors="ignore")

# new for tracking
try:
    from filterpy.kalman import KalmanFilter
except Exception as e:  # keep import error message around
    KalmanFilter = None
    _KALMAN_IMPORT_ERR = e

class KalmanTrack:
    def __init__(self, initial_detection, particle_id, frame, max_missed=5):
        x, y = initial_detection
        if KalmanFilter is None:
            raise ImportError(f"filterpy not available: {_KALMAN_IMPORT_ERR}")
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]], dtype=float)
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]], dtype=float)
        self.kf.R *= 5.0
        self.kf.P *= 1000.0
        self.kf.Q = np.eye(4) * 0.1
        self.kf.x = np.array([x, y, 0.0, 0.0], dtype=float)

        self.id = int(particle_id)
        self.history = [{'frame': int(frame), 'x': float(x), 'y': float(y), 'particle': int(self.id)}]
        self.missed = 0
        self.max_missed = int(max_missed)

    def predict(self):
        self.kf.predict()
        return self.kf.x[:2]

    def update(self, detection, frame):
        self.kf.update(np.asarray(detection, dtype=float))
        x, y = self.kf.x[:2]
        self.history.append({'frame': int(frame), 'x': float(x), 'y': float(y), 'particle': int(self.id)})
        self.missed = 0

    def mark_missed(self):
        self.missed += 1

    def is_dead(self):
        return self.missed > self.max_missed


def kalman_track_blobs(all_blobs: pd.DataFrame, max_distance: float = 15.0, max_missed: int = 5) -> pd.DataFrame:
    """
    all_blobs: DataFrame with columns ['frame','x','y'] (optionally others).
    Returns a DataFrame with ['particle','frame','x','y'].
    """
    if KalmanFilter is None:
        raise ImportError(f"filterpy not available: {_KALMAN_IMPORT_ERR}")

    # ensure expected columns and sort by frame
    need = {'frame', 'x', 'y'}
    if not need.issubset(set(all_blobs.columns)):
        raise ValueError(f"Expected columns {need}, got {set(all_blobs.columns)}")
    all_blobs = all_blobs[['frame', 'x', 'y']].copy()
    all_blobs['frame'] = all_blobs['frame'].astype(int)
    all_blobs = all_blobs.sort_values('frame')

    active_tracks = []
    finished_tracks = []
    next_id = 0

    for frame, detections in all_blobs.groupby('frame', sort=True):
        detected_positions = detections[['x', 'y']].to_numpy(dtype=float)

        # Predict positions for all current tracks
        predictions = [track.predict() for track in active_tracks]
        assigned = set()

        if len(predictions) > 0 and len(detected_positions) > 0:
            tree = cKDTree(detected_positions)
            for i, pred in enumerate(predictions):
                dist, idx = tree.query(pred, distance_upper_bound=float(max_distance))
                if idx != len(detected_positions) and idx not in assigned:
                    active_tracks[i].update(detected_positions[idx], frame)
                    assigned.add(idx)
                else:
                    active_tracks[i].mark_missed()
        else:
            for track in active_tracks:
                track.mark_missed()

        # Start new tracks for unassigned detections
        for i, pos in enumerate(detected_positions):
            if i not in assigned:
                new_track = KalmanTrack(pos, next_id, frame, max_missed)
                active_tracks.append(new_track)
                next_id += 1

        # Remove dead tracks
        still_alive = []
        for track in active_tracks:
            if track.is_dead():
                finished_tracks.append(track)
            else:
                still_alive.append(track)
        active_tracks = still_alive

    # Add remaining active tracks to finished
    finished_tracks.extend(active_tracks)

    # Combine histories
    all_tracks = [row for track in finished_tracks for row in track.history]
    out = pd.DataFrame(all_tracks, columns=['particle','frame','x','y'])
    return out

def filter_tracks_by_net_displacement(tracks_df, displacement_threshold=50, min_frames=10):
    """Filter tracks by net displacement after removing short tracks."""
    if tracks_df.empty:
        return tracks_df

    filtered_tracks = tp.filter_stubs(tracks_df, threshold=min_frames).reset_index(drop=True)
    print(f"[Net Disp] After stub removal: {filtered_tracks['particle'].nunique()} tracks")

    if filtered_tracks.empty:
        return filtered_tracks

    displacements = []
    for pid, group in filtered_tracks.groupby('particle'):
        start = group.iloc[0][['x', 'y']].values.astype(float)
        end   = group.iloc[-1][['x', 'y']].values.astype(float)
        net_disp = np.linalg.norm(end - start)
        displacements.append({'particle': pid, 'net_displacement': net_disp})

    displacement_df = pd.DataFrame(displacements)

    valid_particles = displacement_df.loc[
        displacement_df['net_displacement'] > displacement_threshold,
        'particle'
    ]
    print(f"[Net Disp] Tracks kept: {len(valid_particles)}")

    return filtered_tracks[filtered_tracks['particle'].isin(valid_particles)].reset_index(drop=True)


def count_self_intersections(track_df):
    """Count number of self-intersections for each particle track."""
    results = []

    if track_df.empty:
        return pd.DataFrame(columns=['particle', 'self_intersections'])

    for pid, group in track_df.groupby('particle'):
        coords = group[['x', 'y']].to_numpy(float)

        if len(coords) < 3:
            results.append({'particle': pid, 'self_intersections': 0})
            continue

        segments = [
            LineString([coords[i], coords[i+1]])
            for i in range(len(coords) - 1)
        ]

        crossings = 0
        for i, seg1 in enumerate(segments):
            for j in range(i + 2, len(segments)):  # skip neighbors
                if seg1.crosses(segments[j]):
                    crossings += 1

        results.append({'particle': pid, 'self_intersections': crossings})

    return pd.DataFrame(results)


def filter_tracks_by_intersection_count(tracks_df, max_crossings=10):
    """Keep only tracks whose self-intersection count <= max_crossings."""
    if tracks_df.empty:
        return tracks_df

    intersection_df = count_self_intersections(tracks_df)
    if intersection_df.empty:
        return tracks_df

    valid_particles = intersection_df.loc[
        intersection_df['self_intersections'] <= max_crossings,
        'particle'
    ]
    print(f"[Intersection] Tracks kept: {len(valid_particles)}")

    return tracks_df[tracks_df['particle'].isin(valid_particles)].reset_index(drop=True)
