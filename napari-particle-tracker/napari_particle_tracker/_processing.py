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