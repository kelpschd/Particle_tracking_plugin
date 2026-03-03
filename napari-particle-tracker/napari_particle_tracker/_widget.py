# _widget.py
from __future__ import annotations
from typing import Optional
from napari.layers import Points
from magicgui import magicgui
import numpy as np
import pandas as pd
from napari.utils.notifications import show_info, show_warning, show_error
from napari.qt.threading import thread_worker

from .tracks_table_widget import TracksTableWidget, tracks_layer_to_dataframe

from ._processing import (
    preprocess_stack_single_channel,
    detect_puncta_dask,
    filter_dense_blobs,
    kalman_track_blobs,
    filter_tracks_by_net_displacement,
    filter_tracks_by_intersection_count
)

def _get_active_image_layer_data(viewer):
    if viewer is None:
        return None
    lay = getattr(viewer.layers.selection, "active", None)
    if getattr(lay, "__class__", None) and lay.__class__.__name__ == "Image":
        return lay.data
    for L in viewer.layers:
        if L.__class__.__name__ == "Image":
            return L.data
    return None

@magicgui(
    call_button="Run",
    layout="vertical",
    img_stack={"label": "Input time series"},
    show_preprocessed={"label": "Show preprocessed image series"},
    median_filter_size={"min": 1, "max": 101, "step": 1},
    threshold={"label": "threshold", "min": 0.0, "max": 0.01, "step": 0.00001},
    min_sigma={"min": 0.1, "max": 20.0, "step": 0.1},
    max_sigma={"min": 0.1, "max": 40.0, "step": 0.1},

    enable_density_filter={"label": "Enable density filter"},
    bin_size={"label": "Bin size (px)", "min": 1, "max": 256, "step": 1},
    blob_filter={"label": "Max blobs per bin", "min": 1, "max": 10000, "step": 1},
)

def particle_detection_widget(
    viewer: "napari.viewer.Viewer",
    img_stack: "napari.types.ImageData" = None,   # (t,y,x)
    show_preprocessed: bool = False,
    median_filter_size: int = 10,
    threshold: float = 0.00012,
    min_sigma: float = 1.0,
    max_sigma: float = 3.0,

    enable_density_filter: bool = False,
    bin_size: int = 5,
    blob_filter: int = 10,
):
    """Detect puncta from a 3D single-channel stack (t,y,x) and optionally filter densely clustered blobs."""

    # Auto-select the active image layer if none selected
    if img_stack is None:
        img_stack = _get_active_image_layer_data(viewer)
        if img_stack is None:
            show_info("Select an Image layer in 'Input time series' first.")
            return

    arr = np.asarray(img_stack)
    if arr.ndim != 3:
        show_warning(f"Expected 3D (t,y,x); got {arr.shape}.")
        return

    # --- Preprocess once (NumPy; fast) ---
    try:
        pre = preprocess_stack_single_channel(arr, median_filter_size=median_filter_size)
    except Exception as e:
        show_error(f"Preprocessing failed: {e}")
        return

    if show_preprocessed:
        viewer.add_image(pre, name="Pre-processed images", rgb=False)

    # --- Detect off the UI thread ---
    @thread_worker
    def _run_detection(pre_stack, thr, smin, smax):
        return detect_puncta_dask(
            pre_stack,
            threshold=thr,
            min_sigma=smin,
            max_sigma=smax,
        )

    worker = _run_detection(pre, threshold, min_sigma, max_sigma)

    def _on_error(err):
        show_error(f"Detection failed: {err}")

    def _on_done(det_df: pd.DataFrame):
        if det_df is None or len(det_df) == 0:
            show_info("No puncta detected with current parameters.")
            return

        n_before = len(det_df)
        if enable_density_filter:
            try:
                det_df = filter_dense_blobs(det_df, bin_size=bin_size, blob_filter=blob_filter)
            except Exception as e:
                show_warning(f"Density filtering skipped (error: {e}).")
        n_after = len(det_df)

        if n_after == 0:
            show_info("All puncta were filtered out by density settings.")
            return

        points_data = det_df[["frame", "y", "x"]].to_numpy()
        props = {}
        if "size" in det_df.columns:
            props["size"] = det_df["size"].to_numpy()

        viewer.add_points(
            points_data,
            name="Detected puncta",
            size=30,
            face_color="transparent",
            properties=props,
            border_color="red",
            border_width=0.1,
        )

        if enable_density_filter:
            show_info(f"Puncta detection complete: {n_after} remain (from {n_before}) after density filter.")
        else:
            show_info(f"Puncta detection complete: {n_after} total.")

    worker.errored.connect(_on_error)
    worker.returned.connect(_on_done)
    worker.start()


def _wire_density_filter_controls(func_gui):
    """Show/hide density filter params when the toggle changes."""
    # default visibility
    func_gui.bin_size.visible = func_gui.enable_density_filter.value
    func_gui.blob_filter.visible = func_gui.enable_density_filter.value

    @func_gui.enable_density_filter.changed.connect
    def _toggle(_event=None):
        vis = func_gui.enable_density_filter.value
        func_gui.bin_size.visible = vis
        func_gui.blob_filter.visible = vis

# new
@magicgui(
    call_button="Track",
    layout="vertical",
    points_layer={  # ← no layer_type here
        "label": "Detections (Points layer)",
        "nullable": True, 
    },
    max_distance={"label": "Max link distance (px)", "min": 1.0, "max": 200.0, "step": 1.0},
    max_missed={"label": "Max missed frames", "min": 0, "max": 50, "step": 1},

    use_disp_filter={"label": "Filter by net displacement", "widget_type": "CheckBox"},
    disp_min_frames={"label": "Min frames per track", "min": 1, "max": 1000, "step": 1, "value": 10},
    disp_threshold={"label": "Min net displacement (px)", "min": 0.0, "max": 1_000.0, "step": 1.0, "value": 50.0},

    use_intersection_filter={"label": "Filter by self-intersections", "widget_type": "CheckBox"},
    max_crossings={"label": "Max crossings per track", "min": 0, "max": 200, "step": 1, "value": 10},

    show_tracks_table={"label": "Show tracks table"},
)

def tracking_widget(
    viewer: "napari.viewer.Viewer",
    points_layer: "napari.layers.Points | None" = None,
    max_distance: float = 15.0,
    max_missed: int = 5,

    # filter args
    use_disp_filter: bool = False,
    disp_min_frames: int = 10,
    disp_threshold: float = 50.0,
    use_intersection_filter: bool = False,
    max_crossings: int = 10,

    # table
    show_tracks_table=False,
):
    """
    Build tracks from a points layer with coords [t, y, x].
    Produces a napari Tracks layer.
    """
    
    if show_tracks_table:
        _open_tracks_table(viewer)

    # auto-pick a points layer named "Detected puncta" if not set
    if points_layer is None:
        for lyr in viewer.layers:
            if getattr(lyr, "name", "") == "Detected puncta" and isinstance(lyr, Points):
                points_layer = lyr
                break

    if points_layer is None:
        show_info("Select a Points layer with detections first.")
        return

    data = np.asarray(points_layer.data)
    if data.ndim != 2 or data.shape[1] < 3:
        show_warning("Points layer must be Nx3 with columns [frame, y, x].")
        return

    # build dataframe expected by tracker
    try:
        df = pd.DataFrame({
            "frame": data[:, 0].astype(int),
            "y": data[:, 1].astype(float),
            "x": data[:, 2].astype(float),
        })
    except Exception as e:
        show_error(f"Could not parse points layer data: {e}")
        return

    @thread_worker
    def _run_tracking(
        blobs_df,
        dmax,
        miss,
        use_disp_filter,
        disp_min_frames,
        disp_threshold,
        use_intersection_filter,
        max_crossings,
    ):
        # 1) link detections into tracks
        tracks_df = kalman_track_blobs(
            blobs_df,
            max_distance=float(dmax),
            max_missed=int(miss),
        )

        if tracks_df is None or tracks_df.empty:
            return tracks_df

        # Make sure 'particle' column exists
        if "particle" not in tracks_df.columns:
            raise ValueError("tracks_df must contain a 'particle' column for filtering.")

        # 2) optional net displacement filter
        if use_disp_filter:
            tracks_df = filter_tracks_by_net_displacement(
                tracks_df,
                displacement_threshold=float(disp_threshold),
                min_frames=int(disp_min_frames),
            )
            if tracks_df is None or tracks_df.empty:
                return tracks_df

        # 3) optional self-intersection filter
        if use_intersection_filter:
            tracks_df = filter_tracks_by_intersection_count(
                tracks_df,
                max_crossings=int(max_crossings),
            )
            if tracks_df is None or tracks_df.empty:
                return tracks_df

        return tracks_df

    worker = _run_tracking(
        df,
        max_distance,
        max_missed,
        use_disp_filter,
        disp_min_frames,
        disp_threshold,
        use_intersection_filter,
        max_crossings,
    )

    def _on_error(err):
        show_error(f"Tracking failed: {err}")

    def _on_done(tracks_df: pd.DataFrame):
        if tracks_df is None or tracks_df.empty:
            show_info("No tracks produced (or all removed by filters).")
            return

        # Build napari Tracks arrays
        # data: (N, 4) -> [track_id, t, y, x]
        track_data = tracks_df[["particle", "frame", "y", "x"]].to_numpy(dtype=float)

        # optional properties (per-vertex)
        props = {
            "particle": tracks_df["particle"].to_numpy(),
        }

        viewer.add_tracks(
            track_data,
            name="Tracks",
            properties=props,
            tail_length=10,
            head_length=0,
            blending="translucent",
        )
        n_tracks = tracks_df["particle"].nunique()
        show_info(f"Tracking complete: {n_tracks} tracks.")

    worker.errored.connect(_on_error)
    worker.returned.connect(_on_done)
    worker.start()

def show_tracks_table(viewer):
    # 1) Get your Tracks layer
    try:
        tracks_layer = viewer.layers['tracks']  # or however you name it
    except KeyError:
        show_warning("No 'tracks' layer found.")
        return

    # 2) Build DataFrame from the Tracks layer
    tracks_df = tracks_layer_to_dataframe(tracks_layer)

    # 3) Create the widget and dock it
    widget = TracksTableWidget(viewer, tracks_layer, tracks_df)
    viewer.window.add_dock_widget(widget, name="Tracks Table", area="right")

def _open_tracks_table(viewer):
    # try to get a Tracks layer
    tracks_layers = [ly for ly in viewer.layers if ly.__class__.__name__ == "Tracks"]
    if not tracks_layers:
        show_warning("No Tracks layer found.")
        return

    tracks_layer = tracks_layers[0]  # or pick by name if you want
    tracks_df = tracks_layer_to_dataframe(tracks_layer)

    widget = TracksTableWidget(viewer, tracks_layer, tracks_df)
    viewer.window.add_dock_widget(widget, name="Tracks Table", area="right")
