# _widget.py
from __future__ import annotations
from typing import Optional
from napari.layers import Points
from napari.layers import Image as NapariImage
from magicgui import magicgui
import numpy as np
import pandas as pd
from napari.utils.notifications import show_info, show_warning, show_error
from napari.qt.threading import thread_worker

from ._processing import (
    preprocess_stack_single_channel,
    detect_puncta_dask,
    filter_dense_blobs
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

# Particle detection UI
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

    def _resolve_img_stack(img_stack, viewer):
        from napari.utils.notifications import show_warning

        # unwrap magicgui parameter wrapper
        try:
            if hasattr(img_stack, "value"):
                img_stack = img_stack.value
        except Exception:
            pass

        # If nothing selected, try active image layer
        if img_stack is None:
            candidate = _get_active_image_layer_data(viewer)
            if candidate is None:
                return None
            arr = candidate
        else:
            # If a string layer name
            if isinstance(img_stack, str):
                layer = None
                try:
                    layer = viewer.layers[img_stack]
                except Exception:
                    for ly in viewer.layers:
                        if getattr(ly, "name", None) == img_stack:
                            layer = ly
                            break
                if layer is None:
                    print(f"_resolve_img_stack: no layer named {img_stack!r}")
                    return None
                arr = getattr(layer, "data", None)

            # If napari Image layer
            elif hasattr(img_stack, "data"):
                arr = img_stack.data

            else:
                arr = img_stack

        # Handle dask / lazy arrays
        try:
            import dask.array as da
            if isinstance(arr, da.Array):
                arr = arr.compute()
        except Exception:
            pass

        # Convert to numpy
        try:
            arr = np.asarray(arr)
        except Exception as e:
            print("_resolve_img_stack: np.asarray failed:", e)
            return None

        if arr.size == 0:
            print("_resolve_img_stack: array has size 0")
            return None

        # 2D → treat as single frame
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]

        # 3D → expected case
        if arr.ndim == 3:
            return arr

        # 4D → assume (t, c, y, x) → pick channel 0
        if arr.ndim == 4:
            if arr.shape[1] <= 8:
                show_warning(f"Input shape {arr.shape} looks like (t,c,y,x). Using channel 0.")
                return arr[:, 0, ...]
            raise ValueError(f"Ambiguous 4D shape {arr.shape}")

        raise ValueError(f"Expected 3D (t,y,x); got {arr.shape}")

    # Resolve input
    try:
        arr = _resolve_img_stack(img_stack, viewer)
    except ValueError as e:
        show_warning(str(e))
        return

    if arr is None:
        show_info("Select an Image layer in 'Input time series' first.")
        return

    print("particle_detection resolved array:", type(arr), arr.shape)

    # # Auto-select the active image layer if none selected
    # if img_stack is None:
    #     img_stack = _get_active_image_layer_data(viewer)
    #     if img_stack is None:
    #         show_info("Select an Image layer in 'Input time series' first.")
    #         return

    # arr = np.asarray(img_stack)
    # if arr.ndim != 3:
    #     show_warning(f"Expected 3D (t,y,x); got {arr.shape}.")
    #     return

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

        run_spot_meta = {
            "median_filter_size": int(median_filter_size),
            "threshold": float(threshold),
            "min_sigma": float(min_sigma),
            "max_sigma": float(max_sigma),
            "enable_density_filter": bool(enable_density_filter),
            "bin_size": int(bin_size),
            "blob_filter": int(blob_filter),
            "n_detected_before_filter": int(n_before),
            "n_detected_after_filter": int(n_after),
        }

        pts_layer = viewer.add_points(
            points_data,
            name="Detected puncta",
            size=30,
            face_color="transparent",
            properties=props,
            border_color="red",
            border_width=0.1,
        )
        # assign metadata after creation
        pts_layer.metadata["run_params"] = run_spot_meta

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

