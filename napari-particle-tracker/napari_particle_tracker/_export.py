import os
import numpy as np
import pandas as pd
import cv2
import imageio.v3 as iio
from PIL import Image, ImageDraw

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QLineEdit, QCheckBox, QSpinBox, QGroupBox, QComboBox
)
from napari.utils.notifications import show_info, show_warning, show_error

from ._validation_state import get_validation_state


def add_export_metadata(df: pd.DataFrame, spot_meta=None, track_meta=None, image_meta=None):
    out = df.copy()
    for prefix, meta in [
        ("spot", spot_meta or {}),
        ("track", track_meta or {}),
        ("image", image_meta or {}),
    ]:
        for key, value in meta.items():
            col = f"{prefix}_{key}"
            if np.isscalar(value) or isinstance(value, str):
                out[col] = value
            else:
                value = np.asarray(value)
                out[col] = value if len(value) == len(out) else np.nan
    return out


def normalize_crop_percentile(crop, lower=2, upper=98):
    """
    crop: (T, C, H, W)
    return uint8 (T, C, H, W)
    """
    if crop.ndim != 4:
        raise ValueError(f"Expected (T,C,H,W), got {crop.shape}")

    T, C, H, W = crop.shape
    out = np.zeros((T, C, H, W), dtype=np.uint8)

    for c in range(C):
        data = crop[:, c].astype(np.float32).reshape(-1)
        p_low, p_high = np.percentile(data, [lower, upper])

        if p_high - p_low < 1e-6:
            continue

        norm = (crop[:, c].astype(np.float32) - p_low) / (p_high - p_low)
        out[:, c] = np.clip(norm * 255.0, 0, 255).astype(np.uint8)

    return out


def _to_rgb(gray_u8):
    return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)


def _apply_pseudocolor(gray_u8, rgb_color):
    g = gray_u8.astype(np.float32) / 255.0
    color = np.array(rgb_color, dtype=np.float32) / 255.0
    out = g[..., None] * color[None, None, :]
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _merge_rgb(rgb_list):
    if not rgb_list:
        raise ValueError("No RGB images to merge.")
    acc = np.zeros_like(rgb_list[0], dtype=np.float32)
    for im in rgb_list:
        acc += im.astype(np.float32)
    return np.clip(acc, 0, 255).astype(np.uint8)


def _draw_overlays(rgb_u8, center_xy, track_xy, circle_r=6, line_w=2):
    im = Image.fromarray(rgb_u8)
    dr = ImageDraw.Draw(im)

    cx, cy = center_xy
    dr.ellipse(
        (cx - circle_r, cy - circle_r, cx + circle_r, cy + circle_r),
        outline=(255, 0, 0),
        width=line_w,
    )

    if len(track_xy) >= 2:
        dr.line(track_xy, fill=(255, 255, 0), width=line_w, joint="curve")

    return np.array(im, dtype=np.uint8)


def _resize_nn(rgb_u8, scale):
    if scale == 1:
        return rgb_u8
    im = Image.fromarray(rgb_u8)
    w, h = im.size
    im = im.resize((w * scale, h * scale), resample=Image.Resampling.NEAREST)
    return np.array(im, dtype=np.uint8)


def validate_tracks_df(tracks_df: pd.DataFrame) -> None:
    required = {"frame", "x", "y", "particle"}
    missing = required - set(tracks_df.columns)
    if missing:
        raise ValueError(f"tracks_df missing columns: {missing}")


def _get_pt_meta(layer):
    meta = getattr(layer, "metadata", {}) or {}
    pt_meta = meta.get("particle_tracking", {}) or {}
    return pt_meta if isinstance(pt_meta, dict) else {}


def _collect_channel_colors(image_import_widget, viewer=None):
    """
    Collect channel RGB colors from the imported image metadata.
    Priority:
      1) image_import_widget._channel_info
      2) viewer image layer metadata
      3) fallback defaults
    Returns dict[int, tuple[int,int,int]]
    """
    colors = {}

    channel_info = getattr(image_import_widget, "_channel_info", None) or []
    for info in channel_info:
        try:
            idx = int(info.get("index"))
        except Exception:
            continue

        rgb = info.get("rgb")
        if rgb is None:
            rgb = info.get("channel_rgb")
        if rgb is None:
            continue

        try:
            colors[idx] = tuple(int(v) for v in rgb)
        except Exception:
            continue

    if not colors and viewer is not None:
        for ly in viewer.layers:
            meta = getattr(ly, "metadata", {}) or {}
            pt_meta = meta.get("particle_tracking", {}) or {}
            if pt_meta.get("role") == "raw_image":
                idx = pt_meta.get("channel_index")
                rgb = pt_meta.get("channel_rgb")
                if idx is not None and rgb is not None:
                    try:
                        colors[int(idx)] = tuple(int(v) for v in rgb)
                    except Exception:
                        pass

    if not colors:
        colors = {
            0: (255, 0, 255),
            1: (255, 0, 0),
            2: (0, 255, 0),
        }

    return colors


def _composite_rgb_frame(
    crop_norm,
    t,
    C,
    channel_colors,
    upscale,
):
    """
    Build a pseudocolored composite frame from all available channel colors.
    """
    h, w = crop_norm.shape[2], crop_norm.shape[3]

    pseudo_list = []
    for c in sorted(channel_colors.keys()):
        if 0 <= c < C:
            pseudo_list.append(
                _resize_nn(
                    _apply_pseudocolor(
                        crop_norm[t, c],
                        channel_colors.get(c, (255, 255, 255)),
                    ),
                    upscale,
                )
            )

    if pseudo_list:
        return _merge_rgb(pseudo_list)

    return np.zeros((h * upscale, w * upscale, 3), dtype=np.uint8)


def _save_track_movies_video(
    img_array,
    track_df,
    image_name,
    output_dir,
    *,
    ext,
    fourcc,
    padding=40,
    fps=10,
    channel_colors=None,
    upscale=1,
    norm_lower=2,
    norm_upper=98,
    circle_r=6,
    line_w=2,
    only_track_frames=True,
    pseudocolor_single=False,
    annotate=True, 
):
    if img_array.ndim != 4:
        raise ValueError(f"Expected (T,C,Y,X), got {img_array.shape}")

    os.makedirs(output_dir, exist_ok=True)

    T, C, Y, X = img_array.shape
    base_name = os.path.splitext(os.path.basename(image_name))[0].replace(" ", "_")

    df = track_df.copy()
    df["frame"] = df["frame"].astype(int)
    df = df.sort_values(["particle", "frame"])

    if channel_colors is None:
        channel_colors = {
            0: (255, 0, 255),
            1: (255, 0, 0),
            2: (0, 255, 0),
        }

    for pid, group in df.groupby("particle"):
        group = group.sort_values("frame")

        y_min, y_max = int(np.floor(group["y"].min())), int(np.ceil(group["y"].max()))
        x_min, x_max = int(np.floor(group["x"].min())), int(np.ceil(group["x"].max()))

        y1 = max(y_min - padding, 0)
        y2 = min(y_max + padding, Y)
        x1 = max(x_min - padding, 0)
        x2 = min(x_max + padding, X)

        if y2 <= y1 or x2 <= x1:
            continue

        crop = img_array[:, :, y1:y2, x1:x2]
        crop_norm = normalize_crop_percentile(crop, lower=norm_lower, upper=norm_upper)
        h, w = crop_norm.shape[2], crop_norm.shape[3]

        centers = {
            int(r["frame"]): (float(r["x"]) - x1, float(r["y"]) - y1)
            for _, r in group.iterrows()
            if 0 <= int(r["frame"]) < T
        }

        if only_track_frames:
            frame_list = sorted(centers.keys())
        else:
            frame_list = list(range(T))

        if not frame_list:
            continue

        path_so_far = []
        track_by_frame = {}
        for t in sorted(centers.keys()):
            path_so_far.append(centers[t])
            track_by_frame[t] = path_so_far.copy()

        # composite writers — always write raw, also write annotated if annotate=True
        out_path_raw = os.path.join(output_dir, f"{base_name}_track{pid}_composite_raw{ext}")
        writer_raw = cv2.VideoWriter(
            out_path_raw, cv2.VideoWriter_fourcc(*fourcc),
            fps, (w * upscale, h * upscale), isColor=True,
        )
        if not writer_raw.isOpened():
            raise RuntimeError(f"Could not open video writer for {out_path_raw}")

        writer_ann = None
        if annotate:
            out_path_ann = os.path.join(output_dir, f"{base_name}_track{pid}_composite_annot{ext}")
            writer_ann = cv2.VideoWriter(
                out_path_ann, cv2.VideoWriter_fourcc(*fourcc),
                fps, (w * upscale, h * upscale), isColor=True,
            )
            if not writer_ann.isOpened():
                raise RuntimeError(f"Could not open video writer for {out_path_ann}")

        # single-channel writers — same pattern
        channel_writers = []
        for c in range(C):
            ch_out_raw = os.path.join(output_dir, f"{base_name}_track{pid}_ch{c}_raw{ext}")
            ch_writer_raw = cv2.VideoWriter(
                ch_out_raw, cv2.VideoWriter_fourcc(*fourcc),
                fps, (w * upscale, h * upscale), isColor=True,
            )
            if not ch_writer_raw.isOpened():
                raise RuntimeError(f"Could not open video writer for {ch_out_raw}")

            ch_writer_ann = None
            if annotate:
                ch_out_ann = os.path.join(output_dir, f"{base_name}_track{pid}_ch{c}_annot{ext}")
                ch_writer_ann = cv2.VideoWriter(
                    ch_out_ann, cv2.VideoWriter_fourcc(*fourcc),
                    fps, (w * upscale, h * upscale), isColor=True,
                )
                if not ch_writer_ann.isOpened():
                    raise RuntimeError(f"Could not open video writer for {ch_out_ann}")

            channel_writers.append((c, ch_writer_raw, ch_writer_ann))

        for t in frame_list:
            comp_rgb = _composite_rgb_frame(
                crop_norm=crop_norm, t=t, C=C,
                channel_colors=channel_colors, upscale=upscale,
            )
            writer_raw.write(cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR))

            center = centers.get(t)
            if annotate and writer_ann is not None:
                comp_ann = comp_rgb.copy()
                if center is not None:
                    comp_ann = _draw_overlays(
                        comp_ann,
                        center_xy=(int(center[0] * upscale), int(center[1] * upscale)),
                        track_xy=[
                            (int(p[0] * upscale), int(p[1] * upscale))
                            for p in track_by_frame.get(t, [])
                        ],
                        circle_r=circle_r * upscale,
                        line_w=max(1, line_w * upscale),
                    )
                writer_ann.write(cv2.cvtColor(comp_ann, cv2.COLOR_RGB2BGR))

            for c, ch_writer_raw, ch_writer_ann in channel_writers:
                gray = crop_norm[t, c]
                if pseudocolor_single:
                    rgb = _resize_nn(
                        _apply_pseudocolor(gray, channel_colors.get(c, (255, 255, 255))),
                        upscale,
                    )
                else:
                    rgb = _resize_nn(_to_rgb(gray), upscale)

                ch_writer_raw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                if annotate and ch_writer_ann is not None:
                    rgb_ann = rgb.copy()
                    if center is not None:
                        rgb_ann = _draw_overlays(
                            rgb_ann,
                            center_xy=(int(center[0] * upscale), int(center[1] * upscale)),
                            track_xy=[
                                (int(p[0] * upscale), int(p[1] * upscale))
                                for p in track_by_frame.get(t, [])
                            ],
                            circle_r=circle_r * upscale,
                            line_w=max(1, line_w * upscale),
                        )
                    ch_writer_ann.write(cv2.cvtColor(rgb_ann, cv2.COLOR_RGB2BGR))

        writer_raw.release()
        if writer_ann is not None:
            writer_ann.release()
        for _, ch_writer_raw, ch_writer_ann in channel_writers:
            ch_writer_raw.release()
            if ch_writer_ann is not None:
                ch_writer_ann.release()

        print(f"Saved {out_path_raw}")
        if annotate:
            print(f"Saved {out_path_ann}")


def save_track_movies_mp4(
    img_array,
    track_df,
    image_name,
    output_dir,
    padding=40,
    fps=10,
    channel_colors=None,
    upscale=1,
    norm_lower=2,
    norm_upper=98,
    circle_r=6,
    line_w=2,
    only_track_frames=True,
    pseudocolor_single=False,
    annotate=True,
):
    _save_track_movies_video(
        img_array=img_array,
        track_df=track_df,
        image_name=image_name,
        output_dir=output_dir,
        ext=".mp4",
        fourcc="mp4v",
        padding=padding,
        fps=fps,
        channel_colors=channel_colors,
        upscale=upscale,
        norm_lower=norm_lower,
        norm_upper=norm_upper,
        circle_r=circle_r,
        line_w=line_w,
        only_track_frames=only_track_frames,
        pseudocolor_single=pseudocolor_single,
        annotate=annotate,
    )


def save_track_movies_avi(
    img_array,
    track_df,
    image_name,
    output_dir,
    padding=40,
    fps=10,
    channel_colors=None,
    upscale=1,
    norm_lower=2,
    norm_upper=98,
    circle_r=6,
    line_w=2,
    only_track_frames=True,
    pseudocolor_single=False,
    annotate=True,
):
    _save_track_movies_video(
        img_array=img_array,
        track_df=track_df,
        image_name=image_name,
        output_dir=output_dir,
        ext=".avi",
        fourcc="MJPG",
        padding=padding,
        fps=fps,
        channel_colors=channel_colors,
        upscale=upscale,
        norm_lower=norm_lower,
        norm_upper=norm_upper,
        circle_r=circle_r,
        line_w=line_w,
        only_track_frames=only_track_frames,
        pseudocolor_single=pseudocolor_single,
        annotate=annotate,
    )


def _collect_layer_meta(viewer):
    """Pull run_params from Points and Tracks layers stored at run-time."""
    spot_meta = {}
    track_meta = {}

    pts_layers = []
    trk_layers = []

    for ly in viewer.layers:
        pt_meta = _get_pt_meta(ly)
        role = pt_meta.get("role")

        if ly.__class__.__name__ == "Points" and role == "puncta":
            pts_layers.append(ly)

        if ly.__class__.__name__ == "Tracks" and role == "tracks":
            trk_layers.append(ly)

    if pts_layers:
        spot_meta = _get_pt_meta(pts_layers[0]).get("run_params", {})
        if not spot_meta:
            # fallback: some versions stored directly on metadata
            spot_meta = pts_layers[0].metadata.get("run_params", {})

    if trk_layers:
        track_meta = _get_pt_meta(trk_layers[0]).get("run_params", {})
        if not track_meta:
            track_meta = trk_layers[0].metadata.get("run_params", {})

    return spot_meta, track_meta


def save_tracks_csv(
    tracks_df: pd.DataFrame,
    out_path: str,
    spot_meta=None,
    track_meta=None,
    image_meta=None,
    add_metadata: bool = True,
):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_df = tracks_df.copy()

    if add_metadata:
        out_df = add_export_metadata(
            out_df,
            spot_meta=spot_meta,
            track_meta=track_meta,
            image_meta=image_meta,
        )

    out_df.to_csv(out_path, index=False)


def save_track_gifs(
    img_array,
    track_df,
    image_name,
    output_dir,
    padding=40,
    fps=10,
    upscale=2,
    norm_lower=2,
    norm_upper=98,
    channel_colors=None,
    circle_r=6,
    line_w=2,
    only_track_frames=True,
    pseudocolor_single=False,
    annotate=True,
):
    if img_array.ndim != 4:
        raise ValueError(f"Expected (T,C,Y,X), got {img_array.shape}")

    os.makedirs(output_dir, exist_ok=True)

    T, C, Y, X = img_array.shape
    base_name = os.path.splitext(os.path.basename(image_name))[0].replace(" ", "_")
    duration = 1.0 / fps

    if channel_colors is None:
        channel_colors = {
            0: (255, 0, 255),
            1: (255, 0, 0),
            2: (0, 255, 0),
        }

    df = track_df.copy()
    df["frame"] = df["frame"].astype(int)
    df = df.sort_values(["particle", "frame"])

    for pid, group in df.groupby("particle"):
        group = group.sort_values("frame")

        y_min, y_max = int(np.floor(group["y"].min())), int(np.ceil(group["y"].max()))
        x_min, x_max = int(np.floor(group["x"].min())), int(np.ceil(group["x"].max()))

        y1 = max(y_min - padding, 0)
        y2 = min(y_max + padding, Y)
        x1 = max(x_min - padding, 0)
        x2 = min(x_max + padding, X)

        if y2 <= y1 or x2 <= x1:
            continue

        crop = img_array[:, :, y1:y2, x1:x2]
        crop_norm = normalize_crop_percentile(crop, lower=norm_lower, upper=norm_upper)
        h, w = crop_norm.shape[2], crop_norm.shape[3]

        centers = {
            int(r["frame"]): (float(r["x"]) - x1, float(r["y"]) - y1)
            for _, r in group.iterrows()
            if 0 <= int(r["frame"]) < T
        }

        if only_track_frames:
            frame_list = sorted(centers.keys())
        else:
            frame_list = list(range(T))

        if not frame_list:
            continue

        path_so_far = []
        track_by_frame = {}
        for t in sorted(centers.keys()):
            path_so_far.append(centers[t])
            track_by_frame[t] = path_so_far.copy()

        # composite gif frames, always pseudocolored
        comp_raw = []
        comp_ann = []

        # single-channel gif frame buffers
        single_raw = {c: [] for c in range(C)}
        single_ann = {c: [] for c in range(C)}

        for t in frame_list:
            center = centers.get(t)
            pts = track_by_frame.get(t, [])

            comp_rgb = _composite_rgb_frame(
                crop_norm=crop_norm,
                t=t,
                C=C,
                channel_colors=channel_colors,
                upscale=upscale,
            )
            comp_raw.append(comp_rgb)

            comp_rgb_ann = comp_rgb.copy()
            if annotate and center is not None:
                comp_rgb_ann = _draw_overlays(
                    comp_rgb_ann,
                    center_xy=(int(center[0] * upscale), int(center[1] * upscale)),
                    track_xy=[(int(p[0] * upscale), int(p[1] * upscale)) for p in pts],
                    circle_r=circle_r * upscale,
                    line_w=max(1, line_w * upscale),
                )
            comp_ann.append(comp_rgb_ann)

            for c in range(C):
                gray = crop_norm[t, c]

                if pseudocolor_single:
                    raw = _resize_nn(_apply_pseudocolor(gray, channel_colors.get(c, (255, 255, 255))), upscale)
                else:
                    raw = _resize_nn(_to_rgb(gray), upscale)

                ann = raw.copy()
                if annotate and center is not None:
                    ann = _draw_overlays(
                        ann,
                        center_xy=(int(center[0] * upscale), int(center[1] * upscale)),
                        track_xy=[(int(p[0] * upscale), int(p[1] * upscale)) for p in pts],
                        circle_r=circle_r * upscale,
                        line_w=max(1, line_w * upscale),
                    )

                single_raw[c].append(raw)
                single_ann[c].append(ann)

        fn_raw = os.path.join(output_dir, f"{base_name}_track{pid}_composite_raw.gif")
        iio.imwrite(fn_raw, comp_raw, duration=duration, loop=0)
        if annotate:
            fn_ann = os.path.join(output_dir, f"{base_name}_track{pid}_composite_annot.gif")
            iio.imwrite(fn_ann, comp_ann, duration=duration, loop=0)

        for c in range(C):
            fn_ch_raw = os.path.join(output_dir, f"{base_name}_track{pid}_ch{c}_raw.gif")
            iio.imwrite(fn_ch_raw, single_raw[c], duration=duration, loop=0)
            if annotate:
                fn_ch_ann = os.path.join(output_dir, f"{base_name}_track{pid}_ch{c}_annot.gif")
                iio.imwrite(fn_ch_ann, single_ann[c], duration=duration, loop=0)

        print(f"Saved gifs for track {pid}")


def resample_polyline_equal_arclength(xy: np.ndarray, n_samples: int) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be (N,2)")
    if xy.shape[0] < 2:
        return np.repeat(xy[:1], n_samples, axis=0)
    seg = xy[1:] - xy[:-1]
    d = np.sqrt((seg ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    if total < 1e-6:
        return np.repeat(xy[:1], n_samples, axis=0)
    s_new = np.linspace(0, total, n_samples, dtype=np.float32)
    return np.stack(
        [
            np.interp(s_new, s, xy[:, 0]),
            np.interp(s_new, s, xy[:, 1]),
        ],
        axis=1,
    ).astype(np.float32)


def tangents_and_normals_from_path(path_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(path_xy, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("path_xy must be (L,2) in (x,y)")
    dp = np.zeros_like(p)
    dp[1:-1] = p[2:] - p[:-2]
    dp[0] = p[1] - p[0]
    dp[-1] = p[-1] - p[-2]
    n = np.linalg.norm(dp, axis=1, keepdims=True) + 1e-12
    t_hat = dp / n
    n_hat = np.stack([-t_hat[:, 1], t_hat[:, 0]], axis=1)
    return t_hat.astype(np.float32), n_hat.astype(np.float32)


def kymograph_along_fixed_path(
    stack: np.ndarray,          # (T, C, Y, X) or (T, Y, X)
    frames_0based: np.ndarray,  # (N,)
    path_xy: np.ndarray,        # (L,2)
    width_px: int = 9,
    width_reduce: str = "mean",  # "mean" or "max"
    fill_value: float = np.nan,
) -> np.ndarray:
    if stack.ndim == 3:
        stack = stack[:, None, :, :]
    if stack.ndim != 4:
        raise ValueError(f"stack must be (T,C,Y,X) or (T,Y,X), got {stack.shape}")

    T, C, Y, X = stack.shape
    frames_0based = np.asarray(frames_0based, dtype=int)
    path_xy = np.asarray(path_xy, dtype=np.float32)

    L = path_xy.shape[0]
    if width_px < 1:
        raise ValueError("width_px must be positive")

    _, n_hat = tangents_and_normals_from_path(path_xy)

    u = np.linspace(-(width_px - 1) / 2, (width_px - 1) / 2, width_px, dtype=np.float32)

    kymo = np.full((len(frames_0based), C, L), fill_value, dtype=np.float32)

    x0 = path_xy[:, 0]
    y0 = path_xy[:, 1]
    nx = n_hat[:, 0]
    ny = n_hat[:, 1]

    for i, t_idx in enumerate(frames_0based):
        if t_idx < 0 or t_idx >= T:
            continue

        xs = x0[None, :] + u[:, None] * nx[None, :]
        ys = y0[None, :] + u[:, None] * ny[None, :]

        for c in range(C):
            img = stack[t_idx, c].astype(np.float32)

            vals = cv2.remap(
                img,
                xs.astype(np.float32),
                ys.astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=np.nan,
            )

            if width_reduce == "max":
                prof = np.nanmax(vals, axis=0)
            else:
                prof = np.nanmean(vals, axis=0)

            kymo[i, c, :] = prof.astype(np.float32)

    return kymo


def save_kymos(out_dir, track_id, kymo_ncl, save_float_tif=True,
               save_png=True, make_rgb=True, png_scale=1, png_bitdepth=8,
               channel_colors=None):
    os.makedirs(out_dir, exist_ok=True)
    N, C, L = kymo_ncl.shape

    def _scale(img2d):
        vmin, vmax = np.nanpercentile(img2d, 1), np.nanpercentile(img2d, 99)
        scaled = np.clip((img2d - vmin) / (vmax - vmin + 1e-12), 0, 1)
        scaled = np.nan_to_num(scaled)
        out = (scaled * (65535 if png_bitdepth == 16 else 255)).astype(
            np.uint16 if png_bitdepth == 16 else np.uint8
        )
        if png_scale > 1:
            out = np.kron(out, np.ones((png_scale, png_scale), dtype=out.dtype))
        return out

    for c in range(C):
        img = kymo_ncl[:, c, :]
        if save_float_tif:
            iio.imwrite(
                os.path.join(out_dir, f"track_{track_id}_kymo_ch{c}.tif"),
                img.astype(np.float32)
            )
        if save_png:
            iio.imwrite(os.path.join(out_dir, f"track_{track_id}_kymo_ch{c}.png"), _scale(img))

    if make_rgb and C >= 2 and save_png:
        rgb = np.zeros((N, L, 3), dtype=np.uint8)
        for ch in range(C):
            vmin = np.nanpercentile(kymo_ncl[:, ch, :], 1)
            vmax = np.nanpercentile(kymo_ncl[:, ch, :], 99)
            scaled = np.clip(
                (kymo_ncl[:, ch, :] - vmin) / (vmax - vmin + 1e-12), 0, 1
            )
            gray = np.nan_to_num(scaled).astype(np.float32)

            if channel_colors is not None and ch in channel_colors:
                r, g, b = [v / 255.0 for v in channel_colors[ch]]
            else:
                # cycle through some distinguishable colors for unmapped channels
                _fallback = [
                    (1.0, 0.0, 1.0),  # magenta
                    (1.0, 0.0, 0.0),  # red
                    (0.0, 1.0, 0.0),  # green
                    (0.0, 0.0, 1.0),  # blue
                    (1.0, 1.0, 0.0),  # yellow
                    (0.0, 1.0, 1.0),  # cyan
                ]
                r, g, b = _fallback[ch % len(_fallback)]

            rgb[..., 0] = np.clip(rgb[..., 0] + gray * r * 255, 0, 255).astype(np.uint8)
            rgb[..., 1] = np.clip(rgb[..., 1] + gray * g * 255, 0, 255).astype(np.uint8)
            rgb[..., 2] = np.clip(rgb[..., 2] + gray * b * 255, 0, 255).astype(np.uint8)

        if png_scale > 1:
            rgb = np.kron(rgb, np.ones((png_scale, png_scale, 1), dtype=rgb.dtype))
        iio.imwrite(os.path.join(out_dir, f"track_{track_id}_kymo_rgb.png"), rgb)


def export_all_track_kymographs_from_array(
    img_array, tracks_df, out_root, image_name,
    L=200, W=9, width_reduce="mean", min_len=3,
    include_all_frames=False, frame_base=0,
    channel_colors=None,
):
    validate_tracks_df(tracks_df)
    if img_array.ndim == 3:
        img_array = img_array[:, None, :, :]
    T, C, Y, X = img_array.shape
    image_root = os.path.splitext(os.path.basename(image_name))[0]
    out_root_img = os.path.join(out_root, image_root)
    os.makedirs(out_root_img, exist_ok=True)

    df = tracks_df.copy()
    if "image_name" in df.columns:
        df["_root"] = df["image_name"].astype(str).apply(
            lambda s: os.path.splitext(os.path.basename(s))[0]
        )
        df = df[df["_root"] == image_root].drop(columns=["_root"], errors="ignore")
    if df.empty:
        raise ValueError(f"No tracks found for image '{image_root}'")

    for pid, g in df.groupby("particle", sort=True):
        g = g.sort_values("frame")
        if len(g) < min_len:
            continue
        frames_track = g["frame"].to_numpy(dtype=int) - frame_base
        xy_track = g[["x", "y"]].to_numpy(dtype=np.float32)
        path_xy = resample_polyline_equal_arclength(xy_track, n_samples=L)
        frames_used = np.arange(img_array.shape[0], dtype=int) if include_all_frames else frames_track
        kymo = kymograph_along_fixed_path(
            stack=img_array, frames_0based=frames_used,
            path_xy=path_xy, width_px=W, width_reduce=width_reduce,
        )
        out_dir = os.path.join(out_root_img, f"particle_{pid}")
        save_kymos(
            out_dir=out_dir, track_id=pid, kymo_ncl=kymo,
            save_float_tif=True, save_png=True, make_rgb=True,
            png_scale=3, png_bitdepth=8, channel_colors=channel_colors,
        )
        meta = g[["frame", "x", "y"]].copy()
        meta["frame_0based"] = frames_track
        meta.to_csv(os.path.join(out_dir, f"track_{pid}_meta.csv"), index=False)
        np.savetxt(
            os.path.join(out_dir, f"track_{pid}_path_xy.csv"),
            path_xy, delimiter=",", header="x,y", comments=""
        )
        pd.DataFrame({"frame_0based_used": frames_used}).to_csv(
            os.path.join(out_dir, f"track_{pid}_kymo_frames_used.csv"), index=False
        )


class ExportWidget(QWidget):
    def __init__(self, viewer, image_import_widget, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.image_import_widget = image_import_widget
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # CSV export group
        csv_group = QGroupBox("CSV Export")
        csv_layout = QVBoxLayout(csv_group)

        row1 = QHBoxLayout()
        self.csv_path = QLineEdit()
        self.csv_path.setPlaceholderText("Output CSV path...")
        csv_btn = QPushButton("Browse...")
        csv_btn.clicked.connect(self._pick_csv)
        row1.addWidget(self.csv_path)
        row1.addWidget(csv_btn)
        csv_layout.addLayout(row1)

        self.save_validated_btn = QPushButton("Export validated tracks CSV")
        self.save_validated_btn.setToolTip(
            "Saves only kept tracks — clean ground truth, no status column."
        )
        self.save_validated_btn.clicked.connect(self._save_validated_csv)
        csv_layout.addWidget(self.save_validated_btn)

        self.save_all_btn = QPushButton("Export all tracks with flags")
        self.save_all_btn.setToolTip(
            "Saves kept + deleted + pending tracks, each row has a 'status' column."
        )
        self.save_all_btn.clicked.connect(self._save_all_csv)
        csv_layout.addWidget(self.save_all_btn)

        layout.addWidget(csv_group)

        # Movie / GIF export group
        movie_group = QGroupBox("Movie / GIF Export")
        movie_layout = QVBoxLayout(movie_group)

        row2 = QHBoxLayout()
        self.movie_dir = QLineEdit()
        self.movie_dir.setPlaceholderText("Output folder...")
        movie_btn = QPushButton("Browse...")
        movie_btn.clicked.connect(self._pick_movie_dir)
        row2.addWidget(self.movie_dir)
        row2.addWidget(movie_btn)
        movie_layout.addLayout(row2)

        self.export_format = QComboBox()
        self.export_format.addItems(["MP4", "GIF", "AVI"])
        movie_layout.addWidget(QLabel("Movie format"))
        movie_layout.addWidget(self.export_format)

        self.padding = QSpinBox()
        self.padding.setRange(0, 1000)
        self.padding.setValue(40)

        self.fps = QSpinBox()
        self.fps.setRange(1, 120)
        self.fps.setValue(10)

        self.upscale = QSpinBox()
        self.upscale.setRange(1, 10)
        self.upscale.setValue(2)

        movie_layout.addWidget(QLabel("Padding"))
        movie_layout.addWidget(self.padding)
        movie_layout.addWidget(QLabel("FPS"))
        movie_layout.addWidget(self.fps)
        movie_layout.addWidget(QLabel("Upscale"))
        movie_layout.addWidget(self.upscale)
        self.annotate = QCheckBox("Draw track overlays (circle + path)")
        self.annotate.setChecked(True)
        movie_layout.addWidget(self.annotate)
        self.pseudocolor_single = QCheckBox("Pseudocolor single-channel movies")
        self.pseudocolor_single.setChecked(False)
        movie_layout.addWidget(self.pseudocolor_single)
        self.only_track_frames = QCheckBox("Only frames where track exists")
        self.only_track_frames.setChecked(False)
        movie_layout.addWidget(self.only_track_frames)

        self.export_movie_btn = QPushButton("Export track movies")
        self.export_movie_btn.clicked.connect(self._save_movie)
        movie_layout.addWidget(self.export_movie_btn)

        layout.addWidget(movie_group)

        # Kymograph export group
        kymo_group = QGroupBox("Kymograph Export")
        kymo_layout = QVBoxLayout(kymo_group)

        row3 = QHBoxLayout()
        self.kymo_dir = QLineEdit()
        self.kymo_dir.setPlaceholderText("Output folder...")
        kymo_btn = QPushButton("Browse...")
        kymo_btn.clicked.connect(self._pick_kymo_dir)
        row3.addWidget(self.kymo_dir)
        row3.addWidget(kymo_btn)
        kymo_layout.addLayout(row3)

        self.save_kymo_btn = QPushButton("Export kymographs")
        self.save_kymo_btn.clicked.connect(self._save_kymographs)
        kymo_layout.addWidget(self.save_kymo_btn)

        layout.addWidget(kymo_group)

        layout.addStretch(1)

    def _pick_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV (*.csv)")
        if path:
            self.csv_path.setText(path)

    def _pick_movie_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.movie_dir.setText(path)

    def _pick_kymo_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.kymo_dir.setText(path)

    def _get_state(self):
        state = get_validation_state(self.viewer)
        if state is None or state.is_empty:
            raise ValueError(
                "No validation data found. Run tracking and validate at least one track first."
            )
        return state

    def _get_image_array(self):
        arr = getattr(self.image_import_widget, "_cached_array", None)
        if arr is None:
            raise ValueError("No imported image stack available.")
        return arr

    def _get_shared_meta(self):
        spot_meta, track_meta = _collect_layer_meta(self.viewer)
        if not spot_meta:
            show_warning(
                "Detection parameters not found — re-run detection before exporting "
                "to ensure accurate metadata."
            )
        if not track_meta:
            show_warning(
                "Tracking parameters not found — re-run tracking before exporting "
                "to ensure accurate metadata."
            )
        image_meta = {"image_name": getattr(self.image_import_widget, "nd2_path", "")}
        return spot_meta, track_meta, image_meta

    def _get_channel_colors(self):
        return _collect_channel_colors(self.image_import_widget, self.viewer)

    def _require_csv_path(self):
        path = self.csv_path.text().strip()
        if not path:
            raise ValueError("Choose a CSV output path first.")
        return path

    def _require_movie_dir(self):
        d = self.movie_dir.text().strip()
        if not d:
            raise ValueError("Choose an output directory first.")
        return d
    
    def _require_kymo_dir(self):
        d = self.kymo_dir.text().strip()
        if not d:
            raise ValueError("Choose an output directory first.")
        return d

    def _save_validated_csv(self):
        """Export only kept tracks — clean ground truth, no status column."""
        try:
            out_path = self._require_csv_path()
            state = self._get_state()

            if state.kept_df.empty:
                raise ValueError(
                    "No kept tracks yet. Use the Tracks List to keep tracks before exporting."
                )

            spot_meta, track_meta, image_meta = self._get_shared_meta()
            save_tracks_csv(
                state.validated_df(),
                out_path,
                spot_meta=spot_meta,
                track_meta=track_meta,
                image_meta=image_meta,
                add_metadata=True,
            )
            show_info(
                f"Saved validated CSV: {out_path}  "
                f"({state.n_kept} tracks)"
            )
        except Exception as e:
            show_error(str(e))

    def _save_all_csv(self):
        """Export all tracks with a 'status' column (kept / deleted / pending)."""
        try:
            out_path = self._require_csv_path()
            state = self._get_state()

            spot_meta, track_meta, image_meta = self._get_shared_meta()
            all_df = state.all_tracks_df()

            if all_df.empty:
                raise ValueError("No tracks available to export.")

            save_tracks_csv(
                all_df,
                out_path,
                spot_meta=spot_meta,
                track_meta=track_meta,
                image_meta=image_meta,
                add_metadata=True,
            )
            show_info(
                f"Saved all-tracks CSV: {out_path}  "
                f"(kept={state.n_kept}, deleted={state.n_deleted}, "
                f"pending={state.n_pending})"
            )
        except Exception as e:
            show_error(str(e))

    def _get_export_df(self):
        """Returns kept tracks for movie/gif/kymo export."""
        state = self._get_state()
        if state.kept_df.empty:
            raise ValueError(
                "No kept tracks yet. Validate tracks before exporting movies."
            )
        return state.validated_df()

    def _save_movie(self):
        try:
            out_dir = self._require_movie_dir()
            df = self._get_export_df()
            arr = self._get_image_array()
            image_name = getattr(self.image_import_widget, "nd2_path", "image.nd2")
            channel_colors = self._get_channel_colors()

            fmt = self.export_format.currentText().strip().upper()

            common = dict(
                img_array=arr,
                track_df=df,
                image_name=image_name,
                output_dir=out_dir,
                padding=self.padding.value(),
                fps=self.fps.value(),
                upscale=self.upscale.value(),
                only_track_frames=self.only_track_frames.isChecked(),
                channel_colors=channel_colors,
                pseudocolor_single=self.pseudocolor_single.isChecked(),
                annotate=self.annotate.isChecked(),
            )

            if fmt == "MP4":
                save_track_movies_mp4(**common)
            elif fmt == "GIF":
                save_track_gifs(**common)
            elif fmt == "AVI":
                save_track_movies_avi(**common)
            else:
                raise ValueError(f"Unknown export format: {fmt}")

            show_info(f"{fmt} export complete.")
        except Exception as e:
            show_error(str(e))

    def _save_kymographs(self):
        try:
            out_dir = self._require_kymo_dir()
            df = self._get_export_df()
            arr = self._get_image_array()
            image_name = getattr(self.image_import_widget, "nd2_path", "image.nd2")
            export_all_track_kymographs_from_array(
                img_array=arr, tracks_df=df, out_root=out_dir,
                image_name=image_name, L=200, W=9,
                width_reduce="mean", include_all_frames=False, frame_base=0,
                channel_colors=self._get_channel_colors(),
            )
            show_info("Kymograph export complete.")
        except Exception as e:
            show_error(str(e))