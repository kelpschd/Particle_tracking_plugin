# export.py
import os
import numpy as np
import pandas as pd
import cv2
import imageio.v3 as iio
from PIL import Image, ImageDraw

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox
)
from napari.utils.notifications import show_info, show_warning, show_error
from ._widget import particle_detection_widget
from ._tracking_widget import tracking_widget

def add_export_metadata(df: pd.DataFrame, spot_meta=None, track_meta=None, image_meta=None):
    out = df.copy()

    spot_meta = spot_meta or {}
    track_meta = track_meta or {}
    image_meta = image_meta or {}

    # repeat scalar metadata on every row
    for prefix, meta in [("spot", spot_meta), ("track", track_meta), ("image", image_meta)]:
        for key, value in meta.items():
            col = f"{prefix}_{key}"
            if np.isscalar(value) or isinstance(value, str):
                out[col] = value
            else:
                # if it's already per-row, keep it only if lengths match
                value = np.asarray(value)
                if len(value) == len(out):
                    out[col] = value
                else:
                    out[col] = np.nan

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


def save_track_movies_mp4(
    img_array,
    track_df,
    image_name,
    output_dir,
    padding=40,
    fps=10,
    channel_colors=None,
    composite_channels=(0, 1, 2),
    upscale=1,
    norm_lower=2,
    norm_upper=98,
    circle_r=6,
    line_w=2,
    only_track_frames=True,
):
    """
    img_array: (T, C, Y, X)
    track_df: must contain particle, frame, y, x
    """
    if img_array.ndim != 4:
        raise ValueError(f"Expected (T,C,Y,X), got {img_array.shape}")

    os.makedirs(output_dir, exist_ok=True)

    if channel_colors is None:
        channel_colors = {
            0: (255, 0, 255),
            1: (255, 0, 0),
            2: (0, 255, 0),
        }

    T, C, Y, X = img_array.shape
    base_name = os.path.splitext(os.path.basename(image_name))[0].replace(" ", "_")

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

        for c in range(C):
            out_path = os.path.join(output_dir, f"{base_name}_track{pid}_ch{c}.mp4")
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w * upscale, h * upscale),
                isColor=True,
            )

            for t in frame_list:
                gray = crop_norm[t, c]
                rgb = _to_rgb(gray)
                rgb = _resize_nn(rgb, upscale)

                center = centers.get(t)
                if center is not None:
                    rgb = _draw_overlays(
                        rgb,
                        center_xy=(int(center[0] * upscale), int(center[1] * upscale)),
                        track_xy=[
                            (int(p[0] * upscale), int(p[1] * upscale))
                            for p in track_by_frame.get(t, [])
                        ],
                        circle_r=circle_r * upscale,
                        line_w=max(1, line_w * upscale),
                    )

                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            writer.release()
            print(f"Saved {out_path}")


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
    composite_channels=(0, 1, 2),
    circle_r=6,
    line_w=2,
    only_track_frames=True,
):
    if img_array.ndim != 4:
        raise ValueError(f"Expected (T,C,Y,X), got {img_array.shape}")

    os.makedirs(output_dir, exist_ok=True)

    if channel_colors is None:
        channel_colors = {
            0: (255, 0, 255),
            1: (255, 0, 0),
            2: (0, 255, 0),
        }

    T, C, Y, X = img_array.shape
    base_name = os.path.splitext(os.path.basename(image_name))[0].replace(" ", "_")
    duration = 1.0 / fps

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

        comp_raw = []
        comp_ann = []

        for t in frame_list:
            center = centers.get(t)
            pts = track_by_frame.get(t, [])

            pseudo_list = []
            for c in range(C):
                gray = crop_norm[t, c]
                if c in composite_channels:
                    pseudo_list.append(_resize_nn(_apply_pseudocolor(gray, channel_colors.get(c, (255, 255, 255))), upscale))

            merged = _merge_rgb(pseudo_list) if pseudo_list else np.zeros((h * upscale, w * upscale, 3), dtype=np.uint8)

            comp_raw.append(merged)

            merged_ann = merged
            if center is not None:
                merged_ann = _draw_overlays(
                    merged.copy(),
                    center_xy=(int(center[0] * upscale), int(center[1] * upscale)),
                    track_xy=[(int(p[0] * upscale), int(p[1] * upscale)) for p in pts],
                    circle_r=circle_r * upscale,
                    line_w=max(1, line_w * upscale),
                )
            comp_ann.append(merged_ann)

        fn_raw = os.path.join(output_dir, f"{base_name}_track{pid}_composite_raw.gif")
        fn_ann = os.path.join(output_dir, f"{base_name}_track{pid}_composite_annot.gif")
        iio.imwrite(fn_raw, comp_raw, duration=duration, loop=0)
        iio.imwrite(fn_ann, comp_ann, duration=duration, loop=0)
        print(f"Saved gifs for track {pid}")

class ExportWidget(QWidget):
    def __init__(self, viewer, image_import_widget, tracks_table_widget, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.image_import_widget = image_import_widget
        self.tracks_table_widget = tracks_table_widget
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Export validated tracks"))

        self.csv_path = QLineEdit()
        csv_btn = QPushButton("Browse CSV...")
        csv_btn.clicked.connect(self._pick_csv)
        row1 = QHBoxLayout()
        row1.addWidget(self.csv_path)
        row1.addWidget(csv_btn)
        layout.addLayout(row1)

        self.movie_dir = QLineEdit()
        movie_btn = QPushButton("Browse movie folder...")
        movie_btn.clicked.connect(self._pick_movie_dir)
        row2 = QHBoxLayout()
        row2.addWidget(self.movie_dir)
        row2.addWidget(movie_btn)
        layout.addLayout(row2)

        self.save_csv_btn = QPushButton("Save validated tracks CSV")
        self.save_csv_btn.clicked.connect(self._save_csv)
        layout.addWidget(self.save_csv_btn)

        self.save_mp4_btn = QPushButton("Export MP4 movies")
        self.save_mp4_btn.clicked.connect(self._save_mp4)
        layout.addWidget(self.save_mp4_btn)

        self.save_gif_btn = QPushButton("Export GIFs")
        self.save_gif_btn.clicked.connect(self._save_gif)
        layout.addWidget(self.save_gif_btn)

        self.padding = QSpinBox()
        self.padding.setRange(0, 1000)
        self.padding.setValue(40)

        self.fps = QSpinBox()
        self.fps.setRange(1, 120)
        self.fps.setValue(10)

        self.upscale = QSpinBox()
        self.upscale.setRange(1, 10)
        self.upscale.setValue(2)

        self.only_track_frames = QCheckBox("Only frames where track exists")
        self.only_track_frames.setChecked(True)

        self.annotate = QCheckBox("Include overlays")
        self.annotate.setChecked(True)

        layout.addWidget(QLabel("Padding"))
        layout.addWidget(self.padding)
        layout.addWidget(QLabel("FPS"))
        layout.addWidget(self.fps)
        layout.addWidget(QLabel("Upscale"))
        layout.addWidget(self.upscale)
        layout.addWidget(self.only_track_frames)
        layout.addWidget(self.annotate)

        layout.addStretch(1)

    def _pick_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV (*.csv)")
        if path:
            self.csv_path.setText(path)

    def _pick_movie_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.movie_dir.setText(path)

    def _get_tracks_df(self):
        # 1) Prefer whatever is currently in the table widget
        df = self.tracks_table_widget.dataframe
        if df is not None and not df.empty:
            return df

        # 2) Fallback: try the active Tracks layer
        for ly in self.viewer.layers:
            if ly.__class__.__name__ == "Tracks":
                try:
                    from ._helpers import tracks_layer_to_dataframe
                    df = tracks_layer_to_dataframe(ly)
                    if df is not None and not df.empty:
                        # keep the table in sync for next time
                        self.tracks_table_widget.dataframe = df
                        return df
                except Exception:
                    pass

        raise ValueError("No tracks loaded in the table.")

    def _get_image_array(self):
        arr = getattr(self.image_import_widget, "_cached_array", None)
        if arr is None:
            raise ValueError("No imported image stack available.")
        return arr

    def _save_csv(self):
        try:
            out_path = self.csv_path.text().strip()
            if not out_path:
                raise ValueError("Choose a CSV output path first.")

            df = self._get_tracks_df()

            spot_meta = {
                "median_filter_size": particle_detection_widget.median_filter_size.value,
                "threshold": particle_detection_widget.threshold.value,
                "min_sigma": particle_detection_widget.min_sigma.value,
                "max_sigma": particle_detection_widget.max_sigma.value,
                "enable_density_filter": particle_detection_widget.enable_density_filter.value,
                "bin_size": particle_detection_widget.bin_size.value,
                "blob_filter": particle_detection_widget.blob_filter.value,
            }

            track_meta = {
                "max_distance": tracking_widget.max_distance.value,
                "max_missed": tracking_widget.max_missed.value,
                "disp_min_frames": tracking_widget.disp_min_frames.value,
                "disp_threshold": tracking_widget.disp_threshold.value,
                "max_crossings": tracking_widget.max_crossings.value,
            }

            image_meta = {
                "image_name": getattr(self.image_import_widget, "nd2_path", ""),
            }

            save_tracks_csv(
                df,
                out_path,
                spot_meta=spot_meta,
                track_meta=track_meta,
                image_meta=image_meta,
                add_metadata=self.annotate.isChecked(),
            )
            show_info(f"Saved CSV: {out_path}")
        except Exception as e:
            show_error(str(e))

    def _save_mp4(self):
        try:
            out_dir = self.movie_dir.text().strip()
            if not out_dir:
                raise ValueError("Choose an output directory first.")
            df = self._get_tracks_df()
            arr = self._get_image_array()
            image_name = getattr(self.image_import_widget, "nd2_path", "image.nd2")

            save_track_movies_mp4(
                img_array=arr,
                track_df=df,
                image_name=image_name,
                output_dir=out_dir,
                padding=self.padding.value(),
                fps=self.fps.value(),
                upscale=self.upscale.value(),
                only_track_frames=self.only_track_frames.isChecked(),
            )
            show_info("MP4 export complete.")
        except Exception as e:
            show_error(str(e))

    def _save_gif(self):
        try:
            out_dir = self.movie_dir.text().strip()
            if not out_dir:
                raise ValueError("Choose an output directory first.")
            df = self._get_tracks_df()
            arr = self._get_image_array()
            image_name = getattr(self.image_import_widget, "nd2_path", "image.nd2")

            save_track_gifs(
                img_array=arr,
                track_df=df,
                image_name=image_name,
                output_dir=out_dir,
                padding=self.padding.value(),
                fps=self.fps.value(),
                upscale=self.upscale.value(),
                only_track_frames=self.only_track_frames.isChecked(),
            )
            show_info("GIF export complete.")
        except Exception as e:
            show_error(str(e))