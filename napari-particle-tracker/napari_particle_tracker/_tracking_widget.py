# tracking_widget.py
import numpy as np
import pandas as pd
from functools import partial

from qtpy.QtWidgets import (
    QWidget, 
    QVBoxLayout, 
    QTableWidget, 
    QTableWidgetItem,
    QPushButton, 
    QSizePolicy, 
    QLabel, 
    QAbstractItemView,
    QHeaderView,
    QDialog,
    QHBoxLayout,
    QMessageBox,
    QLineEdit,
    QFormLayout
)
from qtpy.QtCore import Qt

import napari
from napari.layers import Points, Image as NapariImage
try:
    # thread_worker moved around between napari versions
    from napari.qt.threading import thread_worker
except Exception:
    try:
        from napari.utils import thread_worker
    except Exception:
        thread_worker = None  # we'll handle missing decorator later

from magicgui import magicgui
from scipy.spatial import cKDTree
from shapely.geometry import LineString

import trackpy as tp
try:
    from filterpy.kalman import KalmanFilter
except Exception as e:
    KalmanFilter = None
    _KALMAN_IMPORT_ERR = e

from napari.utils.notifications import show_info, show_warning, show_error

from .tracks_table_widget import TracksTableWidget, _open_tracks_table
from ._helpers import tracks_layer_to_dataframe, dataframe_to_tracks_layer_data

# -------------------------
# Kalman tracker
# -------------------------
class KalmanTrack:
    def __init__(self, initial_detection, particle_id, frame, max_missed=5):
        x, y = initial_detection  # note: detection passed in as (x,y) in kalman code
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
        self.kf.R = np.eye(2) * 5.0
        self.kf.P = np.eye(4) * 1000.0
        self.kf.Q = np.eye(4) * 0.1
        # store as (x,y) internally in kf state
        self.kf.x = np.array([x, y, 0.0, 0.0], dtype=float)

        self.id = int(particle_id)
        # store history as (frame, x, y)
        self.history = [{'frame': int(frame), 'x': float(x), 'y': float(y), 'particle': int(self.id)}]
        self.missed = 0
        self.max_missed = int(max_missed)

    def predict(self):
        self.kf.predict()
        return self.kf.x[:2]  # returns (x, y)

    def update(self, detection, frame):
        self.kf.update(np.asarray(detection, dtype=float))
        x, y = self.kf.x[:2]
        self.history.append({'frame': int(frame), 'x': float(x), 'y': float(y), 'particle': int(self.id)})
        self.missed = 0

    def mark_missed(self):
        self.missed += 1

    def is_dead(self):
        return self.missed > self.max_missed


# -------------------------
# Kalman linking function
# -------------------------
def kalman_track_blobs(all_blobs: pd.DataFrame, max_distance: float = 15.0, max_missed: int = 5) -> pd.DataFrame:
    """
    all_blobs: DataFrame with columns ['frame','x','y'] (optionally others).
    Returns a DataFrame with ['particle','frame','y','x'] (note the y,x ordering
    which matches napari's Tracks convention [track_id, t, y, x]).
    """
    if KalmanFilter is None:
        raise ImportError(f"filterpy not available: {_KALMAN_IMPORT_ERR}")

    need = {'frame', 'x', 'y'}
    if not need.issubset(set(all_blobs.columns)):
        raise ValueError(f"Expected columns {need}, got {set(all_blobs.columns)}")

    # operate on a copy
    blobs = all_blobs[['frame', 'x', 'y']].copy()
    blobs['frame'] = blobs['frame'].astype(int)
    blobs = blobs.sort_values('frame')

    active_tracks = []
    finished_tracks = []
    next_id = 0

    for frame, detections in blobs.groupby('frame', sort=True):
        detected_positions = detections[['x', 'y']].to_numpy(dtype=float)  # shape (M, 2) as (x,y)
        # predict
        predictions = [track.predict() for track in active_tracks]  # list of (x,y)
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

        # start new tracks for unassigned detections
        for i, pos in enumerate(detected_positions):
            if i not in assigned:
                new_track = KalmanTrack(pos, next_id, frame, max_missed)
                active_tracks.append(new_track)
                next_id += 1

        # remove dead tracks
        still_alive = []
        for track in active_tracks:
            if track.is_dead():
                finished_tracks.append(track)
            else:
                still_alive.append(track)
        active_tracks = still_alive

    finished_tracks.extend(active_tracks)
    all_tracks = [row for track in finished_tracks for row in track.history]

    # create DataFrame with columns particle, frame, y, x
    if not all_tracks:
        return pd.DataFrame(columns=['particle', 'frame', 'y', 'x'])

    out = pd.DataFrame(all_tracks)
    # ensure ordering and rename x/y -> y,x order expected by napari convention below
    out = out[['particle', 'frame', 'y', 'x']].copy()
    return out


# -------------------------
# Track filters / utils
# -------------------------
def filter_tracks_by_net_displacement(tracks_df, displacement_threshold=50, min_frames=10):
    if tracks_df.empty:
        return tracks_df
    # Expect tracks_df columns: ['particle','frame','y','x']
    filtered_tracks = tp.filter_stubs(tracks_df.rename(columns={'y': 'y', 'x': 'x'}), threshold=min_frames).reset_index(drop=True)
    print(f"[Net Disp] After stub removal: {filtered_tracks['particle'].nunique()} tracks")
    if filtered_tracks.empty:
        return filtered_tracks

    displacements = []
    for pid, group in filtered_tracks.groupby('particle'):
        start = group.iloc[0][['y', 'x']].values.astype(float)
        end   = group.iloc[-1][['y', 'x']].values.astype(float)
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
    results = []
    if track_df.empty:
        return pd.DataFrame(columns=['particle', 'self_intersections'])

    for pid, group in track_df.groupby('particle'):
        coords = group[['y', 'x']].to_numpy(float)  # (y,x)
        if len(coords) < 3:
            results.append({'particle': pid, 'self_intersections': 0})
            continue
        segments = [LineString([coords[i], coords[i+1]]) for i in range(len(coords) - 1)]
        crossings = 0
        for i, seg1 in enumerate(segments):
            for j in range(i + 2, len(segments)):
                if seg1.crosses(segments[j]):
                    crossings += 1
        results.append({'particle': pid, 'self_intersections': crossings})

    return pd.DataFrame(results)


def filter_tracks_by_intersection_count(tracks_df, max_crossings=10):
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

# -------------------------
# UI: Widget helpers
# -------------------------
def centered_cell_widget(button):
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
    layout.addWidget(button)
    return container

# -------------------------
# UI: Tracking widget
# -------------------------
# If thread_worker is None, we will not decorate but run synchronously; most users should have thread_worker.
if thread_worker is None:
    # fallback: define a no-op decorator that returns the original function
    def _noop_decorator(fn):
        return fn
    _thread_worker_decorator = _noop_decorator
else:
    _thread_worker_decorator = thread_worker

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

    try:
        df = pd.DataFrame({
            "frame": data[:, 0].astype(int),
            "y": data[:, 1].astype(float),
            "x": data[:, 2].astype(float),
        })
    except Exception as e:
        show_error(f"Could not parse points layer data: {e}")
        return

    @_thread_worker_decorator
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
        tracks_df = kalman_track_blobs(
            blobs_df,
            max_distance=float(dmax),
            max_missed=int(miss),
        )

        if tracks_df is None or tracks_df.empty:
            return tracks_df

        if use_disp_filter:
            tracks_df = filter_tracks_by_net_displacement(
                tracks_df,
                displacement_threshold=float(disp_threshold),
                min_frames=int(disp_min_frames),
            )
            if tracks_df is None or tracks_df.empty:
                return tracks_df

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

    # connect worker signals if we have a thread_worker object (otherwise function already returned)
    def _on_error(err):
        show_error(f"Tracking failed: {err}")

    def _on_done(tracks_df: pd.DataFrame):
        if tracks_df is None or tracks_df.empty:
            show_info("No tracks produced (or all removed by filters).")
            return

        # ensure columns ordering matches napari Tracks: [track_id, t, y, x]
        tracks_df = tracks_df[['particle', 'frame', 'y', 'x']].copy()
        track_data = tracks_df[['particle', 'frame', 'y', 'x']].to_numpy(dtype=float)

        props = {"particle": tracks_df["particle"].to_numpy()}

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

    if thread_worker is not None:
        worker.errored.connect(_on_error)
        worker.returned.connect(_on_done)
        worker.start()
    else:
        # synchronous fallback: worker already executed and returned a DataFrame
        try:
            tracks_df = worker()
            _on_done(tracks_df)
        except Exception as e:
            _on_error(e)

# -------------------------
# UI: Tracks list widget
# -------------------------
class TracksListWidget(QWidget):
    """
    A simple widget that lists tracks and provides a 'Show' button for each.
    `viewer` should be a napari Viewer instance.
    `tracks_df` must be a pandas.DataFrame with columns ['particle','frame','y','x'].
    """
    def __init__(self, viewer: napari.Viewer, tracks_df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.tracks_df = tracks_df.copy() if tracks_df is not None else None
        self.temp_layer_name = "_selected_track_preview"
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QLabel("Tracks")
        header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setMinimumHeight(40)
        layout.addWidget(header)

        # Table: columns: Track ID | Frames | Length | Show | Delete
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Track ID", "Frames\n(Range)", "Frames\n(Count)", "", "", ""])
        self.table.horizontalHeader().setStretchLastSection(False)

        # Column widths & resize behavior
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.Fixed)
        header_view.setSectionResizeMode(1, QHeaderView.Fixed)
        header_view.setSectionResizeMode(2, QHeaderView.Fixed)
        header_view.setSectionResizeMode(3, QHeaderView.Fixed)
        header_view.setSectionResizeMode(4, QHeaderView.Fixed)
        header_view.setSectionResizeMode(5, QHeaderView.Fixed)

        self.table.setColumnWidth(0, 60)
        self.table.setColumnWidth(1, 60)
        self.table.setColumnWidth(2, 60)
        self.table.setColumnWidth(3, 60)
        self.table.setColumnWidth(4, 60)
        self.table.setColumnWidth(5, 60)

        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)

        layout.addWidget(self.table, stretch = 1)

        if self.tracks_df is not None and not self.tracks_df.empty:
            self.populate_table(self.tracks_df)
        else:
            self.table.setRowCount(0)

    def populate_table(self, tracks_df: pd.DataFrame):
        """
        Populate the QTableWidget with rows for each track.
        Adds Show, Delete, and Edit buttons.
        """
        self.tracks_df = tracks_df.copy() if tracks_df is not None else None

        if self.tracks_df is None or self.tracks_df.empty:
            self.table.clearContents()
            self.table.setRowCount(0)
            return

        # Get authoritative particle ids from viewer Tracks layer if possible
        try:
            tracks_layers = [ly for ly in self.viewer.layers if ly.__class__.__name__ == "Tracks"]
            if tracks_layers:
                layer = tracks_layers[0]
                layer_ids = np.unique(np.asarray(layer.data)[:, 0]).astype(int)
            else:
                layer_ids = None
        except Exception:
            layer_ids = None

        try:
            df_ids = np.unique(self.tracks_df['particle'].to_numpy().astype(int))
        except Exception:
            df_ids = None

        if layer_ids is not None:
            if df_ids is None or not np.array_equal(np.sort(df_ids), np.sort(layer_ids)):
                pids = np.sort(layer_ids)
            else:
                pids = np.sort(df_ids)
        else:
            pids = np.sort(df_ids) if df_ids is not None else np.array([], dtype=int)

        self.table.setUpdatesEnabled(False)
        try:
            self.table.clearContents()
            self.table.setRowCount(len(pids))

            for row_idx, pid in enumerate(pids):
                # extract rows for this pid
                if self.tracks_df is not None and 'particle' in self.tracks_df.columns:
                    group = self.tracks_df[self.tracks_df['particle'].astype(int) == int(pid)]
                    group = group.sort_values('frame') if 'frame' in group.columns else group
                else:
                    group = pd.DataFrame()

                if not group.empty and 'frame' in group.columns:
                    frames = sorted(group['frame'].unique().tolist())
                    frames_text = f"{int(frames[0])} – {int(frames[-1])}"
                    length = int(len(group))
                else:
                    frames_text = "N/A"
                    length = 0

                # Track ID (non-editable, centered)
                item_id = QTableWidgetItem(str(int(pid)))
                item_id.setFlags(item_id.flags() & ~Qt.ItemIsEditable)
                item_id.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row_idx, 0, item_id)

                # Frames (range)
                item_frames = QTableWidgetItem(frames_text)
                item_frames.setFlags(item_frames.flags() & ~Qt.ItemIsEditable)
                item_frames.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row_idx, 1, item_frames)

                # Length (count)
                item_len = QTableWidgetItem(str(length))
                item_len.setFlags(item_len.flags() & ~Qt.ItemIsEditable)
                item_len.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row_idx, 2, item_len)

                # Show button
                show_btn = QPushButton("Show")
                show_btn.setToolTip("Center viewer on this track")
                show_btn.setProperty("particle", int(pid))
                show_btn.setMaximumWidth(55)
                show_btn.setMaximumHeight(25)
                show_btn.clicked.connect(partial(self.on_show_clicked, int(pid)))
                self.table.setCellWidget(row_idx, 3, centered_cell_widget(show_btn))

                # Delete button
                del_btn = QPushButton("Delete")
                del_btn.setToolTip("Remove this track")
                del_btn.setProperty("particle", int(pid))
                del_btn.setMaximumWidth(55)
                del_btn.setMaximumHeight(25)
                del_btn.clicked.connect(partial(self.on_delete_clicked, int(pid)))
                self.table.setCellWidget(row_idx, 4, centered_cell_widget(del_btn))

                # Edit button
                edit_btn = QPushButton("Edit")
                edit_btn.setToolTip("Manually edit spots for this track")
                edit_btn.setMaximumWidth(55)
                edit_btn.setMaximumHeight(25)
                edit_btn.clicked.connect(partial(self._on_edit_clicked, int(pid)))
                self.table.setCellWidget(row_idx, 5, centered_cell_widget(edit_btn))

            # optionally select the first row
            if self.table.rowCount() > 0:
                self.table.selectRow(0)
        finally:
            self.table.setUpdatesEnabled(True)

    def on_delete_clicked(self, particle_id, puncta_radius: float = 5.0):
        """
        Delete a track by particle_id from:
        - self.tracks_df (internal DataFrame)
        - the Napari Tracks layer (rebuild from remaining df)
        - optionally remove associated puncta from a Points layer named 'Detected puncta' within puncta_radius (px)

        puncta_radius: spatial radius in pixels to consider a punctum part of the track point.
        """
        if self.tracks_df is None or self.tracks_df.empty:
            show_warning("No tracks available to delete.")
            return

        pid = int(particle_id)
        # Filter out the particle rows from dataframe
        remaining_df = self.tracks_df[self.tracks_df['particle'].astype(int) != pid].reset_index(drop=True)

        # Update the Tracks layer: rebuild data & properties from remaining_df
        try:
            from ._helpers import dataframe_to_tracks_layer_data
        except Exception:
            show_warning("Could not import helpers to update Tracks layer.")
            return

        # Find the existing tracks layer (authoritative)
        tracks_layers = [ly for ly in self.viewer.layers if ly.__class__.__name__ == "Tracks"]
        if not tracks_layers:
            show_warning("No Tracks layer found to update.")
            # still update internal df and table
            self.set_tracks(remaining_df)
            return

        layer = tracks_layers[0]

        # Build new data/properties from remaining_df
        try:
            new_data, new_props = dataframe_to_tracks_layer_data(remaining_df)
        except Exception as e:
            show_warning(f"Failed to build updated tracks data: {e}")
            return

        # Apply to the layer (best-effort)
        try:
            layer.data = new_data
            # layer.properties assignment may or may not be allowed depending on napari version
            try:
                layer.properties = new_props
            except Exception:
                # fallback: assign individual property arrays if supported
                for k, v in new_props.items():
                    try:
                        layer.properties[k] = v
                    except Exception:
                        pass
        except Exception as e:
            show_warning(f"Failed to update Tracks layer: {e}")
            return

        points_layers = [ly for ly in self.viewer.layers if isinstance(ly, Points) and getattr(ly, "name", "") == "Detected puncta"]
        if points_layers:
            pts_layer = points_layers[0]
            try:
                pts_data = np.asarray(pts_layer.data)  # expect columns [frame, y, x]
                if pts_data.size != 0:
                    # collect indices to remove
                    to_remove_idx = set()
                    # the deleted particle rows
                    removed_rows = self.tracks_df[self.tracks_df['particle'].astype(int) == pid]
                    # for each frame in removed_rows, find points on same frame and within radius
                    from scipy.spatial import cKDTree
                    for _, row in removed_rows.iterrows():
                        f = int(row['frame'])
                        y = float(row['y'])
                        x = float(row['x'])
                        # indices in pts_data for same frame
                        frame_idx = np.where(pts_data[:, 0].astype(int) == f)[0]
                        if frame_idx.size == 0:
                            continue
                        coords = pts_data[frame_idx][:, 1:3].astype(float)  # (y,x)
                        if coords.size == 0:
                            continue
                        tree = cKDTree(coords)
                        d, j = tree.query(np.array([y, x]), k=1)
                        if d <= float(puncta_radius):
                            # mark the original index for removal
                            to_remove_idx.add(frame_idx[int(j)])
                    if to_remove_idx:
                        mask = np.ones(len(pts_data), dtype=bool)
                        mask[list(to_remove_idx)] = False
                        new_pts = pts_data[mask]
                        # also filter properties if any
                        try:
                            new_props_pts = {}
                            for k, v in getattr(pts_layer, "properties", {}).items():
                                arr = np.asarray(v)
                                new_props_pts[k] = arr[mask]
                            pts_layer.data = new_pts
                            try:
                                pts_layer.properties = new_props_pts
                            except Exception:
                                # try assign individual properties
                                for k, v in new_props_pts.items():
                                    try:
                                        pts_layer.properties[k] = v
                                    except Exception:
                                        pass
                        except Exception:
                            # fallback: only set data
                            pts_layer.data = new_pts
            except Exception:
                # don't fail deletion if puncta cleanup fails
                pass

        # Update internal dataframe and refresh table UI
        self.set_tracks(remaining_df)

        show_info(f"Deleted track {pid} (and removed {len(removed_rows) if 'removed_rows' in locals() else 0} associated detected spots).")

    def on_show_clicked(self, particle_id):
        """Zoom/center the viewer to the bounding box of the selected track (robust to camera/dim shapes)."""
        if self.tracks_df is None or self.tracks_df.empty:
            return

        # prefer authoritative coords from the Tracks layer if possible
        layer = None
        try:
            tracks_layers = [ly for ly in self.viewer.layers if ly.__class__.__name__ == "Tracks"]
            layer = tracks_layers[0] if tracks_layers else None
        except Exception:
            layer = None

        # get per-particle rows (DataFrame fallback)
        df_rows = self.tracks_df[self.tracks_df['particle'].astype(int) == int(particle_id)].sort_values('frame')

        # Use layer.data when available (authoritative)
        import numpy as np
        if layer is not None:
            ld = np.asarray(layer.data)
            mask = ld[:, 0].astype(int) == int(particle_id)
            if mask.sum() > 0:
                rows = ld[mask]
                ncols = rows.shape[1]
                # expect last columns to be spatial coords: (..., y, x) or (..., z, y, x)
                if ncols >= 4:
                    y_coords = rows[:, -2].astype(float)
                    x_coords = rows[:, -1].astype(float)
                    frames = rows[:, 1].astype(int)
                else:
                    # fallback to DataFrame if unexpected shape
                    y_coords = df_rows['y'].to_numpy(dtype=float) if 'y' in df_rows.columns else np.array([])
                    x_coords = df_rows['x'].to_numpy(dtype=float) if 'x' in df_rows.columns else np.array([])
                    frames = df_rows['frame'].to_numpy(dtype=int) if 'frame' in df_rows.columns else np.array([])
            else:
                y_coords = df_rows['y'].to_numpy(dtype=float) if 'y' in df_rows.columns else np.array([])
                x_coords = df_rows['x'].to_numpy(dtype=float) if 'x' in df_rows.columns else np.array([])
                frames = df_rows['frame'].to_numpy(dtype=int) if 'frame' in df_rows.columns else np.array([])
        else:
            y_coords = df_rows['y'].to_numpy(dtype=float) if 'y' in df_rows.columns else np.array([])
            x_coords = df_rows['x'].to_numpy(dtype=float) if 'x' in df_rows.columns else np.array([])
            frames = df_rows['frame'].to_numpy(dtype=int) if 'frame' in df_rows.columns else np.array([])

        if len(x_coords) == 0 or len(y_coords) == 0:
            return

        # compute center (cx is x, cy is y) of track
        minx, maxx = float(x_coords.min()), float(x_coords.max())
        miny, maxy = float(y_coords.min()), float(y_coords.max())
        cx = 0.5 * (minx + maxx)
        cy = 0.5 * (miny + maxy)

        # set time/frame
        try:
            if len(frames) > 0:
                # often axis 0 is time/frame
                self.viewer.dims.set_point(0, int(frames[0]))
        except Exception:
            pass

        # center napari viewer respecting camera.center length & ordering
        # Napari camera.center can be 2-tuple (x,y) or 3-tuple (z,y,x) — handle both.
        try:
            cc = tuple(self.viewer.camera.center)
        except Exception:
            cc = None

        if cc is None:
            # fallback: set dims points for spatial axes: try axes 1->y, 2->x
            try:
                self.viewer.dims.set_point(1, int(round(cy)))
                self.viewer.dims.set_point(2, int(round(cx)))
            except Exception:
                pass
        else:
            try:
                if len(cc) == 2:
                    # (x, y)
                    self.viewer.camera.center = (float(cx), float(cy))
                elif len(cc) == 3:
                    # (z, y, x) or similar: preserve non-spatial axis (first) and set y,x in last two slots
                    z_val = float(cc[0])
                    self.viewer.camera.center = (z_val, float(cy), float(cx))
                else:
                    # unknown length: try to set last two entries
                    new_cc = list(cc)
                    new_cc[-2] = float(cy)
                    new_cc[-1] = float(cx)
                    self.viewer.camera.center = tuple(new_cc)
            except Exception:
                # fallback: set dims spatial points
                try:
                    self.viewer.dims.set_point(1, int(round(cy)))
                    self.viewer.dims.set_point(2, int(round(cx)))
                except Exception:
                    pass

        # set zoom to fit bounding box (best-effort)
        try:
            width = max(1.0, maxx - minx)
            height = max(1.0, maxy - miny)
            max_dim = max(width, height)
            if max_dim > 0:
                zoom = max(0.2, min(10.0, 200.0 / max_dim))
                try:
                    self.viewer.camera.zoom = float(zoom)
                except Exception:
                    pass
        except Exception:
            pass

        # highlight the row in the table
        try:
            self._highlight_row_for_particle(particle_id)
        except Exception:
            pass

    def _highlight_row_for_particle(self, particle_id: int):
        """Select and scroll to the table row corresponding to particle_id (no-op if not found)."""
        pid_str = str(int(particle_id))
        # find matching row in column 0 (Track ID column)
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is None:
                continue
            try:
                if item.text() == pid_str:
                    # make this the current row and select it
                    self.table.setCurrentCell(r, 0)
                    self.table.selectRow(r)
                    # center table to selected row
                    # self.table.scrollToItem(item, QAbstractItemView.PositionAtCenter)
                    return
            except Exception:
                continue
    
    def _remove_temp_layer(self):
        for layer in list(self.viewer.layers):
            if getattr(layer, "name", "").startswith(self.temp_layer_name):
                try:
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass

    def set_tracks(self, tracks_df: pd.DataFrame):
        self.tracks_df = tracks_df.copy() if tracks_df is not None else None
        self.table.setRowCount(0)
        if self.tracks_df is not None and not self.tracks_df.empty:
            self.populate_table(self.tracks_df)

    def _on_edit_clicked(self, particle_id: int):
        """Show inline editor for particle_id (load rows into embedded editor)."""
        pid = int(particle_id)
        if self.tracks_df is None or self.tracks_df.empty:
            show_warning("No tracks available.")
            return
        # find rows for pid
        rows = self.tracks_df[self.tracks_df['particle'].astype(int) == pid]
        edit_df = rows[['frame','y','x']].copy().reset_index(drop=True)
        # create editor on demand if not present
        if not hasattr(self, "editor") or self.editor is None:
            self.editor = TrackEditorWidget(parent=self)
            # add editor to layout below table (safe to append)
            try:
                self.layout().addWidget(self.editor, stretch=0)
            except Exception:
                pass
        self.editor.load_track(pid, edit_df)
        # scroll table to the track row
        self._highlight_row_for_particle(pid)

    def _apply_track_edits(self, particle_id: int, new_spots_df: pd.DataFrame):
        """
        Replace the rows for particle_id with new_spots_df (which must have columns frame,y,x),
        then rebuild Tracks layer and optionally update Detected puncta.
        """
        pid = int(particle_id)
        if self.tracks_df is None:
            self.tracks_df = pd.DataFrame(columns=['particle','frame','y','x'])

        # Remove old rows for pid and append new ones with particle column set
        remaining = self.tracks_df[self.tracks_df['particle'].astype(int) != pid].copy()
        new_rows = new_spots_df.copy()
        new_rows['particle'] = pid
        updated = pd.concat([remaining, new_rows], ignore_index=True, sort=False)

        # canonical column order
        canonical = ['particle','frame','y','x']
        other_cols = [c for c in updated.columns if c not in canonical]
        updated = updated[[c for c in canonical if c in updated.columns] + other_cols]

        # update internal df
        self.tracks_df = updated.copy()

        # update Tracks layer via helper
        try:
            data, props = dataframe_to_tracks_layer_data(self.tracks_df)
        except Exception as e:
            show_warning(f"Failed to convert updated tracks: {e}")
            return

        tracks_layers = [ly for ly in self.viewer.layers if ly.__class__.__name__ == "Tracks"]
        if not tracks_layers:
            show_warning("No Tracks layer found to update.")
            self.populate_table(self.tracks_df)
            return

        layer = tracks_layers[0]
        try:
            layer.data = data
            try:
                layer.properties = props
            except Exception:
                for k,v in props.items():
                    try:
                        layer.properties[k] = v
                    except Exception:
                        pass
        except Exception as e:
            show_warning(f"Failed to update Tracks layer: {e}")
            return

        # Optional: update Detected puncta (best-effort)
        try:
            pts_layers = [ly for ly in self.viewer.layers if isinstance(ly, Points) and getattr(ly, "name", "") == "Detected puncta"]
            if pts_layers:
                pts = pts_layers[0]
                pts_data = np.asarray(pts.data)
                # Use rounded integer keys to fuzzy-match points
                def key_of_row(r): return (int(r['frame']), int(round(float(r['y']))), int(round(float(r['x']))))
                existing_keys = { key_of_row({'frame':int(row[0]), 'y':row[1], 'x':row[2]}) for row in pts_data } if pts_data.size else set()
                new_keys = { key_of_row(r) for _, r in new_spots_df.iterrows() }
                old_rows = self.tracks_df[self.tracks_df['particle'].astype(int) == pid]
                old_keys = { key_of_row(r) for _, r in old_rows[['frame','y','x']].iterrows() } if not old_rows.empty else set()

                # Build keep mask removing old_keys not present in new_keys
                keep_mask = []
                for i in range(len(pts_data)):
                    r = pts_data[i]
                    k = (int(r[0]), int(round(float(r[1]))), int(round(float(r[2]))))
                    if k in old_keys and k not in new_keys:
                        keep_mask.append(False)
                    else:
                        keep_mask.append(True)
                new_pts_arr = pts_data[np.array(keep_mask, dtype=bool)] if pts_data.size else pts_data

                # add new points that aren't already present
                rows_to_add = []
                for _, r in new_spots_df.iterrows():
                    k = key_of_row(r)
                    if k not in existing_keys:
                        rows_to_add.append([int(r['frame']), float(r['y']), float(r['x'])])
                if rows_to_add:
                    if new_pts_arr.size == 0:
                        new_pts_arr = np.array(rows_to_add)
                    else:
                        new_pts_arr = np.vstack([new_pts_arr, np.array(rows_to_add)])
                try:
                    pts.data = new_pts_arr
                except Exception:
                    try:
                        pts.data = np.asarray(new_pts_arr)
                    except Exception:
                        pass
        except Exception:
            # non-fatal
            pass

        # refresh UI
        self.populate_table(self.tracks_df)
        show_info(f"Saved edits for track {pid}.")

class TrackEditorWidget(QWidget):
    """
    Embedded editor that shows the spots for a single track and allows
    add/remove/edit/save/cancel. Meant to be placed under TracksListWidget.table.
    """

    def __init__(self, parent: "TracksListWidget" = None):
        super().__init__(parent)
        self.parent_widget: "TracksListWidget" = parent
        self.current_pid = None
        self._build_ui()
        self.setVisible(False)  # hidden until needed

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        title_row = QHBoxLayout()
        self.title_label = QLabel("Edit track: (none)")
        title_row.addWidget(self.title_label)
        title_row.addStretch(1)
        layout.addLayout(title_row)

        # Spots table: now has an extra column for the Show button
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Frame", "Y", "X", ""])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 80)
        self.table.setColumnWidth(3, 64)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.table, stretch=1)

        # add/remove row buttons
        br = QHBoxLayout()
        self.add_btn = QPushButton("Add spot")
        self.remove_btn = QPushButton("Remove selected")
        br.addWidget(self.add_btn)
        br.addWidget(self.remove_btn)
        br.addStretch(1)
        layout.addLayout(br)

        # save / cancel
        br2 = QHBoxLayout()
        br2.addStretch(1)
        self.cancel_btn = QPushButton("Cancel")
        self.save_btn = QPushButton("Save")
        br2.addWidget(self.cancel_btn)
        br2.addWidget(self.save_btn)
        layout.addLayout(br2)

        # wire signals
        self.add_btn.clicked.connect(self._on_add_spot)
        self.remove_btn.clicked.connect(self._on_remove_selected)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.save_btn.clicked.connect(self._on_save)

    def load_track(self, pid: int, spots_df: pd.DataFrame):
        """Populate the editor with the given particle id's rows (frame,y,x)."""
        self.current_pid = int(pid)
        self.title_label.setText(f"Edit track: {self.current_pid}")
        self._populate_from_df(spots_df)
        self.setVisible(True)
        # ensure parent layout shows the editor
        try:
            self.parent_widget.updateGeometry()
        except Exception:
            pass

    def _populate_from_df(self, df: pd.DataFrame):
        self.table.setUpdatesEnabled(False)
        try:
            self.table.clearContents()
            self.table.setRowCount(0)
            if df is None or df.empty:
                return
            for _, row in df.sort_values('frame').iterrows():
                r = self.table.rowCount()
                self.table.insertRow(r)
                # frame editable, y/x editable
                f_item = QTableWidgetItem(str(int(row['frame'])))
                y_item = QTableWidgetItem(f"{float(row['y']):.6f}")
                x_item = QTableWidgetItem(f"{float(row['x']):.6f}")
                self.table.setItem(r, 0, f_item)
                self.table.setItem(r, 1, y_item)
                self.table.setItem(r, 2, x_item)

                # Show button for row
                show_btn = QPushButton("Show")
                show_btn.setMaximumWidth(56)
                show_btn.setMaximumHeight(22)
                # store numeric values on the button as properties for easy access
                show_btn.setProperty("frame", int(row['frame']))
                show_btn.setProperty("y", float(row['y']))
                show_btn.setProperty("x", float(row['x']))
                show_btn.clicked.connect(lambda _checked, b=show_btn: self._on_show_spot_button(b))
                # center the button in the cell
                container = QWidget()
                vlay = QVBoxLayout(container)
                vlay.setContentsMargins(0, 0, 0, 0)
                vlay.setSpacing(0)
                vlay.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                vlay.addWidget(show_btn)
                self.table.setCellWidget(r, 3, container)
        finally:
            self.table.setUpdatesEnabled(True)

    def _on_add_spot(self):
        """
        Enter pick mode: user clicks in the napari canvas to add a spot.
        If pick fails / user prefers manual entry, fall back to the old behavior.
        """
        viewer = getattr(self.parent_widget, "viewer", None) if getattr(self, "parent_widget", None) else None
        if viewer is None:
            # fallback to manual insert if no viewer is available
            self._manual_add_row()
            return

        # toggle pick mode: one-shot by default (pick one point then stop)
        # If already in picking mode, stop it.
        if getattr(self, "_picking", False):
            self._stop_pick_mode()
            return

        # start pick mode
        self._start_pick_mode(viewer)


    def _manual_add_row(self):
        """Original fallback behaviour: insert an editable empty row."""
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem("0"))
        self.table.setItem(r, 1, QTableWidgetItem("0.0"))
        self.table.setItem(r, 2, QTableWidgetItem("0.0"))

        # Add show button for new row and wire it to read values from the row on click
        show_btn = QPushButton("Show")
        show_btn.setMaximumWidth(56)
        show_btn.setMaximumHeight(22)
        show_btn.clicked.connect(lambda _checked, row=r: self._on_show_spot_from_row(row))
        container = QWidget()
        vlay = QVBoxLayout(container)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(0)
        vlay.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        vlay.addWidget(show_btn)
        self.table.setCellWidget(r, 3, container)

        self.table.setCurrentCell(r, 0)
        self.table.editItem(self.table.item(r, 0))
    
    def _start_pick_mode(self, viewer):
        """Start a one-shot pick mode using a temporary Points layer in 'add' mode."""
        # set state
        self._picking = True
        self.add_btn.setText("Click in viewer to add (Esc to cancel)")
        self.add_btn.setEnabled(False)

        viewer = getattr(self.parent_widget, "viewer", None)
        if viewer is None:
            self._manual_add_row()
            return

        # Clean up any previous temp layers first
        for layer in list(viewer.layers):
            if getattr(layer, "name", "") == "_pick_point_temp":
                try:
                    viewer.layers.remove(layer)
                except Exception:
                    pass

        # determine sensible ndim for the temporary points layer (prefer spatial dims)
        try:
            rep_layer = None
            for ly in viewer.layers:
                if hasattr(ly, "data") and getattr(ly, "data") is not None:
                    rep_layer = ly
                    break
            if rep_layer is None:
                ndim = 2
            else:
                ndim = len(np.array(getattr(rep_layer, "data")).shape)
                # ensure at least 2
                ndim = max(2, ndim)
        except Exception:
            ndim = 2

        # try several add_points signatures to be compatible with different napari versions
        tmp = None
        try:
            kwargs_try = dict(
                data=np.empty((0, ndim)),
                name="_pick_point_temp",
                size=6,
                ndim=ndim,
                face_color="transparent",
                edge_color="red",
                properties={},
            )
            try:
                tmp = viewer.add_points(**kwargs_try)
            except TypeError:
                # remove edge_color if rejected
                kwargs_try.pop("edge_color", None)
                try:
                    tmp = viewer.add_points(**kwargs_try)
                except TypeError:
                    # last-resort minimal call (positional)
                    tmp = viewer.add_points(np.empty((0, ndim)), name="_pick_point_temp")
        except Exception as e:
            # couldn't create a points layer — fall back to manual behavior
            QMessageBox.information(self, "Add spot", f"Could not create temporary points layer: {e}\nFalling back to manual add.")
            self._stop_pick_mode()
            self._manual_add_row()
            return

        # keep references for cleanup
        self._tmp_pick_layer = tmp
        self._tmp_pick_handler = None

        # Try to put the points layer into 'add' mode (some napari versions support .mode)
        try:
            tmp.mode = "add"
        except Exception:
            try:
                # older versions sometimes use .interaction or similar; ignore if not supported
                setattr(tmp, "mode", "add")
            except Exception:
                pass

        # install escape shortcut to cancel
        self._install_cancel_shortcut()

        # handler invoked when the layer's data changes (user added a point)
        def _on_tmp_points_changed(event=None):
            try:
                # read the latest data snapshot
                data = np.asarray(tmp.data)
                if data is None or data.size == 0:
                    return
                # pick the last added point row
                last = data[-1]

                # normalize: last may be length ndim ; typical ordering is (z,y,x) or (y,x)
                if last.ndim == 0:
                    raise RuntimeError("unexpected point shape")
                if len(last) >= 2:
                    y_val = float(last[-2])
                    x_val = float(last[-1])
                else:
                    y_val = float(last[0])
                    x_val = float(last[1]) if len(last) > 1 else 0.0

                # determine current frame/time from viewer.dims if available
                try:
                    cur_pts = tuple(viewer.dims.point)
                    frame_idx = int(cur_pts[0]) if len(cur_pts) >= 1 else 0
                except Exception:
                    frame_idx = 0

                # insert into the table (one new row)
                r = self.table.rowCount()
                self.table.insertRow(r)
                self.table.setItem(r, 0, QTableWidgetItem(str(int(frame_idx))))
                self.table.setItem(r, 1, QTableWidgetItem(f"{float(y_val):.6f}"))
                self.table.setItem(r, 2, QTableWidgetItem(f"{float(x_val):.6f}"))

                # add a Show button for the new row (same as manual flow)
                show_btn = QPushButton("Show")
                show_btn.setMaximumWidth(56)
                show_btn.setMaximumHeight(22)
                show_btn.clicked.connect(lambda _checked, row=r: self._on_show_spot_from_row(row))
                container = QWidget()
                vlay = QVBoxLayout(container)
                vlay.setContentsMargins(0, 0, 0, 0)
                vlay.setSpacing(0)
                vlay.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                vlay.addWidget(show_btn)
                self.table.setCellWidget(r, 3, container)

                # disconnect handler if possible
                try:
                    if hasattr(tmp, "events") and hasattr(tmp.events, "data") and getattr(tmp.events, "data") is not None:
                        try:
                            tmp.events.data.disconnect(_on_tmp_points_changed)
                        except Exception:
                            pass
                except Exception:
                    pass

                # remove the temporary layer immediately (best-effort)
                try:
                    if tmp in list(viewer.layers):
                        try:
                            viewer.layers.remove(tmp)
                        except Exception:
                            # some napari versions require removal by name
                            try:
                                viewer.layers.remove("_pick_point_temp")
                            except Exception:
                                pass
                except Exception:
                    pass

                # clear stored refs so _stop_pick_mode doesn't double-remove or try to disconnect again
                try:
                    self._tmp_events_handler = None
                except Exception:
                    pass
                try:
                    self._tmp_pick_layer = None
                except Exception:
                    pass

                # finalize: stop pick mode (restores UI, removes any lingering hooks)
                try:
                    self._stop_pick_mode()
                except Exception:
                    pass

                # select the new row so user can tweak frame if desired
                self.table.setCurrentCell(r, 0)
                self.table.editItem(self.table.item(r, 0))

            except Exception as e:
                # on any failure, ensure pick mode stops and fallback to manual add
                print("Pick handler error:", e)
                # try disconnecting
                try:
                    if hasattr(tmp, "events") and hasattr(tmp.events, "data"):
                        try:
                            tmp.events.data.disconnect(_on_tmp_points_changed)
                        except Exception:
                            pass
                except Exception:
                    pass
                # try to remove temp layer
                try:
                    if tmp in list(viewer.layers):
                        try:
                            viewer.layers.remove(tmp)
                        except Exception:
                            try:
                                viewer.layers.remove("_pick_point_temp")
                            except Exception:
                                pass
                except Exception:
                    pass

                # clear stored refs
                try:
                    self._tmp_events_handler = None
                except Exception:
                    pass
                try:
                    self._tmp_pick_layer = None
                except Exception:
                    pass

                # stop pick mode & fallback
                try:
                    self._stop_pick_mode()
                except Exception:
                    pass
                QMessageBox.information(self, "Add spot", "Could not interpret clicked point. Falling back to manual entry.")
                self._manual_add_row()

            except Exception as e:
                # on any failure while interpreting the picked point, remove temp layer and fallback
                print("Pick handler error:", e)
                try:
                    if getattr(self, "_tmp_pick_layer", None) is not None:
                        try:
                            self.viewer.layers.remove(self._tmp_pick_layer)
                        except Exception:
                            pass
                        self._tmp_pick_layer = None
                except Exception:
                    pass
                self._stop_pick_mode()
                QMessageBox.information(self, "Add spot", f"Could not interpret picked point ({e}). Falling back to manual entry.")
                self._manual_add_row()

        # connect the handler to the points layer data events; different napari versions expose different events
        try:
            # preferred: points layer has .events.data
            self._tmp_pick_handler = _on_tmp_points_changed
            try:
                self._tmp_pick_layer.events.data.connect(self._tmp_pick_handler)
            except Exception:
                # fallback: some versions use 'changed' or 'insert' events
                try:
                    self._tmp_pick_layer.events.changed.connect(self._tmp_pick_handler)
                except Exception:
                    # last fallback: poll via canvas mouse press — but we hope one of the above worked
                    pass
        except Exception:
            # If we couldn't attach any handler, tell the user and fallback to manual
            print("Warning: could not attach points layer change handler; falling back to manual add.")
            try:
                if getattr(self, "_tmp_pick_layer", None) is not None:
                    try:
                        self.viewer.layers.remove(self._tmp_pick_layer)
                    except Exception:
                        pass
                    self._tmp_pick_layer = None
            except Exception:
                pass
            self._stop_pick_mode()
            self._manual_add_row()

    def _stop_pick_mode(self):
        """Stop pick mode and remove any temporary pick layers."""
        self._picking = False

        try:
            self.add_btn.setText("Add spot")
            self.add_btn.setEnabled(True)
        except Exception:
            pass

        viewer = getattr(self.parent_widget, "viewer", None)
        if viewer is None:
            return

        # Remove ALL temp pick layers by name (safest approach)
        for layer in list(viewer.layers):
            if getattr(layer, "name", "") == "_pick_point_temp":
                try:
                    viewer.layers.remove(layer)
                except Exception:
                    pass

        self._tmp_pick_layer = None

        # Remove escape shortcut
        try:
            if getattr(self, "_cancel_sc", None) is not None:
                self._cancel_sc.setEnabled(False)
                self._cancel_sc = None
        except Exception:
            pass

    def _install_cancel_shortcut(self):
        try:
            from qtpy.QtWidgets import QShortcut
            from qtpy.QtGui import QKeySequence
            win = self.window()
            if win is None:
                return
            sc = QShortcut(QKeySequence("Escape"), win)
            sc.activated.connect(self._stop_pick_mode)
            self._cancel_sc = sc
        except Exception:
            self._cancel_sc = None

    def _on_remove_selected(self):
        cur = self.table.currentRow()
        if cur < 0:
            QMessageBox.information(self, "Remove", "No row selected to remove.")
            return
        self.table.removeRow(cur)

    def _gather_table_df(self) -> pd.DataFrame:
        rows = []
        for r in range(self.table.rowCount()):
            f_item = self.table.item(r, 0)
            y_item = self.table.item(r, 1)
            x_item = self.table.item(r, 2)
            if f_item is None or y_item is None or x_item is None:
                continue
            try:
                f = int(float(f_item.text()))
                y = float(y_item.text())
                x = float(x_item.text())
            except Exception as e:
                raise ValueError(f"Invalid numeric value in row {r}: {e}")
            rows.append({'frame': f, 'y': y, 'x': x})
        if not rows:
            return pd.DataFrame(columns=['frame','y','x'])
        return pd.DataFrame(rows)

    def _on_cancel(self):
        self.setVisible(False)
        self.current_pid = None

    def _on_save(self):
        try:
            df = self._gather_table_df()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid data", str(e))
            return
        # delegate application to parent widget
        if self.parent_widget is not None:
            self.parent_widget._apply_track_edits(self.current_pid, df)
        # hide editor after save
        self.setVisible(False)
        self.current_pid = None

    # -------------------------
    # Show button helpers
    # -------------------------
    def _on_show_spot_button(self, button: QPushButton):
        """Show using the coordinates embedded on the button (used when loading from df)."""
        try:
            frame = int(button.property("frame"))
            y = float(button.property("y"))
            x = float(button.property("x"))
        except Exception:
            return
        self._center_view_on_spot(frame, y, x)

    def _on_show_spot_from_row(self, row: int):
        """Read values from a (possibly newly added/edited) table row and show."""
        try:
            f_item = self.table.item(row, 0)
            y_item = self.table.item(row, 1)
            x_item = self.table.item(row, 2)
            if f_item is None or y_item is None or x_item is None:
                return
            frame = int(float(f_item.text()))
            y = float(y_item.text())
            x = float(x_item.text())
        except Exception:
            return
        self._center_view_on_spot(frame, y, x)

    def _center_view_on_spot(self, frame: int, y: float, x: float, debug: bool = False):
        """
        Simple centering assuming axes: 0=time, 1=y, 2=x.
        Prefer dims.set_point; fallback to camera.center.
        """
        viewer = getattr(self.parent_widget, "viewer", None) if getattr(self, "parent_widget", None) else None
        if viewer is None:
            return

        # set frame/time (axis 0)
        try:
            viewer.dims.set_point(0, int(frame))
        except Exception:
            pass

        # Preferred: set spatial axes using dims.set_point (safe, updates viewer instantly)
        try:
            viewer.dims.set_point(1, int(round(y)))
            viewer.dims.set_point(2, int(round(x)))
            return
        except Exception:
            # fallback to camera manipulation
            pass

        # Fallback: try camera.center (handles 2- or 3-tuple camera.center)
        try:
            cc = tuple(viewer.camera.center)
        except Exception:
            cc = None

        try:
            if cc is None or len(cc) == 2:
                viewer.camera.center = (float(x), float(y))
            elif len(cc) >= 3:
                new_cc = list(cc)
                new_cc[-2] = float(y)   # put y in second-last slot
                new_cc[-1] = float(x)   # put x in last slot
                viewer.camera.center = tuple(new_cc)
        except Exception:
            pass

        # optional small zoom reset so spot is visible
        try:
            viewer.camera.zoom = max(0.2, min(10.0, getattr(viewer.camera, "zoom", 1.0)))
        except Exception:
            pass

        if debug:
            print("Simple center attempted: frame", frame, "y", y, "x", x)

    def _on_show_selected(self):
        # parse selected row
        cur = self.table.currentRow()
        if cur < 0:
            QMessageBox.information(self, "Show spot", "No row selected")
            return
        try:
            frame = int(float(self.table.item(cur, 0).text()))
            y = float(self.table.item(cur, 1).text())
            x = float(self.table.item(cur, 2).text())
        except Exception as e:
            QMessageBox.information(self, "Show spot", f"Invalid numeric values: {e}")
            return

        # find the viewer (expect parent_widget to hold it)
        viewer = getattr(self.parent_widget, "viewer", None) if hasattr(self, "parent_widget") else None
        if viewer is None:
            # fallback: try walk up parents
            p = self.parent()
            while p is not None and not hasattr(p, "viewer"):
                p = p.parent()
            viewer = getattr(p, "viewer", None) if p is not None else None

        if viewer is None:
            QMessageBox.information(self, "Show spot", "Could not find viewer to center.")
            return

        # set frame/time first (axis 0 is usually frame)
        try:
            viewer.dims.set_point(0, int(frame))
        except Exception:
            pass

        # set spatial position using dims.set_point (preferred)
        try:
            viewer.dims.set_point(1, int(round(y)))
            viewer.dims.set_point(2, int(round(x)))
            return
        except Exception:
            # fallback to camera.center manipulation if dims.set_point doesn't work
            try:
                cc = tuple(viewer.camera.center)
                if len(cc) == 2:
                    viewer.camera.center = (float(x), float(y))
                elif len(cc) >= 3:
                    # keep leading axis value, replace last two with y,x
                    new_cc = list(cc)
                    new_cc[-2] = float(y)
                    new_cc[-1] = float(x)
                    viewer.camera.center = tuple(new_cc)
            except Exception:
                pass