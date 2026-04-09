# tracking_widget.py
import numpy as np
import pandas as pd
from functools import partial
import trackpy as tp

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
from qtpy.QtCore import Qt, Signal, QObject

import napari
from napari.layers import Points, Image as NapariImage
try:
    from napari.qt.threading import thread_worker
except Exception:
    try:
        from napari.utils import thread_worker
    except Exception:
        thread_worker = None 

from magicgui import magicgui
from scipy.spatial import cKDTree
from shapely.geometry import LineString

try:
    from filterpy.kalman import KalmanFilter
except Exception as e:
    KalmanFilter = None
    _KALMAN_IMPORT_ERR = e

from napari.utils.notifications import show_info, show_warning, show_error

from .tracks_table_widget import TracksTableWidget, _open_tracks_table
from ._helpers import tracks_layer_to_dataframe, dataframe_to_tracks_layer_data
from ._validation_state import get_or_create_validation_state, init_validation_from_tracks

# Kalman tracker
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

# Kalman linking function
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

# Track filters/semi-automated track clean-up
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


# UI: Widget helpers
def centered_cell_widget(button):
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
    layout.addWidget(button)
    return container

if thread_worker is None:
    def _noop_decorator(fn):
        return fn
    _thread_worker_decorator = _noop_decorator
else:
    _thread_worker_decorator = thread_worker

@magicgui(
    call_button="Track",
    layout="vertical",
    # Dropdown to define which layer has the labelled puncta
    points_layer={
        "label": "Detections (Points layer)",
        "nullable": True,
    },
    # Define max distance and max missed for Kalman filter
    max_distance={"label": "Max link distance (px)", "min": 1.0, "max": 200.0, "step": 1.0},
    max_missed={"label": "Max missed frames", "min": 0, "max": 50, "step": 1},
    # Track flitering (displacement and self-intersections)
    use_disp_filter={"label": "Filter by net displacement", "widget_type": "CheckBox"},
    disp_min_frames={"label": "Min frames per track", "min": 1, "max": 1000, "step": 1, "value": 10},
    disp_threshold={"label": "Min net displacement (px)", "min": 0.0, "max": 1_000.0, "step": 1.0, "value": 50.0},
    use_intersection_filter={"label": "Filter by self-intersections", "widget_type": "CheckBox"},
    max_crossings={"label": "Max crossings per track", "min": 0, "max": 200, "step": 1, "value": 10},
    # Show track table
    show_tracks_table={"label": "Show tracks table"},
)

def tracking_widget(
    viewer: "napari.viewer.Viewer",
    # Dropdown to define which layer has the labelled puncta
    points_layer: "napari.layers.Points | None" = None,
    # Define max distance and max missed for Kalman filter (default args)
    max_distance: float = 15.0,
    max_missed: int = 5,
    # Track flitering (displacement and self-intersections) (default args)
    use_disp_filter: bool = True,
    disp_min_frames: int = 10,
    disp_threshold: float = 50.0,
    use_intersection_filter: bool = True,
    max_crossings: int = 10,
    # Show track table
    show_tracks_table=False,
):
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
    def _run_tracking(blobs_df, dmax, miss, use_disp, dmin_f, d_thr, use_int, mx_cross):
        tracks_df = kalman_track_blobs(blobs_df, max_distance=float(dmax), max_missed=int(miss))
        if tracks_df is None or tracks_df.empty:
            return tracks_df
        if use_disp:
            tracks_df = filter_tracks_by_net_displacement(
                tracks_df, displacement_threshold=float(d_thr), min_frames=int(dmin_f)
            )
            if tracks_df is None or tracks_df.empty:
                return tracks_df
        if use_int:
            tracks_df = filter_tracks_by_intersection_count(tracks_df, max_crossings=int(mx_cross))
        return tracks_df

    worker = _run_tracking(
        df, max_distance, max_missed,
        use_disp_filter, disp_min_frames, disp_threshold,
        use_intersection_filter, max_crossings,
    )

    def _on_error(err):
        show_error(f"Tracking failed: {err}")

    def _on_done(tracks_df: pd.DataFrame):
        if tracks_df is None or tracks_df.empty:
            show_info("No tracks produced (or all removed by filters).")
            return

        tracks_df = tracks_df[["particle", "frame", "y", "x"]].copy()
        track_data = tracks_df.to_numpy(dtype=float)
        props = {"particle": tracks_df["particle"].to_numpy()}

        run_track_meta = {
            "max_distance":           float(max_distance),
            "max_missed":             int(max_missed),
            "disp_min_frames":        int(disp_min_frames),
            "disp_threshold":         float(disp_threshold),
            "max_crossings":          int(max_crossings),
            "use_disp_filter":        bool(use_disp_filter),
            "use_intersection_filter": bool(use_intersection_filter),
        }

        tracks_layer = viewer.add_tracks(
            track_data,
            name="Tracks",
            properties=props,
            tail_length=10,
            head_length=0,
            blending="translucent",
        )
        tracks_layer.metadata["run_params"] = run_track_meta

        # Load into ValidationState — preserves any prior validation work
        init_validation_from_tracks(viewer, tracks_df)

        n_tracks = tracks_df["particle"].nunique()
        show_info(f"Tracking complete: {n_tracks} tracks.")

    if thread_worker is not None:
        worker.errored.connect(_on_error)
        worker.returned.connect(_on_done)
        worker.start()
    else:
        try:
            _on_done(worker())
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
    track_kept = Signal()

    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.temp_layer_name = "_selected_track_preview"
        self._build_ui()
        self.refresh_from_state()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Status bar
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.status_label.setMinimumHeight(24)
        layout.addWidget(self.status_label)

        header = QLabel("Tracks")
        header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.setMinimumHeight(40)
        layout.addWidget(header)

        # Table: Track ID | Frames (range) | Count | Show | Keep | Delete | Edit
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Track ID", "Frames\n(Range)", "Count", "", "", "", ""]
        )
        header_view = self.table.horizontalHeader()
        for col, width in enumerate([60, 80, 50, 55, 55, 60, 55]):
            header_view.setSectionResizeMode(col, QHeaderView.Fixed)
            self.table.setColumnWidth(col, width)

        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        layout.addWidget(self.table, stretch=1)

        # Inline editor (hidden until needed)
        self.editor = TrackEditorWidget(parent=self)
        layout.addWidget(self.editor, stretch=0)

    def refresh_from_state(self):
        """Re-populate table from ValidationState.pending_df."""
        state = get_or_create_validation_state(self.viewer)
        self._populate_table(state.pending_df)
        self._update_status_label(state)

    def _update_status_label(self, state):
        self.status_label.setText(
            f"Pending: {state.n_pending}  |  "
            f"Kept: {state.n_kept}  |  "
            f"Deleted: {state.n_deleted}"
        )

    def _populate_table(self, tracks_df: pd.DataFrame):
        self.table.setUpdatesEnabled(False)
        try:
            self.table.clearContents()
            self.table.setRowCount(0)

            if tracks_df is None or tracks_df.empty:
                return

            pids = np.sort(tracks_df["particle"].astype(int).unique())
            self.table.setRowCount(len(pids))

            for row_idx, pid in enumerate(pids):
                group = tracks_df[tracks_df["particle"].astype(int) == int(pid)]
                group = group.sort_values("frame") if "frame" in group.columns else group

                if not group.empty and "frame" in group.columns:
                    frames = sorted(group["frame"].unique().tolist())
                    frames_text = f"{int(frames[0])} – {int(frames[-1])}"
                    length = len(group)
                else:
                    frames_text = "N/A"
                    length = 0

                def _item(text):
                    it = QTableWidgetItem(str(text))
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                    it.setTextAlignment(Qt.AlignCenter)
                    return it

                self.table.setItem(row_idx, 0, _item(int(pid)))
                self.table.setItem(row_idx, 1, _item(frames_text))
                self.table.setItem(row_idx, 2, _item(length))

                # Show
                show_btn = QPushButton("Show")
                show_btn.setMaximumWidth(55); show_btn.setMaximumHeight(25)
                show_btn.clicked.connect(partial(self.on_show_clicked, int(pid)))
                self.table.setCellWidget(row_idx, 3, centered_cell_widget(show_btn))

                # Keep
                keep_btn = QPushButton("Keep")
                keep_btn.setMaximumWidth(55); keep_btn.setMaximumHeight(25)
                keep_btn.setStyleSheet("QPushButton { color: green; font-weight: bold; }")
                keep_btn.clicked.connect(partial(self.on_keep_clicked, int(pid)))
                self.table.setCellWidget(row_idx, 4, centered_cell_widget(keep_btn))

                # Delete
                del_btn = QPushButton("Delete")
                del_btn.setMaximumWidth(55); del_btn.setMaximumHeight(25)
                del_btn.setStyleSheet("QPushButton { color: red; }")
                del_btn.clicked.connect(partial(self.on_delete_clicked, int(pid)))
                self.table.setCellWidget(row_idx, 5, centered_cell_widget(del_btn))

                # Edit
                edit_btn = QPushButton("Edit")
                edit_btn.setMaximumWidth(55); edit_btn.setMaximumHeight(25)
                edit_btn.clicked.connect(partial(self._on_edit_clicked, int(pid)))
                self.table.setCellWidget(row_idx, 6, centered_cell_widget(edit_btn))

            if self.table.rowCount() > 0:
                self.table.selectRow(0)
        finally:
            self.table.setUpdatesEnabled(True)

    def on_keep_clicked(self, particle_id: int):
        state = get_or_create_validation_state(self.viewer)
        state.keep(particle_id)
        self._update_status_label(state)
        self._remove_row_for_particle(particle_id)
        self.track_kept.emit()
        show_info(f"Track {particle_id} kept.")

    def on_delete_clicked(self, particle_id: int, puncta_radius: float = 5.0):
        state = get_or_create_validation_state(self.viewer)

        # Grab rows before deletion for puncta cleanup
        pending = state.pending_df
        removed_rows = (
            pending[pending["particle"].astype(int) == particle_id]
            if not pending.empty else pd.DataFrame()
        )

        state.delete(particle_id)
        self._update_status_label(state)
        self._remove_row_for_particle(particle_id)

        # Remove from napari Tracks layer
        self._remove_particle_from_tracks_layer(particle_id)

        # Optionally remove associated puncta
        self._cleanup_puncta(removed_rows, puncta_radius)

        show_info(f"Track {particle_id} deleted.")

    def _remove_particle_from_tracks_layer(self, particle_id: int):
        tracks_layers = [ly for ly in self.viewer.layers if ly.__class__.__name__ == "Tracks"]
        if not tracks_layers:
            return
        layer = tracks_layers[0]
        try:
            full_df = tracks_layer_to_dataframe(layer)
            remaining = full_df[full_df["particle"].astype(int) != particle_id].reset_index(drop=True)
            new_data, new_props = dataframe_to_tracks_layer_data(remaining)
            layer.data = new_data
            try:
                layer.properties = new_props
            except Exception:
                pass
        except Exception as e:
            show_warning(f"Could not update Tracks layer: {e}")

    def _cleanup_puncta(self, removed_rows: pd.DataFrame, puncta_radius: float):
        if removed_rows.empty:
            return
        pts_layers = [
            ly for ly in self.viewer.layers
            if isinstance(ly, Points) and getattr(ly, "name", "") == "Detected puncta"
        ]
        if not pts_layers:
            return
        pts_layer = pts_layers[0]
        try:
            pts_data = np.asarray(pts_layer.data)
            if pts_data.size == 0:
                return
            to_remove = set()
            for _, row in removed_rows.iterrows():
                f, y, x = int(row["frame"]), float(row["y"]), float(row["x"])
                frame_idx = np.where(pts_data[:, 0].astype(int) == f)[0]
                if frame_idx.size == 0:
                    continue
                coords = pts_data[frame_idx][:, 1:3].astype(float)
                tree = cKDTree(coords)
                d, j = tree.query(np.array([y, x]), k=1)
                if d <= puncta_radius:
                    to_remove.add(frame_idx[int(j)])
            if to_remove:
                mask = np.ones(len(pts_data), dtype=bool)
                mask[list(to_remove)] = False
                pts_layer.data = pts_data[mask]
        except Exception:
            pass

    def on_show_clicked(self, particle_id: int):
        state = get_or_create_validation_state(self.viewer)
        df = state.pending_df
        if df.empty:
            return
        group = df[df["particle"].astype(int) == int(particle_id)].sort_values("frame")
        if group.empty:
            return

        y_coords = group["y"].to_numpy(dtype=float)
        x_coords = group["x"].to_numpy(dtype=float)
        frames   = group["frame"].to_numpy(dtype=int)

        if len(x_coords) == 0:
            return

        cx = 0.5 * (x_coords.min() + x_coords.max())
        cy = 0.5 * (y_coords.min() + y_coords.max())

        try:
            self.viewer.dims.set_point(0, int(frames[0]))
        except Exception:
            pass

        try:
            cc = tuple(self.viewer.camera.center)
            if len(cc) == 2:
                self.viewer.camera.center = (float(cx), float(cy))
            elif len(cc) == 3:
                self.viewer.camera.center = (float(cc[0]), float(cy), float(cx))
            else:
                new_cc = list(cc)
                new_cc[-2] = float(cy)
                new_cc[-1] = float(cx)
                self.viewer.camera.center = tuple(new_cc)
        except Exception:
            try:
                self.viewer.dims.set_point(1, int(round(cy)))
                self.viewer.dims.set_point(2, int(round(cx)))
            except Exception:
                pass

        try:
            width  = max(1.0, x_coords.max() - x_coords.min())
            height = max(1.0, y_coords.max() - y_coords.min())
            zoom   = max(0.2, min(10.0, 200.0 / max(width, height)))
            self.viewer.camera.zoom = float(zoom)
        except Exception:
            pass

        self._highlight_row_for_particle(particle_id)

    def _on_edit_clicked(self, particle_id: int):
        state = get_or_create_validation_state(self.viewer)
        rows = state.pending_df[state.pending_df["particle"].astype(int) == particle_id]
        edit_df = rows[["frame", "y", "x"]].copy().reset_index(drop=True)
        self.editor.load_track(particle_id, edit_df)
        self._highlight_row_for_particle(particle_id)

    def _apply_track_edits(self, particle_id: int, new_spots_df: pd.DataFrame):
        """Called by TrackEditorWidget on save."""
        state = get_or_create_validation_state(self.viewer)
        pid = int(particle_id)

        # Remove old rows for pid from pending
        remaining = state.pending_df[
            state.pending_df["particle"].astype(int) != pid
        ].copy()
        new_rows = new_spots_df.copy()
        new_rows["particle"] = pid
        updated = pd.concat([remaining, new_rows], ignore_index=True)
        state.pending_df = updated.sort_values(["particle", "frame"]).reset_index(drop=True)

        # Sync to Tracks layer
        try:
            all_tracks = pd.concat(
                [state.pending_df, state.kept_df], ignore_index=True
            )
            data, props = dataframe_to_tracks_layer_data(all_tracks)
            tracks_layers = [ly for ly in self.viewer.layers if ly.__class__.__name__ == "Tracks"]
            if tracks_layers:
                layer = tracks_layers[0]
                layer.data = data
                try:
                    layer.properties = props
                except Exception:
                    pass
        except Exception as e:
            show_warning(f"Could not sync edits to Tracks layer: {e}")

        self.refresh_from_state()
        show_info(f"Saved edits for track {pid}.")

    def _remove_row_for_particle(self, particle_id: int):
        pid_str = str(int(particle_id))
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None and item.text() == pid_str:
                self.table.removeRow(r)
                return

    def _highlight_row_for_particle(self, particle_id: int):
        pid_str = str(int(particle_id))
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None and item.text() == pid_str:
                self.table.setCurrentCell(r, 0)
                self.table.selectRow(r)
                return

    def set_tracks(self, tracks_df: pd.DataFrame):
        """Legacy compatibility — loads df into pending and refreshes."""
        state = get_or_create_validation_state(self.viewer)
        state.load_tracks(tracks_df)
        self.refresh_from_state()

    def _remove_temp_layer(self):
        for layer in list(self.viewer.layers):
            if getattr(layer, "name", "").startswith(self.temp_layer_name):
                try:
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass
    

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
            for _, row in df.sort_values("frame").iterrows():
                r = self.table.rowCount()
                self.table.insertRow(r)
                self.table.setItem(r, 0, QTableWidgetItem(str(int(row["frame"]))))
                self.table.setItem(r, 1, QTableWidgetItem(f"{float(row['y']):.6f}"))
                self.table.setItem(r, 2, QTableWidgetItem(f"{float(row['x']):.6f}"))
                show_btn = QPushButton("Show")
                show_btn.setMaximumWidth(56); show_btn.setMaximumHeight(22)
                show_btn.setProperty("frame", int(row["frame"]))
                show_btn.setProperty("y", float(row["y"]))
                show_btn.setProperty("x", float(row["x"]))
                show_btn.clicked.connect(lambda _c, b=show_btn: self._on_show_spot_button(b))
                container = QWidget()
                vlay = QVBoxLayout(container)
                vlay.setContentsMargins(0, 0, 0, 0)
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
        self._picking = True
        self.add_btn.setText("Click in viewer (Esc to cancel)")
        self.add_btn.setEnabled(False)

        for layer in list(viewer.layers):
            if getattr(layer, "name", "") == "_pick_point_temp":
                try:
                    viewer.layers.remove(layer)
                except Exception:
                    pass

        try:
            tmp = viewer.add_points(np.empty((0, 2)), name="_pick_point_temp")
        except Exception as e:
            QMessageBox.information(self, "Add spot", f"Could not create temp layer: {e}")
            self._stop_pick_mode()
            self._manual_add_row()
            return

        self._tmp_pick_layer = tmp
        try:
            tmp.mode = "add"
        except Exception:
            pass
        self._install_cancel_shortcut()

        def _on_data_changed(event=None):
            try:
                data = np.asarray(tmp.data)
                if data is None or data.size == 0:
                    return
                last = data[-1]
                y_val = float(last[-2]) if len(last) >= 2 else 0.0
                x_val = float(last[-1]) if len(last) >= 1 else 0.0
                try:
                    frame_idx = int(tuple(viewer.dims.point)[0])
                except Exception:
                    frame_idx = 0

                r = self.table.rowCount()
                self.table.insertRow(r)
                self.table.setItem(r, 0, QTableWidgetItem(str(frame_idx)))
                self.table.setItem(r, 1, QTableWidgetItem(f"{y_val:.6f}"))
                self.table.setItem(r, 2, QTableWidgetItem(f"{x_val:.6f}"))
                show_btn = QPushButton("Show")
                show_btn.setMaximumWidth(56); show_btn.setMaximumHeight(22)
                show_btn.clicked.connect(lambda _c, row=r: self._on_show_spot_from_row(row))
                container = QWidget()
                vlay = QVBoxLayout(container)
                vlay.setContentsMargins(0, 0, 0, 0)
                vlay.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                vlay.addWidget(show_btn)
                self.table.setCellWidget(r, 3, container)

                try:
                    tmp.events.data.disconnect(_on_data_changed)
                except Exception:
                    pass
                try:
                    viewer.layers.remove(tmp)
                except Exception:
                    pass
                self._tmp_pick_layer = None
                self._stop_pick_mode()
                self.table.setCurrentCell(r, 0)
            except Exception as e:
                print("Pick handler error:", e)
                self._stop_pick_mode()
                self._manual_add_row()

        try:
            tmp.events.data.connect(_on_data_changed)
        except Exception:
            try:
                tmp.events.changed.connect(_on_data_changed)
            except Exception:
                self._stop_pick_mode()
                self._manual_add_row()

    def _stop_pick_mode(self):
        self._picking = False
        try:
            self.add_btn.setText("Add spot")
            self.add_btn.setEnabled(True)
        except Exception:
            pass
        viewer = getattr(self.parent_widget, "viewer", None)
        if viewer:
            for layer in list(viewer.layers):
                if getattr(layer, "name", "") == "_pick_point_temp":
                    try:
                        viewer.layers.remove(layer)
                    except Exception:
                        pass
        self._tmp_pick_layer = None
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
            if win:
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
            if not all([f_item, y_item, x_item]):
                continue
            try:
                rows.append({
                    "frame": int(float(f_item.text())),
                    "y":     float(y_item.text()),
                    "x":     float(x_item.text()),
                })
            except Exception as e:
                raise ValueError(f"Invalid value in row {r}: {e}")
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["frame", "y", "x"])

    def _on_cancel(self):
        self.setVisible(False)
        self.current_pid = None

    def _on_save(self):
        try:
            df = self._gather_table_df()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid data", str(e))
            return
        if self.parent_widget is not None:
            self.parent_widget._apply_track_edits(self.current_pid, df)
        self.setVisible(False)
        self.current_pid = None

    def _on_show_spot_button(self, button: QPushButton):
        try:
            self._center_view_on_spot(
                int(button.property("frame")),
                float(button.property("y")),
                float(button.property("x")),
            )
        except Exception:
            pass

    def _on_show_spot_from_row(self, row: int):
        try:
            frame = int(float(self.table.item(row, 0).text()))
            y     = float(self.table.item(row, 1).text())
            x     = float(self.table.item(row, 2).text())
            self._center_view_on_spot(frame, y, x)
        except Exception:
            pass

    def _center_view_on_spot(self, frame: int, y: float, x: float, debug: bool = False):
        viewer = getattr(self.parent_widget, "viewer", None)
        if viewer is None:
            return
        try:
            viewer.dims.set_point(0, int(frame))
        except Exception:
            pass
        try:
            viewer.dims.set_point(1, int(round(y)))
            viewer.dims.set_point(2, int(round(x)))
            return
        except Exception:
            pass
        try:
            cc = tuple(viewer.camera.center)
            if len(cc) == 2:
                viewer.camera.center = (float(x), float(y))
            else:
                new_cc = list(cc)
                new_cc[-2] = float(y)
                new_cc[-1] = float(x)
                viewer.camera.center = tuple(new_cc)
        except Exception:
            pass