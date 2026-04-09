# _validation_state.py
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

VALIDATION_STATE_KEY = "_pt_validation_state"

_STATE_REGISTRY: dict = {}

@dataclass
class ValidationState:
    """
    Single source of truth for track validation within a napari session.
    Stored on the viewer as viewer._pt_validation_state.

    pending_df  : tracks not yet reviewed
    kept_df     : tracks manually validated/kept
    deleted_df  : tracks manually rejected
    """
    pending_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    kept_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    deleted_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    # Load initial track data 
    def load_tracks(self, tracks_df: pd.DataFrame) -> None:
        """
        Reset state and load all tracks as pending.
        Preserves any already-reviewed tracks if their particle ids are
        present in kept_df / deleted_df (allows re-running tracking on
        a subset without losing prior validation work).
        """
        if tracks_df is None or tracks_df.empty:
            return

        reviewed_ids = set()
        if not self.kept_df.empty and "particle" in self.kept_df.columns:
            reviewed_ids |= set(self.kept_df["particle"].astype(int).unique())
        if not self.deleted_df.empty and "particle" in self.deleted_df.columns:
            reviewed_ids |= set(self.deleted_df["particle"].astype(int).unique())

        if reviewed_ids:
            # only add genuinely new tracks to pending
            new_mask = ~tracks_df["particle"].astype(int).isin(reviewed_ids)
            self.pending_df = tracks_df[new_mask].reset_index(drop=True)
        else:
            self.pending_df = tracks_df.copy().reset_index(drop=True)

    # Track keep and delete actions
    def keep(self, particle_id: int) -> None:
        pid = int(particle_id)
        rows = self._pop_from_pending(pid)
        if rows is not None:
            self.kept_df = pd.concat(
                [self.kept_df, rows], ignore_index=True
            )

    def delete(self, particle_id: int) -> None:
        pid = int(particle_id)
        rows = self._pop_from_pending(pid)
        if rows is not None:
            self.deleted_df = pd.concat(
                [self.deleted_df, rows], ignore_index=True
            )

    def _pop_from_pending(self, particle_id: int) -> pd.DataFrame | None:
        """Remove and return rows for particle_id from pending_df."""
        if self.pending_df.empty or "particle" not in self.pending_df.columns:
            return None
        mask = self.pending_df["particle"].astype(int) == particle_id
        if not mask.any():
            return None
        rows = self.pending_df[mask].copy()
        self.pending_df = self.pending_df[~mask].reset_index(drop=True)
        return rows

    # Track export helpers
    def validated_df(self) -> pd.DataFrame:
        """Return only kept tracks — clean export."""
        return self.kept_df.copy()

    def all_tracks_df(self) -> pd.DataFrame:
        """
        Return all tracks with a 'status' column:
            'kept' | 'deleted' | 'pending'
        Useful for ML pipelines that want explicit labels on every row.
        """
        frames = []
        for df, status in [
            (self.kept_df, "kept"),
            (self.deleted_df, "deleted"),
            (self.pending_df, "pending"),
        ]:
            if df is not None and not df.empty:
                tmp = df.copy()
                tmp["status"] = status
                frames.append(tmp)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        # canonical sort: particle then frame
        if "particle" in out.columns and "frame" in out.columns:
            out = out.sort_values(["particle", "frame"]).reset_index(drop=True)
        return out

    @property
    def is_empty(self) -> bool:
        return (
            self.pending_df.empty
            and self.kept_df.empty
            and self.deleted_df.empty
        )

    @property
    def n_pending(self) -> int:
        if self.pending_df.empty or "particle" not in self.pending_df.columns:
            return 0
        return int(self.pending_df["particle"].nunique())

    @property
    def n_kept(self) -> int:
        if self.kept_df.empty or "particle" not in self.kept_df.columns:
            return 0
        return int(self.kept_df["particle"].nunique())

    @property
    def n_deleted(self) -> int:
        if self.deleted_df.empty or "particle" not in self.deleted_df.columns:
            return 0
        return int(self.deleted_df["particle"].nunique())


# Helper functions - move to helpers.py?
def get_validation_state(viewer) -> ValidationState | None:
    return _STATE_REGISTRY.get(id(viewer))


def get_or_create_validation_state(viewer) -> ValidationState:
    vid = id(viewer)
    if vid not in _STATE_REGISTRY:
        _STATE_REGISTRY[vid] = ValidationState()
    return _STATE_REGISTRY[vid]


def init_validation_from_tracks(viewer, tracks_df: pd.DataFrame) -> ValidationState:
    state = get_or_create_validation_state(viewer)
    state.load_tracks(tracks_df)
    return state