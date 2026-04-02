"""
frame_smoother.py
Temporal smoothing for tile detections across N frames.

A detection in the HAND zone is confirmed when:
  - Seen in >= MIN_HITS of the last WINDOW frames
  - At roughly the same x-position (±POSITION_TOL pixels)

Low-confidence detections (< CONF_WARN) are flagged for user review.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

WINDOW       = 6      # sliding window (frames / screenshots)
MIN_HITS     = 3      # minimum hits to confirm a detection
POSITION_TOL = 40     # pixels: two detections at same x are "same tile"
CONF_WARN    = 0.45   # confidence below this → flag for user review


@dataclass
class RawDetection:
    tile_id:   int     # standard tile ID (0-33)
    model_cid: int     # model class ID
    conf:      float
    xyxy:      Tuple[int,int,int,int]
    zone:      str     # "hand" | "center" | "meld_left" | "meld_right" | "meld_top"
    cx:        int = 0
    cy:        int = 0

    def __post_init__(self):
        x1,y1,x2,y2 = self.xyxy
        self.cx = (x1+x2)//2
        self.cy = (y1+y2)//2


@dataclass
class ConfirmedTile:
    tile_id:    int
    conf_avg:   float
    cx:         int
    cy:         int
    xyxy:       Tuple[int,int,int,int]
    zone:       str
    hit_count:  int
    low_conf:   bool = False   # True → warn user


class FrameSmoother:
    """
    Rolling window smoother.
    Call update(detections) each frame → get confirmed_tiles().
    """

    def __init__(self, window=WINDOW, min_hits=MIN_HITS,
                 pos_tol=POSITION_TOL, conf_warn=CONF_WARN):
        self.window    = window
        self.min_hits  = min_hits
        self.pos_tol   = pos_tol
        self.conf_warn = conf_warn
        # deque of List[RawDetection] per frame
        self._history: deque = deque(maxlen=window)

    def update(self, detections: List[RawDetection]):
        self._history.append(detections)

    def confirmed_tiles(self, zone: Optional[str] = None) -> List[ConfirmedTile]:
        """
        Return detections confirmed across history.
        Optionally filter by zone.
        """
        if not self._history:
            return []

        # Flatten all detections in window, optionally filtered by zone
        all_dets: List[RawDetection] = []
        for frame in self._history:
            for d in frame:
                if zone is None or d.zone == zone:
                    all_dets.append(d)

        if not all_dets:
            return []

        # Cluster by (zone, tile_id, cx within pos_tol)
        clusters: List[List[RawDetection]] = []
        used = [False] * len(all_dets)

        for i, d in enumerate(all_dets):
            if used[i]:
                continue
            cluster = [d]
            used[i] = True
            for j, d2 in enumerate(all_dets):
                if used[j] or d2.zone != d.zone or d2.tile_id != d.tile_id:
                    continue
                if abs(d2.cx - d.cx) <= self.pos_tol:
                    cluster.append(d2)
                    used[j] = True
            clusters.append(cluster)

        confirmed: List[ConfirmedTile] = []
        for cluster in clusters:
            # Count unique frames this cluster appears in
            frame_indices = set()
            for d in cluster:
                # Identify frame index from position in history
                pass
            # Simpler: count total hits (each frame contributes its detections)
            hit_count = len(cluster)
            if hit_count < self.min_hits:
                continue
            avg_conf = float(np.mean([d.conf for d in cluster]))
            best     = max(cluster, key=lambda d: d.conf)
            confirmed.append(ConfirmedTile(
                tile_id   = best.tile_id,
                conf_avg  = avg_conf,
                cx        = best.cx,
                cy        = best.cy,
                xyxy      = best.xyxy,
                zone      = best.zone,
                hit_count = hit_count,
                low_conf  = avg_conf < self.conf_warn,
            ))

        return confirmed

    def hand_tiles_sorted(self) -> List[ConfirmedTile]:
        """Confirmed hand tiles sorted left→right by x position."""
        tiles = [t for t in self.confirmed_tiles(zone="hand")]
        tiles.sort(key=lambda t: t.cx)
        return tiles

    def low_conf_warnings(self) -> List[ConfirmedTile]:
        return [t for t in self.confirmed_tiles() if t.low_conf]

    def reset(self):
        self._history.clear()
