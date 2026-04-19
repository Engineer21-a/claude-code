from __future__ import annotations

from collections import defaultdict

import numpy as np
import supervision as sv

from ..config import TrackerConfig
from ..models import BoundingBox, Detection, FrameDetections


class MultiClassTracker:
    """
    ByteTrack wrapper that maintains one tracker per detection class.
    This prevents track ID collisions across classes (e.g. person track #3
    and license_plate track #3 remain distinct objects).
    """

    def __init__(self, cfg: TrackerConfig) -> None:
        self._cfg = cfg
        self._trackers: dict[str, sv.ByteTrack] = {}

    def _get_tracker(self, class_key: str) -> sv.ByteTrack:
        if class_key not in self._trackers:
            self._trackers[class_key] = sv.ByteTrack(
                track_activation_threshold=self._cfg.track_thresh,
                lost_track_buffer=self._cfg.track_buffer,
                minimum_matching_threshold=self._cfg.match_thresh,
                frame_rate=self._cfg.frame_rate,
            )
        return self._trackers[class_key]

    def update(self, frame_detections: FrameDetections) -> FrameDetections:
        """Assign track IDs to all detections in a single frame."""
        by_class: dict[str, list[Detection]] = defaultdict(list)
        for det in frame_detections.detections:
            by_class[det.detection_class.value].append(det)

        tracked: list[Detection] = []
        for class_key, dets in by_class.items():
            sv_dets = _detections_to_sv(dets)
            tracker = self._get_tracker(class_key)
            sv_tracked = tracker.update_with_detections(sv_dets)
            tracked.extend(
                _sv_to_detections(sv_tracked, dets, frame_detections.frame_index)
            )

        return FrameDetections(
            frame_index=frame_detections.frame_index,
            detections=tracked,
        )

    def reset(self) -> None:
        """Clear all tracker state. Call between videos."""
        self._trackers.clear()


def _detections_to_sv(detections: list[Detection]) -> sv.Detections:
    """Convert our Detection list to a supervision Detections object."""
    if not detections:
        return sv.Detections.empty()
    xyxy = np.array(
        [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in detections],
        dtype=float,
    )
    confidences = np.array([d.confidence for d in detections], dtype=float)
    return sv.Detections(xyxy=xyxy, confidence=confidences)


def _sv_to_detections(
    sv_dets: sv.Detections,
    originals: list[Detection],
    frame_index: int,
) -> list[Detection]:
    """Map supervision tracker output back to our Detection dataclass."""
    result: list[Detection] = []
    for i in range(len(sv_dets)):
        bbox_arr = sv_dets.xyxy[i]
        track_id = (
            int(sv_dets.tracker_id[i]) if sv_dets.tracker_id is not None else None
        )
        orig = originals[i] if i < len(originals) else originals[-1]
        conf = (
            float(sv_dets.confidence[i])
            if sv_dets.confidence is not None
            else orig.confidence
        )
        result.append(
            Detection(
                bbox=BoundingBox(
                    int(bbox_arr[0]),
                    int(bbox_arr[1]),
                    int(bbox_arr[2]),
                    int(bbox_arr[3]),
                ),
                detection_class=orig.detection_class,
                confidence=conf,
                track_id=track_id,
                frame_index=frame_index,
            )
        )
    return result
