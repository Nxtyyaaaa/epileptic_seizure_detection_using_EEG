import numpy as np
from typing import List, Tuple, Dict, Any


def merge_consecutive_detections(
    detection_bounds: List[Tuple[float, float]],
    max_gap_sec: float = 20.0,
) -> List[Tuple[float, float]]:
    """
    Merges adjacent or nearby positive window detections into unified seizure events.
    """
    if not detection_bounds:
        return []

    sorted_bounds = sorted(detection_bounds, key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [sorted_bounds[0]]

    for current in sorted_bounds[1:]:
        prev_start, prev_end = merged[-1]
        cur_start, cur_end = current

        if cur_start <= prev_end + max_gap_sec:
            # Merge
            merged[-1] = (prev_start, max(prev_end, cur_end))
        else:
            merged.append((cur_start, cur_end))

    return merged


def compute_timeline_event_metrics(
    window_predictions: np.ndarray,
    window_bounds: np.ndarray,
    ground_truth_seizures: List[Tuple[float, float]],
    total_duration_sec: float,
    max_gap_sec: float = 20.0,
    collar_sec: float = 10.0,
) -> Dict[str, Any]:
    """
    Reconstructs continuous event-level seizure detection timeline.
    Computes:
    - Event-level Sensitivity (detected seizures / total actual seizures)
    - False Alarms per Hour (FA/h)
    - Total detected events, true alarm count, false alarm count
    """
    # 1. Identify windows predicted as seizure
    positive_indices = np.where(window_predictions == 1)[0]
    raw_pos_bounds = [tuple(window_bounds[idx]) for idx in positive_indices]

    # 2. Merge into discrete events
    detected_events = merge_consecutive_detections(raw_pos_bounds, max_gap_sec=max_gap_sec)

    # 3. Classify detected events against ground truth
    true_alarms = 0
    false_alarms = 0

    # Expand ground truth intervals by collar tolerance if specified
    gt_intervals = [
        (max(0.0, s[0] - collar_sec), s[1] + collar_sec) for s in ground_truth_seizures
    ]

    for det_start, det_end in detected_events:
        is_hit = False
        for gt_start, gt_end in gt_intervals:
            # Overlap check
            if max(det_start, gt_start) < min(det_end, gt_end):
                is_hit = True
                break
        if is_hit:
            true_alarms += 1
        else:
            false_alarms += 1

    # Event sensitivity: how many ground truth seizures were detected
    detected_gt_count = 0
    for gt_start, gt_end in ground_truth_seizures:
        detected = False
        for det_start, det_end in detected_events:
            if max(det_start, gt_start - collar_sec) < min(det_end, gt_end + collar_sec):
                detected = True
                break
        if detected:
            detected_gt_count += 1

    total_gt = len(ground_truth_seizures)
    event_sensitivity = detected_gt_count / total_gt if total_gt > 0 else 1.0

    # Total non-seizure recording time in hours
    seizure_total_sec = sum(max(0.0, s[1] - s[0]) for s in ground_truth_seizures)
    non_seizure_sec = max(1.0, total_duration_sec - seizure_total_sec)
    non_seizure_hours = non_seizure_sec / 3600.0

    false_alarms_per_hour = false_alarms / non_seizure_hours

    return {
        "event_sensitivity": float(event_sensitivity),
        "false_alarms_per_hour": float(false_alarms_per_hour),
        "total_actual_seizures": total_gt,
        "detected_seizures": detected_gt_count,
        "total_detected_events": len(detected_events),
        "true_alarms": true_alarms,
        "false_alarms": false_alarms,
        "total_recording_hours": float(total_duration_sec / 3600.0),
        "detected_event_intervals": detected_events,
    }

