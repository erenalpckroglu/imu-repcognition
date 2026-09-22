"""Utilities for early/middle/late Chest Press push-phase analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

FS = 50.0
BENCH = "Chest Press (rack)"
ACC = ["accel_x_g", "accel_y_g", "accel_z_g"]
SEGMENTS = {"early": (0.0, 1 / 3), "middle": (1 / 3, 2 / 3), "late": (2 / 3, 1.0)}


def load_project_data(data_dir="../data"):
    data_dir = Path(data_dir)
    return (
        pd.read_csv(data_dir / "recordings.csv"),
        pd.read_csv(data_dir / "samples.csv"),
    )


def load_signal(samples, record_uid):
    return (
        samples.loc[samples.record_uid == record_uid]
        .sort_values("sample_index")
        .reset_index(drop=True)
    )


def dominant_axis(signal):
    return max(ACC, key=lambda col: signal[col].max() - signal[col].min())


def smooth(values, cutoff_hz=4.0):
    b, a = butter(2, cutoff_hz / (FS / 2), btype="low")
    return filtfilt(b, a, np.asarray(values))


def autocorr_period(values, min_lag_s=0.8, max_lag_s=6.0):
    centered = np.asarray(values) - np.mean(values)
    if len(centered) < 20:
        return np.nan
    autocorr = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    if autocorr[0] == 0:
        return np.nan
    autocorr /= autocorr[0]
    low = max(1, int(min_lag_s * FS))
    high = min(len(autocorr), int(max_lag_s * FS) + 1)
    if low >= high:
        return np.nan
    return (np.argmax(autocorr[low:high]) + low) / FS


def detect_push_phases(signal):
    """Return dominant axis, raw/smoothed signals, and valid trough-to-peak phases."""
    axis = dominant_axis(signal)
    raw = signal[axis].to_numpy()
    filtered = smooth(raw)
    period_s = autocorr_period(filtered)
    if not np.isfinite(period_s):
        return axis, raw, filtered, []

    distance = max(1, int(0.6 * period_s * FS))
    prominence = max(np.ptp(filtered) * 0.20, 1e-6)
    peaks, _ = find_peaks(filtered, distance=distance, prominence=prominence)
    troughs, _ = find_peaks(-filtered, distance=distance, prominence=prominence)

    if len(troughs) >= 3:
        refined_period_s = np.median(np.diff(troughs)) / FS
        distance = max(1, int(0.6 * refined_period_s * FS))
        peaks, _ = find_peaks(filtered, distance=distance, prominence=prominence)
        troughs, _ = find_peaks(-filtered, distance=distance, prominence=prominence)

    phases = []
    for start in troughs:
        following = peaks[peaks > start]
        if not len(following):
            continue
        end = int(following[0])
        duration_s = (end - start) / FS
        rise_g = filtered[end] - filtered[start]
        if 0.20 <= duration_s <= 4.0 and rise_g >= max(0.03, 0.08 * np.ptp(filtered)):
            phases.append((int(start), end))
    return axis, raw, filtered, phases


def resample_101(values):
    old = np.linspace(0.0, 1.0, len(values))
    new = np.linspace(0.0, 1.0, 101)
    return np.interp(new, old, np.asarray(values))


def phase_segment_rows(samples, record_uid, subject_id):
    signal = load_signal(samples, record_uid)
    axis, raw, filtered, phases = detect_push_phases(signal)
    rows, profiles = [], []
    for rep_index, (start, end) in enumerate(phases, start=1):
        raw_phase = resample_101(raw[start : end + 1])
        smooth_phase = resample_101(filtered[start : end + 1])
        duration_s = (end - start) / FS
        total_rise = smooth_phase[-1] - smooth_phase[0]
        if total_rise <= 0:
            continue

        raw_progress = (smooth_phase - smooth_phase[0]) / total_rise
        # A trough-to-peak phase should represent forward progress. Small signal
        # oscillations can briefly reverse the raw ratio, so use its monotone
        # envelope for a stable regional progress proxy.
        progress = np.maximum.accumulate(np.clip(raw_progress, 0.0, 1.0))
        dynamic = raw_phase - raw_phase[0]
        profiles.append(
            {"record_uid": record_uid, "rep_index": rep_index, "profile": progress}
        )
        for segment, (left, right) in SEGMENTS.items():
            first = int(round(left * 100))
            last = int(round(right * 100))
            segment_progress = progress[first : last + 1]
            segment_dynamic = dynamic[first : last + 1]
            segment_duration_s = duration_s * (right - left)
            rise_fraction = segment_progress[-1] - segment_progress[0]
            rows.append(
                {
                    "record_uid": record_uid,
                    "subject_id": subject_id,
                    "axis": axis,
                    "rep_index": rep_index,
                    "n_reps_detected": len(phases),
                    "phase_duration_s": duration_s,
                    "segment": segment,
                    "rise_fraction": rise_fraction,
                    "progress_rate_per_s": rise_fraction / segment_duration_s,
                    "mean_abs_dynamic_g": np.mean(np.abs(segment_dynamic)),
                    "rms_dynamic_g": np.sqrt(np.mean(segment_dynamic**2)),
                    "peak_dynamic_g": np.max(np.abs(segment_dynamic)),
                }
            )
    return rows, profiles


def build_segment_table(recordings, samples):
    all_rows, all_profiles = [], []
    bench = recordings.loc[recordings.activity_name == BENCH]
    for recording in bench.itertuples():
        rows, profiles = phase_segment_rows(
            samples, recording.record_uid, recording.subject_id
        )
        all_rows.extend(rows)
        all_profiles.extend(profiles)
    return pd.DataFrame(all_rows), pd.DataFrame(all_profiles)


def summarize_set(group):
    n_reps = int(group.rep_index.max())
    comparison_count = max(2, int(np.ceil(0.30 * n_reps)))
    early_reps = group.loc[group.rep_index <= comparison_count]
    late_reps = group.loc[group.rep_index > n_reps - comparison_count]
    rows = []
    for segment in SEGMENTS:
        early = early_reps.loc[early_reps.segment == segment]
        late = late_reps.loc[late_reps.segment == segment]
        early_rate = early.progress_rate_per_s.median()
        late_rate = late.progress_rate_per_s.median()
        rows.append(
            {
                "segment": segment,
                "n_reps_detected": n_reps,
                "comparison_reps_per_end": comparison_count,
                "early_progress_rate": early_rate,
                "late_progress_rate": late_rate,
                "late_to_early_rate_ratio": (
                    late_rate / early_rate if early_rate > 0 else np.nan
                ),
                "early_rms_dynamic_g": early.rms_dynamic_g.median(),
                "late_rms_dynamic_g": late.rms_dynamic_g.median(),
            }
        )

    result = pd.DataFrame(rows)
    valid = result.late_to_early_rate_ratio.replace([np.inf, -np.inf], np.nan)
    candidate = valid.idxmin() if n_reps >= 6 and valid.notna().any() else None
    result["candidate_weak_region"] = False
    # Do not force every set to have a weak region. Require at least a 10%
    # late-set reduction; this is an explicit exploratory threshold that still
    # needs external biomechanical validation.
    if candidate is not None and valid.loc[candidate] < 0.90:
        result.loc[candidate, "candidate_weak_region"] = True
    return result


def build_summary_table(segment_table):
    summaries = []
    grouped = segment_table.groupby(["record_uid", "subject_id"])
    for (record_uid, subject_id), group in grouped:
        summary = summarize_set(group)
        summary.insert(0, "subject_id", subject_id)
        summary.insert(0, "record_uid", record_uid)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)
