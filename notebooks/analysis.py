"""Helper functions for loading the RecoFit subset, detecting Chest Press pushes
and computing the features used in notebooks 03 and 04."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

FS = 50.0
CHEST = "Chest Press (rack)"
SHOULDER = "Squat Rack Shoulder Press"
LATERAL = "Lateral Raise"
EXERCISES = [CHEST, SHOULDER, LATERAL]
ACC = ["accel_x_g", "accel_y_g", "accel_z_g"]
GYR = ["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]

# Sets with fewer labelled repetitions are not treated as repetitive sets.
MIN_LABELLED_REPS = 5


def load_data(data_dir="../data", min_reps=MIN_LABELLED_REPS):
    """Load recordings and samples, dropping sets with fewer than min_reps labelled repetitions."""
    data_dir = Path(data_dir)
    recordings = pd.read_csv(data_dir / "recordings.csv")
    samples = pd.read_csv(data_dir / "samples.csv")
    if min_reps:
        recordings = recordings.loc[recordings.activity_reps >= min_reps].reset_index(drop=True)
        samples = samples.loc[samples.record_uid.isin(recordings.record_uid)].reset_index(drop=True)
    return recordings, samples


def short_sets(data_dir="../data", min_reps=MIN_LABELLED_REPS):
    """Sets excluded by load_data because they have fewer than min_reps labelled repetitions."""
    recordings = pd.read_csv(Path(data_dir) / "recordings.csv")
    short = recordings.loc[recordings.activity_reps < min_reps]
    return short[["record_uid", "activity_name", "activity_reps"]].reset_index(drop=True)


def representative_record(recordings, samples, exercise):
    """Set whose gyroscope magnitude RMS is closest to the median of its exercise."""
    uids = recordings.loc[recordings.activity_name == exercise, "record_uid"]
    rms = pd.Series(
        {
            uid: np.sqrt(np.mean(np.linalg.norm(load_signal(samples, uid)[GYR].to_numpy(), axis=1) ** 2))
            for uid in uids
        }
    )
    return (rms - rms.median()).abs().idxmin()


def load_signal(samples, record_uid):
    return (
        samples.loc[samples.record_uid == record_uid]
        .sort_values("sample_index")
        .reset_index(drop=True)
    )


def smooth(values, fs=FS, cutoff_hz=4.0):
    values = np.asarray(values)
    if len(values) < 10:
        return values.copy()
    b, a = butter(2, cutoff_hz / (fs / 2), btype="low")
    return filtfilt(b, a, values)


def dominant_axis(signal, columns=ACC):
    return max(columns, key=lambda col: np.ptp(signal[col].to_numpy()))


def autocorr_period(values, min_lag_s=0.5, max_lag_s=6.0, fs=FS):
    centered = np.asarray(values) - np.mean(values)
    if len(centered) < 20:
        return np.nan
    autocorr = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    if autocorr[0] == 0:
        return np.nan
    autocorr = autocorr / autocorr[0]
    low = max(1, int(min_lag_s * fs))
    high = min(len(autocorr), int(max_lag_s * fs) + 1)
    if low >= high:
        return np.nan
    return (np.argmax(autocorr[low:high]) + low) / fs


def build_recording_features(recordings, samples):
    """Build one interpretable feature row per exercise set."""
    rows = []
    for recording in recordings.itertuples():
        signal = load_signal(samples, recording.record_uid)
        duration_s = signal.time_s.iloc[-1] - signal.time_s.iloc[0]
        accel = signal[ACC].to_numpy()
        gyro = signal[GYR].to_numpy()
        accel_mag = np.linalg.norm(accel, axis=1)
        gyro_mag = np.linalg.norm(gyro, axis=1)
        axis = dominant_axis(signal)
        axis_smooth = smooth(signal[axis].to_numpy())
        rows.append(
            {
                "record_uid": recording.record_uid,
                "subject_id": recording.subject_id,
                "exercise": recording.activity_name,
                "labelled_reps": recording.activity_reps,
                "n_samples": len(signal),
                "duration_s": duration_s,
                "sample_rate_hz": (len(signal) - 1) / duration_s,
                "seconds_per_labelled_rep": duration_s / recording.activity_reps,
                "accel_mag_mean_g": accel_mag.mean(),
                "accel_mag_std_g": accel_mag.std(),
                "accel_dynamic_rms_g": np.sqrt(np.mean((accel_mag - 1.0) ** 2)),
                "gyro_mag_rms_dps": np.sqrt(np.mean(gyro_mag**2)),
                "dominant_accel_range_g": np.ptp(signal[axis].to_numpy()),
                "tempo_period_s": autocorr_period(axis_smooth),
                **{
                    f"{col}_rms": np.sqrt(np.mean(signal[col].to_numpy() ** 2))
                    for col in ACC + GYR
                },
            }
        )
    return pd.DataFrame(rows)


def detect_push_phases(signal):
    """Detect Chest Press trough-to-peak phases on the dominant accel axis."""
    axis = dominant_axis(signal)
    raw = signal[axis].to_numpy()
    filtered = smooth(raw)
    period_s = autocorr_period(filtered, min_lag_s=0.8)
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
        # A push that contains another trough ran past a missed peak into the
        # next repetition, so it is not a single push.
        if np.any((troughs > start) & (troughs < end)):
            continue
        duration_s = (end - start) / FS
        rise = filtered[end] - filtered[start]
        if 0.20 <= duration_s <= 4.0 and rise >= max(0.03, 0.08 * np.ptp(filtered)):
            phases.append((int(start), end))
    return axis, raw, filtered, phases


def _resample(values, n_points=101):
    old = np.linspace(0.0, 1.0, len(values))
    new = np.linspace(0.0, 1.0, n_points)
    return np.interp(new, old, np.asarray(values))


def build_chest_half_features(recordings, samples):
    """Create one row per half of every detected Chest Press push."""
    rows, profiles = [], []
    chest_recordings = recordings.loc[recordings.activity_name == CHEST]
    for recording in chest_recordings.itertuples():
        signal = load_signal(samples, recording.record_uid)
        axis, _, filtered, phases = detect_push_phases(signal)
        accel_mag = np.linalg.norm(signal[ACC].to_numpy(), axis=1)
        for rep_index, (start, end) in enumerate(phases, start=1):
            smooth_phase = _resample(filtered[start : end + 1])
            accel_mag_phase = _resample(accel_mag[start : end + 1])
            duration_s = (end - start) / FS
            total_rise = smooth_phase[-1] - smooth_phase[0]
            if total_rise <= 0:
                continue
            raw_progress = (smooth_phase - smooth_phase[0]) / total_rise
            progress = np.maximum.accumulate(np.clip(raw_progress, 0.0, 1.0))
            profiles.append(
                {
                    "record_uid": recording.record_uid,
                    "subject_id": recording.subject_id,
                    "rep_index": rep_index,
                    "progress": progress,
                }
            )
            for half, first, last in (("first", 0, 50), ("second", 50, 100)):
                half_progress = progress[first : last + 1]
                half_mag = accel_mag_phase[first : last + 1]
                rise_fraction = half_progress[-1] - half_progress[0]
                half_duration_s = duration_s / 2
                rows.append(
                    {
                        "record_uid": recording.record_uid,
                        "subject_id": recording.subject_id,
                        "axis": axis,
                        "rep_index": rep_index,
                        "labelled_reps": recording.activity_reps,
                        "detected_reps": len(phases),
                        "push_duration_s": duration_s,
                        "half": half,
                        "rise_fraction": rise_fraction,
                        "progress_rate_per_s": rise_fraction / half_duration_s,
                        "dynamic_accel_rms_g": np.sqrt(
                            np.mean((half_mag - 1.0) ** 2)
                        ),
                        "mean_abs_dynamic_accel_g": np.mean(np.abs(half_mag - 1.0)),
                        "peak_dynamic_accel_g": np.max(np.abs(half_mag - 1.0)),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(profiles)


def build_rep_comparison(half_features):
    """Put the first and second half features side-by-side for each repetition."""
    ids = [
        "record_uid",
        "subject_id",
        "rep_index",
        "labelled_reps",
        "detected_reps",
        "push_duration_s",
    ]
    values = ["progress_rate_per_s", "dynamic_accel_rms_g"]
    wide = half_features.pivot(index=ids, columns="half", values=values)
    wide.columns = [f"{feature}_{half}" for feature, half in wide.columns]
    wide = wide.reset_index()
    wide["second_to_first_progress_rate"] = (
        wide.progress_rate_per_s_second / wide.progress_rate_per_s_first
    )
    wide["second_to_first_accel_energy"] = (
        wide.dynamic_accel_rms_g_second / wide.dynamic_accel_rms_g_first
    )
    wide["lower_acceleration_half"] = np.where(
        wide.second_to_first_accel_energy < 1, "second", "first"
    )
    return wide


def build_set_comparison(rep_comparison):
    """Summarize within-rep and early-set/late-set changes for every set."""
    rows = []
    for (record_uid, subject_id), group in rep_comparison.groupby(
        ["record_uid", "subject_id"]
    ):
        group = group.sort_values("rep_index")
        n_reps = len(group)
        edge_count = max(2, int(np.ceil(0.30 * n_reps)))
        first_reps = group.head(edge_count)
        last_reps = group.tail(edge_count)
        first_half_fatigue = (
            last_reps.dynamic_accel_rms_g_first.median()
            / first_reps.dynamic_accel_rms_g_first.median()
        )
        second_half_fatigue = (
            last_reps.dynamic_accel_rms_g_second.median()
            / first_reps.dynamic_accel_rms_g_second.median()
        )
        candidate = "none"
        if n_reps >= 6 and min(first_half_fatigue, second_half_fatigue) < 0.90:
            candidate = (
                "first" if first_half_fatigue < second_half_fatigue else "second"
            )
        rows.append(
            {
                "record_uid": record_uid,
                "subject_id": subject_id,
                "labelled_reps": group.labelled_reps.iloc[0],
                "detected_reps": n_reps,
                "median_push_duration_s": group.push_duration_s.median(),
                "median_first_half_accel_rms_g": group.dynamic_accel_rms_g_first.median(),
                "median_second_half_accel_rms_g": group.dynamic_accel_rms_g_second.median(),
                "median_second_to_first_accel_energy": group.second_to_first_accel_energy.median(),
                "first_half_late_to_early_energy": first_half_fatigue,
                "second_half_late_to_early_energy": second_half_fatigue,
                "candidate_fatigue_sensitive_half": candidate,
            }
        )
    return pd.DataFrame(rows)
