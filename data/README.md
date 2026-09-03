# Data

This folder contains a **3-exercise subset** of Microsoft's RecoFit dataset —
**not** the full dataset. Read this before assuming it's complete.

## Files

- `recordings.csv` — one row per recording (= one set): who, which exercise,
  how many reps, how many sensor samples.
- `samples.csv` — one row per sensor sample (master/right-arm device only):
  `record_uid, activity_name, sample_index, time_s, accel_x_g, accel_y_g,
  accel_z_g, gyro_x_dps, gyro_y_dps, gyro_z_dps`.

## Which exercises, and why only these

Extracted from the RecoFit `singleonly` file
(`exercise_data.50.0000_singleonly.mat`), which holds 94 subjects x 75
possible exercise columns. Only 3 activity columns were read out, because
this project only needed them:

- Chest Press (rack) — 31 recordings
- Squat Rack Shoulder Press — 33 recordings
- Lateral Raise — 35 recordings

99 recordings total, 232,046 sample rows in `samples.csv`.

Only the master (right-arm) sensor stream is included here — the slave/left
arm stream was skipped as redundant for these symmetric two-arm exercises.

## Source & citation

Dataset: Microsoft RecoFit.

> Dan Morris, T. Scott Saponas, Andrew Guillory, Ilya Kelner. "RecoFit: Using
> a Wearable Sensor to Find, Recognize, and Count Repetitive Exercises."
> Proceedings of CHI 2014, ACM. DOI: 10.1145/2556288.2557116

Full dataset and the original MATLAB loader script:
https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors

## License

This data subset is licensed under **CDLA-Permissive-2.0** (full text in
[`LICENSE-DATA.txt`](LICENSE-DATA.txt)), the same license used by the source
repository. This is separate from the MIT license that covers the code
elsewhere in this repository — the code license does not apply to the data,
and this data license does not apply to the code.
