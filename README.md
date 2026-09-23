# imu-repcognition

Wrist-worn IMU analysis project — exercise recognition, rep counting, phase
recognition, and Chest Press weak-region (sticking point) analysis — built
on a 3-exercise subset of Microsoft's RecoFit dataset.

## Contents

- `notebooks/01_legacy_signal_exploration.ipynb` — kept for reference only.
  Shows the project's original signal-exploration approach (PCA,
  gravity-tilt correction, early distance-measurement attempts) before it
  converged on the pipeline below.
- `notebooks/02_ml_pipeline.ipynb` — the machine learning pipeline: exercise
  recognition, rep counting, and concentric/eccentric phase recognition,
  all subject-grouped cross-validated (three trained models).
- `notebooks/03_weak_region_analysis.ipynb` — **main deliverable.**
  Three-exercise comparison and recognition, Chest Press push detection,
  within-repetition weak-half comparison, within-set fatigue analysis, and
  cross-set comparison. Supporting functions live in `notebooks/analysis.py`.
- `data/` — the CSV subset used for the analysis (see `data/README.md` for
  exactly what it is, how it was produced, and its license).
- `outputs/` — CSV tables produced by the notebooks above.

Earlier prototype notebooks and superseded analysis scripts have been
removed from this branch to keep the submission focused. The full project
history remains on the `repcognition-legacy` branch.

## Setup

```
python -m pip install -r requirements.txt
jupyter notebook
```

## Data

This repo ships a small, purpose-extracted subset of Microsoft's RecoFit
dataset — **not** the full dataset. Only 3 exercises are included: Chest
Press (rack), Squat Rack Shoulder Press, and Lateral Raise.

The CSVs were extracted with
[recofit-mat2csv-exercise-filter](https://github.com/erenalpckroglu/recofit-mat2csv-exercise-filter),
a companion tool built for this project.

See [`data/README.md`](data/README.md) for the exact provenance, CSV schema,
and license. The data has its **own license** (CDLA-Permissive-2.0), separate
from the code license below.

Full original dataset and MATLAB loader script:
https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors

## Citation

If you use the RecoFit data, cite the original paper:

> Dan Morris, T. Scott Saponas, Andrew Guillory, Ilya Kelner. "RecoFit: Using
> a Wearable Sensor to Find, Recognize, and Count Repetitive Exercises."
> Proceedings of CHI 2014, ACM. DOI: 10.1145/2556288.2557116

## License

- **Code** in this repository: MIT — see [`LICENSE`](LICENSE).
- **Data** in `data/`: CDLA-Permissive-2.0 — see
  [`data/LICENSE-DATA.txt`](data/LICENSE-DATA.txt). The data license is
  separate from and does not apply to the code, and vice versa.
