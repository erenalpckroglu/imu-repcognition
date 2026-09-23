# imu-repcognition

Exercise recognition, repetition counting and within-repetition analysis of
the bench press from a single wrist-worn IMU, built on Microsoft's RecoFit
dataset.

The central question is where in the push a bench press loses acceleration,
and whether that changes as a set goes on. Bar velocity cannot be recovered
reliably from a wrist accelerometer, so the analysis works with
acceleration-based proxies and treats every weak region it finds as a
candidate rather than a diagnosis.

## Start here

[`notebooks/04_report.ipynb`](notebooks/04_report.ipynb) is the project
report: motivation, data, how the analysis was built, and the results. It is
fully executed and can be read on GitHub without running anything.

## Results at a glance

| Task | Method | Result |
|---|---|---|
| Exercise recognition, Chest Press vs Lateral Raise | Logistic regression, linear SVM, random forest | 1.00 accuracy in every fold |
| Exercise recognition, three exercises | Logistic regression on set-level features | 0.80 mean accuracy |
| Repetition counting, Chest Press | Random forest regression on multi-axis period estimates | MAE 1.57 reps (signal processing baseline 1.93) |
| Repetition counting, Lateral Raise | Same | MAE 1.14 reps (baseline 1.59) |
| Concentric vs eccentric phase from 0.3 s windows | Random forest | 0.80 accuracy (majority baseline 0.55) |
| Push detection, Chest Press | Autocorrelation and peak detection | 414 pushes in 29 of 30 sets |

Every model is evaluated with five-fold cross-validation grouped by
participant, so no person appears in both training and test data.

## Repository layout

```
notebooks/
  01_legacy_signal_exploration.ipynb   signal exploration: dominant axes, PCA,
                                       gravity tilt correction, tempo, double
                                       integration test
  02_ml_pipeline.ipynb                 exercise recognition, repetition counting,
                                       phase recognition
  03_weak_region_analysis.ipynb        Chest Press push detection and comparison
                                       of the two halves of each push
  04_report.ipynb                      project report
  analysis.py                          helper functions used by 03 and 04
data/                                  CSV subset of RecoFit, see data/README.md
outputs/                               tables written by the notebooks
```

Earlier prototypes are kept on the `repcognition-legacy` branch.

## Setup

```
python -m pip install -r requirements.txt
jupyter notebook notebooks/04_report.ipynb
```

Library versions are pinned because the cross-validation folds, and with
them some of the reported accuracies, depend on the scikit-learn version.

## Data

The CSV files were extracted from RecoFit's `singleonly` MATLAB file with
[recofit-mat2csv-exercise-filter](https://github.com/erenalpckroglu/recofit-mat2csv-exercise-filter),
a conversion tool written for this project. Three exercises are included:
Chest Press (rack), Squat Rack Shoulder Press and Lateral Raise, 99 sets from
43 participants recorded at 50 Hz on the right forearm. Sets with fewer than
five labelled repetitions (three sets) are excluded in notebooks 02 to 04.
See [`data/README.md`](data/README.md) for provenance, schema and license.

Full original dataset and MATLAB loader script:
https://github.com/microsoft/Exercise-Recognition-from-Wearable-Sensors

## Limitations

RecoFit contains no ground truth for bar velocity, fatigue, failure or muscle
activation, and its participants were not asked to train close to failure.
All weak-region results rest on acceleration proxies and are exploratory.

## Citation

If you use the RecoFit data, cite the original paper:

> Dan Morris, T. Scott Saponas, Andrew Guillory, Ilya Kelner. "RecoFit: Using
> a Wearable Sensor to Find, Recognize, and Count Repetitive Exercises."
> Proceedings of CHI 2014, ACM. DOI: 10.1145/2556288.2557116

## License

- Code in this repository: MIT, see [`LICENSE`](LICENSE).
- Data in `data/`: CDLA-Permissive-2.0, see
  [`data/LICENSE-DATA.txt`](data/LICENSE-DATA.txt). The data license is
  separate from and does not apply to the code, and vice versa.
