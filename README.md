# imu-repcognition

Small analysis project comparing wrist-worn IMU signal patterns between two
gym exercises — **Chest Press (rack)** vs **Squat Rack Shoulder Press** —
using a 3-exercise subset of Microsoft's RecoFit dataset.

## Contents

- `notebooks/01_bench_press_vs_shoulder_press.ipynb` — main analysis notebook
- `data/` — the CSV subset used for the analysis (see `data/README.md` for
  exactly what it is, how it was produced, and its license)

## Setup

```
python -m pip install -r requirements.txt
jupyter notebook notebooks/01_bench_press_vs_shoulder_press.ipynb
```

## Data

This repo ships a small, purpose-extracted subset of Microsoft's RecoFit
dataset — **not** the full dataset. Only 3 exercises are included: Chest
Press (rack), Squat Rack Shoulder Press, and Lateral Raise.

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
