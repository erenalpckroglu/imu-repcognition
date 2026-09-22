# Sports Data Analytics Project

Open `sports_data_analytics.ipynb` and run the cells from top to bottom.

The notebook compares three exercises from the RecoFit wrist-IMU dataset and
then studies Chest Press repetitions in more detail. Each detected pushing
phase is divided into two halves so that acceleration patterns can be compared
within repetitions, from the beginning to the end of a set, and among sets.

The notebook writes its derived CSV tables to `outputs/`.

From this directory, run:

```bash
jupyter notebook sports_data_analytics.ipynb
```

The supporting `analysis.py` file contains reusable feature-engineering and
Chest Press segmentation functions used by the notebook.
