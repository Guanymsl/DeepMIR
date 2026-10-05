# Music Release-Decade and Release-Market Classification

This project performs two music classification tasks using audio features extracted from **MERT**:

- **Release-Decade Classification**: predicts the release decade of a song.
- **Release-Market Classification**: predicts the release market of a song.

## Environment Setup

Create and activate a Python 3.11 virtual environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip3 install -r requirements.txt
```

## Dataset Preparation

Place **Dataset A** and **Dataset B** under the `data/` directory with the following structure:

```text
data/
├── dataset_A/
│   ├── manifest.csv
│   └── audio/
│       └── ... .wav
└── dataset_B/
    ├── manifest.csv
    └── audio/
        └── ... .wav
```

- **Dataset A** is used for **release-decade classification**.
- **Dataset B** is used for **release-market classification**.

To extract the vocal parts from Dataset B, run:

```bash
python3 separate.py
```

This will generate **Dataset C**, which contains the separated vocal tracks from Dataset B and is used for vocal feature extraction.

## Training

Run the training script with:

```bash
python3 train.py [OPTIONS]
```

### Arguments

- `--dataset A/B`  
  Select the dataset and classification task:
  - `A`: release-decade classification
  - `B`: release-market classification

- `--pooling mean/meanstd`  
  Select the pooling method for MERT features.

- `--layer 0-24/last`  
  Select the MERT layer used for feature extraction.  
  Default: `last`.

- `--acoustic`  
  Enable additional acoustic features.

- `--vocal`  
  Enable additional features extracted from the vocal parts.  
  This option is only available for Dataset B.

- `--augment`  
  Enable data augmentation.

## Submitted Models

To reproduce the submitted models, run:

### Dataset A — Release-Decade Classification

```bash
python3 train.py --dataset A --pooling meanstd --acoustic
```

### Dataset B — Release-Market Classification

```bash
python3 train.py --dataset B --pooling mean --layer 6 --augment
```

## Reproducing All Experiments

To run all experiments included in the report:

```bash
chmod u+x ./run_experiments.sh
./run_experiments.sh
```

## Plotting Results

To generate the line charts used in the report:

```bash
python3 draw_A.py
python3 draw_B.py
```

## Prediction

To generate predictions using the submitted models, run:

```bash
python3 predict.py
```

The prediction script uses the following trained models:

```text
Dataset A: outputs/A/meanstd_last_acoustic
Dataset B: outputs/B/mean_6_aug
```

The predictions are saved to `outputs/b12901024.json`.
