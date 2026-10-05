# Music Era and Release Country Classification

This project performs two music classification tasks using audio features extracted from **MERT**:

- **Era Classification**: predicts the release era of a song (e.g., 1960s, 1970s, ..., 2010s).
- **Release Country Classification**: predicts the country where a song was released.

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

- **Dataset A** is used for **music era classification**.
- **Dataset B** is used for **release country classification**.

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
  - `A`: music era classification
  - `B`: release country classification

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

### Dataset A — Era Classification

```bash
python3 train.py --dataset A --pooling meanstd --acoustic
```

### Dataset B — Release Country Classification

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
