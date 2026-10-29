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

Place **POP909** under the `data/` directory with the following structure:

```text
data/
└── POP909/
    ├── training/
    └── prompts/
        └── basic-b/
        └── bonus/
```

Place **MusDr** under the `data/` directory with the following structure:

```text
MusDr/
└── ...
```

Place **FluidR3_GM.sf2** under the `data/` directory with the following structure:

```text
soundfonts/
└── FluidR3_GM.sf2
```

## Tokenization
Run the tokenization script with:

```bash
python3 tokenization.py
```

## Training
Run the training script with:

```bash
python3 train.py
```

## Prediction
To generate midi files, run:

```bash
python3 predict.py [OPTIONS]
```

### Arguments

- `--task A/B`  
  Select the dataset and generation task:
  - `A`: unconditional generation
  - `B`: conditional generation

## Render

To render .wav files, run:

```bash
python3 predict.py [OPTIONS]
```
- `--task A/B/O`  
  Select the dataset and generation task:
  - `A`: unconditional generation
  - `B`: conditional generation
  - `O`: original dataset

## Plotting Results

To generate the line charts used in the report:

```bash
python3 draw.py
```
