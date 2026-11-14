# Text-to-Music Generation and Evaluation

This project generates music using **Stable Audio Open 1.0** and explores two methods for improving generation:

- **Adaptive Projected Guidance (APG)**: modifies classifier-free guidance (CFG).
- **Prompt Engineering**: rewrites captions using GPT-4.1.

The generated audio is evaluated using **FD-VGG**, **CLAP**, **KL-passt**, and **MQAScore**.

## Environment Setup

Create and activate a Python 3.11 Conda environment:

```bash
conda create -n deepmir_hw3 python=3.11
conda activate deepmir_hw3
```

Install the required packages:

```bash
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Accept the license for `stabilityai/stable-audio-open-1.0` on Hugging Face, then run:

```bash
hf auth login
```

## Dataset Preparation

Place the reference audio and test captions under the project directory:

```text
audio_mp3/
└── ... .mp3
test.csv
```

Run:

```bash
python3 prepare.py
```

This generates the WAV reference audio and evaluation captions:

```text
audio_wav/
└── ... .wav
av.csv
```

## CLAP Retrieval

Place the checkpoint `music_audioset_epoch_15_esc_90.14.pt` under `av-benchmark/weights/`.

Run:

```bash
python3 retrieve.py
```

The retrieval results are saved to `outputs/retrieval.json`.

## Music Generation

To generate music using the CFG baseline:

```bash
python3 baseline.py
```

To generate music using APG:

```bash
python3 apg.py --csv test.csv --scale 9 --eta 0 --momentum -0.5 --norm-threshold 0
```

### Arguments

- `--csv`  
  Select the input caption CSV file.

- `--scale`  
  Set the guidance scale.

- `--eta`  
  Set the parallel guidance coefficient.

- `--momentum`  
  Set the momentum coefficient.

- `--norm-threshold`  
  Set the norm clipping threshold. `0` disables clipping.

## Prompt Engineering

Set the OpenAI API key:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

Generate the Concise, Detailed, and Planned captions:

```bash
python3 prompt.py
```

The rewritten captions can be used with `apg.py` through the `--csv` argument.

## Evaluation

FD-VGG, CLAP, and KL-passt are evaluated using **av-benchmark**. MQAScore uses a custom AF2 backend with Qwen3-1.7B for tag extraction.

Create the MQAScore environments:

```bash
conda env create -f envs/environment_mqascore_tags.yml
conda env create -f envs/environment_mqascore_af2.yml
```

Run:

```bash
python3 evaluate.py --pred <PRED_NAME> --metrics all
```

### Arguments

- `--pred`  
  Select the generated audio directory under `outputs/generated/`.

- `--metrics`  
  Select one or more metrics: `fd-vgg`, `clap`, `kl-passt`, `mqascore`, or `all`.

The evaluation results are saved to `outputs/evaluation/<PRED_NAME>/metrics.json`.

## Reproducing All Experiments

To run the experiments included in the report:

```bash
chmod u+x ./script/*.sh
bash ./script/run_CFG.sh
bash ./script/run_APG_1.sh
bash ./script/run_APG_2.sh
bash ./script/run_APG_3.sh
bash ./script/run_APG_4.sh
```

## Plotting Results

To generate the figures used in the report:

```bash
chmod u+x ./plot/plot.sh
bash ./plot/plot.sh
```

## Audio Conversion

To convert the generated WAV files to MP3:

```bash
python3 convert.py --pred <PRED_NAME>
```

## Submitted Results

The final configuration uses **Original Captions + APG** with scale 9, eta 0, momentum -0.5, and norm threshold 0.

To reproduce the final generation:

```bash
python3 apg.py --csv test.csv --scale 9 --eta 0 --momentum -0.5 --norm-threshold 0
```
