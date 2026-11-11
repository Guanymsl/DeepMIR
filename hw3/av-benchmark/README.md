# Benchmarking for Audio-Text and Audio-Visual Generation

## Overview

This repository supports the evaluations of:

- Fréchet Distances (FD)

    - FD_VGG, with [VGGish](https://github.com/tensorflow/models/blob/master/research/audioset/vggish/README.md), also referred to as FAD

- Mean KL Distances (MKL)

    - KL_PassT, with [PaSST](https://github.com/kkoutini/PaSST)

- CLAP Scores

    - LAION_CLAP, cosine similarity between text and audio embeddings computed by [LAION-CLAP](https://github.com/LAION-AI/CLAP) with the `music_audioset_epoch_15_esc_90.14.pt` model.

## Installation

This repository has been tested on Ubuntu, and requires **Python 3.9+** with **PyTorch 2.5.1+**. Follow the steps below to set up the environment:

### 1. Recommended Setup

We recommend using a [miniforge](https://github.com/conda-forge/miniforge) environment.

### 2. Install PyTorch

Before proceeding, install PyTorch with the appropriate CUDA version from the [official PyTorch website](https://pytorch.org/).

### 3. Local source

The HW3 wrapper runs this local source directly; no editable package installation is required.
Run `python evaluate.py --pred baseline` from `hw3/` using the `deepmir_hw3` environment.

### 4. Download Pretrained Models

Download [music_audioset_epoch_15_esc_90.14.pt](https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt) and put it in `weights`.

## Usage

### Overview

Evaluation is a two-stage process:

1. **Extraction**: extract audio features for ground-truth and predicted audios. The extracted features are saved in `gt_cache` and `pred_cache` respectively.
2. **Evaluation**: compute the desired metrics using the extracted features.

By default, if `gt_cache` or `pred_cache` are not specified, we will use `gt_audio/cache` and `pred_audio/cache`.
`gt_audio` and `pred_audio` should point to a directory containing audio files in either `.wav` or `.flac` formats.

### Extraction

Audio features are automatically extracted during the evaluation stage.

**Manual extraction:**
You can force feature extraction by specifying:
 - `--recompute_gt_cache` for ground-truth audio features.
 - `--recompute_pred_cache` for predicted audio features.

This is useful if the extraction is interrupted or the cache is corrupted.

### Evaluation

```bash
python evaluate.py  --gt_audio <gt audio path> --gt_cache <gt cache path> --pred_audio <pred audio path> --pred_cache <pred cache path> --audio_length=<length of audio wanted in seconds> 
```

Use `--unpaired` for FD-VGG without filename pairing. Paired evaluation also computes KL-PaSST.

For LAION CLAP, supply a CSV with `name` (audio filename without extension) and `caption` columns:

```bash
python evaluate.py --clap_only --captions ../av.csv --pred_audio ../outputs/generated/baseline --pred_cache ../outputs/evaluation/baseline/clap_cache --audio_length 10 --pred_batch_size 8
```

## Supporting Libraries

To address issues with deprecated code in some underlying libraries, we have forked and modified several of them. These forks are included as dependencies to ensure compatibility down the road.

- LAION-CLAP: https://github.com/hkchengrex/CLAP
- PaSST: https://github.com/hkchengrex/passt_hear21


## Citation

This repository is part of the accompanying code for MMAudio. To cite this repository, please use the following BibTeX entry:

```bibtex
@inproceedings{cheng2024taming,
  title={MMAudio: Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis},
  author={Cheng, Ho Kei and Ishii, Masato and Hayakawa, Akio and Shibuya, Takashi and Schwing, Alexander and Mitsufuji, Yuki},
  booktitle={CVPR},
  year={2025}
}
```


## References

Many thanks to
- [PaSST](https://github.com/kkoutini/PaSST)
- [VGGish](https://github.com/tensorflow/models/blob/master/research/audioset/vggish/README.md)
- [passt_hear21](https://github.com/kkoutini/passt_hear21)
- [torchvggish](https://github.com/harritaylor/torchvggish)
- [audioldm_eval](https://github.com/haoheliu/audioldm_eval) -- on which this repository is based on
- [LAION-CLAP](https://github.com/LAION-AI/CLAP)
