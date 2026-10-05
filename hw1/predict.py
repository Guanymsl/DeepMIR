import torch
import torchaudio
import argparse
import json
import joblib
import librosa

import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor

MERT_MODEL = "m-a-p/MERT-v1-330M"
MERT_SAMPLE_RATE = 24000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/b12901024.json")
    return parser.parse_args()

def load_audio(path):
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != MERT_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, MERT_SAMPLE_RATE)

    return waveform.squeeze(0)

def extract_acoustic_features(waveform):
    y = waveform.numpy()

    if len(y) == 0:
        return np.zeros(4, dtype=np.float32)

    peak = float(np.max(np.abs(y)))
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms)) if len(rms) > 0 else 1e-6

    crest_factor = peak / (mean_rms + 1e-6)
    dynamic_range = peak - mean_rms
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=MERT_SAMPLE_RATE, roll_percent=0.85)))

    spec = np.abs(librosa.stft(y)) ** 2
    freqs = librosa.fft_frequencies(sr=MERT_SAMPLE_RATE)
    sub_bass_mask = (freqs >= 20) & (freqs <= 60)
    sub_bass_energy = np.sum(spec[sub_bass_mask, :])
    total_energy = np.sum(spec) + 1e-6
    sub_bass_ratio = float(sub_bass_energy / total_energy)

    return np.array([crest_factor, dynamic_range, rolloff, sub_bass_ratio], dtype=np.float32)

def load_mert(model_name, device):
    print(f"Loading {model_name} ...")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    model.to(device)
    model.eval()

    return processor, model

@torch.no_grad()
def extract_mert_feature(audio_path, processor, model, device, pooling, layer, acoustic=False):
    waveform = load_audio(audio_path)

    inputs = processor(waveform.numpy(), sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt")
    input_values = inputs["input_values"].to(device)

    if layer == "last":
        outputs = model(input_values)
        h = outputs.last_hidden_state[0]
    else:
        captured = {}
        layer = int(layer)

        if layer == 0:
            def hook(module, input):
                captured["h"] = input[0]
            handle = model.encoder.layers[0].register_forward_pre_hook(hook)
        else:
            def hook(module, input, output):
                captured["h"] = output[0] if isinstance(output, tuple) else output
            handle = model.encoder.layers[layer - 1].register_forward_hook(hook)

        model(input_values)
        handle.remove()
        h = captured["h"][0]

    mean = h.mean(dim=0)

    if pooling == "mean":
        mert_feature = mean.cpu().numpy().astype(np.float32)
    else:
        std = h.std(dim=0)
        mert_feature = torch.cat([mean, std], dim=0).cpu().numpy().astype(np.float32)

    if acoustic:
        acoustic_feature = extract_acoustic_features(waveform)
        feature = np.concatenate([mert_feature, acoustic_feature], axis=0)
    else:
        feature = mert_feature

    return feature

def extract_test(df, dataset_dir, processor, model, device, cache_dir, pooling, layer, acoustic):
    features = []
    sample_ids = []

    cache_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting MERT"):
        sample_id = row["sample_id"]
        cache_file = cache_dir / f"{sample_id}.npy"

        if cache_file.exists():
            feature = np.load(cache_file)
        else:
            audio_path = dataset_dir / row["audio_path"]
            feature = extract_mert_feature(audio_path, processor, model, device, pooling, layer, acoustic)
            np.save(cache_file, feature)

        features.append(feature)
        sample_ids.append(sample_id)

    return np.stack(features), sample_ids

def predict_dataset(dataset, model_dir, pooling, layer, acoustic, processor, mert, device):
    dataset_dir = Path(f"data/dataset_{dataset}")
    model_dir = Path(model_dir)

    df = pd.read_csv(dataset_dir / "manifest.csv")
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"\nDataset {dataset}")
    print("Test:", len(test_df))

    X_test, sample_ids = extract_test(test_df, dataset_dir, processor, mert, device, model_dir / "features_test", pooling, layer, acoustic)

    classifier = joblib.load(model_dir / "classifier.joblib")
    scores = classifier.decision_function(X_test)
    classes = classifier.classes_
    top3_indices = np.argsort(scores, axis=1)[:, ::-1][:, :3]

    predictions = {}
    for sample_id, indices in zip(sample_ids, top3_indices):
        predictions[str(sample_id)] = [str(classes[i]) for i in indices]

    return predictions

def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    processor, mert = load_mert(MERT_MODEL, device)

    prediction_a = predict_dataset("A", "outputs/A/meanstd_last_acoustic", "meanstd", "last", True, processor, mert, device)
    prediction_b = predict_dataset("B", "outputs/B/mean_6_aug", "mean", "6", False, processor, mert, device)

    output = {
        "dataset_A": prediction_a,
        "dataset_B": prediction_b,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print("\nSaved:", args.output)

if __name__ == "__main__":
    main()
