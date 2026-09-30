import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torchaudio

from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor

MERT_MODELS = {
    "95M": "m-a-p/MERT-v1-95M",
    "330M": "m-a-p/MERT-v1-330M",
}

MERT_SAMPLE_RATE = 24000
NUM_SEGMENTS = 3

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--mert", type=str, default="95M", choices=["95M", "330M"])
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "mean_std", "segment_mean_std"])
    parser.add_argument("--classifier", type=str, default="linear", choices=["linear", "rbf"])

    return parser.parse_args()

def load_audio(path):
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != MERT_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, MERT_SAMPLE_RATE)

    return waveform.squeeze(0)

def load_mert(model_name, device):
    print(f"Loading {model_name} ...")

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    model.to(device)
    model.eval()

    return processor, model

@torch.no_grad()
def extract_mert_feature(
    audio_path,
    processor,
    model,
    device,
    pooling,
):
    waveform = load_audio(audio_path)

    inputs = processor(
        waveform.numpy(),
        sampling_rate=MERT_SAMPLE_RATE,
        return_tensors="pt",
    )

    input_values = inputs["input_values"].to(device)
    outputs = model(input_values)
    h = outputs.last_hidden_state[0]

    if pooling == "mean":
        feature = h.mean(dim=0)

    elif pooling == "mean_std":
        mean = h.mean(dim=0)
        std = h.std(dim=0)
        feature = torch.cat([mean, std], dim=0)

    elif pooling == "segment_mean_std":
        segments = torch.tensor_split(h, NUM_SEGMENTS, dim=0)
        segment_features = []

        for segment in segments:
            mean = segment.mean(dim=0)
            std = segment.std(dim=0)
            segment_feature = torch.cat([mean, std], dim=0)
            segment_features.append(segment_feature)

        feature = torch.cat(segment_features, dim=0)

    else:
        raise ValueError(f"Unknown pooling method")

    return feature.cpu().numpy().astype(np.float32)

def extract_split(
    df,
    dataset_dir,
    processor,
    model,
    device,
    pooling,
    cache_dir,
):
    features = []
    sample_ids = []

    cache_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Extracting MERT",
    ):
        sample_id = row["sample_id"]
        cache_file = cache_dir / f"{sample_id}.npy"

        if cache_file.exists():
            feature = np.load(cache_file)

        else:
            audio_path = dataset_dir / row["audio_path"]

            feature = extract_mert_feature(
                audio_path,
                processor,
                model,
                device,
                pooling,
            )

            np.save(cache_file, feature)

        features.append(feature)
        sample_ids.append(sample_id)

    X = np.stack(features)

    return X, sample_ids

def main():
    args = parse_args()

    mert_model = MERT_MODELS[args.mert]

    dataset_name = f"dataset_{args.dataset}"
    dataset_dir = Path(f"data/{dataset_name}")

    feature_dir = Path(f"outputs/{dataset_name}/{args.mert}/{args.pooling}")
    output_dir = feature_dir / args.classifier
    cache_dir = feature_dir / "features"

    model_path = output_dir / "classifier.joblib"
    metrics_path = output_dir / "metrics.json"
    prediction_path = output_dir / "predictions.json"

    print(f"Dataset: {dataset_name}")
    print(f"MERT: {args.mert}")
    print(f"Pooling: {args.pooling}")
    print(f"Classifier: {args.classifier}")

    if args.pooling == "segment_mean_std":
        print(f"Segments: {NUM_SEGMENTS}")

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    classifier = joblib.load(model_path)

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        print()
        print("Validation")
        print(f"Top-1: {metrics['val_top1']:.4f}")
        print(f"Top-3: {metrics['val_top3']:.4f}")

    df = pd.read_csv(dataset_dir / "manifest.csv")
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print()
    print("Test:", len(test_df))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print()
    print("Device:", device)

    processor, mert = load_mert(mert_model, device)

    print("\nExtracting TEST features")

    X_test, sample_ids = extract_split(
        test_df,
        dataset_dir,
        processor,
        mert,
        device,
        args.pooling,
        cache_dir,
    )

    print()
    print("Feature shape:", X_test.shape)

    scores = classifier.decision_function(X_test)
    classes = classifier.classes_

    top3_indices = np.argsort(scores, axis=1)[:, ::-1][:, :3]

    predictions = {}

    for i, sample_id in enumerate(sample_ids):
        predictions[str(sample_id)] = classes[top3_indices[i]].tolist()

    with open(
        prediction_path,
        "w",
    ) as f:
        json.dump(
            predictions,
            f,
            indent=2,
        )

    print()
    print("Predictions")

    for sample_id in sample_ids[:10]:
        print(sample_id, predictions[str(sample_id)])

    print()
    print("Saved:")
    print(prediction_path)

if __name__ == "__main__":
    main()
