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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt


MERT_MODELS = {
    "95M": "m-a-p/MERT-v1-95M",
    "330M": "m-a-p/MERT-v1-330M",
}

MERT_LAYERS = {
    "95M": 12,
    "330M": 24,
}

MERT_SAMPLE_RATE = 24000
NUM_SEGMENTS = 3
CROP_SECONDS = 20
RANDOM_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--mert", type=str, default="95M", choices=["95M", "330M"])
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--pooling", type=str, default="mean_std", choices=["mean", "mean_std", "segment_mean_std"])
    parser.add_argument("--classifier", type=str, default="linear", choices=["linear", "rbf"])

    return parser.parse_args()


def load_audio(path):
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != MERT_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, MERT_SAMPLE_RATE)

    return waveform.squeeze(0)


def random_crop(waveform, sample_id):
    crop_length = CROP_SECONDS * MERT_SAMPLE_RATE

    if len(waveform) <= crop_length:
        return waveform

    rng = np.random.default_rng(RANDOM_SEED + int(sample_id))
    max_start = len(waveform) - crop_length
    start = rng.integers(0, max_start + 1)

    return waveform[start:start + crop_length]


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
    layer,
    augment=False,
    sample_id=None,
):
    waveform = load_audio(audio_path)

    if augment:
        waveform = random_crop(waveform, sample_id)

    inputs = processor(
        waveform.numpy(),
        sampling_rate=MERT_SAMPLE_RATE,
        return_tensors="pt",
    )

    input_values = inputs["input_values"].to(device)

    if layer == -1:
        outputs = model(input_values)
        h = outputs.last_hidden_state[0]

    elif layer == 0:
        captured = {}

        def hook(module, input):
            captured["h"] = input[0]

        handle = model.encoder.layers[0].register_forward_pre_hook(hook)
        model(input_values)
        handle.remove()

        h = captured["h"][0]

    else:
        captured = {}

        def hook(module, input, output):
            if isinstance(output, tuple):
                captured["h"] = output[0]
            else:
                captured["h"] = output

        handle = model.encoder.layers[layer - 1].register_forward_hook(hook)
        model(input_values)
        handle.remove()

        h = captured["h"][0]

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
        raise ValueError("Unknown pooling method")

    return feature.cpu().numpy().astype(np.float32)


def extract_split(
    df,
    dataset_dir,
    processor,
    model,
    device,
    pooling,
    layer,
    cache_dir,
):
    features = []
    labels = []
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
                layer,
            )

            np.save(cache_file, feature)

        features.append(feature)
        sample_ids.append(sample_id)

        if pd.notna(row["label"]) and row["label"] != "":
            labels.append(str(row["label"]))

    X = np.stack(features)

    if len(labels) == len(df):
        y = np.array(labels)
    else:
        y = None

    return X, y, sample_ids


def extract_augmented_split(
    df,
    dataset_dir,
    processor,
    model,
    device,
    pooling,
    layer,
    cache_dir,
):
    features = []
    labels = []

    cache_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Extracting augmented MERT",
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
                layer,
                augment=True,
                sample_id=sample_id,
            )

            np.save(cache_file, feature)

        features.append(feature)
        labels.append(str(row["label"]))

    X = np.stack(features)
    y = np.array(labels)

    return X, y


def top_k_accuracy(y_true, scores, classes, k):
    top_indices = np.argsort(scores, axis=1)[:, ::-1][:, :k]

    correct = 0

    for i, label in enumerate(y_true):
        predicted_labels = classes[top_indices[i]]

        if label in predicted_labels:
            correct += 1

    return correct / len(y_true)


def evaluate(classifier, X, y):
    scores = classifier.decision_function(X)
    classes = classifier.classes_

    top1_indices = np.argmax(scores, axis=1)
    y_pred = classes[top1_indices]

    top1 = accuracy_score(y, y_pred)
    top3 = top_k_accuracy(y, scores, classes, k=3)

    return top1, top3, y_pred, scores


def main():
    args = parse_args()

    max_layer = MERT_LAYERS[args.mert]

    if args.layer != -1 and not 0 <= args.layer <= max_layer:
        raise ValueError(f"Layer must be between 0 and {max_layer}, or -1 for final layer")

    mert_model = MERT_MODELS[args.mert]

    if args.layer == -1:
        layer_name = "final"
    else:
        layer_name = f"layer_{args.layer}"

    dataset_name = f"dataset_{args.dataset}"
    dataset_dir = Path(f"data/{dataset_name}")

    feature_dir = Path(f"outputs/{dataset_name}/{args.mert}/{layer_name}/{args.pooling}")

    if args.augment:
        output_dir = feature_dir / f"{args.classifier}_augment"
    else:
        output_dir = feature_dir / args.classifier

    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = feature_dir / "features"
    augment_cache_dir = feature_dir / f"features_crop_{CROP_SECONDS}s"

    print(f"Dataset: {dataset_name}")
    print(f"MERT: {args.mert}")
    print(f"Layer: {args.layer}")
    print(f"Pooling: {args.pooling}")
    print(f"Classifier: {args.classifier}")
    print(f"Augmentation: {args.augment}")

    if args.pooling == "segment_mean_std":
        print(f"Segments: {NUM_SEGMENTS}")

    if args.augment:
        print(f"Crop: {CROP_SECONDS}s")

    df = pd.read_csv(dataset_dir / "manifest.csv")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print()
    print("Samples")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))
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

    print("\nExtracting TRAIN features")

    X_train, y_train, _ = extract_split(
        train_df,
        dataset_dir,
        processor,
        mert,
        device,
        args.pooling,
        args.layer,
        cache_dir,
    )

    original_train_size = len(X_train)

    if args.augment:
        print("\nExtracting AUGMENTED TRAIN features")

        X_aug, y_aug = extract_augmented_split(
            train_df,
            dataset_dir,
            processor,
            mert,
            device,
            args.pooling,
            args.layer,
            augment_cache_dir,
        )

        X_train = np.concatenate([X_train, X_aug], axis=0)
        y_train = np.concatenate([y_train, y_aug], axis=0)

    print("\nExtracting VALIDATION features")

    X_val, y_val, _ = extract_split(
        val_df,
        dataset_dir,
        processor,
        mert,
        device,
        args.pooling,
        args.layer,
        cache_dir,
    )

    print()
    print("Feature shape:", X_train.shape)
    print("Training samples:", len(X_train))

    del mert

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    if args.classifier == "linear":
        C_VALUES = [
            0.00001,
            0.00003,
            0.0001,
            0.0003,
            0.001,
            0.003,
            0.01,
        ]
    else:
        C_VALUES = [
            0.01,
            0.1,
            1,
            10,
            100,
        ]

    best_classifier = None
    best_c = None
    best_top1 = -1

    print("\nTuning SVM...")

    for C in C_VALUES:
        classifier = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel=args.classifier,
                        C=C,
                        gamma="scale",
                    ),
                ),
            ]
        )

        classifier.fit(X_train, y_train)

        train_top1, train_top3, _, _ = evaluate(
            classifier,
            X_train,
            y_train,
        )

        val_top1, val_top3, _, _ = evaluate(
            classifier,
            X_val,
            y_val,
        )

        print(
            f"C={C:<7} "
            f"Train Top-1={train_top1:.4f} "
            f"Train Top-3={train_top3:.4f} "
            f"Val Top-1={val_top1:.4f} "
            f"Val Top-3={val_top3:.4f}"
        )

        if val_top1 > best_top1:
            best_top1 = val_top1
            best_c = C
            best_classifier = classifier

    classifier = best_classifier

    print()
    print("Best C:", best_c)

    train_top1, train_top3, _, _ = evaluate(
        classifier,
        X_train,
        y_train,
    )

    val_top1, val_top3, y_pred, val_scores = evaluate(
        classifier,
        X_val,
        y_val,
    )

    generalization_gap = train_top1 - val_top1

    print()
    print("==============================")
    print("Results")
    print("==============================")
    print(f"Train Top-1 Accuracy: {train_top1:.4f}")
    print(f"Train Top-3 Accuracy: {train_top3:.4f}")
    print(f"Validation Top-1 Accuracy: {val_top1:.4f}")
    print(f"Validation Top-3 Accuracy: {val_top3:.4f}")
    print(f"Generalization Gap: {generalization_gap:.4f}")

    classes = classifier.classes_
    cm = confusion_matrix(y_val, y_pred, labels=classes)

    print("\nConfusion Matrix")
    print(classes)
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes,
    )

    disp.plot(
        xticks_rotation=45,
    )

    plt.tight_layout()

    cm_path = output_dir / "confusion_matrix.png"

    plt.savefig(
        cm_path,
        dpi=200,
    )

    plt.close()

    model_path = output_dir / "classifier.joblib"
    joblib.dump(classifier, model_path)

    metrics = {
        "dataset": args.dataset,
        "mert": args.mert,
        "mert_model": mert_model,
        "layer": args.layer,
        "pooling": args.pooling,
        "augmentation": args.augment,
        "classifier": f"{args.classifier}_svm",
        "svm_c": best_c,
        "feature_dim": int(X_train.shape[1]),
        "original_train_samples": int(original_train_size),
        "train_samples": int(len(X_train)),
        "train_top1": float(train_top1),
        "train_top3": float(train_top3),
        "val_top1": float(val_top1),
        "val_top3": float(val_top3),
        "generalization_gap": float(generalization_gap),
        "classes": classes.tolist(),
    }

    if args.augment:
        metrics["crop_seconds"] = CROP_SECONDS

    if args.pooling == "segment_mean_std":
        metrics["segments"] = NUM_SEGMENTS

    if args.classifier == "rbf":
        metrics["gamma"] = "scale"

    metrics_path = output_dir / "metrics.json"

    with open(
        metrics_path,
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    print()
    print("Saved:")
    print(model_path)
    print(cm_path)
    print(metrics_path)

if __name__ == "__main__":
    main()
