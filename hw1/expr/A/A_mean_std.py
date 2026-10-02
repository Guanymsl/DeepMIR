import torch
import torchaudio
import json
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

MERT_MODEL = "m-a-p/MERT-v1-330M"
MERT_SAMPLE_RATE = 24000
RANDOM_SEED = 42

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
):
    waveform = load_audio(audio_path)

    inputs = processor(waveform.numpy(), sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt")
    input_values = inputs["input_values"].to(device)

    outputs = model(input_values)
    h = outputs.last_hidden_state[0]

    mean = h.mean(dim=0)
    std = h.std(dim=0)
    mert_feature = torch.cat([mean, std], dim=0).cpu().numpy().astype(np.float32)

    return mert_feature

def extract_split(
    df,
    dataset_dir,
    processor,
    model,
    device,
    cache_dir,
):
    features = []
    labels = []
    sample_ids = []

    cache_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting MERT"):
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
    mert_model = MERT_MODEL

    dataset_dir = Path("../../data/dataset_A")
    feature_dir = Path("expr/mean_std")

    output_dir = feature_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = feature_dir / "features"

    df = pd.read_csv(dataset_dir / "manifest.csv")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)

    print("\nSamples")
    print("Train:", len(train_df))
    print("Validation:", len(val_df))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("\nDevice:", device)

    processor, mert = load_mert(mert_model, device)

    print("\nExtracting TRAIN features")
    X_train, y_train, _ = extract_split(
        train_df,
        dataset_dir,
        processor,
        mert,
        device,
        cache_dir,
    )

    print("\nExtracting VALIDATION features")
    X_val, y_val, _ = extract_split(
        val_df,
        dataset_dir,
        processor,
        mert,
        device,
        cache_dir,
    )

    print("\nFeature shape:", X_train.shape)

    del mert
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    best_classifier = None
    best_c = None
    best_top = -1

    C = [0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 1.0]
    for c in C:
        steps = [
            ("scaler", StandardScaler()),
            ("norm", Normalizer(norm="l2")),
            ("svm", LinearSVC(C=c, penalty="l2", dual="auto", class_weight="balanced", max_iter=10000, random_state=RANDOM_SEED))
        ]

        classifier = Pipeline(steps)
        classifier.fit(X_train, y_train)

        train_top1, train_top3, _, _ = evaluate(classifier, X_train, y_train)
        val_top1, val_top3, _, _ = evaluate(classifier, X_val, y_val)

        print(
            f"Param={c:<6} "
            f"Train Top-1={train_top1:.4f} "
            f"Train Top-3={train_top3:.4f} "
            f"Val Top-1={val_top1:.4f} "
            f"Val Top-3={val_top3:.4f}"
        )

        val_score = val_top1 + 0.5 * val_top3
        if val_score > best_top:
            best_top = val_score
            best_c = c
            best_classifier = classifier

    classifier = best_classifier

    print("\nBest C:", best_c)

    train_top1, train_top3, _, _ = evaluate(classifier, X_train, y_train)
    val_top1, val_top3, y_pred, _ = evaluate(classifier, X_val, y_val)
    generalization_gap = train_top1 - val_top1

    print("\n==============================")
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

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    model_path = output_dir / "classifier.joblib"
    joblib.dump(classifier, model_path)

    metrics = {
        "c": best_c,
        "train_top1": float(train_top1),
        "train_top3": float(train_top3),
        "val_top1": float(val_top1),
        "val_top3": float(val_top3),
        "generalization_gap": float(generalization_gap),
        "classes": classes.tolist(),
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved:")
    print(model_path)
    print(cm_path)
    print(metrics_path)

if __name__ == "__main__":
    main()
