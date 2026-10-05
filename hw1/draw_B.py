import json
import re
from pathlib import Path
import matplotlib.pyplot as plt

base_dir = Path("outputs/B")
results = []

for folder in base_dir.iterdir():
    if not folder.is_dir():
        continue

    match = re.fullmatch(r"mean_(\d+)_aug", folder.name)
    if match:
        layer = int(match.group(1))
        if not 0 <= layer <= 24:
            continue
    elif folder.name == "mean_last_aug":
        layer = 25
    else:
        continue

    json_files = list(folder.glob("*.json"))
    if not json_files:
        print(f"No JSON found: {folder}")
        continue

    json_path = json_files[0]
    with open(json_path, "r") as f:
        data = json.load(f)

    train_score = data["train_top1"] + 0.5 * data["train_top3"]
    val_score = data["val_top1"] + 0.5 * data["val_top3"]
    gap = data["generalization_gap"]

    results.append({
        "layer": layer,
        "train_score": train_score,
        "val_score": val_score,
        "gap": gap,
    })

results.sort(key=lambda x: x["layer"])

layers = [r["layer"] for r in results]
train_scores = [r["train_score"] for r in results]
val_scores = [r["val_score"] for r in results]
gaps = [r["gap"] for r in results]

plt.figure(figsize=(10, 6))

plt.plot(
    layers, train_scores,
    marker="o",
    label="Train Score"
)

plt.plot(
    layers, val_scores,
    marker="o",
    label="Validation Score"
)

plt.plot(
    layers, gaps,
    marker="o",
    label="Generalization Gap"
)

plt.xlabel("Layer")
plt.title("Performance vs. MERT Layer")

plt.xticks(
    range(26),
    [str(i) for i in range(25)] + ["last"]
)

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("layer_comparison_B.png", dpi=300)
plt.show()
