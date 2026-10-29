import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HISTORY_PATH = Path("outputs/checkpoints/history.json")
OUTPUT_PATH = Path("outputs/charts/loss_curve.png")

with HISTORY_PATH.open("r", encoding="utf-8") as f:
    history = json.load(f)

epochs = np.array([h["epoch"] for h in history])
train_loss = np.array([h["train_loss"] for h in history])
val_loss = np.array([h["val_loss"] for h in history])

best_idx = np.argmin(val_loss)
best_epoch = epochs[best_idx]
best_loss = val_loss[best_idx]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    epochs, train_loss,
    label="Training Loss",
    linewidth=2,
    color="forestgreen",
    alpha=0.7
)

ax.plot(
    epochs, val_loss,
    label="Validation Loss",
    linewidth=2,
    color="darkorange",
    alpha=0.7
)

ax.scatter(
    best_epoch, best_loss,
    color="red",
    s=25,
    zorder=5
)

ax.annotate(
    f"Best: Epoch {best_epoch} ({best_loss:.4f})",
    xy=(best_epoch, best_loss),
    xytext=(0, 35),
    textcoords="offset points",
    ha="center",
    va="bottom",
    arrowprops=dict(
        arrowstyle="-|>",
        color="red",
        mutation_scale=12,
        linewidth=1.2,
        shrinkA=3,
        shrinkB=4
    ),
    fontsize=11,
    fontweight="bold",
    color="red"
)

ax.set_title(
    "Training and Validation Loss",
    fontsize=16,
    fontweight="bold"
)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Cross-Entropy Loss", fontsize=12)

ax.set_xlim(1, epochs.max())
ax.set_xticks(np.arange(1, epochs.max() + 1, 2))

ax.grid(True, linestyle="--", alpha=0.3)
ax.legend(fontsize=11)

plt.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"Best Epoch: {best_epoch}")
print(f"Best Validation Loss: {best_loss:.4f}")
print(f"Saved to: {OUTPUT_PATH}")
