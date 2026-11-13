import json
import matplotlib.pyplot as plt

from pathlib import Path

ETAS = [0, 0.25, 0.5, 0.75, 1]
METRICS = {
    'FD-VGG': 'FD-VGG ↓',
    'CLAP': 'CLAP ↑',
    'KL-passt': 'KL-PaSST ↓',
    'MQAScore': 'MQAScore ↑'
}

def main():
    results = {metric: [] for metric in METRICS}

    for eta in ETAS:
        path = Path(f'outputs/evaluation/apg_w9_eta{eta}_m-0.5_n0/metrics.json')
        with open(path) as f:
            data = json.load(f)

        for metric in METRICS:
            results[metric].append(data[metric])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (metric, title) in zip(axes.flat, METRICS.items()):
        ax.plot(ETAS, results[metric], marker='o', linewidth=2, color='#0072B2')
        ax.set_title(title)
        ax.set_xlabel('APG Guidance ETA')
        ax.set_ylabel(metric)
        ax.set_xticks(ETAS)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Stable Audio Open 1.0 — APG Guidance ETA Sweep', fontsize=15)
    plt.tight_layout()

    output = Path('outputs/evaluation/apg_scale_sweep_2.png')
    fig.savefig(output, dpi=300, bbox_inches='tight')

    print('Saved:', output)

if __name__ == '__main__':
    main()
