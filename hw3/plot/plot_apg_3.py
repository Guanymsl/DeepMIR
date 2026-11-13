import json
import matplotlib.pyplot as plt

from pathlib import Path

MOMENTUMS = [-0.9, -0.7, -0.5, -0.3, 0, 0.3, 0.5, 0.7, 0.9]
METRICS = {
    'FD-VGG': 'FD-VGG ↓',
    'CLAP': 'CLAP ↑',
    'KL-passt': 'KL-PaSST ↓',
    'MQAScore': 'MQAScore ↑'
}

def main():
    results = {metric: [] for metric in METRICS}

    for momentum in MOMENTUMS:
        path = Path(f'outputs/evaluation/apg_w9_eta0_m{momentum}_n0/metrics.json')
        with open(path) as f:
            data = json.load(f)

        for metric in METRICS:
            results[metric].append(data[metric])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (metric, title) in zip(axes.flat, METRICS.items()):
        ax.plot(MOMENTUMS, results[metric], marker='o', linewidth=2, color='#0072B2')
        ax.set_title(title)
        ax.set_xlabel('APG Guidance MOMENTUM')
        ax.set_ylabel(metric)
        ax.set_xticks(MOMENTUMS)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Stable Audio Open 1.0 — APG Guidance MOMENTUM Sweep', fontsize=15)
    plt.tight_layout()

    output = Path('outputs/evaluation/apg_scale_sweep_3.png')
    fig.savefig(output, dpi=300, bbox_inches='tight')

    print('Saved:', output)

if __name__ == '__main__':
    main()
