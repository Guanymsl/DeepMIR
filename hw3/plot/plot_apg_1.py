
import json
import matplotlib.pyplot as plt

from pathlib import Path

SCALES = [3, 5, 7, 9, 11, 13]
METRICS = {
    'FD-VGG': 'FD-VGG ↓',
    'CLAP': 'CLAP ↑',
    'KL-passt': 'KL-PaSST ↓',
    'MQAScore': 'MQAScore ↑'
}

def main():
    cfg_results = {metric: [] for metric in METRICS}
    apg_results = {metric: [] for metric in METRICS}

    for scale in SCALES:
        path = Path(f'outputs/evaluation/baseline{scale}/metrics.json')
        with open(path) as f:
            data = json.load(f)

        for metric in METRICS:
            cfg_results[metric].append(data[metric])

    for scale in SCALES:
        path = Path(f'outputs/evaluation/apg_w{scale}_eta0_m-0.5_n0/metrics.json')
        with open(path) as f:
            data = json.load(f)

        for metric in METRICS:
            apg_results[metric].append(data[metric])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, (metric, title) in zip(axes.flat, METRICS.items()):
        ax.plot(SCALES, cfg_results[metric], marker='o', linewidth=2, color='#0072B2', label='CFG')
        ax.plot(SCALES, apg_results[metric], marker='o', linewidth=2, color='#E69F00', label='APG')
        ax.set_title(title)
        ax.set_xlabel('Guidance Scale')
        ax.set_ylabel(metric)
        ax.set_xticks(SCALES)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle('Stable Audio Open 1.0 — CFG vs APG Guidance Scale Sweep', fontsize=15)
    plt.tight_layout()

    output = Path('outputs/evaluation/apg_scale_sweep_1.png')
    fig.savefig(output, dpi=300, bbox_inches='tight')

    print('Saved:', output)

if __name__ == '__main__':
    main()
