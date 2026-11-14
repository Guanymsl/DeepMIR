import json
import matplotlib.pyplot as plt

from pathlib import Path

MODES = {
    'Original': 'apg_w9_eta0_m-0.5_n0',
    'Concise': 'concise',
    'Detailed': 'detailed',
    'Planned': 'planned'
}

METRICS = {
    'FD-VGG': 'FD-VGG ↓',
    'CLAP': 'CLAP ↑',
    'KL-passt': 'KL-PaSST ↓',
    'MQAScore': 'MQAScore ↑'
}

COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']

def main():
    results = {metric: [] for metric in METRICS}

    for mode, directory in MODES.items():
        path = Path(f'outputs/evaluation/{directory}/metrics.json')

        with open(path) as f:
            data = json.load(f)

        for metric in METRICS:
            results[metric].append(data[metric])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    labels = list(MODES)
    x = list(range(len(labels)))

    for ax, (metric, title) in zip(axes.flat, METRICS.items()):
        values = results[metric]

        bars = ax.bar(
            x,
            values,
            color=COLORS,
            width=0.4,
            zorder=3
        )

        ax.set_title(title, pad=12)
        ax.set_ylabel(metric)
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylim(0, max(values) * 1.15)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

        for bar in bars:
            height = bar.get_height()

            ax.annotate(
                f'{height:.4f}',
                xy=(
                    bar.get_x() + bar.get_width() / 2,
                    height
                ),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=9
            )

    fig.suptitle(
        'Stable Audio Open 1.0 — APG Prompt Engineering Comparison',
        fontsize=15
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output = Path('outputs/evaluation/apg_prompt_comparison.png')
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('Saved:', output)

if __name__ == '__main__':
    main()
