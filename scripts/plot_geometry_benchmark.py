"""Plot named-reference Way1 estimates without joining missing frames."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peleng.evaluate import _read_truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    truth, _, _ = _read_truth(args.truth)
    count, fps = 1001, 30.
    times = np.arange(count)/fps
    target = np.full((count, 3), np.nan)
    for frame, point in truth.items():
        if 0 <= frame < count:
            target[frame] = np.asarray(point)/100
    variants = [('way1-baseline', 'Исходная версия', '#a35f18'),
                ('way1-generic', 'Исправления', '#2563b5'),
                ('way1-known', 'Исправления + габариты', '#07815b')]
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    for axis in range(3):
        axes[axis].plot(times, target[:, axis], color='#333b46', lw=2, label='Эталон организаторов')
        axes[axis].set_ylabel('XYZ'[axis]+', м')
    for name, label, color in variants:
        folder = sorted((args.root/name).glob('*/observations.jsonl'))[-1]
        rows = [json.loads(s) for s in folder.read_text(encoding='utf-8').splitlines()]
        if [r['source_frame'] for r in rows] != list(range(count)):
            raise ValueError(f'{name}: incomplete run')
        values = np.full((count, 3), np.nan)
        for row in rows:
            world = row['world']
            if world['position_cm'] is not None and world['reference_point']=='front_face_center':
                values[row['source_frame']] = np.asarray(world['position_cm'])/100
        for axis in range(3):
            axes[axis].plot(times, values[:, axis], color=color, lw=1.2, alpha=.85, label=label)
        axes[3].plot(times, np.linalg.norm(values-target, axis=1), color=color, lw=1.2)
    axes[3].set_ylabel('Ошибка 3D, м')
    axes[3].set_xlabel('Время, с')
    for ax in axes:
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .96), ncol=4, frameon=False)
    fig.suptitle('Way1: полные прогоны с одинаковым детектором', fontsize=16, y=.995)
    fig.text(.5, .012, 'Точка: центр передней грани — визуально обоснованное приближение pivot. '
             'Разрывы: нет оценки этой точки; точный pivot не подтверждён.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .925))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
