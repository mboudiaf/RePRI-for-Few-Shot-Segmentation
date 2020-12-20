from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle
import argparse
matplotlib.use("Agg")

colors = ["c", "r", "g", "b", "m", 'y', 'k', 'chartreuse', 'coral', 'gold', 'lavender',
               'silver', 'tan', 'teal', 'wheat', 'orchid', 'orange', 'tomato']

styles = ['--', '-.', ':', '-']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--folder', type=str, help='Folder to search')
    parser.add_argument('--fontsize', type=int, default=11)
    parser.add_argument('--figsize', type=list, default=[10, 10])

    args = parser.parse_args()
    return args


def make_plot(args: argparse.Namespace,
              filename: str) -> None:
    plt.rc('font', size=args.fontsize)

    fig = plt.Figure(args.figsize)
    ax = fig.gca()

    p = Path(args.folder)
    all_files = p.glob(f'**/{filename}')
    for style, color, filepath in zip(cycle(styles), cycle(colors), all_files):
        array = np.load(filepath)
        n_epochs, iter_per_epoch = array.shape

        x = np.linspace(0, n_epochs - 1, (n_epochs * iter_per_epoch))
        y = np.reshape(array, (n_epochs * iter_per_epoch))

        label = filepath

        ax.plot(x, y, label=label, color=color, linestyle=style)

    ax.legend()
    ax.set_xlabel("Epochs")
    ax.set_title(filename.split('.')[0])
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(p.joinpath('{}.png'.format(filename.split('.')[0])))


if __name__ == "__main__":
    args = parse_args()
    for filename in ['val_mIou.npy', 'val_loss.npy', 'train_mIou.npy', 'train_loss.npy']:
        make_plot(args=args, filename=filename)
