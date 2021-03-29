from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from collections import defaultdict
import argparse
plt.style.use('ggplot')

colors = ["g", "m", 'y', 'k', 'chartreuse', 'coral', 'gold', 'lavender',
               'silver', 'tan', 'teal', 'wheat', 'orchid', 'orange', 'tomato']

styles = ['--', '-.', ':', '-']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--folder', type=str, help='Folder to search')
    parser.add_argument('--fontsize', type=int, default=11)
    parser.add_argument('--figsize', type=int, nargs="+", default=[10, 10])
    parser.add_argument('--ablation_plot', action='store_true')

    args = parser.parse_args()
    return args


def make_training_plot(args: argparse.Namespace,
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


def nested_dd():
    return defaultdict(nested_dd)


def make_ablation_plot(args: argparse.Namespace):
    p = Path(args.folder)
    all_files = p.glob(f'**/*.txt')

    sota = {'pascal': {1: 0.608, 5: 0.620},
            'coco': {1: 0.358, 5: 0.390}}
    res_dic = nested_dd()
    for file in all_files:
        shot = eval([part.split('=')[1] for part in file.parts if 'shot' in part][0])
        data = [part.split('=')[1] for part in file.parts if 'data' in part][0]
        split = eval([part.split('=')[1] for part in file.parts if 'split' in part][0])
        tpi = eval(file.stem.split('_')[1])

        res = process_logfile(file)
        res_dic[data][shot][tpi][split] = np.mean(res)

    plt.rc('font', size=30)
    plt.rc('font')
    fig, axes = plt.subplots(1, 2, figsize=args.figsize)
    ax = fig.gca()
    for i, data in enumerate(['pascal', 'coco']):
        ax = axes[i]
        for style, color, shot in zip(cycle(styles), cycle(colors), res_dic[data]):
            tpis = np.array(list(res_dic[data][shot].keys()))
            mean = np.array([np.mean([res_dic[data][shot][tpi][split] for split in res_dic[data][shot][tpi]]) \
                             for tpi in res_dic[data][shot]])

            # x = np.linspace(0, n_epochs - 1, (n_epochs * iter_per_epoch))
            # y = np.reshape(array, (n_epochs * iter_per_epoch))
            sort_index = np.argsort(tpis)
            ax.plot(tpis[sort_index], mean[sort_index],
                    label=f'RePRI ({shot} shot)', color=color, linestyle=style, linewidth=3)
            ax.plot(tpis[sort_index], sota[data][shot] * np.ones(len(tpis)),
                    label=f'SOTA ({shot} shot)', linewidth=3.5, color=color, linestyle=':')
            if data == 'pascal':
                ax.set_ylabel('Average mIoU (over 4 folds)', size=30)
            ax.set_xlabel('$t_\pi$', size=40)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.78, 1.17), ncol=2, shadow=True)
    fig.subplots_adjust(wspace=0.13)
    fig.tight_layout()
    fig.savefig(p.joinpath(f"ablation.pdf"), bbox_inches='tight')


def process_logfile(path: str):
    with open(path, 'r') as f:
        res_lines = [line for line in f.readlines() if line.split('-')[0] == 'mIoU']
        res_lines = [eval(line.split(' ')[-1][:-2]) for line in res_lines]
        res = np.array(res_lines)
    return res


if __name__ == "__main__":
    args = parse_args()
    if args.ablation_plot:
        make_ablation_plot(args)
    else:
        for filename in ['val_mIou.npy', 'val_loss.npy', 'train_mIou.npy', 'train_loss.npy']:
            make_training_plot(args=args, filename=filename)
