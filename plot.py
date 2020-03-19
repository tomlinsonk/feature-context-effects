import glob
import pickle
import matplotlib.pyplot as plt

from experiments import load_wikispeedia, test_lstm_wikispeedia, test_wikispeedia


def plot_loss(fname, axes, row, col):
    with open(fname, 'rb') as f:
        losses = pickle.load(f)

    axes[row, col].plot(range(500), losses)

    if col == 0:
        axes[row, col].set_ylabel('Training losss')

    if row == 2:
        axes[row, col].set_xlabel('Epoch')


def plot_all_lstm_losses():
    loaded_data = load_wikispeedia()

    for dim in [16, 64, 128]:
        plot_all_losses(loaded_data, dim, test_lstm_wikispeedia, f'results/wikispeedia_lstm_losses_{dim}*.pickle',
                        f'lstm_wikispeedia_{dim}.pdf')


def plot_all_history_cdm_losses():
    loaded_data = load_wikispeedia()

    for dim in [16]:
        plot_all_losses(loaded_data, dim, test_wikispeedia, f'results/wikispeedia_losses_{dim}*.pickle',
                        f'history_cdm_wikispeedia_{dim}.pdf')


def plot_all_losses(loaded_data, dim, test_method, loss_file_glob, outfile):
    lrs = ['0.001', '0.005', '0.01']
    wds = ['0', '1e-06', '0.0001']

    fig, axes = plt.subplots(3, 3, sharex='col')

    for fname in glob.glob(loss_file_glob):
        fname_split = fname.split('_')
        lr = fname_split[3]
        wd = fname_split[4].replace('.pickle', '')

        row = lrs.index(lr)
        col = wds.index(wd)

        print(lr, wd, row, col)

        print(fname)
        param_fname = fname.replace('.pickle', '.pt').replace('losses', 'params').replace('results/', 'params/')
        acc, mean_rank, mrr = test_method(param_fname, dim, loaded_data)
        plot_loss(fname, axes, row, col)

        if row == 0:
            axes[row, col].annotate(f'WD: {wd}', xy=(0.5, 1), xytext=(0, 5),
                                    xycoords='axes fraction', textcoords='offset points',
                                    fontsize=14, ha='center', va='baseline')

        if col == 2:
            axes[row, col].annotate(f'LR: {lr}', xy=(1, 0.5), xytext=(-axes[row, col].yaxis.labelpad + 20, 0),
                                    xycoords='axes fraction', textcoords='offset points',
                                    fontsize=14, ha='right', va='center', rotation=270)

        axes[row, col].annotate(f'Val. acc: {acc:.2f}',
                                xy=(0.9, 0.8), xycoords='axes fraction', fontsize=10,
                                ha='right')

    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot_all_history_cdm_losses()
