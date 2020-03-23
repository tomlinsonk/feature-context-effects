import glob
import pickle

import numpy as np
import torch

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE


from experiments import load_wikispeedia, test_lstm_wikispeedia, test_wikispeedia
from models import HistoryCDM


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

    for dim in [16, 64, 128]:
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

        try:
            row = lrs.index(lr)
            col = wds.index(wd)
        except ValueError:
            print('Skipping', fname)
            continue

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


def load_normalized_embeddings(param_fname, n):
    model = HistoryCDM(n, 16, 0.5)
    model.load_state_dict(torch.load(param_fname))
    model.eval()

    history_embedding = model.history_embedding(range(n)).detach().numpy()
    history_embedding = history_embedding / np.linalg.norm(history_embedding, ord=2, axis=1, keepdims=True)

    target_embedding = model.target_embedding(range(n)).detach().numpy()
    target_embedding = target_embedding / np.linalg.norm(target_embedding, ord=2, axis=1, keepdims=True)

    context_embedding = model.context_embedding(range(n)).detach().numpy()
    context_embedding = context_embedding / np.linalg.norm(context_embedding, ord=2, axis=1, keepdims=True)

    return history_embedding, target_embedding, context_embedding


def analyze_embeddings(param_fname):
    graph, train_data, val_data, test_data = load_wikispeedia()
    n = len(graph.nodes)

    model = HistoryCDM(n, 16, 0.5)
    model.load_state_dict(torch.load(param_fname))
    model.eval()

    index_map = ['' for _ in range(n)]
    for node in graph.nodes:
        index_map[graph.nodes[node]['index']] = node

    embeddings = load_normalized_embeddings(param_fname, n)
    names = ['history', 'target', 'context']

    for i in range(3):
        for j in range(i, 3):
            for extreme, dir in ((np.inf, 'smallest'), (-np.inf, 'largest')):
                products = embeddings[i] @ embeddings[j].T

                tri_idx = np.tril_indices(n, -1)
                products[tri_idx] = extreme
                np.fill_diagonal(products, extreme)

                flattened_indices = products.argsort(axis=None)
                if dir == 'largest':
                    flattened_indices = flattened_indices[::-1]
                array_indices = np.unravel_index(flattened_indices, products.shape)

                results = ''

                row_idx, col_idx = array_indices

                for k in range(100):
                    row = row_idx[k]
                    col = col_idx[k]
                    results += f'{products[row, col]}, {index_map[row]}, {index_map[col]}\n'

                result_fname = os.path.basename(param_fname).replace('.pt', '.txt').replace('params', f'embeds_{names[i]}_{names[j]}_{dir}')
                with open(f'results/embedding_stats/{result_fname}', 'w') as f:
                    f.write(results)


def all_tsne(param_fname):
    graph, train_data, val_data, test_data = load_wikispeedia()
    n = len(graph.nodes)

    for embedding in load_normalized_embeddings(param_fname, n):
        tsne_embedding(embedding)


def tsne_embedding(embedding):
    print('Running TSNE...')
    tsne = TSNE(n_components=2, random_state=1).fit_transform(embedding)
    print('Done.')
    plt.scatter(tsne[:, 0], tsne[:, 1])
    plt.show()


if __name__ == '__main__':
    plot_all_history_cdm_losses()
    # analyze_embeddings('params/wikispeedia_params_16_0.005_0.pt')
    # all_tsne('params/wikispeedia_params_16_0.005_0.pt')

