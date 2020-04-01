import glob
import pickle

import numpy as np
from scipy import stats
import torch

import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE


from experiments import load_wikispeedia, test_lstm_wikispeedia, test_wikispeedia
from models import HistoryCDM, HistoryMNL


def plot_loss(fname, axes, row, col):
    with open(fname, 'rb') as f:
        losses = pickle.load(f)

    ax = axes[col] if row is None else axes[row, col]

    ax.plot(range(500), losses)

    if col == 0:
        ax.set_ylabel('Training losss')

    if row == 2:
        ax.set_xlabel('Epoch')


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


def plot_beta_losses(outfile):
    loaded_data = load_wikispeedia()
    betas = ['0', '0.5', '1']
    dims = ['16', '64', '128']

    glob_template = '{}/wikispeedia_mnl_beta_{}_{}_{}_0.005_0.{}'

    fig, axes = plt.subplots(3, 3, sharex='col')

    for row, beta in enumerate(betas):
        for col, dim in enumerate(dims):
            print(row, col)
            acc, mean_rank, mrr = test_wikispeedia(glob_template.format('params', beta, 'params', dim, 'pt'), int(dim), loaded_data, Model=HistoryMNL)
            plot_loss(glob_template.format('results', beta, 'losses', dim, 'pickle'), axes, row, col)

            if row == 0:
                axes[row, col].annotate(f'Dim: {dim}', xy=(0.5, 1), xytext=(0, 5),
                                        xycoords='axes fraction', textcoords='offset points',
                                        fontsize=14, ha='center', va='baseline')
            if col == 2:
                axes[row, col].annotate(f'$\\beta={beta}$', xy=(1, 0.5), xytext=(-axes[row, col].yaxis.labelpad + 20, 0),
                                        xycoords='axes fraction', textcoords='offset points',
                                        fontsize=14, ha='right', va='center', rotation=270)

            axes[row, col].annotate(f'Val. acc: {acc:.2f}',
                                    xy=(0.9, 0.8), xycoords='axes fraction', fontsize=10,
                                    ha='right')

    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight')
    plt.show()


def plot_learn_beta_losses(outfile):
    loaded_data = load_wikispeedia()
    dims = ['16', '64', '128']

    glob_template = '{}/wikispeedia_learn_beta_{}_{}_0.005_0.{}'

    fig, axes = plt.subplots(1, 3, sharex='col', figsize=(6, 2.5))

    for col, dim in enumerate(dims):
        param_fname = glob_template.format('params', 'params', dim, 'pt')
        acc, mean_rank, mrr = test_wikispeedia(param_fname, int(dim), loaded_data)
        plot_loss(glob_template.format('results', 'losses', dim, 'pickle'), axes, None, col)

        axes[col].annotate(f'Dim: {dim}', xy=(0.5, 1), xytext=(0, 5),
                                xycoords='axes fraction', textcoords='offset points',
                                fontsize=14, ha='center', va='baseline')

        model = HistoryCDM(len(loaded_data[0].nodes), int(dim), 0.5)
        model.load_state_dict(torch.load(param_fname))

        axes[col].annotate(f'Val. acc: {acc:.2f}\n$\\beta: {model.beta.item():.2f}$',
                           xy=(0.9, 0.8), xycoords='axes fraction', fontsize=10,
                           ha='right')



    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight')
    plt.show()


def load_normalized_embeddings(param_fname, n, dim):
    model = HistoryCDM(n, dim, 0.5)
    model.load_state_dict(torch.load(param_fname), strict=False)
    model.eval()

    history_embedding = model.history_embedding(range(n)).detach().numpy()
    history_embedding = history_embedding / np.linalg.norm(history_embedding, ord=2, axis=1, keepdims=True)

    target_embedding = model.target_embedding(range(n)).detach().numpy()
    target_embedding = target_embedding / np.linalg.norm(target_embedding, ord=2, axis=1, keepdims=True)

    context_embedding = model.context_embedding(range(n)).detach().numpy()
    context_embedding = context_embedding / np.linalg.norm(context_embedding, ord=2, axis=1, keepdims=True)

    return history_embedding, target_embedding, context_embedding


def analyze_embeddings(param_fname, dim):
    graph, train_data, val_data, test_data = load_wikispeedia()
    n = len(graph.nodes)

    model = HistoryCDM(n, dim, 0.5)
    model.load_state_dict(torch.load(param_fname), strict=False)
    model.eval()

    index_map = ['' for _ in range(n)]
    for node in graph.nodes:
        index_map[graph.nodes[node]['index']] = node

    embeddings = load_normalized_embeddings(param_fname, n, dim)
    names = ['history', 'target', 'context']

    histories, history_lengths, choice_sets, choice_set_lengths, choices = train_data
    choice_indices = choice_sets[torch.arange(choice_sets.size(0)), choices]

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
                with open(f'{result_fname}', 'w') as f:
                    f.write(results)


def analyze_history_effects(param_fname, dim):
    graph, train_data, val_data, test_data = load_wikispeedia()

    data = [torch.cat([train_data[i], val_data[i], test_data[i]]).numpy() for i in range(len(train_data))]
    histories, history_lengths, choice_sets, choice_set_lengths, choices = data
    choice_indices = choice_sets[np.arange(len(choice_sets)), choices]

    n = len(graph.nodes)

    model = HistoryCDM(n, dim, 0.5)
    model.load_state_dict(torch.load(param_fname), strict=False)
    model.eval()

    index_map = ['' for _ in range(n)]
    for node in graph.nodes:
        index_map[graph.nodes[node]['index']] = node

    history_embedding, target_embedding, context_embedding = load_normalized_embeddings(param_fname, n, dim)
    names = ['history', 'target', 'context']

    choice_counts = np.bincount(choice_indices)
    in_choice_set_counts = np.bincount(choice_sets.flatten())
    hit_rates = np.zeros_like(choice_counts, dtype=float)

    chosen_pages = []

    for page in range(n):
        in_choice_set = in_choice_set_counts[page]
        chosen = choice_counts[page]
        # row_in_history = (histories == row).sum(1) > 0
        #
        # conditional_col_in_choice_set = (choice_sets[row_in_history] == col).sum()
        # conditional_col_chosen = (choice_indices[row_in_history] == col).sum()
        #
        # # print('In choice set:', col_in_choice_set)
        # # print('Chosen:', col_chosen)
        # print(index_map[page], chosen / in_choice_set if in_choice_set > 0 else 0)

        if chosen > 0:
            hit_rates[page] = chosen / in_choice_set
            chosen_pages.append(page)

    chosen_pages.sort(key=lambda x: hit_rates[x], reverse=True)
    plt.scatter(range(len(chosen_pages)), hit_rates[chosen_pages], s=10)
    plt.yscale('log')
    plt.ylim(5e-5, 1)
    plt.xlabel('Rank')
    plt.ylabel('Click rate')
    plt.savefig('plots/click_rate_dsn.pdf', bbox_inches='tight')

    plt.show()

    chosen_pages = [page for page in chosen_pages if choice_counts[page] >= 100]

    print(len(chosen_pages), n)
    chosen_pages.sort(key=lambda x: hit_rates[x], reverse=True)

    inner_prods = history_embedding @ target_embedding.T

    rate_diffs = []
    hist_scores = []
    for page in tqdm(chosen_pages[:10]):
        for effect_page in chosen_pages:
            if page == effect_page: continue

            in_history = (histories == effect_page).sum(1) > 0
            conditional_in_choice_set = np.count_nonzero(choice_sets[in_history] == page)
            conditional_chosen = np.count_nonzero(choice_indices[in_history] == page)

            if conditional_in_choice_set >= 10:
                rate_diffs.append((conditional_chosen / conditional_in_choice_set) - hit_rates[page])
                hist_scores.append(inner_prods[effect_page, page])
                # print('Cond. rate', conditional_chosen / conditional_in_choice_set, inner_prods[effect_page, page])

    plt.scatter(rate_diffs, hist_scores, s=10, c='#13085c')
    plt.xlabel('Conditional click rate boost')
    plt.ylabel('$\\langle$history, target$\\rangle$')

    print('Correlation:', stats.pearsonr(rate_diffs, hist_scores))

    slope, intercept, r_value, p_value, std_err = stats.linregress(rate_diffs, hist_scores)
    plt.plot(np.linspace(-0.4, 0.7), slope * np.linspace(-0.4, 0.7) + intercept, c='#ffa600')
    print(r_value, p_value, std_err)

    plt.savefig('plots/history_score_correlation.pdf', bbox_inches='tight')

    plt.show()


    # chosen_pages.sort(key=lambda x: in_choice_set_counts[x], reverse=True)
    # for i in range(10):
    #     print(index_map[chosen_pages[i]], in_choice_set_counts[chosen_pages[i]])
    # plt.scatter(range(len(chosen_pages)), in_choice_set_counts[chosen_pages], s=10)
    # plt.yscale('log')
    # plt.xlabel('Rank')
    # plt.ylabel('# times in choice set')
    #
    # plt.show()
    #
    # chosen_pages.sort(key=lambda x: choice_counts[x], reverse=True)
    # for i in range(10):
    #     print(index_map[chosen_pages[i]], choice_counts[chosen_pages[i]])
    # plt.scatter(range(len(chosen_pages)), choice_counts[chosen_pages], s=10)
    # plt.yscale('log')
    # plt.xlabel('Rank')
    # plt.ylabel('# times chosen')
    #
    # plt.show()


def analyze_context_effects(param_fname, dim):
    graph, train_data, val_data, test_data = load_wikispeedia()

    data = [torch.cat([train_data[i], val_data[i], test_data[i]]).numpy() for i in range(len(train_data))]
    histories, history_lengths, choice_sets, choice_set_lengths, choices = data
    choice_indices = choice_sets[np.arange(len(choice_sets)), choices]

    n = len(graph.nodes)

    model = HistoryCDM(n, dim, 0.5)
    model.load_state_dict(torch.load(param_fname), strict=False)
    model.eval()

    index_map = ['' for _ in range(n)]
    for node in graph.nodes:
        index_map[graph.nodes[node]['index']] = node

    history_embedding, target_embedding, context_embedding = load_normalized_embeddings(param_fname, n, dim)

    choice_counts = np.bincount(choice_indices)
    in_choice_set_counts = np.bincount(choice_sets.flatten())
    hit_rates = np.zeros_like(choice_counts, dtype=float)
    for page in range(n):
        in_choice_set = in_choice_set_counts[page]
        chosen = choice_counts[page]

        if chosen > 0:
            hit_rates[page] = chosen / in_choice_set

    top_20 = np.argsort(choice_counts)[-20:][::-1]
    dot_prods = context_embedding[top_20] @ target_embedding[top_20].T
    np.fill_diagonal(dot_prods, 0)

    page_names = [index_map[idx].replace('_', ' ') for idx in top_20]
    sns.heatmap(dot_prods, center=0, xticklabels=page_names, yticklabels=page_names, linewidths=.5, cmap='RdBu_r')
    plt.xlabel('Target Embeddings')
    plt.ylabel('Context Embeddings')
    plt.savefig('top_20_context_effects.pdf', bbox_inches='tight')
    plt.show()

    hit_rate_diffs = np.zeros_like(dot_prods, dtype=float)

    for i, context_page in tqdm(enumerate(top_20), total=20):
        for j, target_page in enumerate(top_20):
            if context_page == target_page: continue

            in_set = (choice_sets == context_page).sum(1) > 0
            conditional_in_choice_set = np.count_nonzero(choice_sets[in_set] == target_page)
            conditional_chosen = np.count_nonzero(choice_indices[in_set] == target_page)

            conditional_hit_rate = 0 if conditional_in_choice_set == 0 else conditional_chosen / conditional_in_choice_set

            hit_rate_diffs[i, j] = conditional_hit_rate - hit_rates[target_page]

    sns.heatmap(hit_rate_diffs, center=0, xticklabels=page_names, yticklabels=page_names, linewidths=.5, cmap='RdBu_r')
    plt.xlabel('Target Page')
    plt.ylabel('Context Page')
    plt.savefig('top_20_context_hit_rates.pdf', bbox_inches='tight')
    plt.show()

    flat_dot_prods = dot_prods.flatten()
    flat_hit_rate_diffs = hit_rate_diffs.flatten()

    print('Correlation:', stats.pearsonr(flat_dot_prods, flat_hit_rate_diffs))

    slope, intercept, r_value, p_value, std_err = stats.linregress(flat_dot_prods, flat_hit_rate_diffs)
    plt.plot(np.linspace(-0.4, 0.7), slope * np.linspace(-0.4, 0.7) + intercept, c='#ffa600')
    print(r_value, p_value, std_err)

    plt.scatter(flat_dot_prods, flat_hit_rate_diffs)
    plt.xlabel('Context effect score')
    plt.ylabel('Conditional hit rate boost')
    plt.savefig('context_score_correlation.pdf', bbox_inches='tight')
    plt.show()


def all_tsne(param_fname, dim):
    graph, train_data, val_data, test_data = load_wikispeedia()
    n = len(graph.nodes)

    for embedding in load_normalized_embeddings(param_fname, n, dim):
        tsne_embedding(embedding)


def tsne_embedding(embedding):
    print('Running TSNE...')
    tsne = TSNE(n_components=2, random_state=1).fit_transform(embedding)
    print('Done.')
    plt.scatter(tsne[:, 0], tsne[:, 1])
    plt.show()


if __name__ == '__main__':
    # plot_all_history_cdm_losses()
    # analyze_context_effects('params/wikispeedia_params_128_0.005_0.pt', 128)
    # all_tsne('params/wikispeedia_params_128_0.005_0.pt', 128)
    plot_beta_losses('plots/wikispeedia_mnl_vary_beta.pdf')
    # analyze_embeddings('params/wikispeedia_beta_1_params_128_0.005_0.pt', 128)
    # plot_learn_beta_losses('plots/wikispeedia_learn_beta.pdf')

