import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.stats as stats
import matplotlib.ticker as ticker


from datasets import WikispeediaDataset, KosarakDataset, YoochooseDataset, LastFMGenreDataset, ORCIDSwitchDataset, \
    EmailEnronDataset, CollegeMsgDataset, EmailEUDataset, MathOverflowDataset, FacebookWallDataset
from models import HistoryCDM, HistoryMNL, DataLoader, LSTM, FeatureMNL, FeatureCDM


def load_model(Model, n, dim, param_fname):
    if Model is LSTM:
        model = Model(n, dim)
    else:
        model = Model(n, dim, 0.5)

    model.load_state_dict(torch.load(param_fname))
    model.eval()

    return model


def load_feature_model(Model, num_features, param_fname):
    model = Model(num_features)

    model.load_state_dict(torch.load(param_fname))
    model.eval()

    return model


def test_model(model, dataset, loaded_data=None):
    if loaded_data is None:
        graph, train_data, val_data, test_data = dataset.load()
    else:
        graph, train_data, val_data, test_data = loaded_data

    n = len(graph.nodes)
    batch_size = 128

    if model.name == 'lstm':
        data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, sort_batch=True, sort_index=1)
    else:
        data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    count = 0
    correct = 0
    mean_rank = 0
    mrr = 0
    for histories, history_lengths, choice_sets, choice_set_lengths, choices in data_loader:
        choice_pred = model(histories, history_lengths, choice_sets, choice_set_lengths)

        ranks = (torch.argsort(choice_pred, dim=1, descending=True) == choices[:, None]).nonzero()[:, 1] + 1

        vals, idxs = choice_pred.max(1)
        mean_rank += ranks.sum().item() / batch_size
        mrr += (1 / ranks.float()).sum().item() / batch_size
        count += 1
        correct += (idxs == choices).long().sum().item() / batch_size

    return correct / count, mean_rank / count, mrr / count


def plot_loss(fname, axes, row, col):
    with open(fname, 'rb') as f:
        losses = pickle.load(f)

    ax = axes[col] if row is None else axes[row, col]

    ax.plot(range(500), losses)

    if col == 0:
        ax.set_ylabel('Training losss')

    if row == 2:
        ax.set_xlabel('Epoch')


def plot_compare_all():

    fig, axes = plt.subplots(3, 3, sharex='col')

    for row, dataset in enumerate((KosarakDataset, YoochooseDataset, WikispeediaDataset)):
        loaded_data = dataset.load()
        n = len(loaded_data[0].nodes)
        for col, method in enumerate((LSTM, HistoryMNL, HistoryCDM)):
            beta_string = '' if method is LSTM else '_0.5_True'
            param_fname = f'params/{method.name}_{dataset.name}_params_64_0.005_0{beta_string}.pt'
            loss_fname = f'results/{method.name}_{dataset.name}_losses_64_0.005_0{beta_string}.pickle'
            print(method, param_fname)

            if row == 0:
                axes[row, col].annotate(f'{method.name}', xy=(0.5, 1), xytext=(0, 5),
                                        xycoords='axes fraction', textcoords='offset points',
                                        fontsize=14, ha='center', va='baseline')

            if col == 2:
                axes[row, col].annotate(f'{dataset.name}', xy=(1, 0.5), xytext=(-axes[row, col].yaxis.labelpad + 20, 0),
                                        xycoords='axes fraction', textcoords='offset points',
                                        fontsize=14, ha='right', va='center', rotation=270)

            if not os.path.isfile(param_fname):
                continue

            model = load_model(method, n, 64, param_fname)

            acc, mean_rank, mrr = test_model(model, dataset, loaded_data=loaded_data)
            print(f'Accuracy: {acc}')

            with open(loss_fname, 'rb') as f:
                losses = pickle.load(f)

            axes[row, col].plot(range(500), losses)

            beta_string = f'$\\beta={model.beta.item():.2f}$' if method in (HistoryMNL, HistoryCDM) else ''

            axes[row, col].annotate(f'Val. acc: {acc:.2f}\n{beta_string}',
                                    xy=(0.9, 0.72), xycoords='axes fraction', fontsize=10,
                                    ha='right')

    plt.show()


def plot_grid_search(method, dataset):
    lrs = [0.005]
    wds = [0, 1e-5, 1e-4]

    fig, axes = plt.subplots(3, 3, sharex='col')

    loaded_data = dataset.load()
    n = len(loaded_data[0].nodes)

    for row, wd in enumerate(wds):
        for col, lr in enumerate(lrs):
            beta_string = '_None_None' if method is LSTM else '_0.5_True'
            param_fname = f'params/{method.name}_{dataset.name}_params_8_{lr}_{wd}{beta_string}.pt'
            loss_fname = f'results/{method.name}_{dataset.name}_losses_8_{lr}_{wd}{beta_string}.pickle'

            if row == 0:
                axes[row, col].annotate(f'lr={lr}', xy=(0.5, 1), xytext=(0, 5),
                                        xycoords='axes fraction', textcoords='offset points',
                                        fontsize=14, ha='center', va='baseline')

            if col == 2:
                axes[row, col].annotate(f'wd={wd}', xy=(1, 0.5), xytext=(-axes[row, col].yaxis.labelpad + 20, 0),
                                        xycoords='axes fraction', textcoords='offset points',
                                        fontsize=14, ha='right', va='center', rotation=270)

            if not os.path.isfile(param_fname):
                continue

            model = load_model(method, n, 8, param_fname)

            acc, mean_rank, mrr = test_model(model, dataset, loaded_data=loaded_data)
            print(f'Accuracy: {acc}')

            with open(loss_fname, 'rb') as f:
                train_losses, train_accs, val_losses, val_accs = pickle.load(f)

            axes[row, col].plot(range(500), train_accs, label='train')
            axes[row, col].plot(range(500), val_accs, label='val')

            beta_string = f'$\\beta={model.beta.item():.2f}$' if method in (HistoryMNL, HistoryCDM) else ''

            axes[row, col].annotate(f'Val. acc: {acc:.2f}\n{beta_string}',
                                    xy=(0.9, 0.72), xycoords='axes fraction', fontsize=10,
                                    ha='right')

            axes[row, col].set_ylim(0, 1)

    axes[0, 0].legend(loc='best')
    plt.savefig(f'{method.name}_{dataset.name}_grid_search.pdf', bbox_inches='tight')

    plt.close()


def plot_dataset_stats():
    fig, axes = plt.subplots(4, 2, figsize=(6, 8))

    for row, dataset in enumerate((KosarakDataset, YoochooseDataset, WikispeediaDataset, LastFMGenreDataset)):
        graph, train_data, val_data, test_data = dataset.load()
        histories, history_lengths, choice_sets, choice_set_lengths, choices = [torch.cat([train_data[i], val_data[i], test_data[i]]).numpy() for i in range(len(train_data))]

        axes[row, 1].annotate(f'{dataset.name}', xy=(1, 0.5), xytext=(-axes[row, 1].yaxis.labelpad + 20, 0),
                                xycoords='axes fraction', textcoords='offset points',
                                fontsize=14, ha='right', va='center', rotation=270)

        # axes[row, col].annotate(f'{method.name}', xy=(0.5, 1), xytext=(0, 5),
        #                         xycoords='axes fraction', textcoords='offset points',
        #                         fontsize=14, ha='center', va='baseline')

        degree_counts = np.bincount([deg for node, deg in graph.out_degree()])
        axes[row, 0].scatter(range(len(degree_counts)), degree_counts, label='Node Outdegree', s=8, marker='d')

        choice_set_dsn = np.bincount(choice_set_lengths)
        axes[row, 0].scatter(range(len(choice_set_dsn)), choice_set_dsn, label='Choice Set Size', s=8, marker='s')
        axes[row, 0].set_ylabel('Count')

        history_dsn = np.bincount(history_lengths)
        axes[row, 1].scatter(range(len(history_dsn)), history_dsn, label='History Length', s=8, marker='s')

        for col in (0, 1):
            axes[row, col].set_yscale('log')
            axes[row, col].set_xscale('log')

            axes[row, col].set_xlim(0.5)
            axes[row, col].set_ylim(0.5)


        # for col, method in enumerate((LSTM, HistoryMNL, HistoryCDM)):
        #     beta_string = '' if method is LSTM else '_0.5_True'
        #     param_fname = f'params/{method.name}_{dataset.name}_params_64_0.005_0{beta_string}.pt'
        #     loss_fname = f'results/{method.name}_{dataset.name}_losses_64_0.005_0{beta_string}.pickle'
        #     print(method, param_fname)
        #
        #     if row == 0:
        #
        #
        #
        #
        #     if not os.path.isfile(param_fname):
        #         continue
        #
        #     model = load_model(method, n, 64, param_fname)
        #
        #     acc, mean_rank, mrr = test_model(model, dataset, loaded_data=loaded_data)
        #
        #     with open(loss_fname, 'rb') as f:
        #         losses = pickle.load(f)
        #
        #     axes[row, col].plot(range(500), losses)
        #
        #     axes[row, col].annotate(f'Val. acc: {acc:.2f}',
        #                             xy=(0.9, 0.8), xycoords='axes fraction', fontsize=10,
        #                             ha='right')

    axes[0, 0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.02))
    axes[0, 1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.02))
    plt.show()


def examine_email_enron():
    graph, train_data, val_data, test_data = FacebookWallDataset.load()
    histories, history_lengths, choice_sets, choice_set_features, choice_set_lengths, choices = [
        torch.cat([train_data[i], val_data[i], test_data[i]]).numpy() for i in range(len(train_data))]
    # histories, history_lengths, choice_sets, choice_set_features, choice_set_lengths, choices = [x.numpy() for x in train_data]



    in_degree_ratios = []
    out_degree_ratios = []
    reciprocity_ratios = []

    chosen_in_degrees = []
    chosen_out_degrees = []
    chosen_reciprocities = []

    mean_available_in_degrees = []
    mean_available_out_degrees = []
    mean_available_reciprocities = []

    for i in range(len(choice_set_features)):
        choice = choices[i]
        choice_set = choice_set_features[i, :choice_set_lengths[i]]

        mean_available_reciprocities.append(np.mean(choice_set[:, 2]))

        # Convert reciprocities to -1/1 and log-degrees to degrees
        choice_set[:, 0] = np.exp(choice_set[:, 0])
        choice_set[:, 1] = np.exp(choice_set[:, 1])

        in_degree_ratios.append(choice_set[choice, 0] / np.mean(choice_set[:, 0]))
        out_degree_ratios.append(choice_set[choice, 1] / np.mean(choice_set[:, 1]))
        reciprocity_ratios.append(torch.nn.functional.softmax(torch.tensor(choice_set[:, 2]), dim=0)[choice].item() * choice_set_lengths[i])

        chosen_in_degrees.append(choice_set[choice, 0])
        chosen_out_degrees.append(choice_set[choice, 1])
        chosen_reciprocities.append(choice_set[choice, 2])

        mean_available_in_degrees.append(np.mean(choice_set[:, 0]))
        mean_available_out_degrees.append(np.mean(choice_set[:, 1]))

    in_degree_ratios = np.array(in_degree_ratios)
    out_degree_ratios = np.array(out_degree_ratios)
    reciprocity_ratios = np.array(reciprocity_ratios)

    chosen_in_degrees = np.array(chosen_in_degrees)
    chosen_out_degrees = np.array(chosen_out_degrees)
    chosen_reciprocities = np.array(chosen_reciprocities)

    mean_available_in_degrees = np.array(mean_available_in_degrees)
    mean_available_out_degrees = np.array(mean_available_out_degrees)
    mean_available_reciprocities = np.array(mean_available_reciprocities)

    plt.set_cmap('plasma')

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    xlims = [(1, 100), (1, 100), (0.005, 1.2)]

    num_bins = 100

    for j, (x_variable, x_name) in enumerate([(mean_available_in_degrees, 'In-degree'), (mean_available_out_degrees, 'Out-degree'), (mean_available_reciprocities, 'Reciprocity')]):

        values, bins = np.histogram(x_variable, bins=np.logspace(np.log(xlims[j][0]), np.log(xlims[j][1]), num_bins))

        idx = np.digitize(x_variable, bins)

        for i, (y_variable, y_name) in enumerate([(in_degree_ratios, 'Chosen In-degree Ratio'), (out_degree_ratios, 'Chosen Out-degree Ratio'), (reciprocity_ratios, 'Chosen Reciprocity Ratio')]):
            mean_y_vars = np.zeros(num_bins)
            bin_counts = np.zeros(num_bins)

            for bin in range(num_bins):
                mean_y_vars[bin] = np.mean(y_variable[idx == bin])
                bin_counts[bin] = np.count_nonzero(idx == bin)

            log_lengths = np.log(choice_set_lengths)

            scatterplot = axes[i, j].scatter(x_variable, y_variable, s=30, alpha=0.5, marker='.', linewidth=0, c=log_lengths)

            axes[i, j].scatter(bins, mean_y_vars, alpha=1, s=bin_counts ** 0.8, marker='o', color='black')
            axes[i, j].scatter(bins, mean_y_vars, alpha=1, s=1, marker='.', color='white')

            if j == 0:
                axes[i, j].set_ylabel(f'{y_name}')
            else:
                plt.setp(axes[i, j].get_yticklabels(), visible=False)

            if i == 2:
                axes[i, j].set_xlabel(f'Choice Set {x_name}')
                axes[i, j].set_yscale('log')
            else:
                axes[i, j].set_yscale('log')

            axes[i, j].set_xlim(*xlims[j])
            axes[i, j].set_xscale('log')

    # axes[2, 0].set_ylim(0.3, 3)
    # axes[2, 0].set_yticks([0.3, 1, 3])
    # axes[2, 0].get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    # axes[2, 0].get_yaxis().set_minor_formatter(ticker.NullFormatter())

    cbar = fig.colorbar(scatterplot, ax=axes, location='top', shrink=0.3)
    cbar.set_label('log Choice Set Size')

    plt.savefig(f'facebook-wall-log_probs.png', bbox_inches='tight', dpi=200)
    plt.close()

    # model = load_feature_model(FeatureCDM, 3, 'feature_cdm_mathoverflow_params_0.01_0.pt')
    # print(model.contexts, model.weights)




# OLD PLOTTING STUFF
#
# def plot_all_history_cdm_losses(dataset):
#     loaded_data = dataset.load()
#
#     for dim in [16, 64, 128]:
#         plot_all_losses(loaded_data, dim, test_wikispeedia, f'results/wikispeedia_losses_{dim}*.pickle',
#                         f'history_cdm_wikispeedia_{dim}.pdf')
#
#
# def plot_all_losses(loaded_data, dim, test_method, loss_file_glob, outfile):
#     lrs = ['0.001', '0.005', '0.01']
#     wds = ['0', '1e-06', '0.0001']
#
#     fig, axes = plt.subplots(3, 3, sharex='col')
#
#     for fname in glob.glob(loss_file_glob):
#         fname_split = fname.split('_')
#         lr = fname_split[3]
#         wd = fname_split[4].replace('.pickle', '')
#
#         try:
#             row = lrs.index(lr)
#             col = wds.index(wd)
#         except ValueError:
#             print('Skipping', fname)
#             continue
#
#         print(lr, wd, row, col)
#
#         print(fname)
#         param_fname = fname.replace('.pickle', '.pt').replace('losses', 'params').replace('results/', 'params/')
#         acc, mean_rank, mrr = test_method(param_fname, dim, loaded_data)
#         plot_loss(fname, axes, row, col)
#
#         if row == 0:
#             axes[row, col].annotate(f'WD: {wd}', xy=(0.5, 1), xytext=(0, 5),
#                                     xycoords='axes fraction', textcoords='offset points',
#                                     fontsize=14, ha='center', va='baseline')
#
#         if col == 2:
#             axes[row, col].annotate(f'LR: {lr}', xy=(1, 0.5), xytext=(-axes[row, col].yaxis.labelpad + 20, 0),
#                                     xycoords='axes fraction', textcoords='offset points',
#                                     fontsize=14, ha='right', va='center', rotation=270)
#
#         axes[row, col].annotate(f'Val. acc: {acc:.2f}',
#                                 xy=(0.9, 0.8), xycoords='axes fraction', fontsize=10,
#                                 ha='right')
#
#     plt.tight_layout()
#     plt.savefig(outfile, bbox_inches='tight')
#     plt.show()
#
#
# def plot_beta_losses(outfile):
#     loaded_data = load_wikispeedia()
#     betas = ['0', '0.5', '1']
#     dims = ['16', '64', '128']
#
#     glob_template = '{}/wikispeedia_mnl_beta_{}_{}_{}_0.005_0.{}'
#
#     fig, axes = plt.subplots(3, 3, sharex='col')
#
#     for row, beta in enumerate(betas):
#         for col, dim in enumerate(dims):
#             print(row, col)
#             acc, mean_rank, mrr = test_wikispeedia(glob_template.format('params', beta, 'params', dim, 'pt'), int(dim), loaded_data, Model=HistoryMNL)
#             plot_loss(glob_template.format('results', beta, 'losses', dim, 'pickle'), axes, row, col)
#
#             if row == 0:
#                 axes[row, col].annotate(f'Dim: {dim}', xy=(0.5, 1), xytext=(0, 5),
#                                         xycoords='axes fraction', textcoords='offset points',
#                                         fontsize=14, ha='center', va='baseline')
#             if col == 2:
#                 axes[row, col].annotate(f'$\\beta={beta}$', xy=(1, 0.5), xytext=(-axes[row, col].yaxis.labelpad + 20, 0),
#                                         xycoords='axes fraction', textcoords='offset points',
#                                         fontsize=14, ha='right', va='center', rotation=270)
#
#             axes[row, col].annotate(f'Val. acc: {acc:.2f}',
#                                     xy=(0.9, 0.8), xycoords='axes fraction', fontsize=10,
#                                     ha='right')
#
#     plt.tight_layout()
#     plt.savefig(outfile, bbox_inches='tight')
#     plt.show()
#
#
# def plot_learn_beta_losses(outfile):
#     loaded_data = load_wikispeedia()
#     dims = ['16', '64', '128']
#
#     glob_template = '{}/wikispeedia_learn_beta_{}_{}_0.005_0.{}'
#
#     fig, axes = plt.subplots(1, 3, sharex='col', figsize=(6, 2.5))
#
#     for col, dim in enumerate(dims):
#         param_fname = glob_template.format('params', 'params', dim, 'pt')
#         acc, mean_rank, mrr = test_wikispeedia(param_fname, int(dim), loaded_data)
#         plot_loss(glob_template.format('results', 'losses', dim, 'pickle'), axes, None, col)
#
#         axes[col].annotate(f'Dim: {dim}', xy=(0.5, 1), xytext=(0, 5),
#                                 xycoords='axes fraction', textcoords='offset points',
#                                 fontsize=14, ha='center', va='baseline')
#
#         model = HistoryCDM(len(loaded_data[0].nodes), int(dim), 0.5)
#         model.load_state_dict(torch.load(param_fname))
#
#         axes[col].annotate(f'Val. acc: {acc:.2f}\n$\\beta: {model.beta.item():.2f}$',
#                            xy=(0.9, 0.8), xycoords='axes fraction', fontsize=10,
#                            ha='right')
#
#
#
#     plt.tight_layout()
#     plt.savefig(outfile, bbox_inches='tight')
#     plt.show()
#
#
# def load_normalized_embeddings(param_fname, n, dim):
#     model = HistoryCDM(n, dim, 0.5)
#     model.load_state_dict(torch.load(param_fname), strict=False)
#     model.eval()
#
#     history_embedding = model.history_embedding(range(n)).detach().numpy()
#     history_embedding = history_embedding / np.linalg.norm(history_embedding, ord=2, axis=1, keepdims=True)
#
#     target_embedding = model.target_embedding(range(n)).detach().numpy()
#     target_embedding = target_embedding / np.linalg.norm(target_embedding, ord=2, axis=1, keepdims=True)
#
#     context_embedding = model.context_embedding(range(n)).detach().numpy()
#     context_embedding = context_embedding / np.linalg.norm(context_embedding, ord=2, axis=1, keepdims=True)
#
#     return history_embedding, target_embedding, context_embedding
#
#
# def analyze_embeddings(param_fname, dim):
#     graph, train_data, val_data, test_data = load_wikispeedia()
#     n = len(graph.nodes)
#
#     model = HistoryCDM(n, dim, 0.5)
#     model.load_state_dict(torch.load(param_fname), strict=False)
#     model.eval()
#
#     index_map = ['' for _ in range(n)]
#     for node in graph.nodes:
#         index_map[graph.nodes[node]['index']] = node
#
#     embeddings = load_normalized_embeddings(param_fname, n, dim)
#     names = ['history', 'target', 'context']
#
#     histories, history_lengths, choice_sets, choice_set_lengths, choices = train_data
#     choice_indices = choice_sets[torch.arange(choice_sets.size(0)), choices]
#
#     for i in range(3):
#         for j in range(i, 3):
#             for extreme, dir in ((np.inf, 'smallest'), (-np.inf, 'largest')):
#                 products = embeddings[i] @ embeddings[j].T
#
#                 tri_idx = np.tril_indices(n, -1)
#                 products[tri_idx] = extreme
#                 np.fill_diagonal(products, extreme)
#
#                 flattened_indices = products.argsort(axis=None)
#                 if dir == 'largest':
#                     flattened_indices = flattened_indices[::-1]
#                 array_indices = np.unravel_index(flattened_indices, products.shape)
#
#                 results = ''
#
#                 row_idx, col_idx = array_indices
#
#                 for k in range(100):
#                     row = row_idx[k]
#                     col = col_idx[k]
#
#                     results += f'{products[row, col]}, {index_map[row]}, {index_map[col]}\n'
#
#                 result_fname = os.path.basename(param_fname).replace('.pt', '.txt').replace('params', f'embeds_{names[i]}_{names[j]}_{dir}')
#                 with open(f'{result_fname}', 'w') as f:
#                     f.write(results)
#
#
# def analyze_history_effects(param_fname, dim):
#     graph, train_data, val_data, test_data = load_wikispeedia()
#
#     data = [torch.cat([train_data[i], val_data[i], test_data[i]]).numpy() for i in range(len(train_data))]
#     histories, history_lengths, choice_sets, choice_set_lengths, choices = data
#     choice_indices = choice_sets[np.arange(len(choice_sets)), choices]
#
#     n = len(graph.nodes)
#
#     model = HistoryCDM(n, dim, 0.5)
#     model.load_state_dict(torch.load(param_fname), strict=False)
#     model.eval()
#
#     index_map = ['' for _ in range(n)]
#     for node in graph.nodes:
#         index_map[graph.nodes[node]['index']] = node
#
#     history_embedding, target_embedding, context_embedding = load_normalized_embeddings(param_fname, n, dim)
#     names = ['history', 'target', 'context']
#
#     choice_counts = np.bincount(choice_indices)
#     in_choice_set_counts = np.bincount(choice_sets.flatten())
#     hit_rates = np.zeros_like(choice_counts, dtype=float)
#
#     chosen_pages = []
#
#     for page in range(n):
#         in_choice_set = in_choice_set_counts[page]
#         chosen = choice_counts[page]
#         # row_in_history = (histories == row).sum(1) > 0
#         #
#         # conditional_col_in_choice_set = (choice_sets[row_in_history] == col).sum()
#         # conditional_col_chosen = (choice_indices[row_in_history] == col).sum()
#         #
#         # # print('In choice set:', col_in_choice_set)
#         # # print('Chosen:', col_chosen)
#         # print(index_map[page], chosen / in_choice_set if in_choice_set > 0 else 0)
#
#         if chosen > 0:
#             hit_rates[page] = chosen / in_choice_set
#             chosen_pages.append(page)
#
#     chosen_pages.sort(key=lambda x: hit_rates[x], reverse=True)
#     plt.scatter(range(len(chosen_pages)), hit_rates[chosen_pages], s=10)
#     plt.yscale('log')
#     plt.ylim(5e-5, 1)
#     plt.xlabel('Rank')
#     plt.ylabel('Click rate')
#     plt.savefig('plots/click_rate_dsn.pdf', bbox_inches='tight')
#
#     plt.show()
#
#     chosen_pages = [page for page in chosen_pages if choice_counts[page] >= 100]
#
#     print(len(chosen_pages), n)
#     chosen_pages.sort(key=lambda x: hit_rates[x], reverse=True)
#
#     inner_prods = history_embedding @ target_embedding.T
#
#     rate_diffs = []
#     hist_scores = []
#     for page in tqdm(chosen_pages[:10]):
#         for effect_page in chosen_pages:
#             if page == effect_page: continue
#
#             in_history = (histories == effect_page).sum(1) > 0
#             conditional_in_choice_set = np.count_nonzero(choice_sets[in_history] == page)
#             conditional_chosen = np.count_nonzero(choice_indices[in_history] == page)
#
#             if conditional_in_choice_set >= 10:
#                 rate_diffs.append((conditional_chosen / conditional_in_choice_set) - hit_rates[page])
#                 hist_scores.append(inner_prods[effect_page, page])
#                 # print('Cond. rate', conditional_chosen / conditional_in_choice_set, inner_prods[effect_page, page])
#
#     plt.scatter(rate_diffs, hist_scores, s=10, c='#13085c')
#     plt.xlabel('Conditional click rate boost')
#     plt.ylabel('$\\langle$history, target$\\rangle$')
#
#     print('Correlation:', stats.pearsonr(rate_diffs, hist_scores))
#
#     slope, intercept, r_value, p_value, std_err = stats.linregress(rate_diffs, hist_scores)
#     plt.plot(np.linspace(-0.4, 0.7), slope * np.linspace(-0.4, 0.7) + intercept, c='#ffa600')
#     print(r_value, p_value, std_err)
#
#     plt.savefig('plots/history_score_correlation.pdf', bbox_inches='tight')
#
#     plt.show()
#
#
#     # chosen_pages.sort(key=lambda x: in_choice_set_counts[x], reverse=True)
#     # for i in range(10):
#     #     print(index_map[chosen_pages[i]], in_choice_set_counts[chosen_pages[i]])
#     # plt.scatter(range(len(chosen_pages)), in_choice_set_counts[chosen_pages], s=10)
#     # plt.yscale('log')
#     # plt.xlabel('Rank')
#     # plt.ylabel('# times in choice set')
#     #
#     # plt.show()
#     #
#     # chosen_pages.sort(key=lambda x: choice_counts[x], reverse=True)
#     # for i in range(10):
#     #     print(index_map[chosen_pages[i]], choice_counts[chosen_pages[i]])
#     # plt.scatter(range(len(chosen_pages)), choice_counts[chosen_pages], s=10)
#     # plt.yscale('log')
#     # plt.xlabel('Rank')
#     # plt.ylabel('# times chosen')
#     #
#     # plt.show()
#
#
# def analyze_context_effects(param_fname, dim):
#     graph, train_data, val_data, test_data = load_wikispeedia()
#
#     data = [torch.cat([train_data[i], val_data[i], test_data[i]]).numpy() for i in range(len(train_data))]
#     histories, history_lengths, choice_sets, choice_set_lengths, choices = data
#     choice_indices = choice_sets[np.arange(len(choice_sets)), choices]
#
#     n = len(graph.nodes)
#
#     model = HistoryCDM(n, dim, 0.5)
#     model.load_state_dict(torch.load(param_fname), strict=False)
#     model.eval()
#
#     index_map = ['' for _ in range(n)]
#     for node in graph.nodes:
#         index_map[graph.nodes[node]['index']] = node
#
#     history_embedding, target_embedding, context_embedding = load_normalized_embeddings(param_fname, n, dim)
#
#     choice_counts = np.bincount(choice_indices)
#     in_choice_set_counts = np.bincount(choice_sets.flatten())
#     hit_rates = np.zeros_like(choice_counts, dtype=float)
#     for page in range(n):
#         in_choice_set = in_choice_set_counts[page]
#         chosen = choice_counts[page]
#
#         if chosen > 0:
#             hit_rates[page] = chosen / in_choice_set
#
#     top_20 = np.argsort(choice_counts)[-20:][::-1]
#     dot_prods = context_embedding[top_20] @ target_embedding[top_20].T
#     np.fill_diagonal(dot_prods, 0)
#
#     page_names = [index_map[idx].replace('_', ' ') for idx in top_20]
#     sns.heatmap(dot_prods, center=0, xticklabels=page_names, yticklabels=page_names, linewidths=.5, cmap='RdBu_r')
#     plt.xlabel('Target Embeddings')
#     plt.ylabel('Context Embeddings')
#     plt.savefig('top_20_context_effects.pdf', bbox_inches='tight')
#     plt.show()
#
#     hit_rate_diffs = np.zeros_like(dot_prods, dtype=float)
#
#     for i, context_page in tqdm(enumerate(top_20), total=20):
#         for j, target_page in enumerate(top_20):
#             if context_page == target_page: continue
#
#             in_set = (choice_sets == context_page).sum(1) > 0
#             conditional_in_choice_set = np.count_nonzero(choice_sets[in_set] == target_page)
#             conditional_chosen = np.count_nonzero(choice_indices[in_set] == target_page)
#
#             conditional_hit_rate = 0 if conditional_in_choice_set == 0 else conditional_chosen / conditional_in_choice_set
#
#             hit_rate_diffs[i, j] = conditional_hit_rate - hit_rates[target_page]
#
#     sns.heatmap(hit_rate_diffs, center=0, xticklabels=page_names, yticklabels=page_names, linewidths=.5, cmap='RdBu_r')
#     plt.xlabel('Target Page')
#     plt.ylabel('Context Page')
#     plt.savefig('top_20_context_hit_rates.pdf', bbox_inches='tight')
#     plt.show()
#
#     flat_dot_prods = dot_prods.flatten()
#     flat_hit_rate_diffs = hit_rate_diffs.flatten()
#
#     print('Correlation:', stats.pearsonr(flat_dot_prods, flat_hit_rate_diffs))
#
#     slope, intercept, r_value, p_value, std_err = stats.linregress(flat_dot_prods, flat_hit_rate_diffs)
#     plt.plot(np.linspace(-0.4, 0.7), slope * np.linspace(-0.4, 0.7) + intercept, c='#ffa600')
#     print(r_value, p_value, std_err)
#
#     plt.scatter(flat_dot_prods, flat_hit_rate_diffs)
#     plt.xlabel('Context effect score')
#     plt.ylabel('Conditional hit rate boost')
#     plt.savefig('context_score_correlation.pdf', bbox_inches='tight')
#     plt.show()
#
#
# def all_tsne(param_fname, dim):
#     graph, train_data, val_data, test_data = load_wikispeedia()
#     n = len(graph.nodes)
#
#     for embedding in load_normalized_embeddings(param_fname, n, dim):
#         tsne_embedding(embedding)
#
#
# def tsne_embedding(embedding):
#     print('Running TSNE...')
#     tsne = TSNE(n_components=2, random_state=1).fit_transform(embedding)
#     print('Done.')
#     plt.scatter(tsne[:, 0], tsne[:, 1])
#     plt.show()

if __name__ == '__main__':
    examine_email_enron()


    # model = load_feature_model(FeatureCDM, 3, 'feature_cdm_mathoverflow_params_0.01_0.pt')
    # print(model.contexts, model.weights)
    # for param_name, loss_name in zip(('feature_mnl_email-enron_params_0.005_0.pt', 'feature_mnl_email-enron_params_0.005_0.0001.pt', 'feature_mnl_email-enron_params_0.005_0.001.pt'),
    #                               ('feature_mnl_email-enron_losses_0.005_0.pickle', 'feature_mnl_email-enron_losses_0.005_0.0001.pickle', 'feature_mnl_email-enron_losses_0.005_0.001.pickle')):
    #
    #     model = load_feature_model(FeatureMNL, 3, param_name)
    #     print(model.weights)
    #
    #     with open(loss_name, 'rb') as f:
    #         train_losses, train_accs, val_losses, val_accs = pickle.load(f)
    #
    #     print(train_accs[-1], val_accs[-1])

    # plot_dataset_stats()

    # plot_compare_all()

    # plot_grid_search(HistoryCDM, EmailEnronDataset)
    # plot_grid_search(HistoryMNL, EmailEnronDataset)
    # plot_grid_search(LSTM, YoochooseDataset)

    # graph, train_data, val_data, test_data = YoochooseDataset.load()
    #
    # for param_fname in ('params/history_cdm_yoochoose_params_64_0.005_0_0.5_True.pt',):
    #     print(param_fname)
    #
    #     model = load_model(HistoryCDM, len(graph.nodes), 64, param_fname)
    #     print(model.num_items, model.dim, np.mean(model.target_embedding.weight.detach().numpy()))
    #     loaded_data = graph, train_data, val_data, test_data
    #     acc, mean_rank, mrr = test_model(model, YoochooseDataset, loaded_data=loaded_data)
    #     print(f'Accuracy: {acc:.2f}, beta: {model.beta.item():.2f}')

    # for param_fname in ('params/lstm_wikispeedia_params_64_0.005_0.pt',
    #                     'params/wikispeedia_lstm_params_64_0.005_0.pt'):
    #     print(param_fname)
    #
    #     model = load_model(LSTM, len(graph.nodes), 64, param_fname)
    #     print(model.num_items, model.dim)
    #     loaded_data = graph, train_data, val_data, test_data
    #     acc, mean_rank, mrr = test_model(model, WikispeediaDataset, loaded_data=loaded_data)
    #     print(f'Accuracy: {acc:.2f}')

    # with open('results/history_cdm_wikispeedia_losses_64_0.005_0_0.5_True.pickle', 'rb') as f:
    #     losses = pickle.load(f)
    # plt.plot(range(500), losses, label='new_losses')
    #
    # with open('results/wikispeedia_learn_beta_losses_64_0.005_0.pickle', 'rb') as f:
    #     old_losses = pickle.load(f)
    # plt.plot(range(500), old_losses, label='old_losses', ls='dashed')
    #
    # plt.legend()
    # plt.show()
