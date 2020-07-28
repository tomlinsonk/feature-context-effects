import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.ticker as ticker
from scipy.stats import chi2
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter1d


from datasets import WikispeediaDataset, KosarakDataset, YoochooseDataset, LastFMGenreDataset, ORCIDSwitchDataset, \
    EmailEnronDataset, CollegeMsgDataset, EmailEUDataset, MathOverflowDataset, FacebookWallDataset, \
    EmailEnronCoreDataset, EmailW3CDataset, EmailW3CCoreDataset, SMSADataset, SMSBDataset, SMSCDataset, WikiTalkDataset, \
    RedditHyperlinkDataset, BitcoinAlphaDataset, BitcoinOTCDataset, SyntheticMNLDataset, SyntheticCDMDataset, \
    ExpediaDataset, SushiDataset, DistrictDataset
from models import HistoryCDM, HistoryMNL, DataLoader, LSTM, FeatureMNL, FeatureCDM, train_feature_mnl, \
    FeatureContextMixture, train_model, FeatureSelector, RandomSelector, MNLMixture

PARAM_DIR = 'params/triadic-closure-6-standard-feats'
RESULT_DIR = 'results/triadic-closure-6-standard-feats'
PLOT_DIR = 'plots'
CONFIG_DIR = 'config'


def load_model(Model, n, dim, param_fname):
    if Model is LSTM:
        model = Model(n, dim)
    else:
        model = Model(n, dim, 0.5)

    model.load_state_dict(torch.load(param_fname))
    model.eval()

    return model


def load_feature_model(Model, model_param, param_fname):
    model = Model(model_param)

    if Model not in [RandomSelector, FeatureSelector]:
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


def plot_grid_search(dataset):
    with open(f'{CONFIG_DIR}/learning_rate_settings.pickle', 'rb') as f:
        data, lrs = pickle.load(f)

    methods = [FeatureMNL, MNLMixture, FeatureCDM, FeatureContextMixture]
    markers = ['s', '^', 'o', 'P']

    for i, method in enumerate(methods):
        losses = [data[dataset, method, lr] for lr in lrs]
        plt.plot(range(6), losses, '.-', label=method.name, marker=markers[i])


    plt.xticks(range(6), lrs)
    plt.xlabel('Learning Rate')
    plt.ylabel('Total NLL')
    plt.title(dataset.name)
    plt.yscale('log')
    plt.legend()
    plt.show()


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

    axes[0, 0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.02))
    axes[0, 1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.02))
    plt.show()


def plot_binned_mnl(dataset, model_param_fname):
    with open(f'{RESULT_DIR}/{dataset.name}_binned_mnl_params.pickle', 'rb') as f:
        data = pickle.load(f)

    n_feats = dataset.num_features

    model = load_feature_model(FeatureContextMixture, n_feats, model_param_fname)
    slopes = model.slopes.detach().numpy()
    intercepts = model.intercepts.detach().numpy()
    weights = model.weights.detach().numpy()

    plt.set_cmap('plasma')

    fig, axes = plt.subplots(n_feats, n_feats, figsize=(16, 16), sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    y_mins = [np.inf] * n_feats
    y_maxs = [-np.inf] * n_feats

    wls_slopes = torch.zeros(n_feats, n_feats)
    wls_intercepts = torch.zeros(n_feats, n_feats)

    feature_names = ['In-degree', 'Shared Nbrs.', 'Recip. Weight', 'Send Recency', 'Receive Recency', 'Recip. Recency']
    # feature_names = ['Star Rating', 'Review Score', 'Location Score', 'Price', 'On Promotion']

    for col, x_name in enumerate(feature_names):
        bins, mnl_utilities, bin_counts, bin_choice_set_log_lengths, bin_losses = data[col]

        nonempty = bin_counts > 0

        x_min = bins[min([i for i in range(len(bins)) if bin_counts[i] > 0])]
        x_max = bins[max([i for i in range(len(bins)) if bin_counts[i] > 0])]

        for row, y_name in enumerate(feature_names):
            with_const = sm.add_constant(bins[nonempty])
            mod_wls = sm.WLS(mnl_utilities[nonempty, row], with_const, weights=bin_counts[nonempty])
            res_wls = mod_wls.fit()
            wls_intercepts[row, col], wls_slopes[row, col] = res_wls.params

            axes[row, col].scatter(bins, mnl_utilities[:, row], alpha=1, s=bin_counts, marker='o', c=bin_choice_set_log_lengths)
            axes[row, col].scatter(bins, mnl_utilities[:, row], alpha=1, s=1, marker='.', color='white')

            xs = bins
            axes[row, col].plot(bins, list(map(lambda x: intercepts[row, col] + x * slopes[row, col], xs)), label='mixture model')
            axes[row, col].plot(bins, list(map(lambda x: wls_intercepts[row, col] + x * wls_slopes[row, col], xs)), label='WLS')

            if col == 0:
                axes[row, col].set_ylabel(f'{y_name} Utility')
            else:
                plt.setp(axes[row, col].get_yticklabels(), visible=False)

            if row == n_feats - 1:
                axes[row, col].set_xlabel(f'Choice Set {x_name}')
            elif row == 0:
                axes[row, col].set_title(f'Binned MNL NLL: {bin_losses.sum():.0f}\nMixture weight: {np.exp(weights[col]) / np.exp(weights).sum():.2f}')

            axes[row, col].set_xlim(x_min, x_max)

            y_mins[row] = min(y_mins[row], min(mnl_utilities[:, row]))
            y_maxs[row] = max(y_maxs[row], max(mnl_utilities[:, row]))

    for row in range(n_feats):
        axes[row, 0].set_ylim(y_mins[row]-1, y_maxs[row]+1)

    axes[0, 0].legend()

    graph, train_data, val_data, test_data, _, _ = dataset.load_standardized()
    histories, history_lengths, choice_sets, choice_set_features, choice_set_lengths, choices = [
        torch.cat([train_data[i], val_data[i], test_data[i]]) for i in range(len(train_data))]

    sgd_nll = torch.nn.functional.nll_loss(model(choice_set_features, choice_set_lengths), choices, reduction='sum').item()

    model.slopes.data = wls_slopes
    model.intercepts.data = wls_intercepts
    model.weights.data = torch.ones(n_feats)

    all_data = [choice_set_features, choice_set_lengths, choices]
    wls_nll = torch.nn.functional.nll_loss(model(choice_set_features, choice_set_lengths), choices, reduction='sum').item()

    mnl = load_feature_model(FeatureMNL, n_feats, f'{PARAM_DIR}/feature_mnl_{dataset.name}_params_0.005_0.001.pt')
    mnl_nll = torch.nn.functional.nll_loss(mnl(choice_set_features, choice_set_lengths), choices, reduction='sum').item()

    cdm = load_feature_model(FeatureCDM, n_feats, f'{PARAM_DIR}/feature_cdm_{dataset.name}_params_0.005_0.001.pt')
    cdm_nll = torch.nn.functional.nll_loss(cdm(choice_set_features, choice_set_lengths), choices, reduction='sum').item()

    axes[0, 1].text(0.37, 0.67, f'Mix NLL: {sgd_nll:.0f}\nWLS NLL: {wls_nll:.0f}\nMNL NLL: {mnl_nll:.0f}\nCDM NLL: {cdm_nll:.0f}', transform=axes[0, 1].transAxes)

    plt.savefig(f'{dataset.name}-mixture-em-fit-feature-utilities.pdf', bbox_inches='tight')
    plt.close()


def examine_choice_set_size_effects(datasets):
    with open(f'{RESULT_DIR}/all_prediction_results.pickle', 'rb') as f:
        _, _, _, ranks, corrects = pickle.load(f)

    plt.set_cmap('plasma')

    use_methods = [0, 2, 8]
    method_names = ['Feature MNL', 'Feature CDM', 'Context Mixture', 'In-Degree', 'Shared Neighbors', 'Reciprocal Weight', 'Time Since Send', 'Time Since Receive', 'Time Since Reciprocation', 'Random']

    size_fig, size_axes = plt.subplots(1, len(datasets), figsize=(25, 2))
    acc_fig, acc_axes = plt.subplots(1, len(datasets), figsize=(25, 2), sharey=True)

    for i, dataset in enumerate(datasets):

        graph, train_data, val_data, test_data, _, _ = dataset.load_standardized()
        histories, history_lengths, choice_sets, choice_sets_with_features, choice_set_lengths, choices = test_data

        unique_lengths, inverse, counts = np.unique(choice_set_lengths, return_counts=True, return_inverse=True)

        binned_corrects = [[np.mean(corrects[method, i][inverse == idx]) for idx in range(len(unique_lengths))] for method in use_methods]

        size_axes[i].fill_between(unique_lengths, counts)

        for j, method in enumerate(use_methods):
            acc_axes[i].plot(unique_lengths, gaussian_filter1d(binned_corrects[j], sigma=5), label=method_names[method])

        size_axes[i].set_title(dataset.name)
        size_axes[i].set_xlabel('Choice Set Size')
        size_axes[i].set_xscale('log')

        acc_axes[i].set_title(dataset.name)
        acc_axes[i].set_xlabel('Choice Set Size')
        acc_axes[i].set_xscale('log')

        size_axes[i].set_yticks([])

    size_axes[0].set_ylabel('Proportion')
    acc_axes[0].set_ylabel('Accuracy')
    acc_axes[0].legend(bbox_to_anchor=(0, 1.1), loc='lower left')
    acc_axes[0].set_zorder(1)

    plt.figure(size_fig.number)
    plt.savefig(f'{PLOT_DIR}/choice-set-sizes.pdf', bbox_inches='tight')

    plt.figure(acc_fig.number)
    plt.savefig(f'{PLOT_DIR}/choice-set-accs.pdf', bbox_inches='tight')


def compute_all_accuracies(datasets):
    methods = [FeatureMNL, FeatureCDM, MNLMixture, FeatureContextMixture, FeatureContextMixture, FeatureSelector, FeatureSelector, FeatureSelector,
                 FeatureSelector, FeatureSelector, FeatureSelector, RandomSelector]

    losses = [list() for _ in range(len(methods))]
    accs = [list() for _ in range(len(methods))]
    mean_ranks = [list() for _ in range(len(methods))]

    all_correct_preds = [list() for _ in range(len(methods))]
    all_ranks = [list() for _ in range(len(methods))]

    for i, dataset in enumerate(datasets):
        print('Computing accuracies for', dataset.name)
        graph, train_data, val_data, test_data, _, _ = dataset.load_standardized()

        histories, history_lengths, choice_sets, choice_sets_with_features, choice_set_lengths, choices = test_data

        for j, method in enumerate(methods):
            param_fname = f'{PARAM_DIR}/{method.name}_{dataset.name}_params_{dataset.best_lr(method)}_0.001.pt' if j < 4 else f'{PARAM_DIR}/context_mixture_em_{dataset.name}_params.pt'

            model_param = dataset.num_features if j < 5 else j - 5
            model = load_feature_model(method, model_param, param_fname)

            pred = model(choice_sets_with_features, choice_set_lengths)
            train_loss = model.loss(pred, choices)

            ranks = stats.rankdata(-pred.detach().numpy(), method='average', axis=1)[np.arange(len(choices)), choices]
            vals, idxs = pred.max(1)

            correct_preds = (idxs == choices)
            acc = correct_preds.long().sum().item() / len(choices)

            losses[j].append(train_loss.item())
            accs[j].append(acc)
            mean_ranks[j].append(np.mean(ranks / np.array(choice_set_lengths)))
            all_correct_preds[j].append(correct_preds.numpy())
            all_ranks[j].append(ranks)

    with open(f'{RESULT_DIR}/all_prediction_results.pickle', 'wb') as f:
        pickle.dump([np.array(losses), np.array(accs), np.array(mean_ranks), np.array(all_ranks), np.array(all_correct_preds)], f)


def plot_general_choice_dataset_accuracies(dataset):

    methods = [FeatureMNL, FeatureCDM, MNLMixture, FeatureContextMixture, FeatureContextMixture] + ([FeatureSelector] * dataset.num_features) + [RandomSelector]

    losses = []
    accs = []
    mean_ranks = []
    all_correct_preds = []
    all_ranks = []

    graph, train_data, val_data, test_data, _, _ = dataset.load_standardized()

    histories, history_lengths, choice_sets, choice_sets_with_features, choice_set_lengths, choices = test_data

    for j, method in enumerate(methods):
        param_fname = f'{PARAM_DIR}/{method.name}_{dataset.name}_params_{dataset.best_lr(method)}_0.001.pt' if j < 4 else f'{PARAM_DIR}/context_mixture_em_{dataset.name}_params.pt'

        model_param = dataset.num_features if j < 5 else j - 5
        model = load_feature_model(method, model_param, param_fname)

        flip_feats = (dataset == ExpediaDataset and j == 8) or (dataset == SushiDataset and j in (5, 7))

        pred = model(choice_sets_with_features * (-1 if flip_feats else 1), choice_set_lengths)
        train_loss = model.loss(pred, choices)

        ranks = stats.rankdata(-pred.detach().numpy(), method='average', axis=1)[np.arange(len(choices)), choices]
        vals, idxs = pred.max(1)

        correct_preds = (idxs == choices)
        acc = correct_preds.long().sum().item() / len(choices)

        losses.append(train_loss.item())
        accs.append(acc)
        mean_ranks.append(np.mean(ranks / np.array(choice_set_lengths)))
        all_correct_preds.append(correct_preds.numpy())
        all_ranks.append(ranks)

    method_names = ['Feature MNL', 'Feature CDM', 'Mixed MNL', 'Context Mixture', 'Context Mixture EM'] + dataset.feature_names + ['Random']

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    axes[0].bar(range(len(method_names)), losses)
    axes[0].set_xticks(range(len(method_names)))
    axes[0].set_xticklabels(method_names)
    axes[0].set_ylabel('NLL')


    axes[1].bar(range(len(method_names)), accs)
    axes[1].set_xticks(range(len(method_names)))
    axes[1].set_xticklabels(method_names)
    axes[1].set_ylabel('Accuracy')

    axes[2].bar(range(len(method_names)), mean_ranks)
    axes[2].set_xticks(range(len(method_names)))
    axes[2].set_xticklabels(method_names)
    axes[2].set_ylabel('Mean Correct Position')

    plt.savefig(f'{PLOT_DIR}/{dataset.name}_test_results.pdf', bbox_inches='tight')


def plot_all_accuracies(datasets):
    with open(f'{RESULT_DIR}/all_prediction_results.pickle', 'rb') as f:
        losses, accs, mean_ranks, _, _ = pickle.load(f)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))


    bar_width = 0.167
    width_multiplier = 2
    xs = [np.arange(len(datasets)) + (i * bar_width) for i in range(-2, 3)]
    method_names = ['Feature MNL', 'Feature CDM', 'Mixed MNL', 'Context Mixture', 'Context Mixture EM']

    # With baselines
    # bar_width = 0.07
    # width_multiplier = 5.5
    # xs = [np.arange(len(datasets)) + ((i + 0.5) * bar_width) for i in range(-6, 6)]
    # method_names = ['Feature MNL', 'Feature CDM', 'Mixed MNL', 'Context Mixture', 'Context Mixture EM', 'In-Degree', 'Shared Neighbors', 'Reciprocal Weight', 'Time Since Send', 'Time Since Receive', 'Time Since Reciprocation', 'Random']

    min_nll_indices = np.argmin(losses[:len(method_names)], axis=0)
    max_acc_indices = np.argmax(accs[:len(method_names)], axis=0)
    min_mean_rank_indices = np.argmin(mean_ranks[:len(method_names)], axis=0)

    min_nll_xs = (np.arange(len(datasets)) - width_multiplier * bar_width) + (min_nll_indices * bar_width)
    max_acc_xs = (np.arange(len(datasets)) - width_multiplier * bar_width) + (max_acc_indices * bar_width)
    min_mean_rank_xs = (np.arange(len(datasets)) - width_multiplier * bar_width) + (min_mean_rank_indices * bar_width)

    min_nll_ys = losses[min_nll_indices, np.arange(len(datasets))] + 0.2
    max_acc_ys = accs[max_acc_indices, np.arange(len(datasets))] + 0.01
    min_mean_rank_ys = mean_ranks[min_mean_rank_indices, np.arange(len(datasets))] + 0.02

    axes[0].scatter(min_nll_xs, min_nll_ys, marker='*', color='black')
    axes[1].scatter(max_acc_xs, max_acc_ys, marker='*', color='black')
    axes[2].scatter(min_mean_rank_xs, min_mean_rank_ys, marker='*', color='black')

    for i in range(len(method_names)):
        axes[0].bar(xs[i], losses[i], edgecolor='white', label=method_names[i], width=bar_width)
        axes[1].bar(xs[i], accs[i], edgecolor='white', label=method_names[i], width=bar_width)
        axes[2].bar(xs[i], mean_ranks[i], edgecolor='white', label=method_names[i], width=bar_width)

    for i in range(3):
        axes[i].set_xticks(np.arange(len(datasets)))
        axes[i].set_xticklabels([dataset.name for dataset in datasets], rotation=13)

    axes[0].set_ylabel('Mean Test NLL')
    axes[1].set_ylabel('Test Accuracy')
    axes[2].set_ylabel('Test Mean Correct Position')

    axes[1].legend(bbox_to_anchor=(1.01, 0.5), loc='center left')

    plt.savefig(f'{PLOT_DIR}/test_performance.pdf', bbox_inches='tight')


def visualize_context_effects(datasets):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    cmap = mpl.cm.bwr

    all_slopes = []

    for i, dataset in enumerate(datasets):
        row = i // 4
        col = i % 4

        model = load_feature_model(FeatureContextMixture, 6, f'{PARAM_DIR}/feature_context_mixture_{dataset.name}_params_{dataset.best_lr(FeatureContextMixture)}_0.001.pt')
        # model = load_feature_model(FeatureContextMixture, 6, f'{PARAM_DIR}/context_mixture_em_{dataset.name}_params.pt')


        slopes = model.slopes.data.numpy()
        all_slopes.append(slopes)

        axes[row, col].matshow(slopes, cmap=cmap)

        print(dataset.name, slopes)

        axes[row, col].axis('off')
        axes[row, col].set_title(dataset.name, pad=0.1)

    # norm = mpl.colors.Normalize(vmin=-1, vmax=1)

    for col in range(1, 4):
        axes[3, col].axis('off')


    vis = axes[3, 3].matshow(np.mean(all_slopes, axis=0), cmap=cmap)
    axes[3, 3].set_title('Mean', pad=0.1)

    plt.colorbar(vis, ax=axes[:, :])


    # axes[3, 3].matshow(np.std(all_slopes, axis=0), cmap=cmap, vmin=-1, vmax=1)
    # axes[3, 3].set_title('Std Dev', pad=0.1)

    plt.savefig('learned_context_mixture_slopes.pdf', bbox_inches='tight')
    plt.close()


def visualize_context_effects_l1_reg(datasets, method):
    with open(f'{RESULT_DIR}/l1_regularization_grid_search_{method.name}_results.pickle', 'rb') as f:
        results, reg_params = pickle.load(f)

    with open(f'{CONFIG_DIR}/learning_rate_settings.pickle', 'rb') as f:
        grid_search_losses, lrs = pickle.load(f)

    if method == FeatureCDM:
        reg_params.remove(0.001)
    elif method == FeatureContextMixture:
        reg_params = reg_params[:-2]

    fig = plt.figure(figsize=(len(reg_params), len(datasets)*1.1), constrained_layout=False)
    gs = fig.add_gridspec(len(datasets), len(reg_params), wspace=0, hspace=0.1)

    for row, dataset in enumerate(datasets):
        all_slopes = [results[dataset, reg_param, method][0].slopes.data.numpy() if method == FeatureContextMixture else results[dataset, reg_param, method][0].contexts.data.numpy() for reg_param in reg_params]

        max_abs = np.max(np.abs(all_slopes))
        vmin = -max_abs
        vmax = max_abs

        for col, reg_param in enumerate(reg_params):
            ax = fig.add_subplot(gs[row, col])

            if col == 0:
                ax.set_ylabel(dataset.name, rotation='horizontal', ha='right', fontsize=14, va='center')
            if row == 0:
                if col == 0:
                    ax.set_title(f'$\\lambda=${reg_param}', fontsize=12)
                else:
                    ax.set_title(f'{reg_param}', fontsize=12)

            model, loss = results[dataset, reg_param, method]
            ax.matshow(model.slopes.data.numpy() if method == FeatureContextMixture else model.contexts.data.numpy(), cmap=mpl.cm.bwr, vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])

        losses = [results[dataset, reg_param, method][1] for reg_param in reg_params]

        baseline = MNLMixture if method == FeatureContextMixture else FeatureMNL
        baseline_loss = min([grid_search_losses[dataset, baseline, lr] for lr in lrs])

        p = 1e-8
        ddof = dataset.num_features**2
        sig_thresh = baseline_loss - 0.5 * chi2.isf(p, ddof)

        ymax_pcts = 2

        if method == FeatureContextMixture:
            if dataset == EmailEnronDataset:
                ymax_pcts = 4
            elif dataset == EmailW3CDataset:
                ymax_pcts = 10
            elif dataset == SyntheticMNLDataset:
                ymax_pcts = 0.04
        else:
            if dataset in (EmailEnronDataset, EmailW3CDataset, CollegeMsgDataset, MathOverflowDataset, SyntheticCDMDataset, BitcoinOTCDataset):
                ymax_pcts = 4
            elif dataset == SyntheticMNLDataset:
                ymax_pcts = 0.1

        y_min = min(losses + [sig_thresh])
        y_ticks = [y_min, y_min * (1 + ymax_pcts / 200), y_min * (1 + ymax_pcts / 100)]

        loss_ax = fig.add_subplot(gs[row, :])
        loss_ax.plot(range(len(reg_params)), losses, color='black', alpha=0.7)
        loss_ax.set_xlim(-0.5, len(reg_params) - 0.5)

        loss_ax.set_ylim(min(y_ticks) * (1 - ymax_pcts / 1000), max(y_ticks))

        loss_ax.hlines(sig_thresh, -0.5, len(reg_params) - 0.5, linestyles='dotted', colors='green')

        loss_ax.patch.set_visible(False)
        loss_ax.set_xticks([])
        loss_ax.set_yticks(y_ticks)
        loss_ax.set_yticklabels(['', f'+{ymax_pcts/2:.2g}%', f'+{ymax_pcts:.2g}%'])

        loss_ax.yaxis.tick_right()

    plt.suptitle('Linear Mixed Contexts Logit' if method == FeatureContextMixture else 'CDM-Based Model', y=0.91, fontsize=16)

    plt.savefig(f'l1_regularization_{method.name}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    network_datasets = [
        SyntheticMNLDataset, SyntheticCDMDataset,
        WikiTalkDataset, RedditHyperlinkDataset,
        BitcoinAlphaDataset, BitcoinOTCDataset,
        SMSADataset, SMSBDataset, SMSCDataset,
        EmailEnronDataset, EmailEUDataset, EmailW3CDataset,
        FacebookWallDataset, CollegeMsgDataset, MathOverflowDataset
    ]

    all_datasets = [DistrictDataset, ExpediaDataset, SushiDataset] + network_datasets

    for dataset in all_datasets:
        plot_grid_search(dataset)

    # plot_general_choice_dataset_accuracies(ExpediaDataset)

    # visualize_context_effects_l1_reg(network_datasets, FeatureCDM)
    # visualize_context_effects_l1_reg(network_datasets, FeatureContextMixture)

    # visualize_context_effects(network_datasets)
    # compute_all_accuracies(network_datasets)
    # plot_all_accuracies(network_datasets)

    # examine_choice_set_size_effects(datasets)
    #
    #
    # for dataset in network_datasets:
    #     print(dataset.name)
    #     plot_binned_mnl(dataset, f'{PARAM_DIR}/feature_context_mixture_{dataset.name}_params_0.005_0.001.pt')


