import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.ticker as ticker
from tqdm import tqdm

from datasets import WikispeediaDataset, KosarakDataset, YoochooseDataset, LastFMGenreDataset, ORCIDSwitchDataset, \
    EmailEnronDataset, CollegeMsgDataset, EmailEUDataset, MathOverflowDataset, FacebookWallDataset, \
    EmailEnronCoreDataset, EmailW3CDataset, EmailW3CCoreDataset, SMSADataset, SMSBDataset, SMSCDataset
from models import HistoryCDM, HistoryMNL, DataLoader, LSTM, FeatureMNL, FeatureCDM, train_feature_mnl, \
    FeatureContextMixture, train_model


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


    axes[0, 0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.02))
    axes[0, 1].legend(loc='lower center', bbox_to_anchor=(0.5, 1.02))
    plt.show()


def compile_choice_data(dataset):
    graph, train_data, val_data, test_data = dataset.load()

    histories, history_lengths, choice_sets, choice_set_features, choice_set_lengths, choices = [
        torch.cat([train_data[i], val_data[i], test_data[i]]).numpy() for i in range(len(train_data))]

    in_degree_ratios = []
    shared_neighbors_ratios = []
    reciprocity_ratios = []

    chosen_in_degrees = []
    chosen_shared_neighbors = []
    chosen_reciprocities = []

    mean_available_in_degrees = []
    mean_available_shared_neighbors = []
    mean_available_reciprocities = []

    for i in range(len(choice_set_features)):
        choice = choices[i]
        choice_set = choice_set_features[i, :choice_set_lengths[i]]

        # Convert reciprocities to -1/1 and log-degrees to degrees
        # choice_set[:, 0] = np.exp(choice_set[:, 0])
        # choice_set[:, 1] = np.exp(choice_set[:, 1])

        in_degree_ratios.append(choice_set[choice, 0] / np.mean(choice_set[:, 0]))
        shared_neighbors_ratios.append(choice_set[choice, 1] / np.mean(choice_set[:, 1]))
        # reciprocity_ratios.append(torch.nn.functional.softmax(torch.tensor(choice_set[:, 2]), dim=0)[choice].item() * choice_set_lengths[i])

        reciprocity_ratios.append((choice_set[choice, 2] / np.mean(choice_set[:, 2])) if np.mean(choice_set[:, 2]) > 0 else 1)

        chosen_in_degrees.append(choice_set[choice, 0])
        chosen_shared_neighbors.append(choice_set[choice, 1])
        chosen_reciprocities.append(choice_set[choice, 2])

        mean_available_in_degrees.append(np.mean(np.exp(choice_set[:, 0])))
        mean_available_shared_neighbors.append(np.mean(np.exp(choice_set[:, 1])))
        mean_available_reciprocities.append(np.mean(np.exp(choice_set[:, 2])))

    in_degree_ratios = np.array(in_degree_ratios)
    shared_neighbors_ratios = np.array(shared_neighbors_ratios)
    reciprocity_ratios = np.array(reciprocity_ratios)

    chosen_in_degrees = np.array(chosen_in_degrees)
    chosen_shared_neighbors = np.array(chosen_shared_neighbors)
    chosen_reciprocities = np.array(chosen_reciprocities)

    mean_available_in_degrees = np.array(mean_available_in_degrees)
    mean_available_shared_neighbors = np.array(mean_available_shared_neighbors)
    mean_available_reciprocities = np.array(mean_available_reciprocities)

    return in_degree_ratios, shared_neighbors_ratios, reciprocity_ratios, chosen_in_degrees, chosen_shared_neighbors, \
        chosen_reciprocities, mean_available_in_degrees, mean_available_shared_neighbors, mean_available_reciprocities, choice_set_lengths, \
        histories, history_lengths, choice_sets, choice_set_features, choice_set_lengths, choices


def learn_binned_mnl(dataset):
    print(f'Learning binned MNLs for {dataset.name}')
    num_bins = 100

    in_degree_ratios, out_degree_ratios, reciprocity_ratios, chosen_in_degrees, chosen_out_degrees, \
        chosen_reciprocities, mean_available_in_degrees, mean_available_shared_neighbors, mean_available_reciprocities, choice_set_lengths, \
        histories, history_lengths, choice_sets, choice_set_features, choice_set_lengths, choices = compile_choice_data(dataset)

    data = [None, None, None]

    for i, x_var in enumerate([mean_available_in_degrees, mean_available_shared_neighbors, mean_available_reciprocities]):
        x_min = min([x for x in x_var if x > 0]) * 0.8
        x_max = max(x_var) * 1.2

        values, bins = np.histogram(x_var, bins=np.logspace(np.log(x_min), np.log(x_max), num_bins))

        all_bin_idx = np.digitize(x_var, bins)

        mnl_utilities = np.zeros((num_bins, 3))
        bin_counts = np.zeros(num_bins)
        bin_choice_set_log_lengths = np.zeros(num_bins)
        bin_losses = np.zeros(num_bins)

        for bin in tqdm(range(num_bins)):
            bin_idx = all_bin_idx == bin

            bin_counts[bin] = np.count_nonzero(bin_idx)

            if bin_counts[bin] == 0:
                continue

            bin_data = [torch.tensor(choice_set_features[bin_idx]), torch.tensor(choice_set_lengths[bin_idx]), torch.tensor(choices[bin_idx])]
            mnl, train_losses, _, _, _ = train_feature_mnl(bin_data, bin_data, 3, lr=0.01, weight_decay=0.001)
            mnl_utilities[bin] = mnl.weights.detach().numpy()
            bin_choice_set_log_lengths[bin] = np.mean(np.log(choice_set_lengths[bin_idx]))
            bin_losses[bin] = torch.nn.functional.nll_loss(mnl(*bin_data[:-1]), bin_data[-1], reduction='sum').item()

        data[i] = bins, mnl_utilities, bin_counts, bin_choice_set_log_lengths, bin_losses

    with open(f'{dataset.name}_binned_mnl_params.pickle', 'wb') as f:
        pickle.dump(data, f)


def plot_binned_mnl(dataset, model_param_fname):
    with open(f'{dataset.name}_binned_mnl_params.pickle', 'rb') as f:
        data = pickle.load(f)

    model = load_feature_model(FeatureContextMixture, 3, model_param_fname)
    slopes = model.slopes.detach().numpy()
    intercepts = model.intercepts.detach().numpy()
    weights = model.weights.detach().numpy()

    plt.set_cmap('plasma')

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    y_mins = [np.inf, np.inf, np.inf]
    y_maxs = [-np.inf, -np.inf, -np.inf]

    wls_slopes = torch.zeros(3, 3)
    wls_intercepts = torch.zeros(3, 3)

    for col, x_name in enumerate(['In-degree', 'Shared Neighbors', 'Reciprocal Weight']):
        bins, mnl_utilities, bin_counts, bin_choice_set_log_lengths, bin_losses = data[col]

        nonempty = bin_counts > 0

        x_min = bins[min([i for i in range(len(bins)) if bin_counts[i] > 0])]
        x_max = bins[max([i for i in range(len(bins)) if bin_counts[i] > 0])]

        for row, y_name in enumerate(['Log In-degree', 'Log Shared Neighbors', 'Log Reciprocal Weight']):
            with_const = sm.add_constant(np.log(bins[nonempty]))
            mod_wls = sm.WLS(mnl_utilities[nonempty, row], with_const, weights=bin_counts[nonempty])
            res_wls = mod_wls.fit()
            wls_intercepts[row, col], wls_slopes[row, col] = res_wls.params

            axes[row, col].scatter(bins, mnl_utilities[:, row], alpha=1, s=bin_counts, marker='o', c=bin_choice_set_log_lengths)
            axes[row, col].scatter(bins, mnl_utilities[:, row], alpha=1, s=1, marker='.', color='white')

            axes[row, col].plot(bins, list(map(lambda x: intercepts[row, col] + x * slopes[row, col], np.log(bins))), label='mixture model')
            axes[row, col].plot(bins, list(map(lambda x: wls_intercepts[row, col] + x * wls_slopes[row, col], np.log(bins))), label='WLS')

            if col == 0:
                axes[row, col].set_ylabel(f'{y_name} Utility')
            else:
                plt.setp(axes[row, col].get_yticklabels(), visible=False)

            if row == 2:
                axes[row, col].set_xlabel(f'Choice Set {x_name}')
            elif row == 0:
                axes[row, col].set_title(f'Binned MNL NLL: {bin_losses.sum():.0f}\nMixture weight: {np.exp(weights[col]) / np.exp(weights).sum():.2f}')

            axes[row, col].set_xlim(x_min, x_max)

            axes[row, col].set_xscale('log')

            y_mins[row] = min(y_mins[row], min(mnl_utilities[:, row]))
            y_maxs[row] = max(y_maxs[row], max(mnl_utilities[:, row]))

    for row in range(3):
        axes[row, 0].set_ylim(y_mins[row]-1, y_maxs[row]+1)

    axes[0, 0].legend()

    graph, train_data, val_data, test_data = dataset.load()
    histories, history_lengths, choice_sets, choice_set_features, choice_set_lengths, choices = [
        torch.cat([train_data[i], val_data[i], test_data[i]]) for i in range(len(train_data))]

    sgd_nll = torch.nn.functional.nll_loss(model(choice_set_features, choice_set_lengths), choices, reduction='sum').item()

    model.slopes.data = wls_slopes
    model.intercepts.data = wls_intercepts
    model.weights.data = torch.ones(3)

    all_data = [choice_set_features, choice_set_lengths, choices]
    wls_nll = torch.nn.functional.nll_loss(model(choice_set_features, choice_set_lengths), choices, reduction='sum').item()

    mnl = load_feature_model(FeatureMNL, 3, model_param_fname.replace('feature_context_mixture', 'feature_mnl'))
    mnl_nll = torch.nn.functional.nll_loss(mnl(choice_set_features, choice_set_lengths), choices, reduction='sum').item()

    cdm = load_feature_model(FeatureCDM, 3, model_param_fname.replace('feature_context_mixture', 'feature_cdm'))
    cdm_nll = torch.nn.functional.nll_loss(cdm(choice_set_features, choice_set_lengths), choices, reduction='sum').item()

    axes[0, 1].text(0.4, 0.7, f'Mix NLL: {sgd_nll:.0f}\nWLS NLL: {wls_nll:.0f}\nMNL NLL: {mnl_nll:.0f}\nCDM NLL: {cdm_nll:.0f}', transform=axes[0, 1].transAxes)

    plt.savefig(f'{dataset.name}-mixture-fit-feature-utilities.pdf', bbox_inches='tight')
    plt.close()


def examine_choice_set_size_effects(dataset):
    in_degree_ratios, out_degree_ratios, reciprocity_ratios, chosen_in_degrees, chosen_out_degrees, \
        chosen_reciprocities, mean_available_in_degrees, mean_available_out_degrees, mean_available_reciprocities, choice_set_lengths, \
        histories, history_lengths, choice_sets, choice_set_features, choice_set_lengths, choices = compile_choice_data(dataset)

    plt.set_cmap('plasma')

    fig, axes = plt.subplots(3, 1, figsize=(4, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for j, (x_variable, x_name) in enumerate([(mean_available_in_degrees, 'In-degree'), (mean_available_out_degrees, 'Out-degree'), (mean_available_reciprocities, 'Reciprocity')]):

        axes[j].scatter(choice_set_lengths, x_variable, s=10, alpha=0.4, marker='.')
        axes[j].set_ylabel(f'Mean {x_name}')
        axes[j].set_xlabel('Choice Set Size')
        axes[j].set_xscale('log')

        if j < 2:
            axes[j].set_yscale('log')

    plt.savefig(f'{dataset.name}-choice-set-lengths.pdf', bbox_inches='tight')


def plot_all_training_accuracies():
    datasets = [EmailEnronDataset, EmailEUDataset, EmailW3CDataset,
                    SMSADataset, SMSBDataset, SMSCDataset, CollegeMsgDataset, MathOverflowDataset, FacebookWallDataset]
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    losses = [[], [], []]
    accs = [[], [], []]
    mrrs = [[], [], []]

    for i, dataset in enumerate(datasets):
        graph, train_data, val_data, test_data = dataset.load()

        histories, history_lengths, choice_sets, choice_sets_with_features, choice_set_lengths, choices = test_data

        for j, method in enumerate([FeatureMNL, FeatureCDM, FeatureContextMixture]):
            param_fname = f'{method.name}_{dataset.name}_train_params_0.005_0.001.pt'
            model = load_feature_model(method, 3, param_fname)

            pred = model(choice_sets_with_features, choice_set_lengths)
            train_loss = model.loss(pred, choices)

            ranks = pred.argsort(1, descending=True) + 1

            vals, idxs = pred.max(1)
            acc = (idxs == choices).long().sum().item() / len(choices)

            losses[j].append(train_loss.item())
            accs[j].append(acc)
            mrrs[j].append((1 / ranks[torch.arange(len(choices)), choices].float()).sum().item() / len(choices))

    bar_width = 0.25

    xs = [np.arange(9) - bar_width, np.arange(9), np.arange(9) + bar_width]
    method_names = ['Feature MNL', 'Feature CDM', 'Context Mixture']

    losses = np.array(losses)
    accs = np.array(accs)
    mrrs = np.array(mrrs)

    min_nll_indices = np.argmin(losses, axis=0)
    max_acc_indices = np.argmax(accs, axis=0)
    max_mrr_indices = np.argmax(mrrs, axis=0)

    min_nll_xs = (np.arange(9) - bar_width) + (min_nll_indices * bar_width)
    max_acc_xs = (np.arange(9) - bar_width) + (max_acc_indices * bar_width)
    max_mrr_xs = (np.arange(9) - bar_width) + (max_mrr_indices * bar_width)

    min_nll_ys = losses[min_nll_indices, np.arange(9)] + 0.1
    max_acc_ys = accs[max_acc_indices, np.arange(9)] + 0.01
    max_mrr_ys = mrrs[max_mrr_indices, np.arange(9)] + 0.01

    axes[0].scatter(min_nll_xs, min_nll_ys, marker='*', color='black')
    axes[1].scatter(max_acc_xs, max_acc_ys, marker='*', color='black')
    axes[2].scatter(max_mrr_xs, max_mrr_ys, marker='*', color='black')

    for i in range(3):
        axes[0].bar(xs[i], losses[i], edgecolor='white', label=method_names[i], width=bar_width)
        axes[1].bar(xs[i], accs[i], edgecolor='white', label=method_names[i], width=bar_width)
        axes[2].bar(xs[i], mrrs[i], edgecolor='white', label=method_names[i], width=bar_width)

    axes[0].set_xticks(np.arange(9))
    axes[0].set_xticklabels([dataset.name for dataset in datasets])
    axes[1].set_xticks(np.arange(9))
    axes[1].set_xticklabels([dataset.name for dataset in datasets])
    axes[2].set_xticks(np.arange(9))
    axes[2].set_xticklabels([dataset.name for dataset in datasets])

    axes[0].set_ylabel('Mean Test NLL')
    axes[1].set_ylabel('Test Accuracy')
    axes[2].set_ylabel('Test MRR')

    axes[0].legend()

    plt.savefig('plots/test_performance.pdf', bbox_inches='tight')


if __name__ == '__main__':
    plot_all_training_accuracies()

    # for dataset in [FacebookWallDataset, EmailEnronDataset, EmailEUDataset, EmailW3CDataset, CollegeMsgDataset,
    #                 SMSADataset, SMSBDataset, SMSCDataset, MathOverflowDataset]:
        # print(dataset.name)
        # if not os.path.isfile(f'{dataset.name}_binned_mnl_params.pickle'):
        #     learn_binned_mnl(dataset)
        # plot_binned_mnl(dataset, f'feature_context_mixture_{dataset.name}_params_0.005_0.001.pt')

