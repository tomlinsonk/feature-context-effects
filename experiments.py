import pickle
import random

import choix
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets import WikispeediaDataset, KosarakDataset, YoochooseDataset, LastFMGenreDataset, ORCIDSwitchDataset, \
    EmailEnronDataset, CollegeMsgDataset, EmailEUDataset, MathOverflowDataset, FacebookWallDataset, \
    EmailEnronCoreDataset, EmailW3CDataset, EmailW3CCoreDataset, SMSADataset, SMSBDataset, SMSCDataset, WikiTalkDataset, \
    RedditHyperlinkDataset, BitcoinOTCDataset, BitcoinAlphaDataset, SyntheticMNLDataset, SyntheticCDMDataset
from models import train_history_cdm, train_lstm, train_history_mnl, train_feature_mnl, HistoryCDM, HistoryMNL, LSTM, \
    FeatureMNL, FeatureCDM, train_feature_cdm, FeatureContextMixture, train_feature_context_mixture, context_mixture_em

training_methods = {HistoryCDM: train_history_cdm, HistoryMNL: train_history_mnl, LSTM: train_lstm, FeatureMNL: train_feature_mnl,
                    FeatureCDM: train_feature_cdm, FeatureContextMixture: train_feature_context_mixture}


def run_model(method, dataset, dim, lr, wd, beta=None, learn_beta=None):
    graph, train_data, val_data, test_data = dataset.load()

    print(f'Training {method.name} on {dataset.name} (dim={dim}, lr={lr}, wd={wd}, beta={beta}, learn_beta={learn_beta})')

    model, train_losses, train_accs, val_losses, val_accs = training_methods[method](len(graph.nodes), train_data,
                                                                                     val_data, dim=dim, lr=lr, weight_decay=wd,
                                                                                     beta=beta, learn_beta=learn_beta)
    torch.save(model.state_dict(), f'{method.name}_{dataset.name}_params_{dim}_{lr}_{wd}_{beta}_{learn_beta}.pt')
    with open(f'{method.name}_{dataset.name}_losses_{dim}_{lr}_{wd}_{beta}_{learn_beta}.pickle', 'wb') as f:
        pickle.dump((train_losses, train_accs, val_losses, val_accs), f)


def run_feature_model_full_dataset(method, dataset, lr, wd):
    graph, train_data, val_data, test_data, means, stds = dataset.load_standardized()

    all_data = [torch.cat([train_data[i], val_data[i], test_data[i]]) for i in range(3, len(train_data))]

    print(f'Training {method.name} on {dataset.name} (lr={lr}, wd={wd})')

    model, train_losses, train_accs, val_losses, val_accs = training_methods[method](all_data, val_data, dataset.num_features, lr=lr, weight_decay=wd, compute_val_stats=False)
    torch.save(model.state_dict(), f'{method.name}_{dataset.name}_params_{lr}_{wd}.pt')
    with open(f'{method.name}_{dataset.name}_losses_{lr}_{wd}.pickle', 'wb') as f:
        pickle.dump((train_losses, train_accs, val_losses, val_accs), f)

    return model


def run_feature_model_train_data(method, dataset, lr, wd):
    graph, train_data, val_data, test_data, means, stds = dataset.load_standardized()

    print(f'Training {method.name} on {dataset.name}, training data only (lr={lr}, wd={wd})')

    model, train_losses, train_accs, val_losses, val_accs = training_methods[method](train_data[3:], val_data[3:], dataset.num_features, lr=lr, weight_decay=wd, compute_val_stats=True)
    torch.save(model.state_dict(), f'{method.name}_{dataset.name}_train_params_{lr}_{wd}.pt')
    with open(f'{method.name}_{dataset.name}_train_losses_{lr}_{wd}.pickle', 'wb') as f:
        pickle.dump((train_losses, train_accs, val_losses, val_accs), f)

    return model


def compare_methods(dataset):
    run_model(HistoryCDM, dataset, 64, 0.005, 0, 0.5, True)
    run_model(HistoryMNL, dataset, 64, 0.005, 0, 0.5, True)
    run_model(LSTM, dataset, 64, 0.005, 0)


def grid_search(dataset):
    for lr in [0.005]:
        for wd in [0, 1e-4, 5e-4, 1e-3]:
            run_model(HistoryCDM, dataset, 8, lr=lr, wd=wd, beta=0.5, learn_beta=True)
            run_model(HistoryMNL, dataset, 8, lr=lr, wd=wd, beta=0.5, learn_beta=True)
            run_model(LSTM, dataset, 8, lr=lr, wd=wd)


def run_baselines(dataset):
    graph, train_data, val_data, test_data = dataset.load()

    n = len(graph.nodes)

    histories, history_lengths, choice_sets, choice_set_lengths, choices = train_data
    transitions = np.zeros((n, n))
    for i in range(len(histories)):
        transitions[histories[i][0], choice_sets[i][choices[i]]] += 1

    traffic_in = transitions.sum(axis=0)
    traffic_out = transitions.sum(axis=1)

    histories, history_lengths, choice_sets, choice_set_lengths, choices = val_data

    try:
        params = choix.choicerank(graph, traffic_in, traffic_out)

        correct = 0
        total = 0
        for i in range(len(histories)):
            choice_set = choice_sets[i, :choice_set_lengths[i]]
            probs = choix.probabilities(choice_set, params)
            total += 1

            if np.argmax(probs) == choices[i].item():
                correct += 1

        print('ChoiceRank')
        print(f'\tAccuracy: {correct / total}')
    except RuntimeError:
        print('ChoiceRank crashed')

    correct = 0
    total = 0
    for i in range(len(histories)):
        pred = np.random.randint(0, choice_set_lengths[i])
        total += 1

        if pred == choices[i].item():
            correct += 1

    print('Random')
    print(f'\tAccuracy: {correct / total}')

    correct = 0
    total = 0
    most_frequent = np.argmax(transitions, axis=1)

    for i in range(len(histories)):
        total += 1

        if most_frequent[histories[i][0]] == choice_sets[i][choices[i].item()]:
            correct += 1

    print('Pick-most-frequent')
    print(f'\tAccuracy: {correct / total}')

    correct = 0
    total = 0

    for i in range(len(histories)):
        total += 1

        prediction = most_frequent[histories[i][0]]
        if history_lengths[i] > 1:
            prediction = histories[i][1]

        if prediction == choice_sets[i][choices[i].item()]:
            correct += 1

    print('Return-to-previous (default pick-most-frequent)')
    print(f'\tAccuracy: {correct / total}')

    correct = 0
    total = 0

    for i in range(len(histories)):
        total += 1

        if histories[i][0] == choice_sets[i][choices[i].item()]:
            correct += 1

    print('Repeat-current')
    print(f'\tAccuracy: {correct / total}')


def run_triadic_closure_baselines(dataset):
    graph, train_data, val_data, test_data = dataset.load()

    histories, history_lengths, choice_sets, choice_set_lengths, choices = val_data
    index_to_node = {graph.nodes[node]['index']: node for node in graph.nodes}

    # Max indegree
    total = 0
    correct = 0
    for i in range(len(histories)):
        total += 1
        options = [index_to_node[index.item()] for index in choice_sets[i][:choice_set_lengths[i]]]

        max_indegree_node = max(graph.in_degree(options), key=lambda pair: pair[1])[0]
        if choice_sets[i][choices[i]] == graph.nodes[max_indegree_node]['index']:
            correct += 1

    print('Max indegree:', correct / total)

    # Max outdegree
    total = 0
    correct = 0
    for i in range(len(histories)):
        total += 1
        options = [index_to_node[index.item()] for index in choice_sets[i][:choice_set_lengths[i]]]

        max_outdegree_node = max(graph.out_degree(options), key=lambda pair: pair[1])[0]
        if choice_sets[i][choices[i]] == graph.nodes[max_outdegree_node]['index']:
            correct += 1

    print('Max outdegree:', correct / total)

    # Random
    total = 0
    correct = 0
    for i in range(len(histories)):
        total += 1

        if choice_sets[i][choices[i]] == random.choice(choice_sets[i][:choice_set_lengths[i]]):
            correct += 1

    print('Random:', correct / total)


def compile_choice_data(dataset):
    graph, train_data, val_data, test_data, means, stds = dataset.load_standardized()

    n_feats = dataset.num_features

    histories, history_lengths, choice_sets, choice_set_features, choice_set_lengths, choices = [
        torch.cat([train_data[i], val_data[i], test_data[i]]).numpy() for i in range(len(train_data))]

    chosen_item_features = [list() for _ in range(n_feats)]

    choice_set_mean_features = [list() for _ in range(n_feats)]

    for i in range(len(choice_set_features)):
        choice = choices[i]
        choice_set = choice_set_features[i, :choice_set_lengths[i]]

        for feature in range(n_feats):
            chosen_item_features[feature].append(choice_set[choice, feature])
            choice_set_mean_features[feature].append(np.mean(choice_set[:, feature]))

    return histories, history_lengths, choice_sets, choice_set_features, choice_set_lengths, choices, \
        np.array(chosen_item_features), np.array(choice_set_mean_features)


def learn_binned_mnl(dataset):
    print(f'Learning binned MNLs for {dataset.name}')
    num_bins = 100

    n_feats = dataset.num_features

    histories, history_lengths, choice_sets, choice_set_features, choice_set_lengths, choices, \
        chosen_item_features, choice_set_mean_features = compile_choice_data(dataset)

    data = [None for _ in range(n_feats)]

    for i, x_var in enumerate(choice_set_mean_features):
        # First three feats are log feats, so need to treat differently
        x_min = min([x for x in x_var if x > 0]) * 0.8 if i < 3 else min(x_var) * 0.8
        x_max = max(x_var) * 1.2

        values, bins = np.histogram(x_var, bins=np.logspace(np.log(x_min), np.log(x_max), num_bins) if i < 3 else np.linspace(x_min, x_max, num_bins))

        all_bin_idx = np.digitize(x_var, bins)

        mnl_utilities = np.zeros((num_bins, n_feats))
        bin_counts = np.zeros(num_bins)
        bin_choice_set_log_lengths = np.zeros(num_bins)
        bin_losses = np.zeros(num_bins)

        for bin in tqdm(range(num_bins)):
            bin_idx = all_bin_idx == bin

            bin_counts[bin] = np.count_nonzero(bin_idx)

            if bin_counts[bin] == 0:
                continue

            bin_data = [torch.tensor(choice_set_features[bin_idx]), torch.tensor(choice_set_lengths[bin_idx]), torch.tensor(choices[bin_idx])]
            mnl, train_losses, _, _, _ = train_feature_mnl(bin_data, bin_data, n_feats, lr=0.01, weight_decay=0.001)
            mnl_utilities[bin] = mnl.weights.detach().numpy()
            bin_choice_set_log_lengths[bin] = np.mean(np.log(choice_set_lengths[bin_idx]))
            bin_losses[bin] = torch.nn.functional.nll_loss(mnl(*bin_data[:-1]), bin_data[-1], reduction='sum').item()

        data[i] = bins, mnl_utilities, bin_counts, bin_choice_set_log_lengths, bin_losses

    with open(f'{dataset.name}_binned_mnl_params.pickle', 'wb') as f:
        pickle.dump(data, f)


def run_likelihood_ratio_test(dataset, lr, wd):
    torch.random.manual_seed(0)
    np.random.seed(0)
    model = run_feature_model_full_dataset(FeatureMNL, dataset, lr, wd)
    print(model.weights)

    torch.random.manual_seed(0)
    np.random.seed(0)

    model = run_feature_model_full_dataset(FeatureContextMixture, dataset, lr, wd)
    print(model.weights, model.intercepts, model.slopes)

    torch.random.manual_seed(0)
    np.random.seed(0)
    model = run_feature_model_full_dataset(FeatureCDM, dataset, lr, wd)
    print(model.weights, model.contexts)


def train_context_mixture_em(dataset):
    torch.set_num_threads(30)

    print('Running EM for', dataset.name)
    graph, train_data, val_data, test_data, means, stds = dataset.load_standardized()
    all_data = [torch.cat([train_data[i], val_data[i], test_data[i]]) for i in range(3, len(train_data))]

    model = context_mixture_em(all_data, dataset.num_features)
    torch.save(model.state_dict(), f'context_mixture_em_{dataset.name}_params.pt')


if __name__ == '__main__':
    learning_rate = 0.005
    weight_decay = 0.001

    for dataset in [SyntheticCDMDataset, SyntheticMNLDataset,
                    WikiTalkDataset, RedditHyperlinkDataset,
                    BitcoinAlphaDataset, BitcoinOTCDataset,
                    SMSADataset, SMSBDataset, SMSCDataset,
                    EmailEnronDataset, EmailEUDataset, EmailW3CDataset,
                    FacebookWallDataset, CollegeMsgDataset, MathOverflowDataset
                    ]:

        train_context_mixture_em(dataset)

        run_likelihood_ratio_test(dataset, learning_rate, weight_decay)

        torch.random.manual_seed(0)
        np.random.seed(0)
        run_feature_model_train_data(FeatureMNL, dataset, learning_rate, weight_decay)

        torch.random.manual_seed(0)
        np.random.seed(0)
        run_feature_model_train_data(FeatureCDM, dataset, learning_rate, weight_decay)

        torch.random.manual_seed(0)
        np.random.seed(0)
        run_feature_model_train_data(FeatureContextMixture, dataset, learning_rate, weight_decay)
        learn_binned_mnl(dataset)




