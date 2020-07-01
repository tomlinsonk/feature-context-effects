import pickle
import random

import choix
import numpy as np
import torch
import matplotlib.pyplot as plt

from datasets import WikispeediaDataset, KosarakDataset, YoochooseDataset, LastFMGenreDataset, ORCIDSwitchDataset, \
    EmailEnronDataset, CollegeMsgDataset, EmailEUDataset, MathOverflowDataset, FacebookWallDataset, \
    EmailEnronCoreDataset, EmailW3CDataset, EmailW3CCoreDataset, SMSADataset, SMSBDataset, SMSCDataset
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
    graph, train_data, val_data, test_data = dataset.load()

    all_data = [torch.cat([train_data[i], val_data[i], test_data[i]]) for i in range(3, len(train_data))]

    print(f'Training {method.name} on {dataset.name} (lr={lr}, wd={wd})')

    model, train_losses, train_accs, val_losses, val_accs = training_methods[method](all_data, val_data, 3, lr=lr, weight_decay=wd, compute_val_stats=False)
    torch.save(model.state_dict(), f'{method.name}_{dataset.name}_params_{lr}_{wd}.pt')
    with open(f'{method.name}_{dataset.name}_losses_{lr}_{wd}.pickle', 'wb') as f:
        pickle.dump((train_losses, train_accs, val_losses, val_accs), f)

    return model


def run_feature_model_train_data(method, dataset, lr, wd):
    graph, train_data, val_data, test_data = dataset.load()

    context_mixture_em(train_data, 3)

    print(f'Training {method.name} on {dataset.name}, training data only (lr={lr}, wd={wd})')

    model, train_losses, train_accs, val_losses, val_accs = training_methods[method](train_data[3:], val_data[3:], 3, lr=lr, weight_decay=wd, compute_val_stats=True)
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


def run_likelihood_ratio_test(dataset, lr):
    torch.random.manual_seed(0)
    np.random.seed(0)

    model = run_feature_model_full_dataset(FeatureMNL, dataset, lr, 0.001)
    print(model.weights)

    torch.random.manual_seed(0)
    np.random.seed(0)

    model = run_feature_model_full_dataset(FeatureContextMixture, dataset, lr, 0.001)
    print(model.weights, model.intercepts, model.slopes)

    torch.random.manual_seed(0)
    np.random.seed(0)
    model = run_feature_model_full_dataset(FeatureCDM, dataset, lr, 0.001)
    print(model.weights, model.contexts)


if __name__ == '__main__':
    # run_likelihood_ratio_test(FacebookWallDataset, 0.005)
    # run_likelihood_ratio_test(EmailEnronDataset, 0.005)
    # run_likelihood_ratio_test(CollegeMsgDataset, 0.005)
    # run_likelihood_ratio_test(EmailEUDataset, 0.005)
    # run_likelihood_ratio_test(EmailW3CDataset, 0.005)
    # run_likelihood_ratio_test(MathOverflowDataset, 0.005)
    # run_likelihood_ratio_test(SMSADataset, 0.005)
    # run_likelihood_ratio_test(SMSBDataset, 0.005)
    # run_likelihood_ratio_test(SMSCDataset, 0.005)

    # for dataset in [FacebookWallDataset, EmailEnronDataset, EmailEUDataset, EmailW3CDataset, CollegeMsgDataset,
    #                 SMSADataset, SMSBDataset, SMSCDataset, MathOverflowDataset]:
    #
    #
    #
    #     torch.random.manual_seed(0)
    #     np.random.seed(0)
    #     run_feature_model_train_data(FeatureMNL, dataset, 0.005, 0.001)
    #
    #     torch.random.manual_seed(0)
    #     np.random.seed(0)
    #     run_feature_model_train_data(FeatureCDM, dataset, 0.005, 0.001)
    #
    #     torch.random.manual_seed(0)
    #     np.random.seed(0)
    #     run_feature_model_train_data(FeatureContextMixture, dataset, 0.005, 0.001)

    graph, train_data, val_data, test_data = EmailEnronDataset.load()

    all_data = [torch.cat([train_data[i], val_data[i], test_data[i]]) for i in range(len(train_data))]
    context_mixture_em(all_data, 3)



