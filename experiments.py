import pickle

import choix
import numpy as np
import torch

from datasets import WikispeediaDataset, KosarakDataset, YoochooseDataset, LastFMGenreDataset
from models import train_history_cdm, train_lstm, train_history_mnl, HistoryCDM, HistoryMNL, LSTM

training_methods = {HistoryCDM: train_history_cdm, HistoryMNL: train_history_mnl, LSTM: train_lstm}


def run_model(method, dataset, dim, lr, wd, beta=None, learn_beta=None):
    graph, train_data, val_data, test_data = dataset.load()

    print(f'Training {method.name} on {dataset.name} (dim={dim}, lr={lr}, wd={wd}, beta={beta}, learn_beta={learn_beta})')

    model, train_losses, train_accs, val_losses, val_accs = training_methods[method](len(graph.nodes), train_data,
                                                                                     val_data, dim=dim, lr=lr, weight_decay=wd,
                                                                                     beta=beta, learn_beta=learn_beta)
    torch.save(model.state_dict(), f'{method.name}_{dataset.name}_params_{dim}_{lr}_{wd}_{beta}_{learn_beta}.pt')
    with open(f'{method.name}_{dataset.name}_losses_{dim}_{lr}_{wd}_{beta}_{learn_beta}.pickle', 'wb') as f:
        pickle.dump((train_losses, train_accs, val_losses, val_accs), f)


def compare_methods(dataset):
    run_model(HistoryCDM, dataset, 64, 0.005, 0, 0.5, True)
    run_model(HistoryMNL, dataset, 64, 0.005, 0, 0.5, True)
    run_model(LSTM, dataset, 64, 0.005, 0)


def grid_search(dataset):
    for lr in [0.0001, 0.005, 0.01]:
        for wd in [0, 1e-5, 1e-3]:
            run_model(HistoryCDM, dataset, 64, lr=lr, wd=wd, beta=0.5, learn_beta=True)
            run_model(HistoryMNL, dataset, 64, lr=lr, wd=wd, beta=0.5, learn_beta=True)
            run_model(LSTM, dataset, 64, lr=lr, wd=wd)


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


if __name__ == '__main__':
    for dataset in (WikispeediaDataset, KosarakDataset, YoochooseDataset, LastFMGenreDataset):
        grid_search(dataset)

