import pickle

import choix
import numpy as np
import torch

from datasets import WikispeediaDataset, KosarakDataset, YoochooseDataset, LastFMGenreDataset
from models import train_history_cdm, train_lstm, train_history_mnl


def run_history_cdm(dataset, dim, lr, wd, beta, learn_beta):
    graph, train_data, val_data, test_data = dataset.load()

    print(f'Training History CDM on {dataset.name} (dim={dim}, lr={lr}, wd={wd}, beta={beta}, learn_beta={learn_beta})')
    histories, history_lengths, choice_sets, choice_set_lengths, choices = train_data
    model, losses = train_history_cdm(len(graph.nodes), histories, history_lengths, choice_sets, choice_set_lengths,
                                      choices, dim=dim, lr=lr, weight_decay=wd, beta=beta, learn_beta=learn_beta)
    torch.save(model.state_dict(), f'history_cdm_{dataset.name}_params_{dim}_{lr}_{wd}_{beta}_{learn_beta}.pt')
    with open(f'history_cdm_{dataset.name}_losses_{dim}_{lr}_{wd}_{beta}_{learn_beta}.pickle', 'wb') as f:
        pickle.dump(losses, f)


def run_history_mnl(dataset, dim, lr, wd, beta, learn_beta):
    graph, train_data, val_data, test_data = dataset.load()

    print(f'Training History MNL on {dataset.name} (dim={dim}, lr={lr}, wd={wd}, beta={beta}, learn_beta={learn_beta})')
    model, losses = train_history_mnl(len(graph.nodes), *train_data, dim=dim, lr=lr, weight_decay=wd, beta=beta, learn_beta=learn_beta)
    torch.save(model.state_dict(), f'history_mnl_{dataset.name}_params_{dim}_{lr}_{wd}_{beta}_{learn_beta}.pt')
    with open(f'history_mnl_{dataset.name}_losses_{dim}_{lr}_{wd}_{beta}_{learn_beta}.pickle', 'wb') as f:
        pickle.dump(losses, f)


def run_lstm(dataset, dim, lr, wd):
    graph, train_data, val_data, test_data = dataset.load()

    print(f'Training LSTM on {dataset.name} (dim={dim}, lr={lr}, wd={wd})')
    model, losses = train_lstm(len(graph.nodes), *train_data, dim=dim, lr=lr, weight_decay=wd)
    torch.save(model.state_dict(), f'lstm_{dataset.name}_params_{dim}_{lr}_{wd}.pt')
    with open(f'lstm_{dataset.name}_losses_{dim}_{lr}_{wd}.pickle', 'wb') as f:
        pickle.dump(losses, f)


def compare_methods(dataset):
    run_history_cdm(dataset, 64, 0.005, 0, 0.5, True)
    run_history_mnl(dataset, 64, 0.005, 0, 0.5, True)
    run_lstm(dataset, 64, 0.005, 0)


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
    # for dataset in (WikispeediaDataset, KosarakDataset, YoochooseDataset, LastFMGenreDataset):
    #     compare_methods(dataset)

    run_history_cdm(WikispeediaDataset, 64, 0.005, 0, 0.5, True)
