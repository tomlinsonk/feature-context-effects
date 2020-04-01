import collections
import os
import pickle

import choix
import networkx as nx
import numpy as np
import torch
import pandas as pd

from datasets import WikispeediaDataset, KosarakDataset, YoochooseDataset
from models import train_history_cdm, train_lstm, HistoryCDM, DataLoader, LSTM, train_history_mnl


def remove_back(path):
    i = 0
    while i < len(path):
        if path[i] == '<':
            path.pop(i)
            path.pop(i - 1)
            i -= 1
        else:
            i += 1


def load_wikispeedia():
    if os.path.isfile('data/wikispeedia_data.pickle'):
        print('Loading parsed Wikispeedia data from data/wikispeedia_data.pickle...')
        with open('data/wikispeedia_data.pickle', 'rb') as f:
            return pickle.load(f)

    print('Reloading Wikispeedia from raw data...')
    graph = nx.read_edgelist('data/wikispeedia_paths-and-graph/links.tsv', create_using=nx.DiGraph)
    graph.add_node('Wikipedia_Text_of_the_GNU_Free_Documentation_License')

    df = pd.read_csv('data/wikispeedia_paths-and-graph/paths_finished.tsv', sep='\t', comment='#',
                      names=['hashedIpAddress', 'timestamp', 'durationInSec', 'path', 'rating'])

    # Edges not in graph but in paths:
    graph.add_edge('Bird', 'Wikipedia_Text_of_the_GNU_Free_Documentation_License')
    graph.add_edge('Finland', '%C3%85land')
    graph.add_edge('Republic_of_Ireland', '%C3%89ire')
    graph.add_edge('Claude_Monet', '%C3%89douard_Manet')

    paths = []
    for path in df['path']:
        split_path = path.split(';')
        remove_back(split_path)
        paths.append(split_path)

    counter = collections.Counter([len(path) for path in paths])

    # Index nodes
    nodes = []
    for i, node in enumerate(graph.nodes):
        nodes.append(node)
        graph.nodes[node]['index'] = i

    n = len(graph.nodes)
    longest_path = 20
    largest_choice_set = max(graph.out_degree(), key=lambda x: x[1])[1]

    # print(largest_choice_set)

    choice_sets = []
    choice_set_lengths = []
    choices = []
    histories = []
    history_lengths = []

    for path in paths:
        if len(path) > longest_path:
            continue

        for i in range(len(path) - 1):
            neighbors = [graph.nodes[node]['index'] for node in graph.neighbors(path[i])]
            choice_sets.append(neighbors + [n] * (largest_choice_set - len(neighbors)))
            if len(choice_sets[-1]) != largest_choice_set:
                print(len(neighbors), len(choice_sets[-1]))
            choice_set_lengths.append(len(neighbors))

            choices.append(choice_sets[-1].index(graph.nodes[path[i + 1]]['index']))

            histories.append([graph.nodes[node]['index'] for node in path[i::-1]] + [n] * (longest_path - i - 1))
            history_lengths.append(i+1)

    histories = torch.tensor(histories)
    history_lengths = torch.tensor(history_lengths)
    choice_sets = torch.tensor(choice_sets)
    choice_set_lengths = torch.tensor(choice_set_lengths)
    choices = torch.tensor(choices)

    m = len(histories)
    print('num datapoints:', m)
    idx = list(range(m))

    np.random.shuffle(idx)

    train_end = int(m * 0.6)
    val_end = train_end + int(m * 0.2)

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    train_data = [data[train_idx] for data in (histories, history_lengths, choice_sets, choice_set_lengths, choices)]
    val_data = [data[val_idx] for data in (histories, history_lengths, choice_sets, choice_set_lengths, choices)]
    test_data = [data[test_idx] for data in (histories, history_lengths, choice_sets, choice_set_lengths, choices)]

    with open('data/wikispeedia_data.pickle', 'wb') as f:
        pickle.dump((graph, train_data, val_data, test_data), f)

    return graph, train_data, val_data, test_data


def grid_search_wikispeedia():
    dims = [16, 64, 128]
    lrs = [0.005, 0.01, 0.001]
    wds = [0, 0.000001, 0.0001]

    graph, train_data, val_data, test_data = load_wikispeedia()
    n = len(graph.nodes)

    for dim in dims:
        for lr in lrs:
            for wd in wds:
                print(f'Training dim {dim}, lr {lr}, wd {wd}...')
                model, losses = train_history_cdm(n, *train_data, dim=dim, lr=lr, weight_decay=wd)
                torch.save(model.state_dict(), f'wikispeedia_params_{dim}_{lr}_{wd}.pt')
                with open(f'wikispeedia_losses_{dim}_{lr}_{wd}.pickle', 'wb') as f:
                    pickle.dump(losses, f)


def beta_grid_search_wikispeedia():
    dims = [16, 64, 128]
    lr = 0.005
    wd = 0
    betas = [0, 0.5, 1]

    graph, train_data, val_data, test_data = load_wikispeedia()
    n = len(graph.nodes)

    for dim in dims:
        for beta in betas:
            print(f'Training dim {dim}, beta {beta}...')
            model, losses = train_history_cdm(n, *train_data, dim=dim, lr=lr, weight_decay=wd, learn_beta=False, beta=beta)
            torch.save(model.state_dict(), f'wikispeedia_beta_{beta}_params_{dim}_{lr}_{wd}.pt')
            with open(f'wikispeedia_beta_{beta}_losses_{dim}_{lr}_{wd}.pickle', 'wb') as f:
                pickle.dump(losses, f)


def mnl_beta_grid_search_wikispeedia():
    dims = [16, 64, 128]
    lr = 0.005
    wd = 0
    betas = [0, 0.5, 1]

    graph, train_data, val_data, test_data = load_wikispeedia()
    n = len(graph.nodes)

    for dim in dims:
        for beta in betas:
            print(f'Training history MNL dim {dim}, beta {beta}...')
            model, losses = train_history_mnl(n, *train_data, dim=dim, lr=lr, weight_decay=wd, learn_beta=False, beta=beta)
            torch.save(model.state_dict(), f'wikispeedia_mnl_beta_{beta}_params_{dim}_{lr}_{wd}.pt')
            with open(f'wikispeedia_mnl_beta_{beta}_losses_{dim}_{lr}_{wd}.pickle', 'wb') as f:
                pickle.dump(losses, f)


def learn_beta_wikispeedia():
    dims = [128, 16, 64]
    lr = 0.005
    wd = 0

    graph, train_data, val_data, test_data = load_wikispeedia()
    n = len(graph.nodes)

    for dim in dims:
        print(f'Training dim {dim}, learning beta...')
        model, losses = train_history_cdm(n, *train_data, dim=dim, lr=lr, weight_decay=wd, learn_beta=True)
        torch.save(model.state_dict(), f'wikispeedia_learn_beta_params_{dim}_{lr}_{wd}.pt')
        with open(f'wikispeedia_learn_beta_losses_{dim}_{lr}_{wd}.pickle', 'wb') as f:
            pickle.dump(losses, f)


def grid_search_wikispeedia_lstm():
    dims = [16, 64, 128]
    lrs = [0.005, 0.1, 0.0001]
    wds = [0, 0.0001, 0.1]

    graph, train_data, val_data, test_data = load_wikispeedia()
    n = len(graph.nodes)

    for dim in dims:
        for lr in lrs:
            for wd in wds:
                print(f'Training LSTM dim {dim}, lr {lr}, wd {wd}...')
                model, losses = train_lstm(n, *train_data, dim=dim, lr=lr, weight_decay=wd)
                torch.save(model.state_dict(), f'wikispeedia_lstm_params_{dim}_{lr}_{wd}.pt')
                with open(f'wikispeedia_lstm_losses_{dim}_{lr}_{wd}.pickle', 'wb') as f:
                    pickle.dump(losses, f)


def test_wikispeedia(param_fname, dim, loaded_data=None, Model=HistoryCDM):
    if loaded_data is None:
        graph, train_data, val_data, test_data = load_wikispeedia()
    else:
        graph, train_data, val_data, test_data = loaded_data

    n = len(graph.nodes)

    model = Model(n, dim, 0.5)
    model.load_state_dict(torch.load(param_fname))
    model.eval()

    data_loader = DataLoader(val_data, batch_size=128, shuffle=True)

    count = 0
    correct = 0
    mean_rank = 0
    mrr = 0
    total_loss = 0
    for histories, history_lengths, choice_sets, choice_set_lengths, choices in data_loader:
        choice_pred = model(histories, history_lengths, choice_sets, choice_set_lengths)

        ranks = (torch.argsort(choice_pred, dim=1, descending=True) == choices[:, None]).nonzero()[:, 1] + 1

        vals, idxs = choice_pred.max(1)
        mean_rank += ranks.sum().item() / 128
        mrr += (1 / ranks.float()).sum().item() / 128
        count += 1
        correct += (idxs == choices).long().sum().item() / 128

    return correct / count, mean_rank / count, mrr / count


def test_lstm_wikispeedia(param_fname, dim, loaded_data=None):
    if loaded_data is None:
        graph, train_data, val_data, test_data = load_wikispeedia()
    else:
        graph, train_data, val_data, test_data = loaded_data

    n = len(graph.nodes)

    model = LSTM(n, dim)
    model.load_state_dict(torch.load(param_fname))
    model.eval()

    data_loader = DataLoader(val_data, batch_size=128, shuffle=True, sort_batch=True, sort_index=1)

    count = 0
    correct = 0
    mean_rank = 0
    mrr = 0
    total_loss = 0
    for histories, history_lengths, choice_sets, choice_set_lengths, choices in data_loader:
        choice_pred = model(histories, history_lengths, choice_sets, choice_set_lengths)

        ranks = (torch.argsort(choice_pred, dim=1, descending=True) == choices[:, None]).nonzero()[:, 1] + 1

        vals, idxs = choice_pred.max(1)
        mean_rank += ranks.sum().item() / 128
        mrr += (1 / ranks.float()).sum().item() / 128
        count += 1
        correct += (idxs == choices).long().sum().item() / 128

    return correct / count, mean_rank / count, mrr / count


def run_history_cdm(dataset, dim, lr, wd, beta, learn_beta):
    graph, train_data, val_data, test_data = dataset.load()

    print(f'Training History CDM on {dataset.name} (dim={dim}, lr={lr}, wd={wd}, beta={beta}, learn_beta={learn_beta})')
    model, losses = train_history_cdm(len(graph.nodes), *train_data, dim=dim, lr=lr, weight_decay=wd, beta=beta, learn_beta=learn_beta)
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
    for dataset in (KosarakDataset, YoochooseDataset, WikispeediaDataset):
        compare_methods(dataset)
