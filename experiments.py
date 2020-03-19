import collections
import os
import pickle

import choix
import numpy as np
import torch

import networkx as nx

from models import train_history_cdm, train_lstm, HistoryCDM, DataLoader, LSTM


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


def test_wikispeedia(param_fname, dim, loaded_data=None):
    if loaded_data is None:
        graph, train_data, val_data, test_data = load_wikispeedia()
    else:
        graph, train_data, val_data, test_data = loaded_data

    n = len(graph.nodes)

    model = HistoryCDM(n, dim, 0.5)
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


def baseline_wikispeedia():
    graph, train_data, val_data, test_data = load_wikispeedia()

    n = len(graph.nodes)
    traffic_in = np.zeros(n)
    traffic_out = np.zeros(n)

    index_map = {graph.nodes[node]['index']: node for node in graph.nodes}

    histories, history_lengths, choice_sets, choice_set_lengths, choices = train_data
    transitions = np.zeros((n, n))
    for i in range(len(histories)):
        # print(f'History:{[index_map[node.item()] for node in histories[i, :history_lengths[i].item()]]}')
        #
        # print(f'EDGE:{index_map[histories[i][0].item()]};{index_map[choice_sets[i][choices[i]].item()]}')

        transitions[histories[i][0], choice_sets[i][choices[i]]] += 1

    traffic_in = transitions.sum(axis=0)
    traffic_out = transitions.sum(axis=1)
    params = choix.choicerank(graph, traffic_in, traffic_out)

    histories, history_lengths, choice_sets, choice_set_lengths, choices = val_data

    correct = 0
    total = 0
    mrr = 0
    mean_rank = 0
    for i in range(len(histories)):
        choice_set = choice_sets[i, :choice_set_lengths[i]]
        probs = choix.probabilities(choice_set, params)
        total += 1

        if np.argmax(probs) == choices[i].item():
            correct += 1

        rank = (torch.argsort(torch.tensor(probs), descending=True) == choices[i]).nonzero() + 1
        mrr += 1 / rank.item()
        mean_rank += rank.item()

    print('ChoiceRank')
    print(f'Accuracy: {correct / total}')
    print(f'Mean rank: {mean_rank / total}')
    print(f'Mean reciprocal rank: {mrr / total}')

    correct = 0
    total = 0
    for i in range(len(histories)):
        pred = np.random.randint(0, choice_set_lengths[i])
        total += 1

        if pred == choices[i].item():
            correct += 1

    print('Random baseline:')
    print(f'Accuracy: {correct / total}')

    for i in range(len(histories)):
        choice_set = choice_sets[i, :choice_set_lengths[i]]
        transition_counts = [transitions[histories[i][0], node] for node in choice_set]
        total += 1

        if np.argmax(transition_counts) == choices[i].item():
            correct += 1

        rank = (torch.argsort(torch.tensor(transition_counts), descending=True) == choices[i]).nonzero() + 1
        mrr += 1 / rank.item()
        mean_rank += rank.item()

    print('Pick-most-frequent baseline')
    print(f'Accuracy: {correct / total}')
    print(f'Mean rank: {mean_rank / total}')
    print(f'Mean reciprocal rank: {mrr / total}')



if __name__ == '__main__':
    grid_search_wikispeedia()