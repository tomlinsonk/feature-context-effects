import glob
import os
import pickle

import torch
from torch import nn
import numpy as np
import networkx as nx
import pandas as pd
# import matplotlib.pyplot as plt
import collections
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import choix


class DataLoader:
    """
    Simplified, faster DataLoader.
    From https://github.com/arjunsesh/cdm-icml with minor tweaks.
    """
    def __init__(self, data, batch_size=None, shuffle=False, sort_batch=False, sort_index=None):
        self.data = data
        self.data_size = data[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.counter = 0
        self.stop_iteration = False
        self.sort_batch = sort_batch
        self.sort_index = sort_index

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop_iteration:
            self.stop_iteration = False
            raise StopIteration()

        if self.batch_size is None or self.batch_size == self.data_size:
            self.stop_iteration = True
            return self.data
        else:
            i = self.counter
            bs = self.batch_size
            self.counter += 1
            batch = [item[i * bs:(i + 1) * bs] for item in self.data]
            if self.counter * bs >= self.data_size:
                self.counter = 0
                self.stop_iteration = True
                if self.shuffle:
                    random_idx = np.arange(self.data_size)
                    np.random.shuffle(random_idx)
                    self.data = [item[random_idx] for item in self.data]

            if self.sort_batch:
                perm = torch.argsort(batch[self.sort_index], dim=0, descending=True)
                batch = [item[perm] for item in batch]

            return batch


class Embedding(nn.Module):
    """
    Add zero-ed out dimension to Embedding for the padding index.
    From https://github.com/arjunsesh/cdm-icml with minor tweaks.
    """

    def __init__(self, num, dim, pad_idx=None):
        super().__init__()
        self.num = num
        self.dim = dim
        self.pad_idx = pad_idx

        self.weight = nn.Parameter(torch.randn([self.num, self.dim]))

        with torch.no_grad():
            self.weight[self.pad_idx].fill_(0)

    def forward(self, x):
        with torch.no_grad():
            self.weight[self.pad_idx].fill_(0)

        return self.weight[x]


class HistoryCDM(nn.Module):
    def __init__(self, num_items, dim, beta):
        super().__init__()

        self.num_items = num_items
        self.dim = dim
        self.beta = beta

        self.history_embedding = Embedding(
            num=self.num_items + 1,
            dim=self.dim,
            pad_idx=self.num_items
        )

        self.target_embedding = Embedding(
            num=self.num_items + 1,
            dim=self.dim,
            pad_idx=self.num_items
        )

        self.context_embedding = Embedding(
            num=self.num_items + 1,
            dim=self.dim,
            pad_idx=self.num_items
        )

    def forward(self, histories, history_lengths, choice_sets, choice_set_lengths):
        batch_size, max_choice_set_len = choice_sets.size()
        _, max_history_len = histories.size()

        history_vecs = self.history_embedding(histories)
        context_vecs = self.context_embedding(choice_sets)
        target_vecs = self.target_embedding(choice_sets)

        context_sums = context_vecs.sum(1, keepdim=True) - context_vecs
        history_weight = torch.pow(torch.full([max_history_len], self.beta), torch.arange(0, max_history_len))[:, None]
        weighted_history_sums = (history_weight * history_vecs).sum(1, keepdim=True)

        utilities = (target_vecs * (context_sums + weighted_history_sums)).sum(2)
        utilities[torch.arange(max_choice_set_len)[None, :] >= choice_set_lengths[:, None]] = -np.inf

        return nn.functional.log_softmax(utilities, 1)

    def loss(self, y_pred, y):
        """
        The error in inferred log-probabilities given observations
        :param y_pred: log(choice probabilities)
        :param y: observed choices
        :return: the loss
        """

        return nn.functional.nll_loss(y_pred, y)


class LSTM(nn.Module):

    def __init__(self, num_items, dim):
        super().__init__()

        self.num_items = num_items
        self.dim = dim

        self.item_embedding = Embedding(
            num=self.num_items + 1,
            dim=self.dim,
            pad_idx=self.num_items
        )

        self.lstm = nn.LSTM(dim, dim, batch_first=True)

    def forward(self, histories, history_lengths, choice_sets, choice_set_lengths):
        batch_size, max_choice_set_len = choice_sets.size()
        _, max_history_len = histories.size()

        history_vecs = self.item_embedding(histories)
        packed_histories = pack_padded_sequence(history_vecs, history_lengths, batch_first=True)
        packed_output, (h_n, c_n) = self.lstm(packed_histories)

        choice_set_vecs = self.item_embedding(choice_sets)

        utilities = (choice_set_vecs * h_n[-1][:, None, :]).sum(2)
        utilities[torch.arange(max_choice_set_len)[None, :] >= choice_set_lengths[:, None]] = -np.inf

        return nn.functional.log_softmax(utilities, 1)

    def loss(self, y_pred, y):
        """
        The error in inferred log-probabilities given observations
        :param y_pred: log(choice probabilities)
        :param y: observed choices
        :return: the loss
        """

        return nn.functional.nll_loss(y_pred, y)


def toy_example():
    n = 4
    histories = torch.tensor(
        [[0, n, n, n],
         [3, 0, n, n],
         [1, n, n, n],
         [2, 1, n, n],
         [3, 2, 1, n]]
    )

    history_lengths = torch.tensor([1, 2, 1, 2, 3])

    choice_sets = torch.tensor(
        [[3, 1, n, n],
         [2, 0, n, n],
         [2, 1, 3, 0],
         [1, 3, n, n],
         [0, 1, 2, n]]
    )

    choice_set_lengths = torch.tensor([2, 2, 4, 2, 3])

    # Indices into choice_sets
    choices = torch.tensor([0, 0, 0, 1, 2])

    model, losses = train_lstm(n, histories, history_lengths, choice_sets, choice_set_lengths, choices, dim=3, lr=0.01)
    plt.plot(range(500), losses)
    plt.show()


def train_history_cdm(n, histories, history_lengths, choice_sets, choice_set_lengths, choices, dim=64, beta=0.5, lr=1e-4, weight_decay=1e-4):

    model = HistoryCDM(n, dim, beta)
    data_loader = DataLoader((histories, history_lengths, choice_sets, choice_set_lengths, choices),
                             batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay)

    losses = []
    for epoch in tqdm(range(500)):
        total_loss = 0
        count = 0
        for histories, history_lengths, choice_sets, choice_set_lengths, choices in data_loader:
            model.train()
            choice_pred = model(histories, history_lengths, choice_sets, choice_set_lengths)
            loss = model.loss(choice_pred, choices)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            total_loss += loss.item()
            count += 1

        total_loss /= count
        losses.append(total_loss)

    return model, losses


def train_lstm(n, histories, history_lengths, choice_sets, choice_set_lengths, choices, dim=64, lr=1e-4, weight_decay=1e-4):
    # Reverse histories
    for i in range(histories.size(0)):
        histories[i, :history_lengths[i]] = histories[i, :history_lengths[i]].flip(0)

    model = LSTM(n, dim)

    data_loader = DataLoader((histories, history_lengths, choice_sets, choice_set_lengths, choices),
                             batch_size=128, shuffle=True, sort_batch=True, sort_index=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay)

    losses = []
    for epoch in tqdm(range(500)):
        total_loss = 0
        count = 0
        for histories, history_lengths, choice_sets, choice_set_lengths, choices in data_loader:
            model.train()
            choice_pred = model(histories, history_lengths, choice_sets, choice_set_lengths)
            loss = model.loss(choice_pred, choices)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            total_loss += loss.item()
            count += 1

        total_loss /= count
        losses.append(total_loss)

    return model, losses


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

    if os.path.isfile('wikispeedia_data.pickle'):
        print('Loading parsed Wikispeedia data from wikispeedia_data.pickle...')
        with open('wikispeedia_data.pickle', 'rb') as f:
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

    with open('wikispeedia_data.pickle', 'wb') as f:
        pickle.dump((graph, train_data, val_data, test_data), f)

    return graph, train_data, val_data, test_data


def grid_search_wikispeedia():
    dims = [16, 64, 128]
    lrs = [0.005, 0.1, 0.0001]
    wds = [0, 0.0001, 0.1]

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


def test_wikispeedia(param_fname, loaded_data=None):
    if loaded_data is None:
        graph, train_data, val_data, test_data = load_wikispeedia()
    else:
        graph, train_data, val_data, test_data = loaded_data

    n = len(graph.nodes)

    model = HistoryCDM(n, 16, 0.5)
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


def plot_loss(fname, axes, row, col):
    lrs = ['0.1', '0.005', '0.0001']
    wds = ['0.1', '0.0001', '0']

    loaded_data = load_wikispeedia()

    fig, axes = plt.subplots(3, 3, sharex='col')

    for fname in glob.glob('wikispeedia_losses_16*'):
        fname_split = fname.split('_')
        lr = fname_split[3]
        wd = fname_split[4].replace('.pickle', '')

        row = lrs.index(lr)
        col = wds.index(wd)

        print(lr, wd, row, col)

        print(fname)
        param_fname = fname.replace('.pickle', '.pt').replace('losses', 'params')
        acc, mean_rank, mrr = test_wikispeedia(param_fname, loaded_data)
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
    plt.savefig('grid_search_16_dim.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # n, histories, history_lengths, choice_sets, choice_set_lengths, choices, graph = load_wikispeedia()
    #
    # model, losses = train_history_cdm(n, histories, history_lengths, choice_sets, choice_set_lengths, choices)
    # torch.save(model.state_dict(), 'wikispeedia_params.pt')
    # test_wikispeedia()
    # grid_search_wikispeedia()
    # baseline_wikispeedia()

    grid_search_wikispeedia_lstm()
