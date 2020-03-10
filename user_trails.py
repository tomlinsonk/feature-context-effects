import torch
from torch import nn
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm


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


class HistoryCDM(torch.nn.Module):
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

    train_history_cdm(n, histories, history_lengths, choice_sets, choice_set_lengths, choices)


def train_history_cdm(n, histories, history_lengths, choice_sets, choice_set_lengths, choices):

    model = HistoryCDM(n, 64, 0.8)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=True, weight_decay=1e-4)
    for t in tqdm(range(500)):
        choice_pred = model(histories, history_lengths, choice_sets, choice_set_lengths)

        loss = model.loss(choice_pred, choices)
        if t % 100 == 99:
            print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


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

    print('Max out-degree:', max(graph.out_degree()))

    counter = collections.Counter([len(path) for path in paths])

    print('total:', sum([counter[i] for i in range(max(counter)+1)]))
    print('20 or fewer:', sum([counter[i] for i in range(21)]))

    # plt.plot(range(max_count+1), [counter[i] for i in range(max_count+1)])
    #
    # plt.yscale('symlog', linthreshy=1)
    # plt.show()

    # Index nodes
    nodes = []
    for i, node in enumerate(graph.nodes):
        nodes.append(node)
        graph.nodes[node]['index'] = i

    n = len(graph.nodes)
    longest_path = 20
    largest_choice_set = max(graph.out_degree(), key=lambda x: x[1])[1]

    print(largest_choice_set)

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

    model = train_history_cdm(n, histories, history_lengths, choice_sets, choice_set_lengths, choices)
    torch.save(model.state_dict(), 'wikispeedia_params.pt')


if __name__ == '__main__':
    load_wikispeedia()
