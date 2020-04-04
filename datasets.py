import collections
import itertools
import os
import pickle

import numpy as np
import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

import networkx as nx
import pandas as pd
from tqdm import tqdm

DATA_DIR = 'data'


class Dataset(ABC):

    name = ''

    @classmethod
    def load(cls):
        pickle_file = f'{DATA_DIR}/{cls.name}.pickle'

        if not os.path.isfile(pickle_file):
            cls.load_into_pickle(pickle_file)

        with open(pickle_file, 'rb') as f:
            return pickle.load(f)

    @classmethod
    @abstractmethod
    def load_into_pickle(cls, file_name):
        pass

    @classmethod
    def data_split(cls, m, *tensors, train_frac=0.6, val_frac=0.2):
        """
        Split the given data (with m samples) into train, validation, and test sets 
        :param m: number of samples
        :param tensors: a sequence of data tensors
        :param train_frac: the fraction of data to allocate for training
        :param val_frac: the fraction of data to allocate for validation. The rest is used for testing
        :return: three lists of tensors for 1. training, 2. validation, and 3. testing
        """
        idx = list(range(m))
        np.random.shuffle(idx)
        train_end = int(m * train_frac)
        val_end = train_end + int(m * val_frac)

        return [data[idx[:train_end]] for data in tensors], \
               [data[idx[train_end:val_end]] for data in tensors], \
               [data[idx[val_end:]] for data in tensors]

    @classmethod
    def print_stats(cls):
        graph, train_data, val_data, test_data = cls.load()

        data = [torch.cat([train_data[i], val_data[i], test_data[i]]).numpy() for i in range(len(train_data))]
        histories, history_lengths, choice_sets, choice_set_lengths, choices = data

        print(f'Stats for {cls.name} dataset:')
        print(f'\tNodes: {len(graph.nodes)}')
        print(f'\tEdges: {len(graph.edges)}')
        print(f'\tLongest Path: {max(history_lengths) + 1}')
        print(f'\tLargest Choice Set: {max(choice_set_lengths)}')

    @classmethod
    def build_data_from_paths(cls, paths, graph):
        largest_choice_set = max(graph.out_degree(), key=lambda x: x[1])[1]
        longest_path = max(len(path) for path in paths)
        n = len(graph.nodes)

        choice_sets = []
        choice_set_lengths = []
        choices = []
        histories = []
        history_lengths = []

        for path in tqdm(paths):
            for i in range(len(path) - 1):
                neighbors = [graph.nodes[node]['index'] for node in graph.neighbors(path[i])]
                choice_sets.append(neighbors + [n] * (largest_choice_set - len(neighbors)))
                if len(choice_sets[-1]) != largest_choice_set:
                    print(len(neighbors), len(choice_sets[-1]))
                choice_set_lengths.append(len(neighbors))

                choices.append(choice_sets[-1].index(graph.nodes[path[i + 1]]['index']))

                histories.append([graph.nodes[node]['index'] for node in path[i::-1]] + [n] * (longest_path - i - 1))
                history_lengths.append(i + 1)

        histories = torch.tensor(histories)
        history_lengths = torch.tensor(history_lengths)
        choice_sets = torch.tensor(choice_sets)
        choice_set_lengths = torch.tensor(choice_set_lengths)
        choices = torch.tensor(choices)

        return histories, history_lengths, choice_sets, choice_set_lengths, choices


class WikispeediaDataset(Dataset):

    name = 'wikispeedia'

    @classmethod
    def _remove_back(cls, path):
        i = 0
        while i < len(path):
            if path[i] == '<':
                path.pop(i)
                path.pop(i - 1)
                i -= 1
            else:
                i += 1

    @classmethod
    def load_into_pickle(cls, file_name):
        graph = nx.read_edgelist(f'{DATA_DIR}/wikispeedia_paths-and-graph/links.tsv', create_using=nx.DiGraph)
        graph.add_node('Wikipedia_Text_of_the_GNU_Free_Documentation_License')

        df = pd.read_csv(f'{DATA_DIR}/wikispeedia_paths-and-graph/paths_finished.tsv', sep='\t', comment='#',
                         names=['hashedIpAddress', 'timestamp', 'durationInSec', 'path', 'rating'])

        # Edges not in graph but in paths:
        graph.add_edge('Bird', 'Wikipedia_Text_of_the_GNU_Free_Documentation_License')
        graph.add_edge('Finland', '%C3%85land')
        graph.add_edge('Republic_of_Ireland', '%C3%89ire')
        graph.add_edge('Claude_Monet', '%C3%89douard_Manet')

        paths = []
        for path in df['path']:
            split_path = path.split(';')
            cls._remove_back(split_path)
            paths.append(split_path)

        paths = [path for path in paths if len(path) <= 20]

        # Index nodes
        nodes = []
        for i, node in enumerate(graph.nodes):
            nodes.append(node)
            graph.nodes[node]['index'] = i

        histories, history_lengths, choice_sets, choice_set_lengths, choices = cls.build_data_from_paths(paths, graph)
        m = len(histories)
        print('Samples:', m)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_set_lengths, choices)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f)


class YoochooseDataset(Dataset):
    name = 'yoochoose'

    @classmethod
    def load_into_pickle(cls, file_name):
        yoochoose_clicks = np.loadtxt(f'{DATA_DIR}/recsys-challenge-2015/yoochoose-clicks-abbrev.dat', dtype=int)

        paths = []
        current_user = -1
        current_path = []
        for user, item in tqdm(yoochoose_clicks):
            if user == current_user:
                current_path.append(item)
            else:
                paths.append(current_path)
                current_user = user
                current_path = [item]
        paths.append(current_path)

        print('Initial paths', len(paths))
        paths = [path for path in paths if len(path) <= 50]
        print('Long paths removed', len(paths))

        paths = [path for path in paths if len(path) >= 3]
        print('Short paths removed', len(paths))

        edges = [(path[i], path[i + 1]) for path in paths for i in range(len(path) - 1)]
        graph = nx.DiGraph(edges)

        to_remove = set()
        for node, degree in graph.out_degree():
            if degree > 500:
                to_remove.add(node)

        print('Removing', len(to_remove), 'nodes')

        paths = [path for path in paths if all(node not in to_remove for node in path)]

        print('Degree selected paths', len(paths))

        page_counts = np.bincount(list(itertools.chain.from_iterable(paths)))

        len_paths = None
        while len_paths != len(paths):
            len_paths = len(paths)
            paths = [path for path in paths if all(page_counts[page] >= 25 for page in path)]
            page_counts = np.bincount(list(itertools.chain.from_iterable(paths)))
            print('Count selected paths', len(paths))

        edges = [(path[i], path[i + 1]) for path in paths for i in range(len(path) - 1)]

        graph = nx.DiGraph(edges)

        nodes = []
        for i, node in enumerate(graph.nodes):
            nodes.append(node)
            graph.nodes[node]['index'] = i

        largest_choice_set = max(graph.out_degree(), key=lambda x: x[1])[1]
        longest_path = max(len(path) for path in paths)

        print('Largest choice set', largest_choice_set)
        print('Longest path', longest_path)

        histories, history_lengths, choice_sets, choice_set_lengths, choices = cls.build_data_from_paths(paths, graph)
        m = len(histories)
        print('Samples:', m)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_set_lengths,
                                                         choices)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class KosarakDataset(Dataset):
    name = 'kosarak'

    @classmethod
    def load_into_pickle(cls, file_name):

        with open(f'{DATA_DIR}/uchoice-Kosarak/uchoice-Kosarak.txt', 'rb') as f:
            paths = [list(map(int, line.split())) for line in f.readlines()]

        print('Initial paths', len(paths))
        paths = [path for path in paths if len(path) <= 50]
        print('Long paths remove', len(paths))

        paths = [path for path in paths if len(path) >= 3]
        print('Short paths removed', len(paths))

        edges = [(path[i], path[i+1]) for path in paths for i in range(len(path) - 1)]
        graph = nx.DiGraph(edges)

        to_remove = set()
        for node, degree in graph.out_degree():
            if degree > 6000:
                to_remove.add(node)

        print('Removing', len(to_remove), 'nodes')

        paths = [path for path in paths if all(node not in to_remove for node in path)]

        print('Degree selected paths', len(paths))

        page_counts = np.bincount(list(itertools.chain.from_iterable(paths)))

        len_paths = None
        while len_paths != len(paths):
            len_paths = len(paths)
            paths = [path for path in paths if all(page_counts[page] >= 2 for page in path)]
            page_counts = np.bincount(list(itertools.chain.from_iterable(paths)))
            print('Count selected paths', len(paths))

        edges = [(path[i], path[i + 1]) for path in paths for i in range(len(path) - 1)]
        graph = nx.DiGraph(edges)

        nodes = []
        for i, node in enumerate(graph.nodes):
            nodes.append(node)
            graph.nodes[node]['index'] = i

        largest_choice_set = max(graph.out_degree(), key=lambda x: x[1])[1]
        longest_path = max(len(path) for path in paths)

        print('Largest choice set', largest_choice_set)
        print('Longest path', longest_path)

        histories, history_lengths, choice_sets, choice_set_lengths, choices = cls.build_data_from_paths(paths, graph)
        m = len(histories)
        print('Samples:', m)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_set_lengths,
                                                         choices)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class LastFMGenreDataset(Dataset):
    name = 'lastfm-genre'

    @classmethod
    def load_into_pickle(cls, file_name):

        with open(f'{DATA_DIR}/uchoice-Lastfm-Genres/uchoice-Lastfm-Genres.txt', 'rb') as f:
            paths = [list(map(int, line.split())) for line in f.readlines()]

        print('Initial paths', len(paths))
        paths = [path for path in paths if len(path) <= 50]
        print('Long paths remove', len(paths))

        paths = [path for path in paths if len(path) >= 3]
        print('Short paths removed', len(paths))

        edges = [(path[i], path[i+1]) for path in paths for i in range(len(path) - 1)]
        graph = nx.DiGraph(edges)

        # to_remove = set()
        # for node, degree in graph.out_degree():
        #     if degree > 6000:
        #         to_remove.add(node)
        #
        # print('Removing', len(to_remove), 'nodes')
        #
        # paths = [path for path in paths if all(node not in to_remove for node in path)]
        #
        # print('Degree selected paths', len(paths))

        page_counts = np.bincount(list(itertools.chain.from_iterable(paths)))

        len_paths = None
        while len_paths != len(paths):
            len_paths = len(paths)
            paths = [path for path in paths if all(page_counts[page] >= 250 for page in path)]
            page_counts = np.bincount(list(itertools.chain.from_iterable(paths)))
            print('Count selected paths', len(paths))

        edges = [(path[i], path[i + 1]) for path in paths for i in range(len(path) - 1)]
        graph = nx.DiGraph(edges)

        nodes = []
        for i, node in enumerate(graph.nodes):
            nodes.append(node)
            graph.nodes[node]['index'] = i

        largest_choice_set = max(graph.out_degree(), key=lambda x: x[1])[1]
        longest_path = max(len(path) for path in paths)

        print('Largest choice set', largest_choice_set)
        print('Longest path', longest_path)
        print('Nodes', len(graph.nodes))
        print('Edges', len(graph.edges))

        histories, history_lengths, choice_sets, choice_set_lengths, choices = cls.build_data_from_paths(paths, graph)
        m = len(histories)
        print('Samples:', m)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_set_lengths,
                                                         choices)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


if __name__ == '__main__':
    graph, train, val, test = WikispeediaDataset.load()

    print('Nodes', len(graph.nodes))
    print('Edges', len(graph.edges))
    for histories, history_lengths, choice_sets, choice_set_lengths, choices in train, val, test:
        assert len(histories) == len(history_lengths) == len(choice_sets) == len(choice_set_lengths) == len(choices)
        print(len(choices))
