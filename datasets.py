import collections
import itertools
import os
import pickle
import random

import numpy as np
import scipy
import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

import networkx as nx
import pandas as pd
from tqdm import tqdm

DATA_DIR = 'data'


class Dataset(ABC):

    name = ''
    num_features = 6

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
    def data_split(cls, m, *tensors, train_frac=0.6, val_frac=0.2, shuffle=True):
        """
        Split the given data (with m samples) into train, validation, and test sets 
        :param m: number of samples
        :param tensors: a sequence of data tensors
        :param train_frac: the fraction of data to allocate for training
        :param val_frac: the fraction of data to allocate for validation. The rest is used for testing
        :param shuffle: if true, shuffle data. Otherwise take first chunk as train, then val, then test
        :return: three lists of tensors for 1. training, 2. validation, and 3. testing
        """
        idx = list(range(m))
        if shuffle:
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
        histories, history_lengths, choice_sets, _, choice_set_lengths, choices = data

        print(f'Stats for {cls.name} dataset:')
        print(f'\tNodes: {len(graph.nodes)}')
        print(f'\tEdges: {len(graph.edges)}')
        print(f'\tSamples: {len(histories)}')
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

    @classmethod
    def filter_paths_by_length(cls, paths, min_threshold, max_threshold):
        return [path for path in paths if min_threshold <= len(path) <= max_threshold]

    @classmethod
    def filter_paths_by_node_outdegree(cls, paths, max_outdegree):
        edges = [(path[i], path[i + 1]) for path in paths for i in range(len(path) - 1)]
        graph = nx.DiGraph(edges)

        to_remove = set()
        for node, degree in graph.out_degree():
            if degree > max_outdegree:
                to_remove.add(node)

        return [path for path in paths if all(node not in to_remove for node in path)]

    @classmethod
    def filter_paths_by_node_appearances(cls, paths, min_appearances):
        page_counts = np.bincount(list(itertools.chain.from_iterable(paths)))

        len_paths = None
        while len_paths != len(paths):
            len_paths = len(paths)
            paths = [path for path in paths if all(page_counts[page] >= min_appearances for page in path)]
            page_counts = np.bincount(list(itertools.chain.from_iterable(paths)))

        return paths

    @classmethod
    def index_nodes(cls, graph):
        for i, node in enumerate(graph.nodes):
            graph.nodes[node]['index'] = i

    @classmethod
    def build_triadic_closure_data(cls, timestamped_edges):
        timestamped_edges = timestamped_edges[timestamped_edges[:, 2].argsort()]
        graph = nx.DiGraph()

        node_histories = dict()
        last_outgoing_edge = dict()
        last_incoming_edge = dict()

        choice_sets = []
        choice_sets_with_features = []
        histories = []
        choices = []

        for sender, recipient, timestamp in tqdm(timestamped_edges):
            if sender not in node_histories:
                node_histories[sender] = []

            if graph.has_node(sender) and graph.has_node(recipient):
                try:
                    length_2_paths = []
                    for path in nx.shortest_simple_paths(graph, sender, recipient):
                        if len(path) == 3:
                            length_2_paths.append(path)
                        else:
                            break
                    if len(length_2_paths) > 0:
                        sender_neighbors = set(graph.successors(sender)).union(set(graph.predecessors(sender)))
                        intermediate = random.choice(length_2_paths)[1]
                        choice_set = [node for node in graph.neighbors(intermediate) if
                                      not graph.has_edge(sender, node)]

                        # Features: log in-degree, log num shared neighbors, log 1+ reciprocal weight,
                        # log(2+time since outgoing edge from target), log(2+time since incoming edge to target),
                        # log(2+time since target -> chooser interaction)

                        choice_set_features = [[np.log(graph.in_degree(node)),
                                                np.log(len(set(graph.successors(node)).union(set(graph.predecessors(node))).intersection(sender_neighbors))),
                                                0 if not graph.has_edge(node, sender) else np.log(1+graph[node][sender]['weight']),
                                                0 if node not in last_outgoing_edge else 1 / np.log(2+timestamp - last_outgoing_edge[node]),
                                                0 if node not in last_incoming_edge else 1 / np.log(2+timestamp - last_incoming_edge[node]),
                                                0 if not graph.has_edge(node, sender) else 1 / np.log(2+timestamp - graph[node][sender]['last_timestamp'])]
                                               for node in choice_set]

                        choice_sets.append(choice_set)
                        choice_sets_with_features.append(choice_set_features)
                        histories.append(node_histories[sender][:])
                        choices.append(choice_set.index(recipient))

                        node_histories[sender].append(recipient)

                except nx.NetworkXNoPath:
                    pass

            last_outgoing_edge[sender] = timestamp
            last_incoming_edge[recipient] = timestamp

            if graph.has_edge(sender, recipient):
                graph[sender][recipient]['weight'] = graph[sender][recipient]['weight'] + 1
                graph[sender][recipient]['last_timestamp'] = timestamp
            else:
                graph.add_edge(sender, recipient, weight=1, last_timestamp=timestamp)

        longest_history = max(len(history) for history in histories)
        largest_choice_set = max(len(choice_set) for choice_set in choice_sets)

        n = len(graph.nodes)
        cls.index_nodes(graph)

        history_lengths = []
        choice_set_lengths = []

        for history in histories:
            history_lengths.append(len(history))
            history[:] = [graph.nodes[node]['index'] for node in history] + [n] * (longest_history - len(history))

        for choice_set in choice_sets:
            choice_set_lengths.append(len(choice_set))
            choice_set[:] = [graph.nodes[node]['index'] for node in choice_set] + [n] * (
                        largest_choice_set - len(choice_set))

        for choice_set_with_features in choice_sets_with_features:
            choice_set_with_features += [[0] * 6 for _ in range(largest_choice_set - len(choice_set_with_features))]

        histories = torch.tensor(histories)
        history_lengths = torch.tensor(history_lengths)
        choice_sets = torch.tensor(choice_sets)
        choice_sets_with_features = torch.tensor(choice_sets_with_features)
        choice_set_lengths = torch.tensor(choice_set_lengths)
        choices = torch.tensor(choices)

        return graph, histories, history_lengths, choice_sets, choice_sets_with_features, choice_set_lengths, choices


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

        paths = cls.filter_paths_by_length(paths, 0, 20)
        cls.index_nodes(graph)

        histories, history_lengths, choice_sets, choice_set_lengths, choices = cls.build_data_from_paths(paths, graph)
        m = len(histories)
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

        paths = cls.filter_paths_by_length(paths, 3, 50)
        paths = cls.filter_paths_by_node_outdegree(paths, 500)
        paths = cls.filter_paths_by_node_appearances(paths, 25)

        graph = nx.DiGraph([(path[i], path[i + 1]) for path in paths for i in range(len(path) - 1)])
        cls.index_nodes(graph)

        histories, history_lengths, choice_sets, choice_set_lengths, choices = cls.build_data_from_paths(paths, graph)
        m = len(histories)

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

        paths = cls.filter_paths_by_length(paths, 3, 50)
        paths = cls.filter_paths_by_node_outdegree(paths, 6000)
        paths = cls.filter_paths_by_node_appearances(paths, 2)

        edges = [(path[i], path[i + 1]) for path in paths for i in range(len(path) - 1)]
        graph = nx.DiGraph(edges)

        cls.index_nodes(graph)

        histories, history_lengths, choice_sets, choice_set_lengths, choices = cls.build_data_from_paths(paths, graph)
        m = len(histories)

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

        paths = cls.filter_paths_by_length(paths, 3, 50)
        paths = cls.filter_paths_by_node_appearances(paths, 250)

        graph = nx.DiGraph([(path[i], path[i + 1]) for path in paths for i in range(len(path) - 1)])
        cls.index_nodes(graph)

        histories, history_lengths, choice_sets, choice_set_lengths, choices = cls.build_data_from_paths(paths, graph)
        m = len(histories)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_set_lengths,
                                                         choices)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class ORCIDSwitchDataset(Dataset):
    name = 'orcid-switches'

    @classmethod
    def load_into_pickle(cls, file_name):
        switches = pd.read_csv(f'{DATA_DIR}/orcid-switches/all_switches.csv', usecols=('oid', 'from_matched_field', 'to_matched_field')).to_numpy()

        paths = []
        current_user = None
        current_path = []
        for oid, from_field, to_field, in switches:
            if oid == current_user:
                current_path.append(to_field)
            else:
                paths.append(current_path)
                current_user = oid
                current_path = [from_field, to_field]
        paths.append(current_path)

        print('Initial paths', len(paths))
        paths = cls.filter_paths_by_length(paths, 3, np.inf)
        print('Length filtered paths', len(paths))

        graph = nx.DiGraph([(path[i], path[i + 1]) for path in paths for i in range(len(path) - 1)])
        cls.index_nodes(graph)

        histories, history_lengths, choice_sets, choice_set_lengths, choices = cls.build_data_from_paths(paths, graph)
        m = len(histories)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_set_lengths,
                                                         choices)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class EmailEnronDataset(Dataset):
    name = 'email-enron'

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        timestamped_edges = np.loadtxt(f'{DATA_DIR}/email-Enron/email-Enron.txt', usecols=(0, 1, 2), dtype=int)

        graph, histories, history_lengths, choice_sets, \
            choice_sets_with_features, choice_set_lengths, choices = cls.build_triadic_closure_data(timestamped_edges)
        m = len(histories)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class EmailEnronCoreDataset(Dataset):
    name = 'email-enron-core'

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        timestamped_edges = np.loadtxt(f'{DATA_DIR}/email-Enron/email-Enron.txt', usecols=(0, 1, 2), dtype=int)

        filtered_edges = []
        for sender, recipient, timestamp in timestamped_edges:
            if sender <= 148 and recipient <= 148:
                filtered_edges.append([sender, recipient, timestamp])

        timestamped_edges = np.array(filtered_edges)

        graph, histories, history_lengths, choice_sets, \
            choice_sets_with_features, choice_set_lengths, choices = cls.build_triadic_closure_data(timestamped_edges)
        m = len(histories)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class CollegeMsgDataset(Dataset):
    name = 'college-msg'

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        timestamped_edges = np.loadtxt(f'{DATA_DIR}/CollegeMsg/CollegeMsg.txt', usecols=(0, 1, 2), dtype=int)

        graph, histories, history_lengths, choice_sets, \
            choice_sets_with_features, choice_set_lengths, choices = cls.build_triadic_closure_data(timestamped_edges)

        m = len(histories)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class EmailEUDataset(Dataset):
    name = 'email-eu'

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        timestamped_edges = np.loadtxt(f'{DATA_DIR}/email-Eu/email-Eu-core-temporal.txt', usecols=(0, 1, 2), dtype=int)

        graph, histories, history_lengths, choice_sets, \
            choice_sets_with_features, choice_set_lengths, choices = cls.build_triadic_closure_data(timestamped_edges)

        m = len(histories)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class MathOverflowDataset(Dataset):
    name = 'mathoverflow'

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        timestamped_edges = np.loadtxt(f'{DATA_DIR}/mathoverflow/sx-mathoverflow.txt', usecols=(0, 1, 2), dtype=int)

        graph, histories, history_lengths, choice_sets, \
            choice_sets_with_features, choice_set_lengths, choices = cls.build_triadic_closure_data(timestamped_edges)

        m = len(histories)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class FacebookWallDataset(Dataset):
    name = 'facebook-wall'

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        timestamped_edges = np.loadtxt(f'{DATA_DIR}/facebook-wosn-wall/out.facebook-wosn-wall', usecols=(0, 1, 3), dtype=int)

        graph, histories, history_lengths, choice_sets, \
            choice_sets_with_features, choice_set_lengths, choices = cls.build_triadic_closure_data(timestamped_edges)

        m = len(histories)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class EmailW3CDataset(Dataset):
    name = 'email-W3C'

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        timestamped_edges = np.loadtxt(f'{DATA_DIR}/email-W3C/email-W3C.txt', usecols=(0, 1, 2)).astype(int)

        graph, histories, history_lengths, choice_sets, \
            choice_sets_with_features, choice_set_lengths, choices = cls.build_triadic_closure_data(timestamped_edges)
        m = len(histories)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class EmailW3CCoreDataset(Dataset):
    name = 'email-W3C-core'

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        timestamped_edges = np.loadtxt(f'{DATA_DIR}/email-W3C/email-W3C.txt', usecols=(0, 1, 2)).astype(int)
        core = set(np.loadtxt(f'{DATA_DIR}/email-W3C/core-email-W3C.txt', dtype=int))

        filtered_edges = []
        for sender, recipient, timestamp in timestamped_edges:
            if sender in core and recipient in core:
                filtered_edges.append([sender, recipient, timestamp])

        timestamped_edges = np.array(filtered_edges)

        graph, histories, history_lengths, choice_sets, \
            choice_sets_with_features, choice_set_lengths, choices = cls.build_triadic_closure_data(timestamped_edges)
        m = len(histories)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class SMSADataset(Dataset):
    name = 'sms-a'

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        timestamped_edges = np.loadtxt(f'{DATA_DIR}/sms/SD01.txt', usecols=(0, 1, 2)).astype(int)

        graph, histories, history_lengths, choice_sets, \
            choice_sets_with_features, choice_set_lengths, choices = cls.build_triadic_closure_data(timestamped_edges)
        m = len(histories)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class SMSBDataset(Dataset):
    name = 'sms-b'

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        timestamped_edges = np.loadtxt(f'{DATA_DIR}/sms/SD02.txt', usecols=(0, 1, 2)).astype(int)

        graph, histories, history_lengths, choice_sets, \
            choice_sets_with_features, choice_set_lengths, choices = cls.build_triadic_closure_data(timestamped_edges)
        m = len(histories)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class SMSCDataset(Dataset):
    name = 'sms-c'

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        timestamped_edges = np.loadtxt(f'{DATA_DIR}/sms/SD03.txt', usecols=(0, 1, 2)).astype(int)

        graph, histories, history_lengths, choice_sets, \
            choice_sets_with_features, choice_set_lengths, choices = cls.build_triadic_closure_data(timestamped_edges)
        m = len(histories)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class WikiTalkDataset(Dataset):
    name = 'wiki-talk'

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        timestamped_edges = np.loadtxt(f'{DATA_DIR}/wiki-talk/wiki-talk-temporal.txt', usecols=(0, 1, 2)).astype(int)

        # pick out edges from 2004
        timestamped_edges = timestamped_edges[np.logical_and(1072915200 <= timestamped_edges[:, 2],
                                                             timestamped_edges[:, 2] < 1104537600)]

        graph, histories, history_lengths, choice_sets, \
            choice_sets_with_features, choice_set_lengths, choices = cls.build_triadic_closure_data(timestamped_edges)
        m = len(histories)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class RedditHyperlinkDataset(Dataset):
    name = 'reddit-hyperlink'

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        body_df = pd.read_csv(f'{DATA_DIR}/reddit-hyperlinks/soc-redditHyperlinks-body.tsv', sep='\t',
                              parse_dates=['TIMESTAMP'], usecols=['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP'])

        title_df = pd.read_csv(f'{DATA_DIR}/reddit-hyperlinks/soc-redditHyperlinks-title.tsv', sep='\t',
                              parse_dates=['TIMESTAMP'], usecols=['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP'])

        df = pd.concat([body_df, title_df], ignore_index=True)
        stacked = df[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']].stack()
        df[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT']] = pd.Series(stacked.factorize()[0], index=stacked.index).unstack()
        df.sort_values('TIMESTAMP', inplace=True)

        timestamped_edges = df.astype(np.int64).values
        timestamped_edges[:, 2] //= 10**9

        # Select only links before 2015
        timestamped_edges = timestamped_edges[timestamped_edges[:, 2] < 1420070400]

        graph, histories, history_lengths, choice_sets, \
            choice_sets_with_features, choice_set_lengths, choices = cls.build_triadic_closure_data(timestamped_edges)
        m = len(histories)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class BitcoinOTCDataset(Dataset):
    name = 'bitcoin-otc'

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        timestamped_edges = np.loadtxt(f'{DATA_DIR}/bitcoin-otc/soc-sign-bitcoinotc.csv', usecols=(0, 1, 3), delimiter=',').astype(int)

        graph, histories, history_lengths, choice_sets, \
            choice_sets_with_features, choice_set_lengths, choices = cls.build_triadic_closure_data(timestamped_edges)
        m = len(histories)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class BitcoinAlphaDataset(Dataset):
    name = 'bitcoin-alpha'

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        timestamped_edges = np.loadtxt(f'{DATA_DIR}/bitcoin-alpha/soc-sign-bitcoinalpha.csv', usecols=(0, 1, 3), delimiter=',').astype(int)

        graph, histories, history_lengths, choice_sets, \
            choice_sets_with_features, choice_set_lengths, choices = cls.build_triadic_closure_data(timestamped_edges)
        m = len(histories)

        train_data, val_data, test_data = cls.data_split(m, histories, history_lengths, choice_sets, choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class SyntheticMNLDataset(Dataset):
    name = 'synthetic-mnl'

    @classmethod
    def generate(cls):
        random.seed(0)
        np.random.seed(0)

        nodes = list(range(20000))
        target_triangle_closures = 50000
        p_triangle_closure = 0.25

        triangle_closures = 0

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)

        timestamp = 0

        last_outgoing_edge = dict()
        last_incoming_edge = dict()

        choice_sets = []
        choice_sets_with_features = []
        choices = []

        mnl_utilities = np.array([2, 1, 3, 1, 3, 5])

        with tqdm(total=target_triangle_closures) as pbar:
            while triangle_closures < target_triangle_closures:
                timestamp += np.random.poisson(5)
                chooser = np.random.choice(nodes)

                target = None

                if graph.out_degree(chooser) > 0:
                    intermediate = np.random.choice(list(graph.neighbors(chooser)))
                    choice_set = [node for node in graph.neighbors(intermediate) if node != chooser]
                    if len(choice_set) > 2 and np.random.random() < p_triangle_closure:
                        triangle_closures += 1
                        pbar.update(1)
                        sender_neighbors = set(graph.successors(chooser)).union(set(graph.predecessors(chooser)))

                        choice_set_features = [[np.log(graph.in_degree(node)),
                                                np.log(len(set(graph.successors(node)).union(
                                                    set(graph.predecessors(node))).intersection(sender_neighbors))),
                                                0 if not graph.has_edge(node, chooser) else np.log(
                                                    1 + graph[node][chooser]['weight']),
                                                0 if node not in last_outgoing_edge else 1 / np.log(
                                                    2 + timestamp - last_outgoing_edge[node]),
                                                0 if node not in last_incoming_edge else 1 / np.log(
                                                    2 + timestamp - last_incoming_edge[node]),
                                                0 if not graph.has_edge(node, chooser) else 1 / np.log(
                                                    2 + timestamp - graph[node][chooser]['last_timestamp'])]
                                               for node in choice_set]

                        utilities = (np.array(choice_set_features) * mnl_utilities).sum(axis=1)

                        target = np.random.choice(choice_set, p=scipy.special.softmax(utilities))

                        choice_sets.append(choice_set)
                        choice_sets_with_features.append(choice_set_features)
                        choices.append(choice_set.index(target))

                if target is None:
                    target = chooser
                    while target == chooser:
                        target = np.random.choice(nodes)

                if graph.has_edge(chooser, target):
                    graph[chooser][target]['weight'] = graph[chooser][target]['weight'] + 1
                    graph[chooser][target]['last_timestamp'] = timestamp
                else:
                    graph.add_edge(chooser, target, weight=1, last_timestamp=timestamp)

                last_outgoing_edge[chooser] = timestamp
                last_incoming_edge[target] = timestamp

        largest_choice_set = max(len(choice_set) for choice_set in choice_sets)

        n = len(graph.nodes)
        cls.index_nodes(graph)

        choice_set_lengths = []

        for choice_set in choice_sets:
            choice_set_lengths.append(len(choice_set))
            choice_set[:] = [graph.nodes[node]['index'] for node in choice_set] + [n] * (
                    largest_choice_set - len(choice_set))

        for choice_set_with_features in choice_sets_with_features:
            choice_set_with_features += [[0] * cls.num_features for _ in range(largest_choice_set - len(choice_set_with_features))]

        choice_sets = torch.tensor(choice_sets)
        choice_sets_with_features = torch.tensor(choice_sets_with_features)
        choice_set_lengths = torch.tensor(choice_set_lengths)
        choices = torch.tensor(choices)

        return graph, choice_sets, choice_sets_with_features, choice_set_lengths, choices

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        graph, choice_sets, choice_sets_with_features, choice_set_lengths, choices = SyntheticMNLDataset.generate()
        m = len(choices)

        train_data, val_data, test_data = cls.data_split(m, choice_sets,
                                                         choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)



if __name__ == '__main__':
    # for dataset in [WikiTalkDataset, RedditHyperlinkDataset,
    #                 BitcoinAlphaDataset, BitcoinOTCDataset,
    #                 SMSADataset, SMSBDataset, SMSCDataset,
    #                 EmailEnronDataset, EmailEUDataset, EmailW3CDataset,
    #                 FacebookWallDataset, CollegeMsgDataset, MathOverflowDataset]:
    #     dataset.print_stats()

    SyntheticMNLDataset.print_stats()
