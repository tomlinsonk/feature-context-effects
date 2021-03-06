import os
import pickle
import random
from abc import ABC, abstractmethod
from zipfile import ZipFile

import networkx as nx
import numpy as np
import pandas as pd
import scipy
import torch
from tqdm import tqdm

DATA_DIR = 'data'
CONFIG_DIR = 'hyperparams'


class Dataset(ABC):

    name = ''
    num_features = 6
    feature_names = ['in-degree', 'shared neighbors', 'reciprocal weight', 'send recency', 'receive recency', 'reciprocal recency']

    @classmethod
    def load(cls):
        pickle_file = f'{DATA_DIR}/{cls.name}.pickle'

        if not os.path.isfile(pickle_file):
            cls.load_into_pickle(pickle_file)

        with open(pickle_file, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def load_standardized(cls):
        graph, train_data, val_data, test_data = cls.load()
        data = [torch.cat([train_data[i], val_data[i], test_data[i]]) for i in range(len(train_data))]
        histories, history_lengths, choice_sets, choice_set_features, choice_set_lengths, choices = data

        all_feature_vecs = choice_set_features[torch.arange(choice_sets.size(1))[None, :] < choice_set_lengths[:, None]]
        means = all_feature_vecs.mean(0)
        stds = all_feature_vecs.std(0)

        train_data[3][torch.arange(train_data[3].size(1))[None, :] < train_data[4][:, None]] -= means
        train_data[3][torch.arange(train_data[3].size(1))[None, :] < train_data[4][:, None]] /= stds

        val_data[3][torch.arange(val_data[3].size(1))[None, :] < val_data[4][:, None]] -= means
        val_data[3][torch.arange(val_data[3].size(1))[None, :] < val_data[4][:, None]] /= stds

        test_data[3][torch.arange(test_data[3].size(1))[None, :] < test_data[4][:, None]] -= means
        test_data[3][torch.arange(test_data[3].size(1))[None, :] < test_data[4][:, None]] /= stds

        return graph, train_data, val_data, test_data, means, stds

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
        print(f'\tLargest Choice Set: {max(choice_set_lengths)}')
        print(f'\tMean Choice Set: {np.mean(choice_set_lengths)}')


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
                        # 1 / log(2+time since outgoing edge from target), 1 / log(2+time since incoming edge to target),
                        # 1 / log(2+time since target -> chooser interaction)

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


    @classmethod
    def best_lr(cls, method):
        with open(f'{CONFIG_DIR}/learning_rate_settings.pickle', 'rb') as f:
            grid_search_losses, lrs = pickle.load(f)

        return lrs[np.argmin([grid_search_losses[cls, method, lr] for lr in lrs])]

    @classmethod
    def best_val_lr_wd(cls, method):
        with open(f'{CONFIG_DIR}/validation_loss_lr_wd_settings.pickle', 'rb') as f:
            losses, datasets, methods, lrs, wds = pickle.load(f)

        # Pick lr, wd pair that has lowest max loss over the last 5 epochs
        loss_grid = np.array([[max(losses[cls, method, lr, wd][2][-5:]) for wd in wds] for lr in lrs])
        min_lr_idx, min_wd_idx = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)

        return lrs[min_lr_idx], wds[min_wd_idx]

    @classmethod
    def pickle_to_zip(cls):
        zip_fname = f'{cls.name}.zip'
        features_fname = f'{DATA_DIR}/txt-files/{cls.name}-features.txt'
        choices_fname = f'{DATA_DIR}/txt-files/{cls.name}-choices.txt'
        choice_sets_fname = f'{DATA_DIR}/txt-files/{cls.name}-choice-sets.txt'
        readme_fname = f'{DATA_DIR}/txt-files/{cls.name}-README.txt'

        graph, train_data, val_data, test_data = cls.load()
        data = [torch.cat([train_data[i], val_data[i], test_data[i]]) for i in range(len(train_data))]
        histories, history_lengths, choice_sets, choice_set_features, choice_set_lengths, choices = data

        with open(choice_sets_fname, 'w') as f:
            f.write('\n'.join(';'.join(' '.join(f'{x.item()}' for x in item) for item in choice_set_features[i, :choice_set_lengths[i]]) for i in range(len(choices))))

        with open(choices_fname, 'w') as f:
            f.write('\n'.join(f'{x.item()}' for x in choices))

        with open(features_fname, 'w') as f:
            f.write(', '.join(cls.feature_names))

        with open(readme_fname, 'w') as f:
            f.write('This dataset contains three files:\n'
                    f'1. {cls.name}-choice-sets.txt\n'
                    f'2. {cls.name}-choices.txt\n'
                    f'3. {cls.name}-features.txt\n'
                    'Each line in file 1 is a single choice instance. Each item in the choice set\n'
                    'is represented by numerical features (space separated) and each item is \n'
                    'semicolon-separated. The index of the item selected from each choice set is in\n'
                    'file 2. The names of the features are in file 3.')

        with ZipFile(zip_fname, 'w') as z:
            z.write(features_fname, os.path.basename(features_fname))
            z.write(choices_fname, os.path.basename(choices_fname))
            z.write(choice_sets_fname, os.path.basename(choice_sets_fname))
            z.write(readme_fname, os.path.basename(readme_fname))


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
    name = 'email-w3c'

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

        nodes = list(range(1000))
        target_triangle_closures = 50000
        p_triangle_closure = 0.1

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
        choice_sets_with_features = torch.tensor(choice_sets_with_features).float()
        choice_set_lengths = torch.tensor(choice_set_lengths)
        choices = torch.tensor(choices)

        return graph, choice_sets, choice_sets_with_features, choice_set_lengths, choices

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        graph, choice_sets, choice_sets_with_features, choice_set_lengths, choices = cls.generate()
        m = len(choices)

        train_data, val_data, test_data = cls.data_split(m, torch.zeros_like(choices),
                                                         torch.zeros_like(choices),
                                                         choice_sets,
                                                         choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class SyntheticLCLDataset(Dataset):
    name = 'synthetic-lcl'

    @classmethod
    def generate(cls):
        random.seed(0)
        np.random.seed(0)

        nodes = list(range(1000))
        target_triangle_closures = 50000
        p_triangle_closure = 0.1

        triangle_closures = 0

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)

        timestamp = 0

        last_outgoing_edge = dict()
        last_incoming_edge = dict()

        choice_sets = []
        choice_sets_with_features = []
        choices = []

        base_utilities = np.array([2, 1, 3, 1, 3, 5])

        context_effects = np.array([[0, 0, 0, 0, 0, 100],
                                    [0, 0, 5, 0, 0, 0],
                                    [0, -5, 0, 0, 0, 0],
                                    [-5, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0],
                                    [0, 0, 5, 0, 0, 0]])

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

                        choice_set_features_np = np.array(choice_set_features)
                        mean_feature_vector = np.mean(choice_set_features_np, axis=0)
                        utilities = (choice_set_features_np * (base_utilities + context_effects @ mean_feature_vector)).sum(axis=1)

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
        choice_sets_with_features = torch.tensor(choice_sets_with_features).float()
        choice_set_lengths = torch.tensor(choice_set_lengths)
        choices = torch.tensor(choices)

        return graph, choice_sets, choice_sets_with_features, choice_set_lengths, choices

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        graph, choice_sets, choice_sets_with_features, choice_set_lengths, choices = cls.generate()
        m = len(choices)

        train_data, val_data, test_data = cls.data_split(m, torch.zeros_like(choices),
                                                         torch.zeros_like(choices),
                                                         choice_sets,
                                                         choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((graph, train_data, val_data, test_data), f, protocol=4)


class ExpediaDataset(Dataset):
    name = 'expedia'
    num_features = 5

    feature_names = ['Star Rating', 'Review Score', 'Location Score', 'Price', 'On Promotion']

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        feature_names = ['prop_starrating', 'prop_review_score', 'prop_location_score1', 'price_usd', 'promotion_flag']

        df = pd.read_csv(f'{DATA_DIR}/expedia-personalized-sort/train.csv', usecols=['srch_id', 'prop_id', 'booking_bool'] + feature_names)

        # Select only searches that result in a booking
        df = df[df.groupby(['srch_id'])['booking_bool'].transform(max) == 1]

        max_choice_set_size = df['srch_id'].value_counts().max()
        samples = df['srch_id'].nunique()
        n_feats = 5

        choice_sets = torch.full((samples, max_choice_set_size), -1, dtype=torch.long)
        choice_sets_with_features = torch.zeros((samples, max_choice_set_size, n_feats), dtype=torch.float)
        choice_set_lengths = torch.zeros(samples, dtype=torch.long)
        choices = torch.zeros(samples, dtype=torch.long)

        for i, (srch_id, group) in tqdm(enumerate(df.groupby('srch_id')), total=samples):
            choice_set_length = len(group.index)
            choice_set_lengths[i] = choice_set_length

            choice_sets[i, :choice_set_length] = torch.as_tensor(group['prop_id'].values)
            features = torch.as_tensor(group[feature_names].values)
            features[torch.isnan(features)] = 0

            choice_sets_with_features[i, :choice_set_length] = features
            torch.isnan(choice_sets_with_features[i, :choice_set_length])

            choices[i] = torch.from_numpy(np.where(group['booking_bool'] == 1)[0])

        train_data, val_data, test_data = cls.data_split(samples, torch.zeros_like(choices),
                                                         torch.zeros_like(choices),
                                                         choice_sets,
                                                         choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=False)

        with open(file_name, 'wb') as f:
            pickle.dump((nx.DiGraph(), train_data, val_data, test_data), f, protocol=4)


class SushiDataset(Dataset):
    name = 'sushi'
    num_features = 6

    feature_names = ['Is Maki', 'Is Seafood', 'Oiliness', 'Popularity', 'Price', 'Availability']

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        rankings = np.loadtxt(f'{DATA_DIR}/sushi3-2016/sushi3b.5000.10.order', skiprows=1, usecols=range(2, 12), dtype=int)
        features = np.loadtxt(f'{DATA_DIR}/sushi3-2016/sushi3.idata', usecols=(2, 3, 5, 6, 7, 8))

        # Flip first two binary features
        features[:, 0] = 1 - features[:, 0]
        features[:, 1] = 1 - features[:, 1]
        features = torch.from_numpy(features)

        samples = len(rankings)
        max_choice_set_size = 10

        choice_sets = torch.full((samples, max_choice_set_size), -1, dtype=torch.long)
        choice_sets_with_features = torch.zeros((samples, max_choice_set_size, cls.num_features), dtype=torch.float)
        choice_set_lengths = torch.zeros(samples, dtype=torch.long)
        choices = torch.zeros(samples, dtype=torch.long)

        for i, row in enumerate(rankings):
            choice_set = torch.from_numpy(row)

            choice_sets[i] = choice_set
            choice_sets_with_features[i] = features[choice_set]
            choice_set_lengths[i] = 10
            choices[i] = 0

        train_data, val_data, test_data = cls.data_split(samples, torch.zeros_like(choices),
                                                         torch.zeros_like(choices),
                                                         choice_sets,
                                                         choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=True)

        with open(file_name, 'wb') as f:
            pickle.dump((nx.DiGraph(), train_data, val_data, test_data), f, protocol=4)


class DistrictDataset(Dataset):
    name = 'district'
    num_features = 27

    feature_names = ['points', 'var_xcoord', 'var_ycoord', 'varcoord_ratio', 'avgline', 'varline', 'boyce', 'lenwid',
                     'jagged', 'parts', 'hull', 'bbox', 'reock', 'polsby', 'schwartzberg', 'circle_area',
                     'circle_perim', 'hull_area', 'hull_perim', 'orig_area', 'district_perim', 'corners', 'xvar',
                     'yvar', 'cornervar_ratio', 'sym_x', 'sym_y']

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        features = pd.read_csv(f'{DATA_DIR}/district-compactness/features.csv')

        comparisons = pd.read_csv(f'{DATA_DIR}/district-compactness/paired_comparisons.csv')

        feature_dict = {row['district']: row[cls.feature_names].to_numpy(dtype=float) for index, row in features.iterrows()}
        district_to_index = {district: i for i, district in enumerate(sorted(feature_dict.keys()))}

        samples = len(comparisons)
        max_choice_set_size = 2

        choice_sets = torch.full((samples, max_choice_set_size), -1, dtype=torch.long)
        choice_sets_with_features = torch.zeros((samples, max_choice_set_size, cls.num_features), dtype=torch.float)
        choice_set_lengths = torch.zeros(samples, dtype=torch.long)
        choices = torch.zeros(samples, dtype=torch.long)

        for index, row in comparisons.iterrows():
            district1 = row['alternate_id_1']
            district2 = row['alternate_id_2']
            chosen = row['alternate_id_winner']
            choice_set = [district_to_index[district1], district_to_index[district2]]

            choice_sets[index] = torch.as_tensor(choice_set)
            choices[index] = choice_set.index(district_to_index[chosen])
            choice_set_lengths[index] = 2
            choice_sets_with_features[index] = torch.as_tensor([feature_dict[district1], feature_dict[district2]])

        train_data, val_data, test_data = cls.data_split(samples, torch.zeros_like(choices),
                                                         torch.zeros_like(choices),
                                                         choice_sets,
                                                         choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=True)

        with open(file_name, 'wb') as f:
            pickle.dump((nx.DiGraph(), train_data, val_data, test_data), f, protocol=4)


class DistrictSmartDataset(DistrictDataset):
    name = 'district-smart'
    num_features = 6
    feature_names = ['hull', 'bbox', 'reock', 'polsby', 'sym_x', 'sym_y']


class CarADataset(Dataset):
    name = 'car-a'
    num_features = 4
    features_names = ['SUV', 'Automatic', 'Engine Displacement', 'Hybrid']

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        features = np.loadtxt(f'{DATA_DIR}/car/exp1-prefs/items1.csv', skiprows=1, delimiter=',')
        comparisons = np.loadtxt(f'{DATA_DIR}/car/exp1-prefs/prefs1.csv', skiprows=1, delimiter=',')

        feature_dict = {item: [body_type-1, transmission-1, displacement, 2-non_hybrid] for item, body_type, transmission, displacement, non_hybrid in features}

        comparisons = comparisons[np.logical_not(comparisons[:, 3])]

        samples = len(comparisons)
        max_choice_set_size = 2

        choice_sets = torch.full((samples, max_choice_set_size), -1, dtype=torch.long)
        choice_sets_with_features = torch.zeros((samples, max_choice_set_size, cls.num_features), dtype=torch.float)
        choice_set_lengths = torch.zeros(samples, dtype=torch.long)
        choices = torch.zeros(samples, dtype=torch.long)

        for index, row in enumerate(comparisons):
            user, chosen_car, other_car, is_control = row

            if not is_control:
                choice_set = [chosen_car-1, other_car-1]
                choice_sets[index] = torch.as_tensor(choice_set)
                choices[index] = 0
                choice_set_lengths[index] = 2
                choice_sets_with_features[index] = torch.as_tensor([feature_dict[chosen_car], feature_dict[other_car]])

        train_data, val_data, test_data = cls.data_split(samples, torch.zeros_like(choices),
                                                         torch.zeros_like(choices),
                                                         choice_sets,
                                                         choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=True)

        with open(file_name, 'wb') as f:
            pickle.dump((nx.DiGraph(), train_data, val_data, test_data), f, protocol=4)


class CarBDataset(Dataset):
    name = 'car-b'
    num_features = 7
    features_names = ['Sedan', 'SUV', 'Hatchback', 'Automatic', 'Engine Displacement', 'Hybrid', 'All-wheel-drive']

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        features = np.loadtxt(f'{DATA_DIR}/car/exp2-prefs/items2.csv', skiprows=1, delimiter=',')
        comparisons = np.loadtxt(f'{DATA_DIR}/car/exp2-prefs/prefs2.csv', skiprows=1, delimiter=',')
        comparisons = comparisons[np.logical_not(comparisons[:, 3])]

        feature_dict = {item: [body_type == 1, body_type == 2, body_type == 3, transmission-1, displacement, 2-non_hybrid, 2-awd] for item, body_type, transmission, displacement, non_hybrid, awd in features}

        samples = len(comparisons)
        max_choice_set_size = 2

        choice_sets = torch.full((samples, max_choice_set_size), -1, dtype=torch.long)
        choice_sets_with_features = torch.zeros((samples, max_choice_set_size, cls.num_features), dtype=torch.float)
        choice_set_lengths = torch.zeros(samples, dtype=torch.long)
        choices = torch.zeros(samples, dtype=torch.long)

        for index, row in enumerate(comparisons):
            user, chosen_car, other_car, is_control = row

            if not is_control:
                choice_set = [chosen_car-1, other_car-1]
                choice_sets[index] = torch.as_tensor(choice_set)
                choices[index] = 0
                choice_set_lengths[index] = 2
                choice_sets_with_features[index] = torch.as_tensor([feature_dict[chosen_car], feature_dict[other_car]])

        train_data, val_data, test_data = cls.data_split(samples, torch.zeros_like(choices),
                                                         torch.zeros_like(choices),
                                                         choice_sets,
                                                         choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=True)

        with open(file_name, 'wb') as f:
            pickle.dump((nx.DiGraph(), train_data, val_data, test_data), f, protocol=4)


class CarAltDataset(Dataset):
    name = 'car-alt'
    num_features = 21
    feature_names = ['Price / ln(income)', 'Range', 'Acceleration', 'Top speed', 'Pollution', 'Size', '"Big enough"',
                      'Luggage space', 'Operating cost', 'Station availability', 'SUV', 'Sports car',
                      'Station wagon', 'Truck', 'Van', 'EV', 'Commute < 5 x EV', 'College x EV',
                      'CNG', 'Methanol', 'College x methanol']

    @classmethod
    def load_into_pickle(cls, file_name):
        random.seed(0)
        np.random.seed(0)

        data = np.loadtxt(f'{DATA_DIR}/car-alt-data/xmat.txt').reshape((4654, 156))

        samples = len(data)
        max_choice_set_size = 6

        choice_sets = torch.full((samples, max_choice_set_size), -1, dtype=torch.long)
        choice_sets_with_features = torch.zeros((samples, max_choice_set_size, cls.num_features), dtype=torch.float)
        choice_set_lengths = torch.full([samples], 6, dtype=torch.long)
        choices = torch.zeros(samples, dtype=torch.long)

        for i, row in enumerate(data):
            for item in range(6):
                features = torch.zeros(cls.num_features, dtype=torch.float)

                for feature in range(cls.num_features):
                    features[feature] = row[feature * 6 + item]

                choice_sets_with_features[i, item] = features
            choices[i] = list(row[132:138]).index(1)

        train_data, val_data, test_data = cls.data_split(samples, torch.zeros_like(choices),
                                                         torch.zeros_like(choices),
                                                         choice_sets,
                                                         choice_sets_with_features,
                                                         choice_set_lengths, choices, shuffle=True)

        with open(file_name, 'wb') as f:
            pickle.dump((nx.DiGraph(), train_data, val_data, test_data), f, protocol=4)



SYNTHETIC_DATASETS = [SyntheticMNLDataset, SyntheticLCLDataset]
REAL_NETWORK_DATASETS = [
    WikiTalkDataset, RedditHyperlinkDataset,
    BitcoinAlphaDataset, BitcoinOTCDataset,
    SMSADataset, SMSBDataset, SMSCDataset,
    EmailEnronDataset, EmailEUDataset, EmailW3CDataset,
    FacebookWallDataset, CollegeMsgDataset, MathOverflowDataset
]
REAL_GENERAL_DATASETS = [DistrictDataset, DistrictSmartDataset, ExpediaDataset, SushiDataset, CarADataset, CarBDataset,
                    CarAltDataset]

NETWORK_DATASETS = SYNTHETIC_DATASETS + REAL_NETWORK_DATASETS
ALL_DATASETS = REAL_GENERAL_DATASETS + NETWORK_DATASETS


if __name__ == '__main__':
    for dataset in ALL_DATASETS:
        dataset.print_stats()



