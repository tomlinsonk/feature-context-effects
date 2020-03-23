import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm


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
    def __init__(self, num_items, dim, beta=0.5, learn_beta=False):
        super().__init__()

        self.num_items = num_items
        self.dim = dim
        self.beta = nn.Parameter(torch.tensor([beta]), requires_grad=learn_beta)

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
        history_weight = torch.pow(self.beta.repeat(max_history_len), torch.arange(0, max_history_len))[:, None]
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

    model, losses = train_history_cdm(n, histories, history_lengths, choice_sets, choice_set_lengths, choices, dim=3, lr=0.005, weight_decay=0)
    plt.plot(range(500), losses)
    plt.show()

    print(model.beta)


def train_history_cdm(n, histories, history_lengths, choice_sets, choice_set_lengths, choices, dim=64, beta=0.5, lr=1e-4, weight_decay=1e-4, learn_beta=False):

    model = HistoryCDM(n, dim, beta, learn_beta)
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

            with torch.no_grad():
                model.beta.data = model.beta.clamp(0, 1)

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


if __name__ == '__main__':
    toy_example()
