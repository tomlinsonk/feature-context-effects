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

    name = 'history_cdm'

    def __init__(self, num_items, dim, beta=0.5, learn_beta=False):
        super().__init__()

        self.num_items = num_items
        self.dim = dim
        self.beta = nn.Parameter(torch.tensor([beta]), requires_grad=learn_beta)
        self.learn_beta = learn_beta

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


class HistoryMNL(nn.Module):

    name = 'history_mnl'

    def __init__(self, num_items, dim, beta=0.5, learn_beta=False):
        super().__init__()

        self.num_items = num_items
        self.dim = dim
        self.beta = nn.Parameter(torch.tensor([beta]), requires_grad=learn_beta)
        self.learn_beta = learn_beta

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

        self.page_utilities = Embedding(
            num=self.num_items + 1,
            dim=1,
            pad_idx=self.num_items
        )

    def forward(self, histories, history_lengths, choice_sets, choice_set_lengths):
        batch_size, max_choice_set_len = choice_sets.size()
        _, max_history_len = histories.size()

        history_vecs = self.history_embedding(histories)
        page_utilities = self.page_utilities(choice_sets)
        target_vecs = self.target_embedding(choice_sets)

        history_weight = torch.pow(self.beta.repeat(max_history_len), torch.arange(0, max_history_len))[:, None]
        weighted_history_sums = (history_weight * history_vecs).sum(1, keepdim=True)

        utilities = (target_vecs * weighted_history_sums).sum(2) + page_utilities.squeeze()
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

    name = 'lstm'

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

    train_data = histories, history_lengths, choice_sets, choice_set_lengths, choices

    model, train_losses, train_accs, val_losses, val_accs = train_history_mnl(n, train_data, train_data, dim=3, lr=0.005, weight_decay=0)
    plt.plot(range(500), train_losses)
    plt.plot(range(500), train_accs)

    plt.show()

    print(model.beta)


def train_history_model(model, train_data, val_data, lr=1e-4, weight_decay=1e-4):
    print(f'Training {model.name} dim={model.dim}, lr={lr}, wd={weight_decay}, beta={model.beta.item()}, learn_beta={model.learn_beta}...')

    batch_size = 128
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in tqdm(range(500)):
        train_loss = 0
        train_count = 0
        train_correct = 0
        for histories, history_lengths, choice_sets, choice_set_lengths, choices in train_data_loader:
            model.train()
            choice_pred = model(histories, history_lengths, choice_sets, choice_set_lengths)

            loss = model.loss(choice_pred, choices)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.beta.data = model.beta.clamp(0, 1)

            model.eval()
            vals, idxs = choice_pred.max(1)
            train_correct += (idxs == choices).long().sum().item() / choice_pred.size(0)
            train_loss += loss.item()
            train_count += 1

        train_accs.append(train_correct / train_count)
        train_losses.append(train_loss / train_count)

        val_loss = 0
        val_count = 0
        val_correct = 0
        model.eval()
        for histories, history_lengths, choice_sets, choice_set_lengths, choices in val_data_loader:
            choice_pred = model(histories, history_lengths, choice_sets, choice_set_lengths)
            loss = model.loss(choice_pred, choices)
            vals, idxs = choice_pred.max(1)
            val_correct += (idxs == choices).long().sum().item() / choice_pred.size(0)
            val_loss += loss.item()
            val_count += 1

        val_losses.append(val_loss / val_count)
        val_accs.append(val_correct / val_count)

    return model, train_losses, train_accs, val_losses, val_accs


def train_history_cdm(n, train_data, val_data, dim=64, beta=0.5, lr=1e-4, weight_decay=1e-4, learn_beta=False):
    model = HistoryCDM(n, dim, beta, learn_beta)
    return train_history_model(model, train_data, val_data, lr, weight_decay)


def train_history_mnl(n, train_data, val_data, dim=64, beta=0.5, lr=1e-4, weight_decay=1e-4, learn_beta=False):
    model = HistoryMNL(n, dim, beta, learn_beta)
    return train_history_model(model, train_data, val_data, lr, weight_decay)


def train_lstm(n, train_data, val_data, dim=64, lr=1e-4, weight_decay=1e-4, beta=None, learn_beta=None):
    print(f'Training LSTM dim={dim}, lr={lr}, wd={weight_decay}...')
    batch_size = 128

    train_reverse_history = train_data[0].clone().detach()
    val_reverse_history = val_data[0].clone().detach()
    # Reverse histories
    for i in range(train_reverse_history.size(0)):
        train_reverse_history[i, :train_data[1][i]] = train_reverse_history[i, :train_data[1][i]].flip(0)

    for i in range(val_reverse_history.size(0)):
        val_reverse_history[i, :val_data[1][i]] = val_reverse_history[i, :val_data[1][i]].flip(0)

    train_data = train_reverse_history, train_data[1], train_data[2], train_data[3], train_data[4]
    val_data = val_reverse_history, val_data[1], val_data[2], val_data[3], val_data[4]

    model = LSTM(n, dim)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, sort_batch=True, sort_index=1)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, sort_batch=True, sort_index=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in tqdm(range(500)):
        train_loss = 0
        train_count = 0
        train_correct = 0
        for histories, history_lengths, choice_sets, choice_set_lengths, choices in train_data_loader:
            model.train()
            choice_pred = model(histories, history_lengths, choice_sets, choice_set_lengths)
            loss = model.loss(choice_pred, choices)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            vals, idxs = choice_pred.max(1)
            train_correct += (idxs == choices).long().sum().item() / choice_pred.size(0)
            train_loss += loss.item()
            train_count += 1

        train_accs.append(train_correct / train_count)
        train_losses.append(train_loss / train_count)

        val_loss = 0
        val_count = 0
        val_correct = 0
        model.eval()
        for histories, history_lengths, choice_sets, choice_set_lengths, choices in val_data_loader:
            choice_pred = model(histories, history_lengths, choice_sets, choice_set_lengths)
            loss = model.loss(choice_pred, choices)
            vals, idxs = choice_pred.max(1)
            val_correct += (idxs == choices).long().sum().item() / choice_pred.size(0)
            val_loss += loss.item()
            val_count += 1

        val_losses.append(val_loss / val_count)
        val_accs.append(val_correct / val_count)

    return model, train_losses, train_accs, val_losses, val_accs


if __name__ == '__main__':
    toy_example()
