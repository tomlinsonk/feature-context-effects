import time

import numpy as np
import torch
from torch import nn, jit
from tqdm import tqdm


# From https://github.com/pytorch/pytorch/issues/31829
@jit.script
def logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))


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
            return [item for item in self.data]
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


class MNL(nn.Module):

    name = 'mnl'

    def __init__(self, num_features):
        super().__init__()

        self.num_features = num_features
        self.theta = nn.Parameter(torch.ones(self.num_features), requires_grad=True)

    def forward(self, choice_set_features, choice_set_lengths):
        batch_size, max_choice_set_len, num_feats = choice_set_features.size()

        utilities = (self.theta * choice_set_features).sum(-1)

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


class LCL(nn.Module):

    name = 'lcl'

    def __init__(self, num_features, l1_reg=0):
        super().__init__()

        self.num_features = num_features
        self.theta = nn.Parameter(torch.ones(self.num_features), requires_grad=True)
        self.A = nn.Parameter(torch.zeros(self.num_features, self.num_features), requires_grad=True)

        self.l1_reg = l1_reg

    def forward(self, choice_set_features, choice_set_lengths):
        batch_size, max_choice_set_len, num_feats = choice_set_features.size()

        # Compute mean of each feature over each choice set
        mean_choice_set_features = choice_set_features.sum(1) / choice_set_lengths.unsqueeze(-1)

        # Compute context effect in each sample
        context_effects = self.A @ mean_choice_set_features.unsqueeze(-1)

        # Compute context-adjusted utility of every item
        utilities = ((self.theta.unsqueeze(-1) + context_effects).view(batch_size, 1, -1) * choice_set_features).sum(-1)
        utilities[torch.arange(max_choice_set_len).unsqueeze(0) >= choice_set_lengths.unsqueeze(-1)] = -np.inf

        return nn.functional.log_softmax(utilities, 1)

    def loss(self, y_pred, y):
        """
        The error in inferred log-probabilities given observations
        :param y_pred: log(choice probabilities)
        :param y: observed choices
        :return: the loss
        """

        return nn.functional.nll_loss(y_pred, y) + self.l1_reg * self.A.norm(1)


class DLCL(nn.Module):

    name = 'dlcl'

    def __init__(self, num_features, l1_reg=0):
        super().__init__()

        self.num_features = num_features

        # Context effect slopes
        self.A = nn.Parameter(torch.zeros(self.num_features, self.num_features), requires_grad=True)

        # Context effect intercepts
        self.B = nn.Parameter(torch.zeros(self.num_features, self.num_features), requires_grad=True)

        self.mixture_weights = nn.Parameter(torch.ones(self.num_features), requires_grad=True)

        self.l1_reg = l1_reg

    def forward(self, choice_set_features, choice_set_lengths):
        batch_size, max_choice_set_len, num_feats = choice_set_features.size()

        # Compute mean of each feature over each choice set
        mean_choice_set_features = choice_set_features.sum(1) / choice_set_lengths[:, None]

        # Use learned linear context model to compute utility matrices for each sample
        utility_matrices = self.B + self.A * (torch.ones(self.num_features, 1) @ mean_choice_set_features[:, None, :])

        # Compute utility of each item under each feature MNL
        utilities = choice_set_features @ utility_matrices
        utilities[torch.arange(max_choice_set_len)[None, :] >= choice_set_lengths[:, None]] = -np.inf

        # Compute MNL log-probs for each feature
        log_probs = nn.functional.log_softmax(utilities, 1)

        # Combine the MNLs into single probability using weights
        # This is what I want to do, but logsumexp produces nan gradients when there are -infs
        # https://github.com/pytorch/pytorch/issues/31829
        # return torch.logsumexp(log_probs + torch.log(self.weights / self.weights.sum()), 2)

        # So, I'm instead using the fix in the issue linked above
        return logsumexp(log_probs + torch.log_softmax(self.mixture_weights, 0), 2)

    def loss(self, y_pred, y):
        """
        The error in inferred log-probabilities given observations
        :param y_pred: log(choice probabilities)
        :param y: observed choices
        :return: the loss
        """

        return nn.functional.nll_loss(y_pred, y) + self.l1_reg * self.A.norm(1)


class MixedLogit(nn.Module):

    name = 'mixed_logit'

    def __init__(self, num_features):
        super().__init__()

        self.num_features = num_features
        self.thetas = nn.Parameter(torch.rand(self.num_features, self.num_features), requires_grad=True)
        self.mixture_weights = nn.Parameter(torch.ones(self.num_features), requires_grad=True)

    def forward(self, choice_set_features, choice_set_lengths):
        batch_size, max_choice_set_len, num_feats = choice_set_features.size()

        # Compute utility of each item under each MNL
        utilities = choice_set_features @ self.thetas.T
        utilities[torch.arange(max_choice_set_len)[None, :] >= choice_set_lengths[:, None]] = -np.inf

        # Compute MNL log-probs for each feature
        log_probs = nn.functional.log_softmax(utilities, 1)

        # Combine the MNLs into single probability using weights
        # This is what I want to do, but logsumexp produces nan gradients when there are -infs
        # https://github.com/pytorch/pytorch/issues/31829
        # return torch.logsumexp(log_probs + torch.log(self.weights / self.weights.sum()), 2)

        # So, I'm instead using the fix in the issue linked above
        return logsumexp(log_probs + torch.log_softmax(self.mixture_weights, 0), 2)

    def loss(self, y_pred, y):
        """
        The error in inferred log-probabilities given observations
        :param y_pred: log(choice probabilities)
        :param y: observed choices
        :return: the loss
        """

        return nn.functional.nll_loss(y_pred, y)


class EMAlgorithmQ(nn.Module):

    def __init__(self, num_features, r, mean_cs_features, B_init, C_init):
        super().__init__()

        self.num_features = num_features
        self.r = r
        self.mean_cs_features = mean_cs_features

        self.B = nn.Parameter(B_init, requires_grad=True)
        self.C = nn.Parameter(C_init, requires_grad=True)

    def forward(self, choice_set_features, choice_set_lengths, choices, indices):
        batch_size, max_choice_set_len, _ = choice_set_features.size()

        # Use linear context model to compute preference matrices for each sample
        preference_matrices = self.B + self.C * (torch.ones(self.num_features, 1) @ self.mean_cs_features[indices, None, :])

        # Compute utility of each item under each feature MNL
        utilities = choice_set_features @ preference_matrices
        utilities[torch.arange(max_choice_set_len)[None, :] >= choice_set_lengths[:, None]] = -np.inf

        # Compute MNL log-probs for each feature
        log_probs = nn.functional.log_softmax(utilities, 1)[torch.arange(batch_size), choices]

        return - (self.r[indices] * log_probs).sum()


def train_model(model, train_data, val_data, lr, weight_decay, compute_val_stats=True, timeout_min=60, only_context_effect=None):
    torch.set_num_threads(1)

    batch_size = 128
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    start_time = time.time()

    for epoch in tqdm(range(500)):
        train_loss = 0
        train_count = 0
        train_correct = 0
        total_loss = 0

        if time.time() - start_time > timeout_min * 60:
            break

        for batch in train_data_loader:
            choices = batch[-1]
            model.train()

            choice_pred = model(*batch[:-1])
            loss = model.loss(choice_pred, choices)

            total_loss += nn.functional.nll_loss(choice_pred, choices, reduction='sum').item()

            optimizer.zero_grad()
            loss.backward()

            # If we only want to train one context effect, zero every other gradient
            if only_context_effect is not None:
                tmp = model.A.grad.data[only_context_effect].item()
                model.A.grad.data.zero_()
                model.A.grad.data[only_context_effect] = tmp
            optimizer.step()

            model.eval()
            vals, idxs = choice_pred.max(1)
            train_correct += (idxs == choices).long().sum().item() / choice_pred.size(0)
            train_loss += loss.item()
            train_count += 1

        train_accs.append(train_correct / train_count)
        train_losses.append(total_loss)

        if compute_val_stats:
            total_val_loss = 0
            val_loss = 0
            val_count = 0
            val_correct = 0
            model.eval()
            for batch in val_data_loader:
                choices = batch[-1]
                choice_pred = model(*batch[:-1])
                loss = model.loss(choice_pred, choices)
                vals, idxs = choice_pred.max(1)
                val_correct += (idxs == choices).long().sum().item() / choice_pred.size(0)
                val_loss += loss.item()

                total_val_loss += nn.functional.nll_loss(choice_pred, choices, reduction='sum').item()
                val_count += 1

            val_losses.append(val_loss)
            val_accs.append(val_correct / val_count)

    return model, train_losses, train_accs, val_losses, val_accs


def train_mnl(train_data, val_data, num_features, lr=1e-4, weight_decay=1e-4, compute_val_stats=False):
    model = MNL(num_features)
    return train_model(model, train_data, val_data, lr, weight_decay, compute_val_stats=compute_val_stats)


def train_lcl(train_data, val_data, num_features, lr=1e-4, weight_decay=1e-4, compute_val_stats=False, l1_reg=0):
    model = LCL(num_features, l1_reg=l1_reg)
    return train_model(model, train_data, val_data, lr, weight_decay, compute_val_stats=compute_val_stats)


def train_dlcl(train_data, val_data, num_features, lr=1e-4, weight_decay=1e-4, compute_val_stats=False, l1_reg=0):
    model = DLCL(num_features, l1_reg=l1_reg)
    return train_model(model, train_data, val_data, lr, weight_decay, compute_val_stats=compute_val_stats)


def train_mixed_logit(train_data, val_data, num_features, lr=1e-4, weight_decay=1e-4, compute_val_stats=False):
    model = MixedLogit(num_features)
    return train_model(model, train_data, val_data, lr, weight_decay, compute_val_stats=compute_val_stats)


def context_mixture_em(train_data, num_features, lr=0.005, epochs=100, detailed_return=False, timeout_min=60, initialize_from=None):
    torch.set_num_threads(1)

    n = num_features

    choice_set_features, choice_set_lengths, choices = train_data

    B = torch.ones(n, n, requires_grad=False).float()
    C = torch.zeros(n, n, requires_grad=False).float()
    alpha = torch.ones(n, requires_grad=False).float() / n

    if initialize_from is not None:
        B = initialize_from.B.clone().detach()
        C = initialize_from.A.clone().detach()
        alpha = torch.softmax(initialize_from.thetas.clone().detach(), 0)

    # Compute mean of each feature over each choice set
    batch_size, max_choice_set_len, _ = choice_set_features.size()
    mean_choice_set_features = choice_set_features.sum(1) / choice_set_lengths[:, None]
    nan_idx = torch.arange(max_choice_set_len)[None, :] >= choice_set_lengths[:, None]

    train_data_loader = DataLoader([choice_set_features, choice_set_lengths, choices, torch.arange(len(choices))], batch_size=128)

    start_time = time.time()
    iter_times = []
    losses = []

    model = DLCL(num_features)

    model.B.data = B
    model.A.data = C
    model.mixture_weights.data = torch.log(alpha)

    # Loop until timeout or gradient norm < 10^-6
    while True:

        # Use learned linear context model to compute utility matrices for each sample
        utility_matrices = B + C * (torch.ones(n, 1) @ mean_choice_set_features[:, None, :])

        # Compute utility of each item under each feature MNL
        utilities = choice_set_features @ utility_matrices
        utilities[nan_idx] = -np.inf

        # Compute MNL log-probs for each feature, pick out only chosen items
        log_probs = nn.functional.log_softmax(utilities, 1)[torch.arange(batch_size), choices]

        responsibilities = nn.functional.softmax(log_probs + torch.log(alpha), 1)
        alpha = responsibilities.sum(0) / batch_size

        Q = EMAlgorithmQ(num_features, responsibilities, mean_choice_set_features, B, C)
        optimizer = torch.optim.Adam(Q.parameters(), lr=lr, weight_decay=0, amsgrad=True)

        prev_loss = np.inf
        total_loss = np.inf

        for epoch in range(epochs):
            prev_loss = total_loss
            total_loss = 0
            for batch in train_data_loader:
                loss = Q(*batch)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # print(total_loss)

            if timeout_min is not None:
                if time.time() - start_time > timeout_min * 60:
                    break

        if timeout_min is not None:
            if time.time() - start_time > timeout_min * 60:
                break

        B = Q.B.clone().detach()
        C = Q.C.clone().detach()

        model.B.data = B
        model.A.data = C
        model.mixture_weights.data = torch.log(alpha)

        nll = torch.nn.functional.nll_loss(model(choice_set_features, choice_set_lengths), choices, reduction='sum')
        # print('Loss:', nll.item())

        losses.append(nll.item())
        iter_times.append(time.time() - start_time)

        model.zero_grad()
        nll.backward()
        grad_norm = torch.cat(
            [model.B.grad.flatten(), model.A.grad.flatten(), model.mixture_weights.grad.flatten()], dim=0).norm(
            p=2).item()
        if grad_norm < 10 ** -6:
            break

    if detailed_return:
        return model, losses, iter_times

    return model


def toy_example():
    x_1 = [-1, 2, -1]
    x_2 = [-1, 1, -2]
    x_3 = [1, 0, 1]
    x_4 = [1, -3, 2]
    z = [0, 0, 0]

    choice_set_features = torch.tensor(
        [[x_1, x_2, z, z],
         [x_1, x_3, z, z],
         [x_1, x_2, x_3, x_4],
         [x_2, x_3, z, z],
         [x_1, x_4, z, z],
         [x_2, x_3, x_4, z]]
    ).float()

    choice_set_lengths = torch.tensor([2, 2, 1, 2, 2, 3])

    # Indices into choice_sets
    choices = torch.tensor([0, 0, 3, 1, 0, 2])

    train_data = choice_set_features, choice_set_lengths, choices

    model, train_losses, train_accs, val_losses, val_accs = train_lcl(train_data, train_data, num_features=3, lr=0.1, weight_decay=0.001)
    print(model.theta.data, model.A.data)


if __name__ == '__main__':
    toy_example()
