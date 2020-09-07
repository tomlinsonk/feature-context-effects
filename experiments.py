import pickle
from multiprocessing.pool import Pool

import pickle
from multiprocessing.pool import Pool

import numpy as np
import torch
from tqdm import tqdm

from datasets import ALL_DATASETS, SushiDataset, ExpediaDataset, CarAltDataset
from models import train_mnl, MNL, LCL, train_lcl, DLCL, train_dlcl, context_mixture_em, train_mixed_logit, MixedLogit, \
    train_model

training_methods = {MNL: train_mnl, LCL: train_lcl, DLCL: train_dlcl, MixedLogit: train_mixed_logit}

CONFIG_DIR = 'hyperparams'


def run_feature_model_full_dataset(method, dataset, lr, wd):
    graph, train_data, val_data, test_data, means, stds = dataset.load_standardized()

    all_data = [torch.cat([train_data[i], val_data[i], test_data[i]]) for i in range(3, len(train_data))]

    print(f'Training {method.name} on {dataset.name} (lr={lr}, wd={wd})')

    model, train_losses, train_accs, val_losses, val_accs = training_methods[method](all_data, val_data, dataset.num_features, lr=lr, weight_decay=wd, compute_val_stats=False)
    torch.save(model.state_dict(), f'{method.name}_{dataset.name}_params_{lr}_{wd}.pt')
    with open(f'{method.name}_{dataset.name}_losses_{lr}_{wd}.pickle', 'wb') as f:
        pickle.dump((train_losses, train_accs, val_losses, val_accs), f)

    return model


def run_feature_model_train_data(method, dataset, lr, wd):
    graph, train_data, val_data, test_data, means, stds = dataset.load_standardized()

    print(f'Training {method.name} on {dataset.name}, training data only (lr={lr}, wd={wd})')

    model, train_losses, train_accs, val_losses, val_accs = training_methods[method](train_data[3:], val_data[3:], dataset.num_features, lr=lr, weight_decay=wd, compute_val_stats=False)
    torch.save(model.state_dict(), f'{method.name}_{dataset.name}_train_params_{lr}_{wd}.pt')
    with open(f'{method.name}_{dataset.name}_train_losses_{lr}_{wd}.pickle', 'wb') as f:
        pickle.dump((train_losses, train_accs, val_losses, val_accs), f)

    return model


def compile_choice_data(dataset):
    graph, train_data, val_data, test_data, means, stds = dataset.load_standardized()

    n_feats = dataset.num_features

    histories, history_lengths, choice_sets, choice_set_features, choice_set_lengths, choices = [
        torch.cat([train_data[i], val_data[i], test_data[i]]).numpy() for i in range(len(train_data))]

    chosen_item_features = [list() for _ in range(n_feats)]

    choice_set_mean_features = [list() for _ in range(n_feats)]

    for i in range(len(choice_set_features)):
        choice = choices[i]
        choice_set = choice_set_features[i, :choice_set_lengths[i]]

        for feature in range(n_feats):
            chosen_item_features[feature].append(choice_set[choice, feature])
            choice_set_mean_features[feature].append(np.mean(choice_set[:, feature]))

    return histories, history_lengths, choice_sets, choice_set_features, choice_set_lengths, choices, \
        np.array(chosen_item_features), np.array(choice_set_mean_features)


def learn_binned_mnl(dataset):
    print(f'Learning binned MNLs for {dataset.name}')
    num_bins = 100

    n_feats = dataset.num_features

    histories, history_lengths, choice_sets, choice_set_features, choice_set_lengths, choices, \
        chosen_item_features, choice_set_mean_features = compile_choice_data(dataset)

    data = [None for _ in range(n_feats)]

    for i, x_var in enumerate(choice_set_mean_features):
        values, bins = np.histogram(x_var, bins=np.linspace(min(x_var), max(x_var), num_bins))

        all_bin_idx = np.digitize(x_var, bins)

        mnl_utilities = np.zeros((num_bins, n_feats))
        bin_counts = np.zeros(num_bins)
        bin_choice_set_log_lengths = np.zeros(num_bins)
        bin_losses = np.zeros(num_bins)

        for bin in tqdm(range(num_bins)):
            bin_idx = all_bin_idx == bin

            bin_counts[bin] = np.count_nonzero(bin_idx)

            if bin_counts[bin] == 0:
                continue

            bin_data = [torch.tensor(choice_set_features[bin_idx]), torch.tensor(choice_set_lengths[bin_idx]), torch.tensor(choices[bin_idx])]
            mnl, train_losses, _, _, _ = train_mnl(bin_data, bin_data, n_feats, lr=0.01, weight_decay=0.001)
            mnl_utilities[bin] = mnl.thetas.detach().numpy()
            bin_choice_set_log_lengths[bin] = np.mean(np.log(choice_set_lengths[bin_idx]))
            bin_losses[bin] = torch.nn.functional.nll_loss(mnl(*bin_data[:-1]), bin_data[-1], reduction='sum').item()

        data[i] = bins, mnl_utilities, bin_counts, bin_choice_set_log_lengths, bin_losses

    with open(f'{dataset.name}_binned_mnl_params.pickle', 'wb') as f:
        pickle.dump(data, f)


def run_all_models(dataset, wd, models):
    for model in models:
        torch.random.manual_seed(0)
        np.random.seed(0)
        run_feature_model_full_dataset(model, dataset, dataset.best_lr(model), wd)


def train_context_mixture_em(dataset):
    print('Running EM for', dataset.name)
    graph, train_data, val_data, test_data, means, stds = dataset.load_standardized()
    all_data = [torch.cat([train_data[i], val_data[i], test_data[i]]) for i in range(3, len(train_data))]

    model = context_mixture_em(all_data, dataset.num_features)
    torch.save(model.state_dict(), f'context_mixture_em_{dataset.name}_params.pt')


def learning_rate_grid_search_helper(args):
    dataset, method, lr = args
    graph, train_data, val_data, test_data, means, stds = dataset.load_standardized()
    all_data = [torch.cat([train_data[i], val_data[i], test_data[i]]) for i in range(3, len(train_data))]

    torch.random.manual_seed(0)
    np.random.seed(0)
    model, train_losses, train_accs, val_losses, val_accs = training_methods[method](all_data, val_data,
                                                                                     dataset.num_features,
                                                                                     lr=lr, weight_decay=0.001,
                                                                                     compute_val_stats=False)
    return args, train_losses[-1]


def learning_rate_grid_search(datasets, methods, update=False):
    lrs = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    params = {(dataset, method, lr) for dataset in datasets for lr in lrs for method in methods}

    results = dict()

    pool = Pool(30)

    for args, loss in tqdm(pool.imap_unordered(learning_rate_grid_search_helper, params), total=len(params)):
        results[args] = loss

    pool.close()
    pool.join()

    filename = f'{CONFIG_DIR}/learning_rate_settings.pickle'

    if update:
        with open(filename, 'rb') as f:
            old_results, old_lrs = pickle.load(f)

        old_results.update(results)
        results = old_results
        lrs = sorted(set(old_lrs).union(lrs))

    with open(filename, 'wb') as f:
        pickle.dump((results, lrs), f)


def validation_loss_grid_search_helper(args):
    dataset, method, lr, wd = args
    graph, train_data, val_data, test_data, means, stds = dataset.load_standardized()

    torch.random.manual_seed(0)
    np.random.seed(0)
    model, train_losses, train_accs, val_losses, val_accs = training_methods[method](train_data[3:], val_data[3:],
                                                                                     dataset.num_features,
                                                                                     lr=lr, weight_decay=wd,
                                                                                     compute_val_stats=True)
    return args, (train_losses, train_accs, val_losses, val_accs)


def validation_loss_grid_search(datasets, methods, update=False):
    lrs = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    wds = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01]

    params = {(dataset, method, lr, wd) for dataset in datasets for lr in lrs for method in methods for wd in wds}

    results = dict()

    pool = Pool(30)

    for args, losses in tqdm(pool.imap_unordered(validation_loss_grid_search_helper, params), total=len(params)):
        results[args] = losses

    pool.close()
    pool.join()

    filename = f'{CONFIG_DIR}/validation_loss_lr_wd_settings.pickle'

    if update:
        with open(filename, 'rb') as f:
            old_results, old_datasets, old_methods, old_lrs, old_wds = pickle.load(f)

        old_results.update(results)
        results = old_results
        datasets = list(set(old_datasets).union(datasets))
        lrs = sorted(set(old_lrs).union(lrs))
        wds = sorted(set(old_wds).union(wds))

    with open(filename, 'wb') as f:
        pickle.dump((results, datasets, methods, lrs, wds), f)


def l1_regularization_helper(args):
    dataset, reg_param, method = args
    graph, train_data, val_data, test_data, means, stds = dataset.load_standardized()
    all_data = [torch.cat([train_data[i], val_data[i], test_data[i]]) for i in range(3, len(train_data))]

    torch.random.manual_seed(0)
    np.random.seed(0)
    model, train_losses, train_accs, val_losses, val_accs = training_methods[method](all_data, val_data,
        dataset.num_features,
        lr=dataset.best_lr(method),
        weight_decay=0.001,
        compute_val_stats=False,
        l1_reg=reg_param)

    filename = f'l1_reg_{method.name}_{dataset.name}_{reg_param}.pickle'

    with open(filename, 'wb') as f:
        pickle.dump((model, train_losses), f)


def train_with_l1_regularization(datasets, method):
    reg_params = [0, 0.001, 0.005, 0.01, 0.05, 0.1]
    params = [(dataset, reg_param, method) for dataset in datasets for reg_param in reg_params]
    pool = Pool(30)

    for _ in tqdm(pool.imap_unordered(l1_regularization_helper, params), total=len(params)):
        pass

    pool.close()
    pool.join()


def all_experiments_helper(dataset):
    learn_binned_mnl(dataset)
    train_context_mixture_em(dataset)
    run_all_models(dataset, 0.001, [LCL, DLCL, MixedLogit, MNL])


def train_data_training_helper(args):
    dataset, method = args
    torch.random.manual_seed(0)
    np.random.seed(0)
    lr, wd = dataset.best_val_lr_wd(method)
    run_feature_model_train_data(method, dataset, lr, wd)


def train_data_training(datasets, methods):
    pool = Pool(30)
    params = {(dataset, method) for dataset in datasets for method in methods}

    for _ in tqdm(pool.imap_unordered(train_data_training_helper, params), total=len(params)):
        pass

    pool.close()
    pool.join()


def all_experiments(datasets):
    pool = Pool(30)
    pool.map(all_experiments_helper, datasets)
    pool.close()
    pool.join()


def em_grid_search_helper(args):
    dataset, lr, epochs = args
    graph, train_data, val_data, test_data, means, stds = dataset.load_standardized()
    all_data = [torch.cat([train_data[i], val_data[i], test_data[i]]) for i in range(3, len(train_data))]

    torch.random.manual_seed(0)
    np.random.seed(0)
    model, losses, times = context_mixture_em(all_data, dataset.num_features, lr=lr, epochs=epochs, detailed_return=True)

    return args, (model, losses, times)


def em_grid_search(datasets):
    lrs = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    epochs = [5, 10, 50, 100]

    params = {(dataset, lr, epoch) for dataset in datasets for lr in lrs for epoch in epochs}

    results = dict()
    pool = Pool(30)

    for args, losses in tqdm(pool.imap_unordered(em_grid_search_helper, params), total=len(params)):
        results[args] = losses

    pool.close()
    pool.join()

    filename = f'{CONFIG_DIR}/em_lr_epoch_settings.pickle'
    with open(filename, 'wb') as f:
        pickle.dump((results, datasets, lrs, epochs), f)


def check_lcl_identifiability(datasets):
    for dataset in datasets:
        graph, train_data, val_data, test_data, means, stds = dataset.load_standardized()
        choice_set_features, choice_set_lengths = [torch.cat([train_data[i], val_data[i], test_data[i]]).numpy() for i in range(3, len(train_data)-1)]

        n, max_choice_set_len, d = np.shape(choice_set_features)
        m = np.sum(choice_set_lengths)

        means = np.sum(choice_set_features, axis=1) / choice_set_lengths[:, None]

        repeated_means = np.repeat(means, choice_set_lengths, axis=0)
        repeated_mean_1s = np.append(repeated_means, np.ones((m, 1)), axis=1)

        mean_1s = np.append(means, np.ones((n, 1)), axis=1)
        choice_sets_rank = np.linalg.matrix_rank(mean_1s)

        all_shifted_feat_vecs = choice_set_features[np.arange(max_choice_set_len)[None, :] < choice_set_lengths[:, None]] - repeated_means

        # Kronecker product along axis (https://stackoverflow.com/questions/50676698/numpy-kron-along-a-given-axis)
        krons = (repeated_mean_1s[:, :, None] * all_shifted_feat_vecs[:, None, :]).reshape(m, -1)
        kron_rank = np.linalg.matrix_rank(krons)

        kron_text = f'\\textbf{{{kron_rank}/{d**2 + d}}}' if kron_rank == d**2 + d else f'{kron_rank}/{d**2 + d}'
        choice_set_text = f'\\textbf{{{choice_sets_rank}/{d + 1}}}' if choice_sets_rank == d+1 else f'{choice_sets_rank}/{d+1}'

        print(f'\\textsc{{{dataset.name}}} & {kron_text} & {choice_set_text}\\\\')


def biggest_context_effect_helper(args):
    dataset, index, row, col, A_pq = args

    graph, train_data, val_data, test_data, means, stds = dataset.load_standardized()
    all_data = [torch.cat([train_data[i], val_data[i], test_data[i]]) for i in range(3, len(train_data))]

    model, train_losses, _, _, _ = train_model(LCL(dataset.num_features), all_data, all_data,
                                                        dataset.best_lr(LCL), 0.001, False, 60, (row, col))

    return (dataset, index), (row, col, A_pq, model.A[row, col].item(), train_losses)


def biggest_context_effects(datasets, num=5):
    params = []

    for dataset in datasets:
        model = LCL(dataset.num_features)
        model.load_state_dict(torch.load(f'params/lcl_{dataset.name}_params_{dataset.best_lr(LCL)}_0.001.pt'))
        contexts = model.A.data.numpy()

        max_abs_idx = np.dstack(np.unravel_index(np.argsort(-abs(contexts).ravel()), contexts.shape))[0]

        for index, (row, col) in enumerate(max_abs_idx[:num]):
            params.append((dataset, index, row, col, contexts[row, col]))

    results = dict()
    pool = Pool(30)

    for key, val in tqdm(pool.imap_unordered(biggest_context_effect_helper, params), total=len(params)):
        results[key] = val

    pool.close()
    pool.join()

    filename = f'biggest_context_effects.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    methods = [MixedLogit, MNL, DLCL, LCL]

    # Fix for RuntimeError: received 0 items of ancdata (https://github.com/pytorch/pytorch/issues/973)
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


    validation_loss_grid_search(ALL_DATASETS, methods, update=False)
    train_data_training(ALL_DATASETS, methods)

    learning_rate_grid_search(ALL_DATASETS, methods, update=False)
    train_with_l1_regularization(ALL_DATASETS, LCL)
    train_with_l1_regularization(ALL_DATASETS, DLCL)

    em_grid_search(ALL_DATASETS)

    all_experiments(ALL_DATASETS)

    biggest_context_effects([SushiDataset, ExpediaDataset, CarAltDataset])

    check_lcl_identifiability(ALL_DATASETS)




