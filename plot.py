import pickle

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import torch
from matplotlib import patheffects
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import chi2
from sklearn.manifold import TSNE

import datasets
from datasets import WikiTalkDataset, RedditHyperlinkDataset, BitcoinAlphaDataset, BitcoinOTCDataset, SMSADataset, \
    SMSBDataset, SMSCDataset, EmailEnronDataset, EmailEUDataset, EmailW3CDataset, FacebookWallDataset, \
    CollegeMsgDataset, MathOverflowDataset, SyntheticLCLDataset, SyntheticMNLDataset, SushiDataset, ExpediaDataset, \
    CarAltDataset, DistrictSmartDataset
from models import MNL, LCL, DLCL, MixedLogit

PARAM_DIR = 'params'
RESULT_DIR = 'results'
CONFIG_DIR = 'hyperparams'


def load_model(Model, model_param, param_fname):
    model = Model(model_param)
    model.load_state_dict(torch.load(param_fname))
    model.eval()

    return model


def compute_all_accuracies(datasets):
    methods = [MNL, LCL, MixedLogit, DLCL]

    losses = [list() for _ in range(len(methods))]
    accs = [list() for _ in range(len(methods))]
    mean_correct_positions = [list() for _ in range(len(methods))]

    all_correct_preds = [list() for _ in range(len(methods))]
    all_correct_positions = [list() for _ in range(len(methods))]

    for i, dataset in enumerate(datasets):
        print('Computing accuracies for', dataset.name)
        graph, train_data, val_data, test_data, _, _ = dataset.load_standardized()

        histories, history_lengths, choice_sets, choice_sets_with_features, choice_set_lengths, choices = test_data

        non_singleton_set = choice_set_lengths > 1
        choice_sets = choice_sets[non_singleton_set]
        choice_sets_with_features = choice_sets_with_features[non_singleton_set]
        choice_set_lengths = choice_set_lengths[non_singleton_set]
        choices = choices[non_singleton_set]

        for j, method in enumerate(methods):
            lr, wd = dataset.best_val_lr_wd(method)
            param_fname = f'{PARAM_DIR}/{method.name}_{dataset.name}_train_params_{lr}_{wd}.pt'

            model_param = dataset.num_features
            model = load_model(method, model_param, param_fname)

            pred = model(choice_sets_with_features, choice_set_lengths)
            train_loss = model.loss(pred, choices)

            ranks = stats.rankdata(-pred.detach().numpy(), method='average', axis=1)[np.arange(len(choices)), choices] - 1
            vals, idxs = pred.max(1)

            correct_preds = (idxs == choices)
            acc = correct_preds.long().sum().item() / len(choices)

            losses[j].append(train_loss.item())
            accs[j].append(acc)
            mean_correct_positions[j].append(np.mean(ranks / (np.array(choice_set_lengths) - 1)))
            all_correct_preds[j].append(correct_preds.numpy())
            all_correct_positions[j].append(ranks / (np.array(choice_set_lengths) - 1))

    with open(f'{RESULT_DIR}/all_prediction_results.pickle', 'wb') as f:
        pickle.dump([np.array(losses), np.array(accs), np.array(mean_correct_positions), np.array(all_correct_positions), np.array(all_correct_preds)], f)

def sci_not(num):
    string = f'{num:#.2g}'
    if 'e' in string:
        split = string.split('e')
        string = f'{split[0]} \\times 10^{{{int(split[1])}}}'

    if num < 10**-16:
        string = '< 10^{-16}'
    return string


def make_prediction_table(datasets):
    print('\n\nPrediction table:')

    with open(f'{RESULT_DIR}/all_prediction_results.pickle', 'rb') as f:
        losses, accs, mean_correct_positions, all_correct_positions, all_correct = pickle.load(f)


    n_cols = 8
    fig, axes = plt.subplots(3, n_cols, figsize=(15, 4.5))
    fig_num = 0

    colors = ["#dc622c", "#8b4acc", "#84c440", "#d93b8e"]
    names = ['CL', 'LCL', 'Mixed logit', 'DLCL']
    markers = ['o', '^', 's', 'D']

    for j, dataset in enumerate(datasets):
        mnl = mean_correct_positions[0, j]
        lcl = mean_correct_positions[1, j]
        mixed_mnl = mean_correct_positions[2, j]
        dlcl = mean_correct_positions[3, j]

        mnl_positions = all_correct_positions[0, j]
        lcl_positions = all_correct_positions[1, j]
        mixed_mnl_positions = all_correct_positions[2, j]
        dlcl_positions = all_correct_positions[3, j]

        for i, mrrs in enumerate([mnl_positions, lcl_positions, mixed_mnl_positions, dlcl_positions]):
            axes[fig_num // n_cols, fig_num % n_cols].errorbar(i, np.mean(mrrs), yerr=stats.sem(mrrs), color=colors[i],
                                                               fmt=markers[i], capsize=2, label=names[i])



        # axes[fig_num // n_cols, fig_num % n_cols].text(0.5, 0.85, dataset.name.replace('-mnl', '-cl'), ha='center', va='center',
        #                                                bbox=dict(facecolor='white', alpha=0.8, pad=0, edgecolor='white',
        #                                                          boxstyle='round,pad=0.2'),
        #                                                transform=axes[fig_num // n_cols, fig_num % n_cols].transAxes)
        axes[fig_num // n_cols, fig_num % n_cols].set_title(dataset.name.replace('-mnl', '-cl'), fontsize=11, pad=3)
        axes[fig_num // n_cols, fig_num % n_cols].set_xticks([])
        axes[fig_num // n_cols, fig_num % n_cols].set_xlim(-0.5, 3.5)
        # axes[fig_num // n_cols, fig_num % n_cols].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        axes[fig_num // n_cols, fig_num % n_cols].yaxis.set_tick_params(labelsize=8)

        stds = [np.std(mnl_positions), np.std(lcl_positions), np.std(mixed_mnl_positions), np.std(dlcl_positions)]

        try:
            mnl_lcl_wilcoxon_W, mnl_lcl_wilcoxon_p = stats.wilcoxon(mnl_positions, lcl_positions)
        except ValueError:
            mnl_lcl_wilcoxon_W = ''
            mnl_lcl_wilcoxon_p = 1

        try:
            mmnl_dlcl_wilcoxon_W, mmnl_dlcl_wilcoxon_p = stats.wilcoxon(mixed_mnl_positions, dlcl_positions)
        except ValueError:
            mmnl_dlcl_wilcoxon_W = ''
            mmnl_dlcl_wilcoxon_p = 1
        p_thresh = 0.001

        print(f'\\textsc{{{dataset.name.replace("-mnl", "-cl")}}}', end='')
        for i, val in enumerate([mnl, lcl, mixed_mnl, dlcl]):
            sig_mark = ''
            if i == 1:
                if mnl_lcl_wilcoxon_p < p_thresh:
                    sig_mark = '$^*$'
                    # txt = axes[fig_num // n_cols, fig_num % n_cols].text(1, lcl, '*', ha='center', va='top', color='white', fontsize=11)
                    # txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='black')])

                else:
                    sig_mark = '\\phantom{$^*$}'
            elif i == 3:
                if mmnl_dlcl_wilcoxon_p < p_thresh:
                    sig_mark = '$^{\dagger}$'
                    # txt = axes[fig_num // n_cols, fig_num % n_cols].text(3, dlcl, '**', ha='center', va='top', color='white', fontsize=11)
                    # txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='black')])

                else:
                    sig_mark = '\\phantom{$^{\dagger}$}'

            if round(val, 4) == min(round(x, 4) for x in [mnl, lcl, mixed_mnl, dlcl]):
                print(f' & \\textbf{{{val:.4f}}}{sig_mark} ({stds[i]:.4f})'.replace('0.', '.'), end='')
            else:
                print(f' & {val:.4f}{sig_mark} ({stds[i]:.4f})'.replace('0.', '.'), end='')

            if i == 1:
                print(f' & {int(mnl_lcl_wilcoxon_W) if mnl_lcl_wilcoxon_W else "---"} & ${sci_not(mnl_lcl_wilcoxon_p)}$', end='')
            elif i == 3:
                print(f' & {int(mmnl_dlcl_wilcoxon_W) if mmnl_dlcl_wilcoxon_W else "---"} & ${sci_not(mmnl_dlcl_wilcoxon_p)}$', end='')

        print('\\\\')
        
        fig_num += 2 if j == 6 or j == 13 else 1


    axes[0, 7].legend(*axes[0, 0].get_legend_handles_labels(), prop={'size': 9})
    axes[0, 7].axis('off')
    axes[1, 7].axis('off')

    axes[0, 0].set_ylabel('Mean Rel. Rank', fontsize=10)
    axes[1, 0].set_ylabel('Mean Rel. Rank', fontsize=10)
    axes[2, 0].set_ylabel('Mean Rel. Rank', fontsize=10)


    plt.subplots_adjust(wspace=0.4, hspace=0.2)
    # plt.show()
    plt.savefig('lcl_prediction.pdf', bbox_inches='tight')
    plt.close()

def visualize_context_effects(datasets):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    cmap = mpl.cm.bwr

    all_contexts = []

    for i, dataset in enumerate(datasets):
        row = i // 4
        col = i % 4

        model = load_model(LCL, 6, f'{PARAM_DIR}/lcl_{dataset.name}_params_{dataset.best_lr(LCL)}_0.001.pt')

        contexts = model.A.data.numpy()
        all_contexts.append(contexts)

        axes[row, col].matshow(contexts, cmap=cmap)

        print(dataset.name, contexts)

        axes[row, col].axis('off')
        axes[row, col].set_title(dataset.name.replace('-mnl', '-cl'), pad=0.1)

    # norm = mpl.colors.Normalize(vmin=-1, vmax=1)

    for col in range(1, 4):
        axes[3, col].axis('off')

    vis = axes[3, 3].matshow(np.mean(all_contexts, axis=0), cmap=cmap)
    axes[3, 3].set_title('Mean', pad=0.1)

    plt.colorbar(vis, ax=axes[:, :])

    plt.savefig('learned_lcl_contexts.pdf', bbox_inches='tight')
    plt.close()


def visualize_context_effects_l1_reg(datasets, method):
    reg_params = [0, 0.001, 0.005, 0.01, 0.05, 0.1]

    results = dict()

    for dataset in datasets:
        for reg_param in reg_params:
            filename = f'{RESULT_DIR}/l1_reg_{method.name}_{dataset.name}_{reg_param}.pickle'
            with open(filename, 'rb') as f:
                results[dataset, reg_param, method] = pickle.load(f)

    with open(f'{CONFIG_DIR}/learning_rate_settings.pickle', 'rb') as f:
        grid_search_losses, lrs = pickle.load(f)

    if method == LCL:
        reg_params.remove(0.001)
    elif method == DLCL:
        reg_params = reg_params[:-2]

    fig = plt.figure(figsize=(len(reg_params), len(datasets)*1.1), constrained_layout=False)
    gs = fig.add_gridspec(len(datasets), len(reg_params), wspace=0, hspace=0.1)

    for row, dataset in enumerate(datasets):
        all_slopes = [results[dataset, reg_param, method][0].A.data.numpy() if method == DLCL else
                      results[dataset, reg_param, method][0].A.data.numpy() for reg_param in reg_params]

        max_abs = np.max(np.abs(all_slopes))
        vmin = -max_abs
        vmax = max_abs

        for col, reg_param in enumerate(reg_params):
            ax = fig.add_subplot(gs[row, col])

            if col == 0:
                ax.set_ylabel(dataset.name.replace('-mnl', '-cl'), rotation='horizontal', ha='right', fontsize=14, va='center')
            if row == 0:
                if col == 0:
                    ax.set_title(f'$\\lambda=${reg_param}', fontsize=12)
                else:
                    ax.set_title(f'{reg_param}', fontsize=12)

            model, loss = results[dataset, reg_param, method]
            ax.matshow(model.A.data.numpy() if method == DLCL else model.A.data.numpy(), cmap=mpl.cm.bwr, vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])

        losses = [results[dataset, reg_param, method][1][-1] for reg_param in reg_params]

        baseline = MixedLogit if method == DLCL else MNL
        baseline_loss = min([grid_search_losses[dataset, baseline, lr] for lr in lrs])

        p = 0.001
        ddof = dataset.num_features**2
        sig_thresh = baseline_loss - 0.5 * chi2.isf(p, ddof)

        ymax_pcts = 2

        if method == DLCL:
            if dataset == EmailEnronDataset:
                ymax_pcts = 4
            elif dataset == EmailW3CDataset:
                ymax_pcts = 10
            elif dataset == SyntheticMNLDataset:
                ymax_pcts = 0.04
        else:
            if dataset in (EmailW3CDataset, CollegeMsgDataset, SyntheticLCLDataset, BitcoinOTCDataset, SMSBDataset, SMSADataset, EmailEUDataset, BitcoinAlphaDataset):
                ymax_pcts = 4
            elif dataset == SyntheticMNLDataset:
                ymax_pcts = 0.1
            elif dataset in [MathOverflowDataset, EmailEnronDataset]:
                ymax_pcts = 6

        y_min = min(losses + [sig_thresh])
        y_ticks = [y_min, y_min * (1 + ymax_pcts / 200), y_min * (1 + ymax_pcts / 100)]

        loss_ax = fig.add_subplot(gs[row, :])
        loss_ax.plot(range(len(reg_params)), losses, color='black', alpha=0.7)
        loss_ax.set_xlim(-0.5, len(reg_params) - 0.5)

        loss_ax.set_ylim(min(y_ticks) * (1 - ymax_pcts / 1000), max(y_ticks))

        loss_ax.hlines(sig_thresh, -0.5, len(reg_params) - 0.5, linestyles='dotted', colors='green')

        loss_ax.patch.set_visible(False)
        loss_ax.set_xticks([])
        loss_ax.set_yticks(y_ticks)
        loss_ax.set_yticklabels(['', f'+{ymax_pcts/2:.2g}%', f'+{ymax_pcts:.2g}%'])

        loss_ax.yaxis.tick_right()

    plt.savefig(f'l1-regularization-{method.name}.pdf', bbox_inches='tight')
    plt.close()


def plot_binned_mnl_example():
    # _, _, _, _, mathoverflow_means, mathoverflow_stds = MathOverflowDataset.load_standardized()
    # _, _, _, _, synthetic_mnl_means, synthetic_mnl_stds = SyntheticMNLDataset.load_standardized()
    # _, _, _, _, enron_means, enron_stds = EmailEnronDataset.load_standardized()

    # Hard-coded to avoid loading whole datasets
    means = [[4.9191475 , 4.656493  , 0.2162834 , 0.13189632, 0.13194366, 0.02068096],
             [2.6338944e+00, 1.5371358e+00, 2.6449988e-02, 4.7664855e-02, 4.7703274e-02, 1.7460630e-03],
             [1.0776899,  0.5040105,  0.02805469, 0.04287676, 0.07075235, 0.00146797]]

    stds = [[0.46722016, 0.8068568 , 0.35599306, 0.03198218, 0.03193424, 0.03406612],
            [1.3703023 , 1.255706  , 0.1655428 , 0.0230962 , 0.01190723, 0.01028514],
            [1.2532787,  0.75487214, 0.21800406, 0.05251113, 0.02918332, 0.00996732]]

    with open(f'{RESULT_DIR}/{datasets.MathOverflowDataset.name}_binned_mnl_params.pickle', 'rb') as f:
        mathoverflow_data = pickle.load(f)

    with open(f'{RESULT_DIR}/{datasets.SyntheticMNLDataset.name}_binned_mnl_params.pickle', 'rb') as f:
        synthetic_mnl_data = pickle.load(f)

    with open(f'{RESULT_DIR}/{datasets.EmailEnronDataset.name}_binned_mnl_params.pickle', 'rb') as f:
        enron_data = pickle.load(f)

    fig, axes = plt.subplots(3, 2, figsize=(4.5, 5.5), sharey=True, sharex='col')

    for row, data in enumerate([synthetic_mnl_data, mathoverflow_data, enron_data]):
        for col in range(2):
            bins, mnl_utilities, bin_counts, bin_choice_set_log_lengths, bin_losses = data[col]

            # De-normalize
            bins *= stds[row][col]
            bins += means[row][col]

            nonempty = bin_counts > 0

            with_const = sm.add_constant(bins[nonempty])
            mod_wls = sm.WLS(mnl_utilities[nonempty, 1], with_const, weights=bin_counts[nonempty])
            res_wls = mod_wls.fit()
            wls_intercept, wls_slope = res_wls.params

            xs = np.exp(bins)

            # axes[row, col].scatter(xs, mnl_utilities[:, 1], alpha=0.1, s=bin_counts, marker='o', linewidths=0)
            axes[row, col].scatter(xs, mnl_utilities[:, 1], alpha=1, s=bin_counts**0.5, marker='.', color='black')

            axes[row, col].plot(xs, list(map(lambda x: wls_intercept + np.log(x) * wls_slope, xs)),
                                label='WLS', color='red')

            axes[row, col].text(0.95, 0.95, f'${wls_slope:.2f}\;\log \;x {"+" if wls_intercept > 0 else ""}{wls_intercept:.2f}$\n$r^2 = {res_wls.rsquared:.2f}$', ha='right', va='top', transform=axes[row, col].transAxes)

            axes[row, col].set_xscale('log')

    axes[0, 1].text(1.05, 0.5, 'synthetic-cl', rotation=270, size=12, ha='left', va='center', transform=axes[0, 1].transAxes)
    axes[1, 1].text(1.05, 0.5, 'mathoverflow', rotation=270, size=12, ha='left', va='center', transform=axes[1, 1].transAxes)
    axes[2, 1].text(1.05, 0.5, 'email-enron', rotation=270, size=12, ha='left', va='center', transform=axes[2, 1].transAxes)

    axes[2, 0].set_xlabel('Choice Set In-Degree')
    axes[2, 1].set_xlabel('Choice Set Shared Nbrs.')

    axes[0, 0].set_ylabel('Shared Nbrs. Coef.')
    axes[1, 0].set_ylabel('Shared Nbrs. Coef.')
    axes[2, 0].set_ylabel('Shared Nbrs. Coef.')

    axes[2, 0].set_xticks([1, 10, 100, 1000])
    axes[2, 0].set_xticklabels([1, 10, 100, ''])
    axes[2, 1].set_xticks([1, 10, 100, 1000])
    axes[2, 1].set_xticklabels([1, 10, 100, 1000])

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig('context-effect-example.pdf', bbox_inches='tight')
    plt.close()


def make_likelihood_table(all_datasets):
    print('\n\nLikelihood table:')

    with open(f'{CONFIG_DIR}/learning_rate_settings.pickle', 'rb') as f:
        data, lrs = pickle.load(f)

    methods = [MNL, LCL, MixedLogit, DLCL]

    for dataset in all_datasets:
        print(f'\\textsc{{{dataset.name.replace("-mnl", "-cl")}}}', end='')

        best_nll_overall = min(int(data[dataset, method, lr]) for lr in lrs for method in methods)

        for method in methods:
            best_nll = min(int(data[dataset, method, lr]) for lr in lrs)
            if best_nll == best_nll_overall:
                print(f' & \\textbf{{{best_nll}}}', end='')
            else:
                print(f' & {best_nll}', end='')

            if method in [LCL, DLCL]:
                sig_thresh = 0.001

                baseline = MixedLogit if method == DLCL else MNL
                baseline_nll = min([data[dataset, baseline, lr] for lr in lrs])

                if stats.chi2.sf(2 * (baseline_nll - best_nll), dataset.num_features ** 2) < sig_thresh:
                    print('$^*$' if method == LCL else '$^{\dagger}$', end='')
                else:
                    print('\phantom{$^*$}' if method == LCL else '\phantom{$^{\dagger}$}', end='')

        print(r'\\')


def make_big_likelihood_table(all_datasets):
    print('\n\nBig likelihood table:')
    with open(f'{CONFIG_DIR}/learning_rate_settings.pickle', 'rb') as f:
        data, lrs = pickle.load(f)

    methods = [MNL, LCL, MixedLogit, DLCL]

    for dataset in all_datasets:
        print(f'\\textsc{{{dataset.name.replace("-mnl", "-cl")}}}', end='')

        best_nll_overall = min(int(data[dataset, method, lr]) for lr in lrs for method in methods)

        for method in methods:
            best_nll = min(int(data[dataset, method, lr]) for lr in lrs)
            if best_nll == best_nll_overall:
                print(f' & \\textbf{{{best_nll}}}', end='')
            else:
                print(f' & {best_nll}', end='')

            if method in [LCL, DLCL]:
                sig_thresh = 0.001

                baseline = MixedLogit if method == DLCL else MNL
                baseline_nll = min([data[dataset, baseline, lr] for lr in lrs])

                statistic = 2 * (baseline_nll - best_nll)
                p_val = stats.chi2.sf(statistic, dataset.num_features ** 2)
                if p_val < sig_thresh:
                    print('$^*$' if method == LCL else '$^{\dagger}$', end='')
                else:
                    print('\phantom{$^*$}' if method == LCL else '\phantom{$^{\dagger}$}', end='')

                print(f' & ${round(statistic)}$ & ${sci_not(p_val)}$', end='')

        print(r'\\')


def visualize_context_effects_l1_reg_general_choice_dataset(dataset, method):
    reg_params = [0, 0.001, 0.005, 0.01, 0.05, 0.1]

    results = dict()

    for reg_param in reg_params:
        filename = f'{RESULT_DIR}/l1_reg_{method.name}_{dataset.name}_{reg_param}.pickle'
        with open(filename, 'rb') as f:
            results[dataset, reg_param, method] = pickle.load(f)

    with open(f'{CONFIG_DIR}/learning_rate_settings.pickle', 'rb') as f:
        grid_search_losses, lrs = pickle.load(f)

    if method == LCL:
        reg_params.remove(0.001)
    elif method == DLCL:
        reg_params = reg_params[:-2]

    fig = plt.figure(figsize=(len(reg_params), 1), constrained_layout=False)
    gs = fig.add_gridspec(1, len(reg_params), wspace=0, hspace=0)

    all_slopes = [results[dataset, reg_param, method][0].A.data.numpy() if method == DLCL else
                  results[dataset, reg_param, method][0].A.data.numpy() for reg_param in reg_params]

    max_abs = np.max(np.abs(all_slopes))
    vmin = -max_abs
    vmax = max_abs

    for col, reg_param in enumerate(reg_params):
        ax = fig.add_subplot(gs[col])

        if col == 0:
            ax.set_ylabel(dataset.name.replace('-mnl', '-cl'), rotation='horizontal', ha='right', fontsize=14, va='center')
            ax.set_title(f'$\\lambda=${reg_param}', fontsize=12)
        else:
            ax.set_title(f'{reg_param}', fontsize=12)

        model, loss = results[dataset, reg_param, method]

        ax.matshow(model.A.data.numpy() if method == DLCL else model.A.data.numpy(), cmap=mpl.cm.seismic, vmin=vmin, vmax=vmax, interpolation='nearest')

        ax.set_xticks([])
        ax.set_yticks([])

    losses = [results[dataset, reg_param, method][1][-1] for reg_param in reg_params]

    baseline = MixedLogit if method == DLCL else MNL
    baseline_loss = min([grid_search_losses[dataset, baseline, lr] for lr in lrs])

    p = 0.001
    ddof = dataset.num_features**2
    sig_thresh = baseline_loss - 0.5 * chi2.isf(p, ddof)

    ymax_pcts = 2

    if method == DLCL:
        if dataset == DistrictSmartDataset:
            ymax_pcts = 4
        elif dataset == ExpediaDataset:
            ymax_pcts = 0.2
    else:
        if dataset == DistrictSmartDataset:
            ymax_pcts = 4
        elif dataset == ExpediaDataset:
            ymax_pcts = 0.4
        elif dataset == CarAltDataset:
            ymax_pcts = 12

    y_min = min(losses + [sig_thresh])
    y_ticks = [y_min, y_min * (1 + ymax_pcts / 200), y_min * (1 + ymax_pcts / 100)]

    loss_ax = fig.add_subplot(gs[:])
    loss_ax.plot(range(len(reg_params)), losses, color='black', alpha=0.7)
    loss_ax.set_xlim(-0.5, len(reg_params) - 0.5)

    loss_ax.set_ylim(min(y_ticks) * (1 - ymax_pcts / 1000), max(y_ticks))

    loss_ax.hlines(sig_thresh, -0.5, len(reg_params) - 0.5, linestyles='dotted', colors='green')

    loss_ax.patch.set_visible(False)
    loss_ax.set_xticks([])
    loss_ax.set_yticks(y_ticks)
    loss_ax.set_yticklabels(['', f'+{ymax_pcts/2:.2g}%', f'+{ymax_pcts:.2g}%'])

    loss_ax.yaxis.tick_right()

    plt.savefig(f'l1_regularization_{dataset.name}_{method.name}.pdf', bbox_inches='tight')
    plt.close()


def make_context_effect_significance_table(dataset, num=5):

    with open(f'{RESULT_DIR}/context_effect_significance.pickle', 'rb') as f:
        data = pickle.load(f)

    theta, theta_std, theta_p, A, A_std, A_p = data[dataset]

    sorted_rows, sorted_cols = np.unravel_index(np.argsort(np.abs(A).ravel()), (dataset.num_features, dataset.num_features))

    for i in range(num):
        row, col = sorted_rows[-1-i], sorted_cols[-1-i]
        print(f'\\emph{{{dataset.feature_names[col].lower()}}} on \\emph{{{dataset.feature_names[row].lower()}}} & ${A[row, col]:.2f}$ $({A_std[row, col]:.2f}) $ & ${sci_not(A_p[row, col])}$\\\\')

    print('base uts:')
    for i, name in enumerate(dataset.feature_names):
        print(name, f'{theta[i]:.2f} \pm {theta_std[i]:.2g}')
    print()


def compare_em_to_sgd(datasets):
    print('\n\nEM vs SGD table:')
    with open(f'{CONFIG_DIR}/learning_rate_settings.pickle', 'rb') as f:
        data, lrs = pickle.load(f)

    em_lrs = [0.001, 0.01, 0.1]
    em_epochs = [10, 50, 100]

    with open(f'{CONFIG_DIR}/em_lr_epoch_settings.pickle', 'rb') as f:
        em_data, _, em_lrs, em_epochs = pickle.load(f)

    for dataset in datasets:
        sgd_nll = min(int(data[dataset, DLCL, lr]) for lr in lrs)

        em_nll = min([int(em_data[dataset, lr, epochs][1][-1]) for lr in em_lrs for epochs in em_epochs if len(em_data[dataset, lr, epochs][1]) > 0])

        sgd = f'{sgd_nll}'
        if sgd_nll == min(sgd_nll, em_nll):
            sgd = f'\\textbf{{{sgd_nll}}}'

        em = f'{em_nll}'
        if em_nll == min(sgd_nll, em_nll):
            em = f'\\textbf{{{em_nll}}}'

        print(f'\\textsc{{{dataset.name.replace("-mnl", "-cl")}}} & {sgd} & {em}\\\\')


if __name__ == '__main__':
    make_context_effect_significance_table(ExpediaDataset)
    make_context_effect_significance_table(SushiDataset)

    make_likelihood_table(datasets.ALL_DATASETS)
    make_big_likelihood_table(datasets.ALL_DATASETS)
    compare_em_to_sgd(datasets.ALL_DATASETS)

    plot_binned_mnl_example()

    visualize_context_effects_l1_reg_general_choice_dataset(SushiDataset, LCL)
    visualize_context_effects_l1_reg_general_choice_dataset(DistrictSmartDataset, LCL)
    visualize_context_effects_l1_reg_general_choice_dataset(ExpediaDataset, LCL)
    visualize_context_effects_l1_reg_general_choice_dataset(CarAltDataset, LCL)

    visualize_context_effects_l1_reg(datasets.NETWORK_DATASETS, LCL)
    visualize_context_effects_l1_reg(datasets.NETWORK_DATASETS, DLCL)

    compute_all_accuracies(datasets.ALL_DATASETS)
    make_prediction_table(datasets.ALL_DATASETS)


