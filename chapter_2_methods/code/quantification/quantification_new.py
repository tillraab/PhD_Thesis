import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from plottools.colors import *
from plottools.tag import tag
colors_params(colors_muted, colors_tableau)
from IPython import embed
import pandas as pd
import os
import sys
import scipy.stats as scp
from IPython import embed
from tqdm import tqdm

def gauss(t, shift, sigma, size, norm = False):
    g = np.exp(-((t - shift) / sigma) ** 2 / 2) * size
    if norm:
        g = g / np.sum(g) / (t[1] - t[0])
        # print(np.sum(g) * (t[1] - t[0]))
    return g

def boltzmann(t, alpha=0.25, beta=0.0, x0=4, dx=0.85):
    """
    Calulates a boltzmann function.

    Parameters
    ----------
    t: array
        time vector.
    alpha: float
        max value of the boltzmann function.
    beta: float
        min value of the boltzmann function.
    x0: float
        time where the turning point of the boltzmann function occurs.
    dx: float
        slope of the boltzman function.

    Returns
    -------
    array
        boltzmann function of the given time array base on the other parameters given.
    """

    boltz = (alpha - beta) / (1. + np.exp(- (t - x0) / dx)) + beta
    return boltz

def kde(target_param, altern_param, max_signal, sigma):

    error_steps = np.linspace(0, max_signal * 501 / 500, 500)

    kde_target = np.zeros(len(error_steps))
    for e in tqdm(target_param, desc='target'):
        kde_target += gauss(error_steps, e, sigma, 1, norm=True)

    kde_altern = np.zeros(len(error_steps))
    for e in tqdm(altern_param, desc='altern'):
        kde_altern += gauss(error_steps, e, sigma, 1, norm=True)

    return error_steps, kde_target, kde_altern

def hist_kde(target_param, altern_param, sigma_factor = 1/10, max_signal = None):
    if max_signal == None:
        help_array = np.concatenate((target_param, altern_param))
        error_steps = np.linspace(0, np.max(help_array) * 501 / 500, 500)
    else:
        error_steps = np.linspace(0, max_signal * 501 / 500, 500)

    kde_target = np.zeros(len(error_steps))
    for e in tqdm(target_param, desc='target'):
        kde_target += gauss(error_steps, e, np.std(target_param) * sigma_factor, 1, norm=True)

    kde_altern = np.zeros(len(error_steps))
    for e in tqdm(altern_param, desc='altern'):
        kde_altern += gauss(error_steps, e, np.std(altern_param) * sigma_factor, 1, norm=True)


    bin_edges = np.linspace(0, np.max(help_array), int(5 * (1/sigma_factor)))

    n_tar, _ = np.histogram(target_param, bin_edges)
    n_tar = n_tar / np.sum(n_tar) / (bin_edges[1] - bin_edges[0])
    n_alt, _ = np.histogram(altern_param, bin_edges)
    n_alt = n_alt / np.sum(n_alt) / (bin_edges[1] - bin_edges[0])

    return error_steps, kde_target, kde_altern, bin_edges, n_tar, n_alt

def roc_analysis(error_steps, target_param, altern_param):
    roc_steps = np.sort(np.unique(np.concatenate((target_param, altern_param))))
    # embed()
    # quit()
    roc_steps = np.hstack([[0], roc_steps, [roc_steps[-1]*1.01]])

    true_pos = np.ones(len(roc_steps))
    false_pos = np.ones(len(roc_steps))

    for i in tqdm(range(len(roc_steps))):
        true_pos[i] = len(np.array(target_param)[np.array(target_param) < roc_steps[i]]) / len(target_param)
        false_pos[i] = len(np.array(altern_param)[np.array(altern_param) < roc_steps[i]]) / len(altern_param)
    auc_value = np.sum(true_pos[:-1] * np.diff(false_pos))

    return true_pos, false_pos, auc_value, roc_steps

def main(data_set = None):
    if data_set == None:
        folder = './2016-04-10_5min'
    elif data_set == 'f':
        folder = './2016-04-10_full'
    else:
        quit()

    a_error_dist = np.load(os.path.join(folder, 'quantification/a_error_dist.npy'), allow_pickle=True)
    error_col_load = np.load(os.path.join(folder, 'quantification/error_col.npy'), allow_pickle=True).item()
    error_col_load = pd.DataFrame(error_col_load)
    error_col = error_col_load[(~np.isnan(error_col_load.alternID)) & (~np.isnan(error_col_load.targetID))]


    #########################################
    fig = plt.figure(figsize=(8.5 / 2.54, 16 / 2.54))
    gs = gridspec.GridSpec(6, 2, left=0.15, bottom=0.075, right=0.98, top=0.95, wspace=0.5, hspace=0.8, height_ratios=[5, 5, 1, 5, 5, 5], width_ratios=[4, 1])

    ax = []
    ax_auc = []
    for i in [0, 1, 3, 4, 5]:
        ax.append(fig.add_subplot(gs[i, 0]))
        ax_auc.append(fig.add_subplot(gs[i, 1]))
        ax_auc[-1].set_axis_off()

    ax_m = []
    ax_m.append(ax[0].twinx())
    ax_m.append(ax[1].twinx())
    ax_m[0].plot(np.linspace(0, 3, 1000),
                 boltzmann(np.linspace(0, 3, 1000), alpha=1, beta=0, x0=.35, dx=.08),
                 color='k')
    ax_m[0].set_ylim(bottom=0)
    ax_m[0].set_yticks([0, 1])
    ax_m[0].set_ylabel(r'$\varepsilon_{f}$', fontsize=10)

    ax_m[1].plot(a_error_dist[np.argsort(a_error_dist)], np.linspace(0, 1, len(a_error_dist)), color='k')
    ax_m[1].set_ylim(bottom=0)
    ax_m[1].set_yticks([0, 1])
    ax_m[1].set_ylabel(r'$\varepsilon_{S}$', fontsize=10)

    for enu, key0, key1, name in zip(np.arange(5), ['target_dfreq', 'target_dfield', 'target_freq_e', 'target_field_e', 'target_signal_e'],
                                     ['altern_dfreq', 'altern_dfield', 'altern_freq_e', 'altern_field_e', 'altern_signal_e'],
                                     ['dfreq', 'dfield', 'freq_e', 'field_e', 'signal_e']):
        # hist_kde(error_col[key0], error_col[key0], sigma = 0.2)
        if enu < 2:
            sigma, max_signal = 0.2, 3
        else:
            sigma, max_signal = 0.02, 0.7

        error_steps, kde_target, kde_altern = kde(error_col[key0], error_col[key1], max_signal, sigma)

        true_pos, false_pos, auc_value, roc_steps = roc_analysis(error_steps, error_col[key0], error_col[key1])

        ax[enu].plot(error_steps, kde_target / len(error_col[key0]), lw=2)
        ax[enu].plot(error_steps, kde_altern / len(error_col[key1]), lw=2)

        ax[enu].set_ylabel('KDE')
        ax[enu].set_xlim(error_steps[0], error_steps[-1])


        ax_auc[enu].set_xlim(0, 1)
        ax_auc[enu].set_ylim(0, 1)
        ax_auc[enu].text(0.5, 0.5, '%.2f' % (auc_value * 100) + '%', fontsize=10, ha='center', va='center')


    ax[0].set_xlim(0, 3)
    ax[1].set_xlim(0, 3)
    ax[0].set_ylim(0, 4)
    ax[1].set_ylim(0, 2.5)

    for a in ax[2:]:
        a.set_xlim(0, .6)
        a.set_ylim(-1, 30)
    # ax[2].set_xlim(0, .6)
    # ax[3].set_xlim(0, .6)
    # ax[4].set_xlim(0, .6)

    ax[0].set_xlabel(r'$\Delta f$ [Hz]', fontsize=10, labelpad=-1)
    ax[1].set_xlabel(r'$\Delta S$', fontsize=10, labelpad=-1)
    ax[2].set_xlabel(r'$\varepsilon_{f}$', fontsize=10, labelpad=-1)
    ax[3].set_xlabel(r'$\varepsilon_{S}$', fontsize=10, labelpad=-1)
    ax[4].set_xlabel(r'$\varepsilon$', fontsize=10, labelpad=-1 )

    ax[0].set_title('signal differences', fontsize=10, fontweight='bold')
    ax[2].set_title('signal errors', fontsize=10, fontweight='bold')

    fig.align_ylabels(ax)

    ax_auc[0].set_title('AUC', fontsize=10, fontweight='bold')

    for a in ax:
        a.tick_params(labelsize=9)
    plt.show()

if __name__ == '__main__':
    full = None
    if len(sys.argv) > 1:
        full = 'f' if sys.argv[1] == 'f' else None
    main(full)