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
from tqdm import tqdm

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
    def plot_signal_difference():

        fig = plt.figure(figsize=(15 / 2.54, 10 / 2.54))
        gs = gridspec.GridSpec(2, 2, left=0.15, bottom=0.15, right=0.95, top=0.95, wspace=0.6, hspace=0.6,
                               width_ratios=[2, 1])
        ax = []
        ax.append(fig.add_subplot(gs[0, 0]))
        ax.append(fig.add_subplot(gs[1, 0]))
        ax_auc = []
        ax_auc.append(fig.add_subplot(gs[0, 1]))
        ax_auc.append(fig.add_subplot(gs[1, 1]))

        ax_m = []
        ax_m.append(ax[0].twinx())
        ax_m.append(ax[1].twinx())
        # ax_m = ax[0].twinx()
        ax_m[0].plot(np.linspace(0, 2.5, 1000),
                     boltzmann(np.linspace(0, 2.5, 1000), alpha=1, beta=0, x0=.35, dx=.08),
                     color='k')
        ax_m[0].set_ylim(bottom=0)
        ax_m[0].set_yticks([0, 1])
        ax_m[0].set_ylabel(r'$\varepsilon_{f}$', fontsize=10)

        ax_m[1].plot(a_error_dist[np.argsort(a_error_dist)], np.linspace(0, 1, len(a_error_dist)), color='k')
        ax_m[1].set_ylim(bottom=0)
        ax_m[1].set_yticks([0, 1])
        ax_m[1].set_ylabel(r'$\varepsilon_{S}$', fontsize=10)

        for enu, key0, key1, name in zip(np.arange(2), ['target_dfreq', 'target_dfield'],
                                         ['altern_dfreq', 'altern_dfield'],
                                         ['dfreq', 'dfield']):
            # for enu, key0, key1 in zip(np.arange(2), ['target_freq_e', 'target_field_e'], ['altern_freq_e', 'altern_field_e']):

            error_steps = np.load(os.path.join(folder, './quantification/error_steps_%s.npy' % name),
                                  allow_pickle=True)
            bin_edges = np.load(os.path.join(folder, './quantification/bin_edges_%s.npy' % name), allow_pickle=True)

            kde_target = np.load(os.path.join(folder, './quantification/kde_target_%s.npy' % name),
                                 allow_pickle=True)
            kde_altern = np.load(os.path.join(folder, './quantification/kde_altern_%s.npy' % name),
                                 allow_pickle=True)
            n_tar = np.load(os.path.join(folder, './quantification/n_tar_%s.npy' % name), allow_pickle=True)
            n_alt = np.load(os.path.join(folder, './quantification/n_alt_%s.npy' % name), allow_pickle=True)

            true_pos = np.load(os.path.join(folder, './quantification/true_pos_%s.npy' % name), allow_pickle=True)
            false_pos = np.load(os.path.join(folder, './quantification/false_pos_%s.npy' % name), allow_pickle=True)
            auc_value = np.load(os.path.join(folder, './quantification/auc_value_%s.npy' % name), allow_pickle=True)

            # true_pos, false_pos, auc_value, roc_steps = roc_analysis(error_steps, error_col[key0], error_col[key1])
            #
            # np.save(os.path.join(folder, './quantification/true_pos_%s.npy' % name), true_pos)
            # np.save(os.path.join(folder, './quantification/false_pos_%s.npy' % name), false_pos)
            # np.save(os.path.join(folder, './quantification/auc_value_%s.npy' % name), auc_value)

            print('')
            print(name)
            print('correct: %.0f; %.2f' % (len(error_col[error_col[key0] < error_col[key1]]),
                                           len(error_col[error_col[key0] < error_col[key1]]) / len(
                                               error_col) * 100) + '%')
            print('all: %.0f' % len(error_col))
            print('AUC: %.2f' % (auc_value * 100.))

            target_handle, = ax[enu].plot(error_steps, kde_target / len(error_col[key0]), lw=2)
            altern_handle, = ax[enu].plot(error_steps, kde_altern / len(error_col[key1]), lw=2)

            ax[enu].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, n_tar,
                        width=(bin_edges[1] - bin_edges[0]) * 0.8, alpha=0.4, color=target_handle.get_color(),
                        align='center')
            ax[enu].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, n_alt,
                        width=(bin_edges[1] - bin_edges[0]) * 0.8, alpha=0.4, color=altern_handle.get_color(),
                        align='center')

            ax[enu].set_ylabel('KDE')
            ax[enu].set_xlim(error_steps[0], error_steps[-1])
            ax[enu].set_ylim(0, np.max(np.concatenate((n_tar, n_alt))) * 1.1)

            ax_auc[enu].fill_between(false_pos, np.zeros(len(false_pos)), true_pos, color='#999999')
            ax_auc[enu].plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=2)
            ax_auc[enu].text(0.95, 0.05, '%.1f' % (auc_value * 100) + ' %', fontsize=9, color='k', ha='right',
                             va='bottom')
            ax_auc[enu].set_xlim(0, 1)
            ax_auc[enu].set_ylim(0, 1)

            ax_auc[enu].set_xticks([0, 1])
            ax_auc[enu].set_yticks([0, 1])

        ax[0].set_xlabel(r'$\Delta f$ [Hz]', fontsize=10)
        ax[1].set_xlabel(r'field difference ($\Delta S$)', fontsize=10)

        ax_auc[0].set_ylabel('true positive', fontsize=10)
        ax_auc[1].set_ylabel('true positive', fontsize=10)
        ax_auc[1].set_xlabel('false positive', fontsize=10)

        for a in np.hstack([ax, ax_auc, ax_m]):
            a.tick_params(labelsize=9)
        fig.tag(axes=[ax], labels=['A', 'B'], fontsize=15, yoffs=2, xoffs=-6)

        if data_set == None:
            plt.savefig(os.path.join(folder, 'freq_field_difference.pdf'))
        elif data_set == 'f':
            plt.savefig(os.path.join(folder, 'freq_field_difference_full.pdf'))
        else:
            pass

    def plot_signal_errors():
        fig = plt.figure(figsize=(15 / 2.54, 14 / 2.54))
        gs = gridspec.GridSpec(3, 2, left=0.15, bottom=0.1, right=0.95, top=0.95, hspace=0.6, wspace=0.4,
                               height_ratios=[4, 4, 4], width_ratios=[2.5, 1])
        ax = []
        ax.append(fig.add_subplot(gs[0, 0]))
        ax.append(fig.add_subplot(gs[1, 0]))
        ax.append(fig.add_subplot(gs[2, 0]))

        ax_roc = []
        ax_roc.append(fig.add_subplot(gs[0, 1]))
        ax_roc.append(fig.add_subplot(gs[1, 1]))
        ax_roc.append(fig.add_subplot(gs[2, 1]))

        # ax_auc = fig.add_subplot(fig.add_subplot(gs[-1, :]))

        for enu, key0, key1, name in zip(np.arange(5),
                                         ['target_freq_e', 'target_field_e', 'target_signal_e'],
                                         ['altern_freq_e', 'altern_field_e', 'altern_signal_e'],
                                         ['freq_e', 'field_e', 'signal_e']):

            error_steps = np.load(os.path.join(folder, './quantification/error_steps_%s.npy' % name), allow_pickle=True)
            bin_edges = np.load(os.path.join(folder, './quantification/bin_edges_%s.npy' % name), allow_pickle=True)

            kde_target = np.load(os.path.join(folder, './quantification/kde_target_%s.npy' % name), allow_pickle=True)
            kde_altern = np.load(os.path.join(folder, './quantification/kde_altern_%s.npy' % name), allow_pickle=True)
            n_tar = np.load(os.path.join(folder, './quantification/n_tar_%s.npy' % name), allow_pickle=True)
            n_alt = np.load(os.path.join(folder, './quantification/n_alt_%s.npy' % name), allow_pickle=True)

            true_pos = np.load(os.path.join(folder, './quantification/true_pos_%s.npy' % name), allow_pickle=True)
            false_pos = np.load(os.path.join(folder, './quantification/false_pos_%s.npy' % name), allow_pickle=True)
            auc_value = np.load(os.path.join(folder, './quantification/auc_value_%s.npy' % name), allow_pickle=True)

            # true_pos, false_pos, auc_value, roc_steps = roc_analysis(error_steps, error_col[key0], error_col[key1])
            #
            # np.save(os.path.join(folder, './quantification/true_pos_%s.npy' % name), true_pos)
            # np.save(os.path.join(folder, './quantification/false_pos_%s.npy' % name), false_pos)
            # np.save(os.path.join(folder, './quantification/auc_value_%s.npy' % name), auc_value)

            print('')
            print(name)
            print('correct: %.0f; %.2f' % (len(error_col[error_col[key0] < error_col[key1]]),
                                           len(error_col[error_col[key0] < error_col[key1]]) / len(
                                               error_col) * 100) + '%')
            print('all: %.0f' % len(error_col))
            print('AUC: %.2f' % (auc_value * 100.))

            target_handle, = ax[enu].plot(error_steps, kde_target / len(error_col[key0]))
            altern_handle, = ax[enu].plot(error_steps, kde_altern / len(error_col[key1]))

            ax[enu].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, n_tar,
                        width=(bin_edges[1] - bin_edges[0]) * 0.8, alpha=0.5, color=target_handle.get_color())
            ax[enu].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, n_alt,
                        width=(bin_edges[1] - bin_edges[0]) * 0.8, alpha=0.5, color=altern_handle.get_color())

            ax[enu].set_ylabel('KDE', fontsize=10)
            # ax.set_xlabel(key0)
            help_array = np.concatenate((error_col[key0], error_col[key1]))
            ax[enu].set_xlim(0, np.percentile(help_array, 95))
            ax[enu].set_ylim(0, np.max(np.concatenate((n_tar, n_alt))) * 1.1)

            ax_roc[enu].fill_between(false_pos, np.zeros(len(false_pos)), true_pos, color='#999999')
            ax_roc[enu].plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=2)
            ax_roc[enu].text(0.95, 0.05, '%.1f' % (auc_value * 100), fontsize=10, color='k', ha='right', va='bottom')
            ax_roc[enu].set_xlim(0, 1)
            ax_roc[enu].set_ylim(0, 1)
            ax_roc[enu].set_xticks([0, 1])
            ax_roc[enu].set_yticks([0, 1])
            ax_roc[enu].set_ylabel('true positive', fontsize=10)
            if enu == 2:
                ax_roc[enu].set_xlabel('false positive', fontsize=10)

        ax[0].set_xlabel(r'$\varepsilon_{f}$', fontsize=10)
        ax[1].set_xlabel(r'$\varepsilon_{S}$', fontsize=10)
        ax[2].set_xlabel(r'$\varepsilon$', fontsize=10)

        for a in np.concatenate((ax, ax_roc)):
            a.tick_params(labelsize=9)
        fig.tag(axes=ax, fontsize=15, yoffs=1, xoffs=-6)

        if data_set == None:
            plt.savefig(os.path.join(folder, 'freq_field_signal_error.pdf'))
        elif data_set == 'f':
            plt.savefig(os.path.join(folder, 'freq_field_signal_error_full.pdf'))
        else:
            pass

    if data_set == None:
        folder = './2016-04-10_5min'
    elif data_set == 'f':
        folder = './2016-04-10_full'

        # ToDo: load ident_v and disregard the tracking hell !!!
        # ToDo: eliminate those where any ID is nan
    else:
        quit()

    a_error_dist = np.load(os.path.join(folder, 'quantification/a_error_dist.npy'), allow_pickle=True)
    error_col_load = np.load(os.path.join(folder, 'quantification/error_col.npy'), allow_pickle=True).item()

    error_col_load = pd.DataFrame(error_col_load)
    error_col = error_col_load[(~np.isnan(error_col_load.alternID)) & (~np.isnan(error_col_load.targetID))]
    # embed()
    # quit()

    print(folder)
    print('n = %.0f connections' % len(error_col))

    #plot_signal_difference()

    #plot_signal_errors()

    ########################################################################################################

    fig = plt.figure(figsize=(9 / 2.54, 17.5 / 2.54))
    gs = gridspec.GridSpec(6, 2, left=0.15, bottom=0.075, right=1, top=0.95, wspace=0.4, hspace=0.8, height_ratios=[5, 5, 1, 5, 5, 5], width_ratios=[4, 1])

    ax = []
    ax_auc = []
    for i in [0, 1, 3, 4, 5]:
        ax.append(fig.add_subplot(gs[i, 0]))
        ax_auc.append(fig.add_subplot(gs[i, 1]))
        ax_auc[-1].set_axis_off()

    ax_m = []
    ax_m.append(ax[0].twinx())
    ax_m.append(ax[1].twinx())
    ax_m[0].plot(np.linspace(0, 2.5, 1000),
                 boltzmann(np.linspace(0, 2.5, 1000), alpha=1, beta=0, x0=.35, dx=.08),
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
        error_steps = np.load(os.path.join(folder, './quantification/error_steps_%s.npy' % name),
                              allow_pickle=True)
        bin_edges = np.load(os.path.join(folder, './quantification/bin_edges_%s.npy' % name), allow_pickle=True)

        kde_target = np.load(os.path.join(folder, './quantification/kde_target_%s.npy' % name),
                             allow_pickle=True)
        kde_altern = np.load(os.path.join(folder, './quantification/kde_altern_%s.npy' % name),
                             allow_pickle=True)
        n_tar = np.load(os.path.join(folder, './quantification/n_tar_%s.npy' % name), allow_pickle=True)
        n_alt = np.load(os.path.join(folder, './quantification/n_alt_%s.npy' % name), allow_pickle=True)

        true_pos = np.load(os.path.join(folder, './quantification/true_pos_%s.npy' % name), allow_pickle=True)
        false_pos = np.load(os.path.join(folder, './quantification/false_pos_%s.npy' % name), allow_pickle=True)
        auc_value = np.load(os.path.join(folder, './quantification/auc_value_%s.npy' % name), allow_pickle=True)

        true_pos, false_pos, auc_value, roc_steps = roc_analysis(error_steps, error_col[key0], error_col[key1])
        # np.save(os.path.join(folder, './quantification/true_pos_%s.npy' % name), true_pos)
        # np.save(os.path.join(folder, './quantification/false_pos_%s.npy' % name), false_pos)
        # np.save(os.path.join(folder, './quantification/auc_value_%s.npy' % name), auc_value)
        print('')
        print(name)
        print('correct: %.0f; %.2f' % (len(error_col[error_col[key0] < error_col[key1]]),
                                       len(error_col[error_col[key0] < error_col[key1]]) / len(
                                           error_col) * 100) + '%')
        print('all: %.0f' % len(error_col))
        print('AUC: %.2f' % (auc_value * 100.))

        target_handle, = ax[enu].plot(error_steps, kde_target / len(error_col[key0]), lw=2)
        altern_handle, = ax[enu].plot(error_steps, kde_altern / len(error_col[key1]), lw=2)

        ax[enu].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, n_tar,
                    width=(bin_edges[1] - bin_edges[0]) * 0.8, alpha=0.4, color=target_handle.get_color(),
                    align='center')
        ax[enu].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, n_alt,
                    width=(bin_edges[1] - bin_edges[0]) * 0.8, alpha=0.4, color=altern_handle.get_color(),
                    align='center')

        ax[enu].set_ylabel('KDE')
        ax[enu].set_xlim(error_steps[0], error_steps[-1])
        ax[enu].set_ylim(0, np.max(np.concatenate((n_tar, n_alt))) * 1.1)

        ax_auc[enu].set_xlim(0, 1)
        ax_auc[enu].set_ylim(0, 1)
        ax_auc[enu].text(0.5, 0.5, '%.2f' % (auc_value * 100) + '%', fontsize=10, ha='center', va='center')

    ax[0].set_xlabel(r'$\Delta f$ [Hz]', fontsize=10)
    ax[1].set_xlabel(r'field difference ($\Delta S$)', fontsize=10)
    ax[2].set_xlabel(r'$\varepsilon_{f}$', fontsize=10)
    ax[3].set_xlabel(r'$\varepsilon_{S}$', fontsize=10)
    ax[4].set_xlabel(r'$\varepsilon$', fontsize=10)

    ax[0].set_title('signal differences', fontsize=10, fontweight='bold')
    ax[2].set_title('signal errors', fontsize=10, fontweight='bold')

    ax_auc[0].set_title('AUC', fontsize=10, fontweight='bold')
    for a in ax:
        a.tick_params(labelsize=9)
    fig.tag(axes=ax, fontsize=15, yoffs=1, xoffs=-6)


    plt.savefig('./signal_diff_error.pdf')
    plt.show()
    embed()
    quit()

if __name__ == '__main__':
    full = None
    if len(sys.argv) > 1:
        full = 'f' if sys.argv[1] == 'f' else None
    main(full)