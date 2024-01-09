import sys
import os
import time
import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
from thunderfish.version import __version__
from thunderfish.powerspectrum import decibel, next_power_of_two, spectrogram
from thunderfish.dataloader import open_data, fishgrid_grids, fishgrid_spacings
from thunderfish.harmonics import harmonic_groups, fundamental_freqs
# from signal_tracker import freq_tracking_v5, plot_tracked_traces, Emit_progress
from thunderfish.eventdetection import hist_threshold
from plottools.tag import tag
import multiprocessing
from functools import partial

from IPython import embed
def load(filename):
    data = open_data(filename, -1, 60, 10)
    samplerate = data.samplerate
    channels = data.channels
    return data, samplerate, channels

def plot_spec(psd, freq):
    fig = plt.figure(figsize=(20/2.54, 20/2.54))
    gs = gridspec.GridSpec(8, 8, left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)
    ax = []
    for i in range(8):
        for j in range(8):
            if len(ax) == 0:
                ax.append(fig.add_subplot(gs[j, i]))
            else:
                ax.append(fig.add_subplot(gs[j, i], sharex=ax[0], sharey=ax[0]))

            if i != 0:
                plt.setp(ax[-1].get_yticklabels(), visible=False)
                ax[-1].yaxis.set_ticks_position('none')
            if j != 7:
                plt.setp(ax[-1].get_xticklabels(), visible=False)
                ax[-1].xaxis.set_ticks_position('none')
    for i in range(len(psd)):
        ax[i].plot(freq, decibel(psd[i]), 'k')
    ax[0].set_xlim(400, 3000)


    # fig = plt.figure()
    # gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1, top=0.95, right=0.95)
    # ax = fig.add_subplot(gs[0, 0])
    # ax.plot(freq, decibel(psd[39]))
    # ax.set_xlim(400, 3000)
    # plt.show()

def calc_specs(data, samplerate, idx0, channels):
    # embed()
    # quit()
    psd = []
    sum_spec = None

    core_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(core_count - 1)
    nfft = next_power_of_two(samplerate / 0.5)

    func = partial(spectrogram, ratetime=samplerate, freq_resolution=0.5, overlap_frac=0.9)
    a = pool.map(func, [data[idx0: idx0 + 180*samplerate, ch] for ch in range(64)])

    spectra = [a[channel][0] for channel in range(len(a))]
    spec_freqs = a[0][1]
    spec_times = a[0][2]
    pool.terminate()

    comb_spectra = np.sum(spectra, axis=0)

    # fig, ax = plt.subplots()
    # ax.imshow(decibel(comb_spectra)[::-1], extent=[spec_times[0], spec_times[-1], spec_freqs[0], spec_freqs[-1]],
    #           aspect='auto', alpha=0.7, cmap='jet', vmax = -50, vmin = -110, interpolation='gaussian', zorder=1)
    # ax.set_ylim(880, 950)
    #
    # fig = plt.figure(figsize=(20/2.54, 20/2.54))
    # gs = gridspec.GridSpec(8, 8, left=0.1, bottom=0.1, right=0.95, top=0.95)
    # ax = []
    # for i in range(8):
    #     for j in range(8):
    #         if (i, j) == (0, 0):
    #             ax.append(fig.add_subplot(gs[i, j]))
    #         else:
    #             ax.append(fig.add_subplot(gs[i, j], sharex=ax[0], sharey=ax[0]))
    #         # if i == 3 and j == 3:
    #         #     break
    #         ax[-1].imshow(decibel(a[int(i*4 + j)][0])[::-1], extent=[spec_times[0], spec_times[-1], spec_freqs[0], spec_freqs[-1]],
    #           aspect='auto', alpha=0.7, cmap='jet', vmax = -50, vmin = -120, interpolation='gaussian', zorder=1)
    # ax[-1].set_ylim(880, 950)
    dt = spec_times[1] - spec_times[0]
    time_idx = np.arange(len(spec_times))[spec_times >= 50][0]

    # embed()
    # quit()

    fig = plt.figure(figsize=(17.5/2.54, 20/2.54))
    gs = gridspec.GridSpec(1, 8, left=0.1, bottom=0.8, right=1, top=0.975, width_ratios=[3, 1, 0.5, 3, 1, 0.5, 3, 1], wspace=0)
    ax_e_sp = []
    ax_e_psd = []
    for i in range(3):
        ax_e_sp.append(fig.add_subplot(gs[0, int(0 + i*3)]))
        ax_e_psd.append(fig.add_subplot(gs[0, int(1 + i*3)]))

    gs2 = gridspec.GridSpec(2, 1, left=0.2, bottom=0.1, right=0.825, top=0.65, hspace=0.6)
    ax_main = fig.add_subplot(gs2[0, 0])
    ax = fig.add_subplot(gs2[1, 0])



    # fig = plt.figure(figsize=(17.5/2.54, 10/2.54))
    # gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.3, right=0.55, top=0.85)
    # ax_main = fig.add_subplot(gs[0, 0])
    #
    # gs2 = gridspec.GridSpec(3, 2, left=0.7, bottom=0.2, right=0.95, top=0.95, width_ratios=[3, 1], wspace=0, hspace=0.3)
    # ax_e_sp = []
    # ax_e_psd = []
    # for i in range(3):
    #     ax_e_sp.append(fig.add_subplot(gs2[i, 0]))
    #     ax_e_psd.append(fig.add_subplot(gs2[i, 1]))

    

    ax_main.imshow(decibel(comb_spectra)[::-1], extent=[0, 180, spec_freqs[0], spec_freqs[-1]],
              aspect='auto', alpha=0.7, cmap='jet', vmax = -50, vmin = -110, interpolation='gaussian', zorder=1)
    ax_main.plot([50, 50], [890, 930], linestyle='dotted', lw=2, color='k')
    # ax_main.set_ylim(875, 950)
    ax_main.set_ylim(890, 930)
    ax_main.set_xticks(np.arange(0, 181, 60))
    ax_main.set_xticklabels([])
    # ax_main.set_xticklabels(np.arange(0, 4, 1))

    max_x = None
    for enu, i in enumerate(np.array([23, 43, 61], dtype=int)):
        ax_e_sp[enu].imshow(decibel(a[i][0])[::-1], extent=[0, 180, spec_freqs[0], spec_freqs[-1]],
                            aspect='auto', alpha=0.7, cmap='jet', vmax=-50, vmin=-110, interpolation='gaussian', zorder=1)
        ax_e_sp[enu].plot([50, 50], [890, 930], linestyle='dotted', lw=2, color='k')
        ax_e_sp[enu].set_yticks([890, 910, 930])
        if enu != 0:
            ax_e_sp[enu].set_yticklabels([])
        ax_e_sp[enu].set_ylim(890, 930)
        ax_e_sp[enu].set_xticks(np.arange(0, 181, 60))
        # ax_e_sp[enu].set_xticklabels(np.arange(0, 4, 1))
        ax_e_sp[enu].set_xticklabels([])
        # ax_e_sp[enu].set_xticks([])

        ax_e_psd[enu].plot(a[i][0][:, time_idx], spec_freqs, 'k')
        ax_e_psd[enu].set_yticks([])
        ax_e_psd[enu].set_ylim(890, 930)

        if enu != 2:
            plt.setp(ax_e_psd[enu].get_xticklabels(), visible=False)

        if not hasattr(max_x, '__len__'):
            max_x = np.max(a[23][0][:, time_idx])
        ax_e_psd[enu].set_xlim(0, max_x/2)

        ax_e_psd[enu].set_axis_off()

        ax_e_sp[enu].tick_params(labelsize=10)
        ax_e_psd[enu].tick_params(labelsize=10)

    ax_main.tick_params(labelsize=9)

    ax_main.text(90, 935, r'$\sum{power(E_n)}$', ha = 'center', va='center', fontsize=11)
    ax_e_sp[0].text(180, 931, r'$power(E_{23})$', ha='right', va='bottom', fontsize=8)
    ax_e_sp[1].text(180, 931, r'$power(E_{43})$', ha='right', va='bottom', fontsize=8)
    ax_e_sp[2].text(180, 931, r'$power(E_{61})$', ha='right', va='bottom', fontsize=8)


    # ax_main.set_ylabel(r'EOD$f$ [Hz]', fontsize=10)
    ax_main.set_ylabel('frequency [Hz]', fontsize=10)
    # ax_main.set_xlabel('time [min]', fontsize=10)

    # ax_e_sp[0].set_ylabel(r'EOD$f$ [Hz]', fontsize=10)
    ax_e_sp[0].set_ylabel('frequency [Hz]', fontsize=10)
    # ax_e_sp[0].set_xlabel('time [min]', fontsize=10)
    # ax_e_sp[1].set_xlabel('time [min]', fontsize=10)
    # ax_e_sp[2].set_xlabel('time [min]', fontsize=10)

    ax_e_sp[0].tick_params(labelsize=9)
    ax_e_sp[1].tick_params(labelsize=9)
    ax_e_sp[2].tick_params(labelsize=9)


    # plt.savefig('elec_specs.pdf')
    # plt.savefig('elec_specs2.png', dpi = 300)
    # ax[0].set_ylim(400, 2000)


    ###################################################

    res = harmonic_groups(spec_freqs, comb_spectra[:, time_idx], low_threshold_factor=6.0, high_threshold_factor=10.0,
                          min_freq=400, max_freq=2000, mains_freq=50, min_group_size=2)
    # res = harmonic_groups(spec_freqs, comb_spectra[:, time_idx], check_freqs = [920.5, 917.5, 910.3, 904.6, 895.4],
    #                       low_threshold_factor=6.0, high_threshold_factor=10.0, min_freq=400, max_freq=2000,
    #                       mains_freq=50, min_group_size=2)

    #fund_freqs = fundamental_freqs(res[0])

    # fig = plt.figure(figsize=(20/2.54, 12/2.54))
    # gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.15, right=0.95, top=0.9)
    # ax = fig.add_subplot(gs[0, 0])
    ax.plot(spec_freqs, decibel(comb_spectra[:, time_idx+1]), color='midnightblue', lw=1)

    res[0][0] = res[0][0][1:]
    # for fish in res[0]:
    for i in [1, 2, 3, 0, 4, 5]:
        fish = res[0][i]
        c = np.random.rand(3)
        for enu, h in enumerate(fish):
            if enu == 0:
                ax.plot(h[0], decibel(h[1]), 'o', color=c, label='%.1f' % h[0])
            else:
                ax.plot(h[0], decibel(h[1]), 'o', color=c)

    ax.plot([500, 1200], [-50, -50], color='grey', lw=5)
    ax.text(850, -49, 'fundamental', fontsize=9, va='bottom', ha='center')
    ax.plot([500, 500], [-50, -110], lw=1, linestyle='dotted', color='grey')
    ax.plot([1200, 1200], [-50, -110], lw=1, linestyle='dotted', color='grey')

    ax.plot([1300, 1900], [-50, -50], color='grey', lw=5)
    ax.text(1600, -49, r'$1^{st}$ harmonic', fontsize=9, va='bottom', ha='center')
    ax.plot([1300, 1300], [-50, -110], lw=1, linestyle='dotted', color='grey')
    ax.plot([1900, 1900], [-50, -110], lw=1, linestyle='dotted', color='grey')

    ax.plot([2000, 3000], [-50, -50], color='grey', lw=5)
    ax.text(2500, -49, r'$2^{nd}$ harmonic', fontsize=9, va='bottom', ha='center')
    ax.plot([2000, 2000], [-50, -110], lw=1, linestyle='dotted', color='grey')
    ax.plot([2995, 2995], [-50, -110], lw=1, linestyle='dotted', color='grey')

    ax.set_xlim(400, 3000)
    ax.set_ylim(-110, -45)
    #ax.legend(loc='upper center', frameon=False, ncol=3, bbox_to_anchor=(0.4, 1.3), fontsize=9)
    ax.legend(loc=1, frameon=False, ncol=1, bbox_to_anchor=(1.2, 0.9), fontsize=9)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xlabel('freqeuncy [Hz]', fontsize=10)
    ax.set_ylabel('power [dB]', fontsize=10)
    ax.tick_params(labelsize=9)

    con = ConnectionPatch(xyA=(0, 890), xyB=(0, 930), coordsA="data", coordsB="data",
                          axesA=ax_e_sp[0], axesB=ax_main, color="k", linestyle='--', zorder=10, lw=1.5)
    ax_main.add_artist(con)

    con = ConnectionPatch(xyA=(180, 890), xyB=(180, 930), coordsA="data", coordsB="data",
                          axesA=ax_e_sp[2], axesB=ax_main, color="k", linestyle='--', zorder=10, lw=1.5)
    ax_main.add_artist(con)

    con = ConnectionPatch(xyA=(50, 890), xyB=(400, -45), coordsA="data", coordsB="data",
                          axesA=ax_main, axesB=ax, color="k", linestyle='--', zorder=10, lw=1.5)
    ax_main.add_artist(con)

    con = ConnectionPatch(xyA=(50, 890), xyB=(3000, -45), coordsA="data", coordsB="data",
                          axesA=ax_main, axesB=ax, color="k", linestyle='--', zorder=10, lw=1.5)
    ax_main.add_artist(con)

    # ax_e_sp[0].plot([120, 180], [885, 885], lw=3, clip_on=False)
    for enu, Cax in enumerate(ax_e_sp):
        shift=0
        if enu == 2:
            shift = 120
        Cax.fill_between([120-shift, 180-shift], [886, 886], [887.5, 887.5], color='k', clip_on=False)
        Cax.text(150-shift, 885.5, '1 min', fontsize=9, va='top',  ha='center', clip_on=False)
    ax_main.fill_between([150, 180], [886, 886], [887, 887], color='k', clip_on=False)
    ax_main.text(165, 885.5, '30 sec', fontsize=9, va='top', ha='center', clip_on=False)

    fig.tag(axes=[ax_e_sp[0]], labels=['A'], fontsize=15, yoffs=2, xoffs=-6)
    fig.tag(axes=[ax_main, ax], labels=['B', 'C'], fontsize=15, yoffs=2, xoffs=-14)

    fig.align_ylabels([ax_main, ax])

    plt.savefig('signal_extraction.pdf')
    plt.savefig('signal_extraction2.png', dpi=300)
def main():
    #data_col = np.load('./colombia_example_15sec.npy')
    #data_col = np.load('./tube_comp_5min_10rises.npy')

    # tube competition
    # data, samplerate, channels = load('/home/raab/data/2019_tube_competition/2020-05-28-10_00/traces-grid1.raw')
    # idx0 = (89.5) * 60 * samplerate
    # didx = 20 * 0.001 * samplerate # 20 ms


    #colmbia file
    data, samplerate, channels = load('/home/raab/data/2016-colombia/2016-04-10-11_12/traces-grid1.raw')
    idx0 = int(285 * 60 * samplerate) # min 285
    didx = 20 * 0.001 * samplerate # 20 ms
    # samplerate_col=20000
    # channels_col=15
    # idx0 = samplerate_col * 180

    calc_specs(data, samplerate, idx0, channels)




if __name__ == '__main__':
    main()
