import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch

from thunderfish.powerspectrum import decibel
from plottools.tag import tag
from IPython import embed



def boltzmann(t, alpha=0.25, beta=0.0, x0=4, dx=0.85):
    # boltzmann(np.arange(0, 2.5, 0.001), alpha=1, beta=0, x0=.35, dx=.08)
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

def plot_th_indicate(spec, times, idx_v, fund_v, part_spec, part_times, part_freqs, ax):
    # ax.imshow(decibel(spec)[::-1], extent=[times[0], times[-1], 0, 2000],
    #           aspect='auto', alpha=0.7, cmap='Greys', vmax=-50, vmin=-110, interpolation='gaussian', zorder=1)
    ax.imshow(decibel(part_spec)[::-1], extent=[part_times[0]-2, part_times[-1]-2, part_freqs[0], part_freqs[-1]], aspect='auto', alpha=0.7, cmap='Greys', vmax=-50, vmin=-110, interpolation='gaussian', zorder=1)

    rest_mask = np.ones(len(idx_v), dtype=bool)
    # for i, c in zip([2394], ['royalblue']):
    for i, c in zip([2394], ['cornflowerblue']):
    #for i, c in zip([2394, 2319], ['royalblue', 'royalblue']):
        m = np.ones(len(idx_v), dtype=bool)
        m[fund_v > fund_v[i] + 2.5] = 0
        m[fund_v < fund_v[i] - 2.5] = 0
        m[times[idx_v] > times[idx_v[i]] + 10] = 0
        m[times[idx_v] < times[idx_v[i]] - 10] = 0
        rest_mask[m] = 0
        m[i] = 0

        ax.plot(times[idx_v[i]], fund_v[i], '.', color='midnightblue', markersize=10, zorder=4)
        ax.plot(times[idx_v[m]], fund_v[m], '.', color=c)

        x0, x1 = times[idx_v[i]] - 10, times[idx_v[i]] + 10
        y0, y1 = fund_v[i] - 2.5, fund_v[i] + 2.5

        ax.plot([x0, x1], [y0, y0], lw=1, color='k')
        ax.plot([x0, x1], [y1, y1], lw=1, color='k')
        ax.plot([x0, x0], [y0, y1], lw=1, color='k')
        ax.plot([x1, x1], [y0, y1], lw=1, color='k')

        ax.text(x0 + (x1-x0)/2, y1, r'$2 \times \Delta t_{thresh}$', ha='center', va='bottom', fontsize=10)
        ax.text(x0, y0 + (y1 - y0) / 2, r'$2 \times \Delta f_{thresh}$', ha='right', va='center', rotation=90)

    ax.plot(times[idx_v[rest_mask]], fund_v[rest_mask], '.', color='k', markersize=3, alpha=0.2)

    #ax.set_xlim(50, 110)
    ax.set_xlim(times[idx_v[2394]]-15, times[idx_v[2394]]+15)
    #ax.set_ylim(890, 930)
    #ax.set_ylim(915, 925)
    ax.set_ylim(fund_v[2394] - 3.5, fund_v[2394] + 3.5)

    ax.set_xlabel('time [s]', fontsize=14)
    ax.set_ylabel('frequency [Hz]', fontsize=14)
    ax.tick_params(labelsize=12)

    y0 = fund_v[2394]
    x0 = times[idx_v[2394]]
    return y0, x0

def plot_f_error(y_off, ax):
    f_error = boltzmann(np.arange(0, 2.5, 0.001), alpha=1, beta=0, x0=.35, dx=.08)
    ax.plot(f_error, np.arange(0, 2.5, 0.001) + y_off, color='midnightblue', lw=2)
    ax.plot(f_error, -1 * np.arange(0, 2.5, 0.001) + y_off, color='midnightblue', lw=2)

    ax.set_ylim(y_off-3.5, y_off+3.5)
    ax.set_xlim(0, 1.05)
    ax.set_yticks([y_off - 3, y_off - 2, y_off - 1, y_off, y_off + 1, y_off + 2, y_off + 3])
    ax.set_yticklabels([3, 2, 1, 0, 1, 2, 3])


    ax.set_ylabel(r'$\Delta freq$ [Hz]', fontsize=14)
    ax.set_xlabel(r'$\varepsilon_{f}$', fontsize=14)
    ax.tick_params(labelsize=12)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

def plot_amplitude_error_dist(sign_v, a_error_dist, ex, ex_i0, ex_i1, ax):
    mask = np.argsort(a_error_dist)
    X, Y = np.meshgrid(np.arange(8), np.arange(8))
    ex_color = ['forestgreen', 'gold', 'darkorange', 'firebrick']

    for enu, i0, i1 in zip(np.arange(len(ex_i0)), ex_i0, ex_i1):
        s0 = sign_v[i0].reshape(8, 8)
        s0 = (s0 - np.min(s0)) / (np.max(s0) - np.min(s0))
        # , aspect='auto'
        ax[enu * 2].imshow(s0[::-1], alpha=0.7, cmap='jet', vmax=1, vmin=0, interpolation='gaussian', zorder=1)
        s1 = sign_v[i1].reshape(8, 8)
        s1 = (s1 - np.min(s1)) / (np.max(s1) - np.min(s1))
        ax[enu * 2 + 1].imshow(s1[::-1], alpha=0.7, cmap='jet', vmax=1, vmin=0, interpolation='gaussian',
                               zorder=1)
        for x, y in zip(X, Y):
            ax[enu * 2].plot(x, y, '.', color='k', markersize=2)
            ax[enu * 2 + 1].plot(x, y, '.', color='k', markersize=2)
        y0, y1 = ax[enu * 2].get_ylim()
        # ax[enu * 2].arrow(8.5, 3.5, 2, 0, head_width=.7, head_length=.7, clip_on=False, color=ex_color[enu], lw=2.5)
        ax[enu * 2].arrow(3.5, 8.25, 0, .8, head_width=.7, head_length=.7, clip_on=False, color=ex_color[enu], lw=2)
        ax[enu * 2].set_ylim(y0, y1)

        ax[enu * 2].set_xticks([])
        ax[enu * 2 + 1].set_xticks([])

        ax[enu * 2].set_yticks([])
        ax[enu * 2 + 1].set_yticks([])

    ax[-1].plot(a_error_dist[mask], np.linspace(0, 1, len(a_error_dist)), color='midnightblue', clip_on=False)
    for enu in range(4):
        ax[-1].plot(a_error_dist[mask[ex[enu]]], np.linspace(0, 1, len(a_error_dist))[ex[enu]], 'o',
                    color=ex_color[enu], clip_on=False, markersize=6)
    ax[-1].set_ylim(0, 1)
    ax[-1].set_yticks([0, 1])

    ax[-1].set_xlim(0, np.max(a_error_dist))
    ax[-1].set_ylabel(r'$\varepsilon_{S}$', fontsize=14)
    ax[-1].set_xlabel(r'field difference ($\Delta S$)', fontsize=14)
    ax[-1].tick_params(labelsize=12)

def main():

    folder = "2016-04-10-11_12"

    if os.path.exists(os.path.join(folder, 'fund_v.npy')):

        part_spec = np.load(os.path.join(folder, 'part_spec.npy'))
        part_times = np.load(os.path.join(folder, 'part_times.npy'))
        part_freqs = np.load(os.path.join(folder, 'part_freqs.npy'))

        fund_v = np.load(os.path.join(folder, 'fund_v.npy'))
        sign_v = np.load(os.path.join(folder, 'sign_v.npy'))
        idx_v = np.load(os.path.join(folder, 'idx_v.npy'))
        times = np.load(os.path.join(folder, 'times.npy'))
        spec = np.load(os.path.join(folder, 'spec.npy'))
        a_error_dist = np.load(os.path.join(folder, 'a_error_dist.npy'))
        start_time, end_time = np.load(os.path.join(folder, 'meta.npy'))


        ex_in_dist = np.array([10, 126467, 252924, 379361])
        ex_i0 = np.array([693, 684, 746, 915])
        ex_i1 = np.array([705, 916, 963, 966])

        times -= times[0]

    ##########################################################################################################

    fig = plt.figure(figsize=(17.5 / 2.54, 17.5 / 2.54))
    gs = gridspec.GridSpec(1, 2, left=0.1, bottom=0.65, right=0.9, top=0.95, width_ratios=[2.5, 1], wspace=0.2)
    ax0 = []
    ax0.append(fig.add_subplot(gs[0, 0]))
    ax0.append(fig.add_subplot(gs[0, 1]))


    gs = gridspec.GridSpec(3, 4, left=0.1, bottom=0.1, right=0.9, top=0.5, hspace=0.4, wspace=0.4, height_ratios=[2, 2, 1.5])
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[1, 0]))

    ax.append(fig.add_subplot(gs[0, 1]))
    ax.append(fig.add_subplot(gs[1, 1]))

    ax.append(fig.add_subplot(gs[0, 2]))
    ax.append(fig.add_subplot(gs[1, 2]))

    ax.append(fig.add_subplot(gs[0, 3]))
    ax.append(fig.add_subplot(gs[1, 3]))

    ax.append(fig.add_subplot(gs[2, :]))

    ##########################################################################################################

    # fig = plt.figure(figsize=(17.5/2.54, 12/2.54))
    # gs = gridspec.GridSpec(1, 1, left=0.15, bottom=0.15, right=0.95, top=0.95)
    # ax = fig.add_subplot(gs[0, 0])

    ##########################################################################################################

    y0, x0 = plot_th_indicate(spec, times, idx_v, fund_v, part_spec, part_times, part_freqs, ax=ax0[0])
    #

    plot_f_error(y0, ax0[1])


    for x, y in zip([68.486, 70.78], [920.408, 919.49]):
        xy0 = (x, y)
        df = np.abs(y0 - y)
        f_e = boltzmann(df, alpha=1, beta=0, x0=.35, dx=.08)
        xy1 = (f_e, y)

        con = ConnectionPatch(xyA=xy0, xyB=xy1, coordsA="data", coordsB="data",
                              axesA=ax0[0], axesB=ax0[1], color="firebrick", linestyle='-')
        ax0[1].add_artist(con)
        ax0[1].plot([f_e, f_e], [y, y0 - 3.5], color='firebrick', linestyle='-')

    xy0 = (x0 + 10, y0 + 2.5)
    xy1 = (1.05, y0 + 2.5)
    con = ConnectionPatch(xyA=xy0, xyB=xy1, coordsA="data", coordsB="data",
                          axesA=ax0[0], axesB=ax0[1], color="k", linestyle=(0, (5, 7)), lw=0.5)
    ax0[1].add_artist(con)
    xy0 = (x0 + 10, y0 - 2.5)
    xy1 = (1.05, y0 - 2.5)
    con = ConnectionPatch(xyA=xy0, xyB=xy1, coordsA="data", coordsB="data",
                          axesA=ax0[0], axesB=ax0[1], color="k", linestyle=(0, (5, 7)), lw=0.5)
    ax0[1].add_artist(con)

    plot_amplitude_error_dist(sign_v, a_error_dist, ex_in_dist, ex_i0, ex_i1, ax)

    plt.savefig('tracking_features3.jpg', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()