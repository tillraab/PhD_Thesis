import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from thunderfish.powerspectrum import decibel
from plottools.tag import tag
from IPython import embed

def ex_traces(ax0, ax1):
    times = np.load('./ex_times.npy')
    t_contact = np.load('./ex_contact.npy')
    t_chasings = np.load('./ex_chasing.npy')
    ex_W_freq = np.load('./ex_W_freq.npy')
    ex_L_freq = np.load('./ex_L_freq.npy')
    ex_W_rise_idx = np.load('./ex_W_rise_idx.npy')
    ex_L_rise_idx = np.load('./ex_L_rise_idx.npy')

    Wc, Lc = 'darkgreen', '#3673A4'

    ax0.plot(times / 3600, ex_W_freq, color=Wc, zorder=2)
    ax0.plot(times / 3600, ex_L_freq, color=Lc, zorder=2)

    ax0.fill_between([0, 3], [590, 600], [910, 910], color='#aaaaaa', zorder=1)
    ax0.set_ylim(600, 900)
    ax0.set_xlim(0, 6)
    ax0.set_xlabel('time [h]', fontsize=12)
    ax0.set_ylabel('EODf [Hz]', fontsize=12)

    ###############################################

    ax1.plot(times[ex_W_rise_idx] / 3600, np.ones(len(ex_W_rise_idx)) - 1, '|', color=Wc, markersize=7)
    ax1.plot(times[ex_L_rise_idx] / 3600, np.ones(len(ex_L_rise_idx)), '|', color=Lc, markersize=7)

    ax1.plot(t_contact / 3600, np.ones(len(t_contact)) + 4, '|', color='firebrick', markersize=7)
    ax1.plot(t_chasings / 3600, np.ones(len(t_chasings)) + 2, '|', color='darkorange', markersize=7)

    ax1.set_ylim(-0.75, 5.5)
    ax1.set_yticks([0.5, 3, 5])
    ax1.set_yticklabels(['rises', 'chasing', 'contact'])

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.xaxis.set_ticks_position('none')
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax0.tick_params(labelsize=10)
    ax1.tick_params(labelsize=10)

def plot_rise_specs(ax0, ax1):

    ax0.plot([69, 89], [676, 676], lw=4, color='k', clip_on=False)
    ax0.text(79, 669, '20 sec', fontsize=11, ha='center', va='center')

    ax0.set_xlim(right=220)

    ax1.plot([168, 218], [695, 695], lw=4, color='k', clip_on=False)
    ax1.text(193, 687, '50 sec', fontsize=11, ha='center', va='center')

    rise_spec1 = np.load('../rise_size/rise_spec1.npy', allow_pickle=True)
    rise_spec_extent1 = np.load('../rise_size/rise_spec1_extent.npy', allow_pickle=True)

    rise_spec2 = np.load('../rise_size/rise_spec2.npy', allow_pickle=True)
    rise_spec_extent2 = np.load('../rise_size/rise_spec2_extent.npy', allow_pickle=True)

    for a, rise_spec, rise_spec_extent in zip([ax0, ax1], [rise_spec1, rise_spec2], [rise_spec_extent1, rise_spec_extent2]):

        a.imshow(decibel(rise_spec[::-1]),extent=[rise_spec_extent[0] - rise_spec_extent[0],
                                                   rise_spec_extent[1] - rise_spec_extent[0], rise_spec_extent[2],
                                                   rise_spec_extent[3]],
                 aspect='auto', alpha=0.7, cmap='jet', interpolation='gaussian', vmin=-120, vmax=-50)
        a.set_xlim(rise_spec_extent[0] - rise_spec_extent[0], rise_spec_extent[1] - rise_spec_extent[0])
        a.set_ylim(rise_spec_extent[2], rise_spec_extent[3])

    ax0.plot(39., 733, 'v', color='k', markersize=6)
    X, Y = [49.5, 65, 91, 97, 118, 126, 148, 159, 168, 185], [743, 750, 739, 744, 746, 750, 747, 743, 739, 743]
    for x, y in zip(X, Y):
        ax1.plot(x, y, 'v', color='k', markersize=6)

    ax0.set_ylim(680, 740)
    ax0.set_yticks(np.arange(680, 741, 20))
    #
    # ax[1].set_ylim(700, 775)
    ax1.set_yticks(np.arange(700, 776, 25))

    ax0.set_xticks([])
    ax1.set_xticks([])

    ax0.set_ylabel('EODf [Hz]', fontsize=12)
    ax1.set_ylabel('EODf [Hz]', fontsize=12)

    ax0.tick_params(labelsize=10)
    ax1.tick_params(labelsize=10)

def main():
    fig = plt.figure(figsize=(17.5 / 2.54, 20 / 2.54))
    #gs = gridspec.GridSpec(2, 1, left=0.125, bottom=0.6, right=0.95, top=0.975, height_ratios=[1, 2.5], hspace=0)
    gs = gridspec.GridSpec(2, 1, left=0.125, bottom=0.075, right=0.95, top=0.425, height_ratios=[1, 2.5], hspace=0)
    ax_trace = []

    ax_trace.append(fig.add_subplot(gs[1, 0]))
    ax_trace.append(fig.add_subplot(gs[0, 0], sharex=ax_trace[0]))
    ex_traces(ax_trace[0], ax_trace[1])

    #gs2 = gridspec.GridSpec(2, 2, left=0.125, bottom=0.05, right=0.95, top=0.5, hspace=0.4, wspace=0.4)
    gs2 = gridspec.GridSpec(2, 2, left=0.125, bottom=0.525, right=0.95, top=0.975, hspace=0.4, wspace=0.4)
    ax_beh = []
    ax_beh.append(fig.add_subplot(gs2[0, 1]))
    ax_beh.append(fig.add_subplot(gs2[1, 1]))

    ax_beh[0].set_axis_off()
    ax_beh[1].set_axis_off()

    ax_specs = []
    ax_specs.append(fig.add_subplot(gs2[0, 0]))
    ax_specs.append(fig.add_subplot(gs2[1, 0]))

    plot_rise_specs(ax_specs[0], ax_specs[1])

    fig.tag(axes=[ax_specs[0], ax_specs[1], ax_beh[0], ax_beh[1], ax_trace[1]], fontsize=15, yoffs=2, xoffs=-8)
    # plt.savefig('./example_trial_rise_fight.pdf')
    plt.show()


if __name__ == '__main__':
    main()