import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from thunderfish.powerspectrum import decibel
from plottools.tag import tag
from IPython import embed

def ex_traces(ax0, ax1, fs = 12):
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

    ax0.fill_between([0, 3], [590, 600], [930, 930], color='#aaaaaa', zorder=1)
    ax0.set_ylim(600, 920)
    ax0.set_xlim(0, 6)
    ax0.set_xlabel('time [h]', fontsize=fs+2)
    ax0.set_ylabel('frequency [Hz]', fontsize=fs+2)

    ###############################################

    ax1.plot(times[ex_W_rise_idx] / 3600, np.ones(len(ex_W_rise_idx)) - 1, '|', color=Wc, markersize=10)
    ax1.plot(times[ex_L_rise_idx] / 3600, np.ones(len(ex_L_rise_idx)), '|', color=Lc, markersize=10)

    ax1.plot(t_contact / 3600, np.ones(len(t_contact)) + 4, '|', color='firebrick', markersize=10)
    ax1.plot(t_chasings / 3600, np.ones(len(t_chasings)) + 2, '|', color='darkorange', markersize=10)

    ax1.set_ylim(-0.75, 5.5)
    ax1.set_yticks([0.5, 3, 5])
    ax1.set_yticklabels(['rises', 'chase', 'contact'])

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.xaxis.set_ticks_position('none')
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax0.tick_params(labelsize=fs)
    ax1.tick_params(labelsize=fs)

def main():

    fig = plt.figure(figsize=(18 / 2.54, 18 * (14/20) / 2.54))
    gs = gridspec.GridSpec(2, 1, left=0.15, bottom=0.15, right=0.975, top=0.975, height_ratios=[1, 2.5], hspace=0)
    ax_trace = []
    ax_trace.append(fig.add_subplot(gs[1, 0]))
    ax_trace.append(fig.add_subplot(gs[0, 0], sharex=ax_trace[0]))
    ex_traces(ax_trace[0], ax_trace[1], fs=18)

    plt.savefig('poster_example_comp.pdf')
    plt.show()

if __name__ == '__main__':
    main()