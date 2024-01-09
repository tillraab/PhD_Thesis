import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from thunderfish.dataloader import open_data, fishgrid_grids, fishgrid_spacings
from plottools.tag import tag


def plot_raw_data_col(data, channels, samplerate, idx0, didx,):
    fig = plt.figure(figsize=(13 / 2.54, 13 * (12/20) / 2.54))

    gs = gridspec.GridSpec(8, 8, left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)
    ax = []
    for i in range(8):
        for j in range(8):
            if len(ax) == 0:
                ax.append(fig.add_subplot(gs[j, i]))
            else:
                ax.append(fig.add_subplot(gs[j, i], sharex=ax[0], sharey=ax[0]))

            ax[-1].set_axis_off()

    for i in range(channels):
        ax[i].plot(np.arange(len(data[int(idx0):int(idx0+didx/2), i]))/samplerate, data[int(idx0):int(idx0+didx/2), i], color='midnightblue', lw=1)
    #plt.show()

    if True:
        x0, x1 = ax[7].get_xlim()
        y0, y1 = ax[7].get_ylim()
        offset = 0.005
        ax[7].plot([0-offset, 0.01-offset], [0-offset, 0-offset], color='k', lw=2, clip_on=False)
        ax[7].plot([0-offset, 0-offset], [0-offset, 0.01-offset], color='k', lw=2, clip_on=False)

        ax[7].text(0-offset + 0.005, 0 - offset*1.2, r'10$\,$ms', ha='center', va='top', fontsize=8)
        ax[7].text(0-offset*1.2, 0 - offset + 0.005, r'10$\,$mV', ha='right', va='center', rotation=90, fontsize=8)
        ax[7].set_xlim(x0, x1)
        ax[7].set_ylim(y0, y1)

    for a in ax:
        a.set_axis_off()

    plt.savefig('col_raw_traces.jpg', dpi=300)


def plot_raw_data_comp(data2, channels2, samplerate2, idx02, didx2):
    fig = plt.figure(figsize=(13 / 2.54, 13 * (12/20) / 2.54))

    gs = gridspec.GridSpec(3, 6, left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)
    ax2 = []
    for i in range(3):
        for j in range(6):
            if (i, j) == (0, 0) or (i, j) == (0, 5) or (i, j) == (2, 0) or (i, j) == (2, 5):
                continue
            if len(ax2) == 0:
                ax2.append(fig.add_subplot(gs[i, j]))
            else:
                ax2.append(fig.add_subplot(gs[i, j], sharex=ax2[0], sharey=ax2[0]))

            ax2[-1].set_axis_off()

    ax_help = fig.add_subplot(gs[0, 0])
    ax_help.set_axis_off()
    ax_help2 = fig.add_subplot(gs[2, 0], sharex=ax2[0], sharey=ax2[0])
    ax_help2.set_axis_off()

    for a in ax2:
        a.set_axis_off()

    gs = gridspec.GridSpec(1, 1, left=0.1 + 2.5*(0.85/6), bottom=0.1 + 1*(0.85/3), right=0.1 + 3.5*(0.85/6), top=0.1 + 2*(0.85/3), wspace=0, hspace=0)

    ax2.append(fig.add_subplot(gs[0, 0], sharex= ax2[0], sharey=ax2[0]))
    ax2[-1].set_xticks([])
    ax2[-1].set_yticks([])
    #
    for i in range(channels2):
        ax2[i].plot(np.arange(len(data2[int(idx02):int(idx02+didx2/2), i]))/samplerate2, data2[int(idx02):int(idx02+didx2/2), i], color='midnightblue', lw=1)

    for key in ['left', 'right', 'bottom', 'top']:
        ax2[-1].spines[key].set_linewidth(2)
        ax2[-1].spines[key].set_color('grey')

    if True:
        x0, x1 = ax_help2.get_xlim()
        y0, y1 = ax_help2.get_ylim()
        #offset = 0.0025
        offset = 0.0040
        ax_help2.plot([0-offset, 0.010-offset], [0-offset, 0-offset], color='k', lw=2, clip_on=False)
        ax_help2.plot([0-offset, 0-offset], [0-offset, 0.004-offset], color='k', lw=2, clip_on=False)

        ax_help2.text(0-offset + 0.005, 0 - offset*1.1, r'10$\,$ms', ha='center', va='top', fontsize=8)
        ax_help2.text(0-offset*1.2, 0 - offset + 0.002, r'2$\,$mV', ha='right', va='center', rotation=90, fontsize=8)

        ax_help2.set_xlim(x0, x1)
        ax_help2.set_ylim(y0, y1)

    y0, y1 = ax2[-1].get_ylim()
    ax2[-1].set_ylim(y0 - (y1 - y0)*0.1, y1 + (y1 - y0)*0.1)

    plt.savefig('comp_raw_traces.jpg', dpi=300)
    plt.show()

def main():
    pass

    data_col = np.load('./colombia_example_15sec.npy')
    samplerate_col=20000
    channels_col=64
    idx0_col = 0
    didx_col = 20 * 0.001 * samplerate_col  # 20 ms

    #data_tue = np.load('./tube_comp_5min.npy')
    data_tue = np.load('./tube_comp_5min_10rises.npy')
    samplerate_tue=20000
    channels_tue=15
    idx0_tue = 0
    didx_tue = 20 * 0.001 * samplerate_tue  # 20 ms

    plot_raw_data_col(data_col, channels_col, samplerate_col, idx0_col, didx_col)

    plot_raw_data_comp(data_tue, channels_tue, samplerate_tue, idx0_tue, didx_tue)

    plt.show()




if __name__ == '__main__':
    main()
