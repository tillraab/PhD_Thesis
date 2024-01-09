import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from thunderfish.dataloader import open_data, fishgrid_grids, fishgrid_spacings
from plottools.tag import tag
from IPython import embed

def load(filename):
    data = open_data(filename, -1, 60, 10)
    samplerate = data.samplerate
    channels = data.channels
    return data, samplerate, channels

def plot_raw_data(data, channels, samplerate, idx0, didx, data2, channels2, samplerate2, idx02, didx2):
    fig = plt.figure(figsize=(17.5/2.54, 14/2.54))
    gs = gridspec.GridSpec(8, 8, left=0.15, bottom=0.55, right=0.95, top=0.95, wspace=0, hspace=0)
    ax = []
    for i in range(8):
        for j in range(8):
            if len(ax) == 0:
                ax.append(fig.add_subplot(gs[j, i]))
            else:
                ax.append(fig.add_subplot(gs[j, i], sharex=ax[0], sharey=ax[0]))

            ax[-1].set_axis_off()

    for i in range(channels):
        ax[i].plot(np.arange(len(data[int(idx0):int(idx0+didx/2), i]))/samplerate, data[int(idx0):int(idx0+didx/2), i], color='midnightblue')
    #plt.show()

    if True:
        x0, x1 = ax[7].get_xlim()
        y0, y1 = ax[7].get_ylim()
        offset = 0.005
        ax[7].plot([0-offset, 0.01-offset], [0-offset, 0-offset], color='k', lw=3, clip_on=False)
        ax[7].plot([0-offset, 0-offset], [0-offset, 0.01-offset], color='k', lw=3, clip_on=False)

        ax[7].text(0-offset + 0.005, 0 - offset*1.2, '10ms', ha='center', va='top')
        ax[7].text(0-offset*1.2, 0 - offset + 0.005, '10mV/cm', ha='right', va='center', rotation=90)
        ax[7].set_xlim(x0, x1)
        ax[7].set_ylim(y0, y1)

    gs = gridspec.GridSpec(3, 6, left=0.15, bottom=0.1, right=0.95, top=0.4, wspace=0, hspace=0)
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

    gs = gridspec.GridSpec(1, 1, left=0.55 - 0.8/12, bottom=0.2, right=0.55 + 0.8/12, top=0.3, wspace=0, hspace=0)
    ax2.append(fig.add_subplot(gs[0, 0], sharex= ax2[0], sharey=ax2[0]))
    ax2[-1].set_xticks([])
    ax2[-1].set_yticks([])

    for i in range(channels2):
        ax2[i].plot(np.arange(len(data2[int(idx02):int(idx02+didx2/2), i]))/samplerate2, data2[int(idx02):int(idx02+didx2/2), i], color='midnightblue')

    for key in ['left', 'right', 'bottom', 'top']:
        ax2[-1].spines[key].set_linewidth(2)
        ax2[-1].spines[key].set_color('grey')

    if True:
        x0, x1 = ax_help2.get_xlim()
        y0, y1 = ax_help2.get_ylim()
        offset = 0.0025
        offset = 0.0040
        ax_help2.plot([0-offset, 0.010-offset], [0-offset, 0-offset], color='k', lw=3, clip_on=False)
        ax_help2.plot([0-offset, 0-offset], [0-offset, 0.004-offset], color='k', lw=3, clip_on=False)

        ax_help2.text(0-offset + 0.005, 0 - offset*1.2, '10ms', ha='center', va='top')
        ax_help2.text(0-offset*1.2, 0 - offset + 0.002, '2mV/cm', ha='right', va='center', rotation=90)

        ax_help2.set_xlim(x0, x1)
        ax_help2.set_ylim(y0, y1)

    fig.tag(axes=[ax[0], ax_help], fontsize=15, yoffs=2, xoffs=-8)
    #plt.savefig('raw_data.pdf')

def plot_raw_data_col(data, channels, samplerate, idx0, didx, fig, gs):

    # gs = gridspec.GridSpec(8, 8, left=0.075, bottom=0.05, right=0.475, top=0.3, wspace=0, hspace=0)
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

        ax[7].text(0-offset + 0.005, 0 - offset*1.2, '10 ms', ha='center', va='top', fontsize=8)
        ax[7].text(0-offset*1.2, 0 - offset + 0.005, '10 mV', ha='right', va='center', rotation=90, fontsize=8)
        ax[7].set_xlim(x0, x1)
        ax[7].set_ylim(y0, y1)

    return ax

def plot_raw_data_comp(data2, channels2, samplerate2, idx02, didx2, fig, gs, gs2):
    # gs = gridspec.GridSpec(3, 6, left=0.55, bottom=0.05, right=0.95, top=0.3, wspace=0, hspace=0)
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

    #gs = gridspec.GridSpec(1, 1, left=0.75 - 0.8/12, bottom=0.1, right=0.75 + 0.8/12, top=0.15, wspace=0, hspace=0)
    #gs = gridspec.GridSpec(1, 1, left=0.75 - 0.4/12, bottom=0.05+0.25/3, right=0.75 + 0.4/12, top=0.05 + 0.25/3*2, wspace=0, hspace=0)
    ax2.append(fig.add_subplot(gs2[2:4, 5:7], sharex= ax2[0], sharey=ax2[0]))
    ax2[-1].set_xticks([])
    ax2[-1].set_yticks([])

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

        ax_help2.text(0-offset + 0.005, 0 - offset*1.2, '10 ms', ha='center', va='top', fontsize=8)
        ax_help2.text(0-offset*1.2, 0 - offset + 0.002, '2 mV', ha='right', va='center', rotation=90, fontsize=8)

        ax_help2.set_xlim(x0, x1)
        ax_help2.set_ylim(y0, y1)

    y0, y1 = ax2[-1].get_ylim()
    ax2[-1].set_ylim(y0 - (y1 - y0)*0.1, y1 + (y1 - y0)*0.1)

    return ax_help

def main():

    def foto_zoom(img, zoom_fac):
        shape = np.shape(img)
        x_pixels = int(shape[0] * zoom_fac/2)
        y_pixels = int(shape[1] * zoom_fac/2)
        crop_img = img[x_pixels:shape[0] - x_pixels, y_pixels:shape[1]-y_pixels, :]
        return crop_img
    # #colmbia file
    # data, samplerate, channels = load('/home/raab/data/2016-colombia/2016-04-10-11_12/traces-grid1.raw')
    # idx0 = int(285 * 60 * samplerate) # min 285
    # didx = 20 * 0.001 * samplerate # 20 ms

    # tube competition
    # data, samplerate, channels = load('/home/raab/data/2019_tube_competition/2020-05-28-10_00/traces-grid1.raw')
    # idx0 = int((3*60 + 11) * 60 * samplerate) # min 285
    # #idx0 = int((1*60 + 11) * 60 * samplerate) # min 285
    # didx = 20 * 0.001 * samplerate # 20 ms

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

    plot_raw_data(data_col, channels_col, samplerate_col, idx0_col, didx_col,
                  data_tue, channels_tue, samplerate_tue, idx0_tue, didx_tue)

    ########################################################################################
    plt.close('all')

    setups = mpimg.imread('./IMG_0322.png')
    electode = mpimg.imread('./electrode.png')
    competition_grid = mpimg.imread('./competition_grid.png')
    colombia_grid = mpimg.imread('colombia_grid.png')

    crop_setups = foto_zoom(setups, .2)
    crop_elec = foto_zoom(electode, .25)

    fig = plt.figure(figsize=(18/2.54, 18/2.54))

    gs = gridspec.GridSpec(3, 2, left = 0.05, bottom = 0.05, right=0.975, top=0.975)
    ax = []

    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[0, 1]))
    ax.append(fig.add_subplot(gs[1, 0]))
    ax.append(fig.add_subplot(gs[2, 0]))
    ax.append(fig.add_subplot(gs[1, 1]))
    ax.append(fig.add_subplot(gs[2, 1]))

    ax[0].imshow(crop_setups)
    ax[1].imshow(crop_elec)
    ax[2].imshow(colombia_grid)
    ax[3].imshow(competition_grid[400:, :, :])

    gssub_col = gs[1, 1].subgridspec(8, 8, wspace=0, hspace=0)
    gssub_comp = gs[2, 1].subgridspec(3, 6, wspace=0, hspace=0)
    gssub_comp2 = gs[2, 1].subgridspec(6, 12, wspace=0, hspace=0)

    _ = plot_raw_data_col(data_col, channels_col, samplerate_col, idx0_col, didx_col, fig, gssub_col)

    _ = plot_raw_data_comp(data_tue, channels_tue, samplerate_tue, idx0_tue, didx_tue, fig, gssub_comp, gssub_comp2)

    #ax[3].imshow(competition_grid)


    for a in ax:
        a.set_axis_off()

    fig.tag(axes=[ax[0], ax[1], ax[2], ax[4], ax[3], ax[5]], fontsize=15, yoffs=.5, xoffs=-3)
    # fig.tag(axes=[ax_col[0]], labels=['E'], fontsize=15, yoffs=2, xoffs=-5)
    plt.savefig('./setups_and_raw_traces_new.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()