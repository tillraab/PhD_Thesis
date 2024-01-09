import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from thunderfish.powerspectrum import decibel
from plottools.tag import tag
from IPython import embed


def plot_rise_specs(ax1):
    fs = 12

    ax1.plot([168, 218], [697.5, 697.5], lw=4, color='k', clip_on=False)
    ax1.text(193, 692, '50 sec', fontsize=11, ha='center', va='center')


    rise_spec2 = np.load('../rise_size/rise_spec2.npy', allow_pickle=True)
    rise_spec_extent2 = np.load('../rise_size/rise_spec2_extent.npy', allow_pickle=True)

    ax1.imshow(decibel(rise_spec2[::-1]),extent=[rise_spec_extent2[0] - rise_spec_extent2[0],
                                                 rise_spec_extent2[1] - rise_spec_extent2[0], rise_spec_extent2[2]-1,
                                                 rise_spec_extent2[3]],
             aspect='auto', alpha=0.7, cmap='jet', interpolation='gaussian', vmin=-120, vmax=-50)
    ax1.set_xlim(rise_spec_extent2[0] - rise_spec_extent2[0], rise_spec_extent2[1] - rise_spec_extent2[0])
    ax1.set_ylim(rise_spec_extent2[2], rise_spec_extent2[3])


    X, Y = [49.5, 65, 91, 97, 118, 126, 148, 159, 168, 185], [743, 750, 739, 744, 746, 750, 747, 743, 739, 743]
    for x, y in zip(X, Y):
        ax1.plot(x, y, 'v', color='k', markersize=6)

    ax1.set_yticks(np.arange(700, 776, 25))
    ax1.set_xticks([])

    ax1.set_ylabel('frequency [Hz]', fontsize=fs + 2)
    ax1.tick_params(labelsize=fs)

    plt.savefig('rise_example_jet_pres.jpg', dpi=300)

def plot_rise_specs_plain():

    fig = plt.figure(figsize=(8 /2.54, 8 * (10/20) / 2.54))
    gs = gridspec.GridSpec(1, 1, left=0, bottom = 0, right = 1, top= 1)
    ax1 = fig.add_subplot(gs[0, 0])

    rise_spec2 = np.load('../rise_size/rise_spec2.npy', allow_pickle=True)
    rise_spec_extent2 = np.load('../rise_size/rise_spec2_extent.npy', allow_pickle=True)

    ax1.imshow(decibel(rise_spec2[::-1]),extent=[rise_spec_extent2[0] - rise_spec_extent2[0],
                                                 rise_spec_extent2[1] - rise_spec_extent2[0], rise_spec_extent2[2]-1,
                                                 rise_spec_extent2[3]],
             aspect='auto', alpha=0.7, cmap='Greys', interpolation='gaussian', vmin=-120, vmax=-90)

    ax1.set_xlim(30, 220)
    ax1.set_ylim(705, 747.5)

    plt.savefig('rises_grey_ex.jpg', dpi=300)
    plt.show()
def main():
    fig = plt.figure(figsize=(10 /2.54, 10 * (12/20) / 2.54))
    gs = gridspec.GridSpec(1, 1, left=0.2   , bottom = 0.15, right = 0.975, top=0.975)
    ax = fig.add_subplot(gs[0, 0])
    plot_rise_specs(ax)
    plt.show()
    # plt.close()

    plot_rise_specs_plain()



if __name__ == '__main__':
    main()