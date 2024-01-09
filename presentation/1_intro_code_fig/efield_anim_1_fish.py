import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from thunderfish.efield import *
from thunderfish.fakefish import *
from thunderfish.fishshapes import *
from IPython import embed
from plottools.tag import tag


def main():
    # grid params
    maxx = 20 * 0.9 * 2 #  fig width * axis size (size_hint = 0.8) * upscale
    maxy = 12 * 0.85 * 2

    x = np.linspace(-maxx, maxx, 200)
    y = np.linspace(-maxy, maxy, 200)
    xx, yy = np.meshgrid(x, y)

    # fish params
    fish1 = ((0, 0), (1, 0), 20.0, 0)
    poles1 = efish_monopoles(*fish1)
    pot1 = epotential_meshgrid(xx, yy, None, poles1)

    pot = pot1

    thresh = 0.65
    levels = np.linspace(-thresh, thresh, 16)
    zz = squareroot_transform(pot/200, thresh)

    eodf = 1
    samplerate = 30
    duration = 2
    wavefish = wavefish_eods('Alepto', eodf, samplerate, duration, noise_std=0, phase0=-0.5 * np.pi)[::-1]
    wavefish = wavefish / np.max(wavefish) * 0.9

    potential = zz[120, 105] * wavefish

    for i in range(len(wavefish)-1):
        fig = plt.figure(figsize=(20 / 2.54, 12 / 2.54))
        gs = gridspec.GridSpec(1, 1, left=0.05, bottom=0.1, right=0.95, top=0.95)
        ax = []
        ax.append(fig.add_subplot(gs[0, 0]))
        gs2 = gridspec.GridSpec(1, 1, left=.7, bottom=.1, right=0.95, top=0.3)
        ax2 = fig.add_subplot(gs2[0, 0])

        ax[0].contourf(x, y, -zz * wavefish[i], levels, zorder=1, cmap='RdYlBu')
        ax[0].contour(x, y, -zz * wavefish[i], levels, zorder=2, colors='#707070', linewidths=0.1, linestyles='solid')

        ax[0].plot(xx[120, 120], yy[105, 105], 'o', color='firebrick', markersize=4, markeredgecolor='k', mew=.5)

        ax2.plot(np.arange(len(potential))/samplerate, potential, color='k', lw=2)
        ax2.plot(i/samplerate, potential[i], 'o', color='firebrick', markersize=4, markeredgecolor='k', mew=.5)

        ax2.set_xlim(0, (len(wavefish) - 1)/samplerate)
        ax2.patch.set_alpha(0)
        ax2.set_axis_off()

        # fish shadow
        plot_fish(ax[0], "Alepto_top", pos=fish1[0], direction=fish1[1], size=fish1[2], bend=fish1[3],
                  bodykwargs=dict(lw=1, edgecolor='k', facecolor='k'),
                  finkwargs=dict(lw=1, edgecolor='k', facecolor='k'))
        file_name = ('./field_animaltion_1f/%3.f' % int(i)).replace(' ', '0')
        plt.savefig(file_name + '.jpg', dpi=300)
        plt.close()

    embed()
    quit()

    #### -- replace x and y with bar
    # xlims = ax[0].get_xlim()
    # ylims = ax[0].get_ylim()
    #
    # x0, x1 = xlims[0], xlims[0] + 10
    # y0, y1 = ylims[0] - (ylims[1] - ylims[0]) * 0.05, ylims[0] - (ylims[1] - ylims[0]) * 0.03
    #
    # # ax[0].fill_between([x0, x1], [y0, y0], [y1, y1], color='k', clip_on=False)
    # ax[0].hlines(y0, x0, x1, color='k', clip_on=False, lw=5)
    # ax[0].text(x0 + (x1 - x0) / 2, y0 - 0.8, '10 cm', fontsize=10, ha='center', va='top')
    #
    # ax[0].set_xlim(xlims)
    # ax[0].set_ylim(ylims)
    #
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])



    plt.savefig('test.pdf')
    plt.show()


    pass


if __name__ == '__main__':
    main()