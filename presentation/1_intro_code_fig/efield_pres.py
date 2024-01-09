import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from thunderfish.efield import *
from thunderfish.fakefish import *
from thunderfish.fishshapes import *
from IPython import embed
from plottools.tag import tag

def main():

    fig = plt.figure(figsize=(15/2.54, 15 * (12/20) /2.54))
    #gs = gridspec.GridSpec(1, 1, left=0.025, bottom=0.1, right=0.975, top=0.975)
    gs = gridspec.GridSpec(1, 1, left=0, bottom=0, right=1, top=1)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))

    # maxx = 15 * 0.95 * 2.5
    # maxy = 15 * (12/20) * 0.875 * 2.5

    maxx = 15  * 2.5
    maxy = 15 * (12/20) * 2.5

    x = np.linspace(-maxx, maxx, 200)
    y = np.linspace(-maxy, maxy, 200)
    xx, yy = np.meshgrid(x, y)
    fish1 = ((-10, -10), (1, 0.5), 18.0, -25)
    fish2 = ((13, -2), (0.8, 1), 15.0, 20)
    fish3 = ((-7, 10), (-1, -.5), 20.0, -50)
    poles1 = efish_monopoles(*fish1)
    poles2 = efish_monopoles(*fish2)
    poles3 = efish_monopoles(*fish3)

    pot1 = epotential_meshgrid(xx, yy, None, poles1) * -1
    pot2 = epotential_meshgrid(xx, yy, None, poles2)
    pot3 = epotential_meshgrid(xx, yy, None, poles3)
    pot = pot1 + pot2 + pot3

    thresh = 0.65
    zz = squareroot_transform(pot/200, thresh)

    levels = np.linspace(-thresh, thresh, 16)
    ax[0].contourf(x, y, -zz, levels, cmap='RdYlBu')
    ax[0].contour(x, y, -zz, levels, zorder=1, colors='#707070',
                  linewidths=0.1, linestyles='solid')

    species = "Alepto_top"
    for fish in [fish1, fish2, fish3]:
        plot_fish(ax[0], species, pos = fish[0], direction=fish[1], size=fish[2], bend=fish[3],
                  bodykwargs=dict(lw=1, edgecolor='k', facecolor='k'), finkwargs=dict(lw=1, edgecolor='k', facecolor='k'))

    xlims = ax[0].get_xlim()
    ylims = ax[0].get_ylim()

    x0, x1 = xlims[0], xlims[0] + 10
    y0, y1 = ylims[0] - (ylims[1] - ylims[0]) * 0.05, ylims[0] - (ylims[1] - ylims[0]) * 0.03

    # ax[0].fill_between([x0, x1], [y0, y0], [y1, y1], color='k', clip_on=False)
    ax[0].hlines(y0, x0, x1, color='k', clip_on=False, lw=5)
    ax[0].text(x0 + (x1 - x0) / 2, y0 - 0.8, '10 cm', fontsize=10, ha='center', va='top')

    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)

    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ############
    ax_i = []

    gs3 = gridspec.GridSpec(1, 1, left=0.45, right=0.55, bottom=0.15, top=0.25)
    ax_i.append(fig.add_subplot(gs3[0, 0], facecolor='None'))

    gs3 = gridspec.GridSpec(1, 1, left=0.75, right=0.85, bottom=0.35, top=0.45)
    ax_i.append(fig.add_subplot(gs3[0, 0], facecolor='None'))

    gs3 = gridspec.GridSpec(1, 1, left=0.25, right=0.35, bottom=0.8, top=0.9)
    ax_i.append(fig.add_subplot(gs3[0, 0], facecolor='None'))


    samplerate = 40000.0 # in Hz
    duration = 0.01      # in sec

    inset_len = 0.01     # in sec
    inset_indices = int(inset_len*samplerate)
    ws_fac = 0.1         # whitespace factor or ylim (between 0. and 1.)

    # generate data:
    eodf = 200.0


    wavefish = wavefish_eods('Alepto', eodf, samplerate, duration, noise_std=0, phase0=-0.5*np.pi)[::-1]
    wavefish = wavefish / np.max(wavefish) * 0.9

    wavefish2 = wavefish_eods('Alepto', eodf, samplerate, duration, noise_std=0, phase0=np.pi)
    wavefish2 = wavefish2 / np.max(wavefish2) * 0.75

    wavefish3 = wavefish_eods('Alepto', eodf, samplerate, duration, noise_std=0, phase0=-0.5*np.pi)
    wavefish3 /= np.max(wavefish3)


    time = np.arange(len(wavefish))/samplerate

    for w, a, i, enu in zip([wavefish, wavefish2, wavefish3], ax_i, [130, 100, 150], [1, 2, 3]):
        a.plot(time, w, color='midnightblue', lw=1.5)
        a.plot(time[i], w[i], 'o', color='firebrick', markersize=6, clip_on=False)
        a.set_xlim(time[0], time[-1])
        a.text(time[-1] * -0.5, 0, r'F$_{%s}$' % enu, fontsize=12, clip_on=False)
        a.set_axis_off()
        a.set_ylim(-1, 1)

    plt.savefig('efield_solo.jpg', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()