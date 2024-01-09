import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from thunderfish.efield import *
from thunderfish.fakefish import *
from thunderfish.fishshapes import *
from IPython import embed
from tqdm import tqdm
from plottools.tag import tag
import glob
import os


def first_stage_1_fish(x, y, dir, ax, ax2, eodf = 1., samplerate=30, duration=8):
    xx, yy = np.meshgrid(x, y)

    fish1 = ((-10, 0), (1, 0), 20.0, 0)
    poles1 = efish_monopoles(*fish1)
    pot1 = epotential_meshgrid(xx, yy, None, poles1)

    thresh = 0.65 * 3
    levels = np.linspace(-thresh, thresh, 16)
    zz1 = squareroot_transform(pot1/200, thresh)

    wavefish = wavefish_eods('Alepto', eodf, samplerate, duration, noise_std=0)
    potential = zz1[100, 105] * wavefish
    potential *=-1

    for i in tqdm(np.arange(len(wavefish)-1)):
        ax[0].clear()
        ax2.clear()

        ax[0].contourf(x, y, zz1 * wavefish[i], levels, zorder=1, cmap='RdYlBu')
        ax[0].contour(x, y, zz1 * wavefish[i], levels, zorder=2, colors='#707070', linewidths=0.1, linestyles='solid')
        ax[0].plot(xx[100, 100], yy[105, 105], 'o', color='firebrick', markersize=4, markeredgecolor='k', mew=.5)

        # fish shadow
        plot_fish(ax[0], "Alepto_top", pos=fish1[0], direction=fish1[1], size=fish1[2], bend=fish1[3],
                  bodykwargs=dict(lw=1, edgecolor='k', facecolor='k'),
                  finkwargs=dict(lw=1, edgecolor='k', facecolor='k'))

        ax2.plot(np.arange(len(potential))/samplerate, potential, color='k', lw=2)
        ax2.plot(i/samplerate, potential[i], 'o', color='firebrick', markersize=4, markeredgecolor='k', mew=.5)
        ax2.set_xlim(0, (len(wavefish) - 1)/samplerate)

        fig_cosmetics(ax, ax2)

        ax[0].text(0.01, 0.96, 'Speed:', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')
        ax[0].text(0.125, 0.96, '1/2000', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')

        ax[0].text(0.01, 0.89, 'EODf:', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')
        ax[0].text(0.125, 0.89, '1000 Hz', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')

        file_count = len(glob.glob(os.path.join(dir, '*.jpg')))
        file_name = os.path.join(dir, ('%4.f.jpg' % (file_count+1)).replace(' ', '0'))
        plt.savefig(file_name, dpi=300)


def second_stage_2_fish(x, y, dir, ax, ax2, eodf1 = 1., eodf2 = .8, samplerate=30, duration=8):
    xx, yy = np.meshgrid(x, y)

    # fish params
    fish1 = ((-10, 0), (1, 0), 20.0, 0)
    poles1 = efish_monopoles(*fish1)
    pot1 = epotential_meshgrid(xx, yy, None, poles1)

    fish2 = ((10, 7.5), (-1, -0.5), 15.0, -45)
    poles2 = efish_monopoles(*fish2)
    pot2 = epotential_meshgrid(xx, yy, None, poles2)


    thresh = 0.65 * 3
    levels = np.linspace(-thresh, thresh, 16)
    zz1 = squareroot_transform(pot1/200, thresh)
    zz2 = squareroot_transform(pot2/200, thresh)

    wavefish1 = wavefish_eods('Alepto', eodf1, samplerate, duration, noise_std=0)
    wavefish2 = wavefish_eods('Alepto', eodf2, samplerate, duration, noise_std=0)

    wavefish = wavefish1 + wavefish2
    potential = zz1[100, 105] * wavefish1 + zz2[100, 105] * wavefish2
    potential *=-1

    for i in tqdm(np.arange(len(wavefish)-1)):
        ax[0].clear()
        ax2.clear()

        ax[0].contourf(x, y, zz1 * wavefish1[i] + (zz2 * wavefish2[i]), levels, zorder=1, cmap='RdYlBu')
        ax[0].contour(x, y, zz1 * wavefish1[i] + (zz2 * wavefish2[i]), levels, zorder=2, colors='#707070', linewidths=0.1, linestyles='solid')

        ax[0].plot(xx[100, 100], yy[105, 105], 'o', color='firebrick', markersize=4, markeredgecolor='k', mew=.5)

        # fish shadow
        plot_fish(ax[0], "Alepto_top", pos=fish1[0], direction=fish1[1], size=fish1[2], bend=fish1[3],
                  bodykwargs=dict(lw=1, edgecolor='k', facecolor='k'),
                  finkwargs=dict(lw=1, edgecolor='k', facecolor='k'))
        plot_fish(ax[0], "Alepto_top", pos=fish2[0], direction=fish2[1], size=fish2[2], bend=fish2[3],
                  bodykwargs=dict(lw=1, edgecolor='k', facecolor='k'),
                  finkwargs=dict(lw=1, edgecolor='k', facecolor='k'))

        ax2.plot(np.arange(len(potential))/samplerate, potential, color='k', lw=2)
        ax2.plot(i/samplerate, potential[i], 'o', color='firebrick', markersize=4, markeredgecolor='k', mew=.5)
        ax2.set_xlim(0, (len(wavefish) - 1)/samplerate)

        fig_cosmetics(ax, ax2)

        ax[0].text(0.01, 0.96, 'Speed:', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')
        ax[0].text(0.125, 0.96, '1/2000', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')

        ax[0].text(0.01, 0.89, 'EODf$_{1}$:', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')
        ax[0].text(0.125, 0.89, '1000 Hz', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')

        ax[0].text(0.01, 0.82, r'EODf$_{2}$:', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')
        ax[0].text(0.125, 0.82, '800 Hz', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')

        file_count = len(glob.glob(os.path.join(dir, '*.jpg')))
        file_name = os.path.join(dir, ('%4.f.jpg' % (file_count+1)).replace(' ', '0'))
        plt.savefig(file_name, dpi=300)


def thrid_stage_2_fish_zoom(x, y, dir, ax, ax2, eodf1 = 1., eodf2 = .8, samplerate=30, duration=8):
    xx, yy = np.meshgrid(x, y)

    # fish params
    fish1 = ((-10, 0), (1, 0), 20.0, 0)
    poles1 = efish_monopoles(*fish1)
    pot1 = epotential_meshgrid(xx, yy, None, poles1)


    fish2 = ((10, 7.5), (-1, -0.5), 15.0, -45)
    poles2 = efish_monopoles(*fish2)
    pot2 = epotential_meshgrid(xx, yy, None, poles2)


    thresh = 0.65 * 3
    levels = np.linspace(-thresh, thresh, 16)
    zz1 = squareroot_transform(pot1/200, thresh)
    zz2 = squareroot_transform(pot2/200, thresh)

    wavefish1 = wavefish_eods('Alepto', eodf1, samplerate, duration, noise_std=0)
    wavefish2 = wavefish_eods('Alepto', eodf2, samplerate, duration, noise_std=0)

    wavefish = wavefish1 + wavefish2
    potential = zz1[100, 105] * wavefish1 + zz2[100, 105] * wavefish2
    potential *=-1

    duration_b = duration*5
    wavefish1_b = wavefish_eods('Alepto', eodf1, samplerate, duration_b, noise_std=0)
    wavefish2_b = wavefish_eods('Alepto', eodf2, samplerate, duration_b, noise_std=0)
    potential_b = zz1[100, 105] * wavefish1_b + zz2[100, 105] * wavefish2_b
    potential_b *=-1

    for i in tqdm(np.arange(len(wavefish)-1)):
        ax[0].clear()
        ax2.clear()

        ax[0].contourf(x, y, zz1 * wavefish1[i] + (zz2 * wavefish2[i]), levels, zorder=1, cmap='RdYlBu')
        ax[0].contour(x, y, zz1 * wavefish1[i] + (zz2 * wavefish2[i]), levels, zorder=2, colors='#707070', linewidths=0.1, linestyles='solid')

        ax[0].plot(xx[100, 100], yy[105, 105], 'o', color='firebrick', markersize=4, markeredgecolor='k', mew=.5)

        # fish shadow
        plot_fish(ax[0], "Alepto_top", pos=fish1[0], direction=fish1[1], size=fish1[2], bend=fish1[3],
                  bodykwargs=dict(lw=1, edgecolor='k', facecolor='k'),
                  finkwargs=dict(lw=1, edgecolor='k', facecolor='k'))
        plot_fish(ax[0], "Alepto_top", pos=fish2[0], direction=fish2[1], size=fish2[2], bend=fish2[3],
                  bodykwargs=dict(lw=1, edgecolor='k', facecolor='k'),
                  finkwargs=dict(lw=1, edgecolor='k', facecolor='k'))

        ax2.plot(np.arange(len(potential_b)) / samplerate, potential_b, color='k', lw=2)
        ax2.set_xlim(0, duration + (i / len(wavefish)) * (duration_b - duration))

        fig_cosmetics(ax, ax2)

        ax[0].text(0.01, 0.96, 'Speed:', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')
        ax[0].text(0.125, 0.96, '1/2000', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')

        ax[0].text(0.01, 0.89, 'EODf$_{1}$:', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')
        ax[0].text(0.125, 0.89, '1000 Hz', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')

        ax[0].text(0.01, 0.82, r'EODf$_{2}$:', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')
        ax[0].text(0.125, 0.82, '800 Hz', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')

        file_count = len(glob.glob(os.path.join(dir, '*.jpg')))
        file_name = os.path.join(dir, ('%4.f.jpg' % (file_count+1)).replace(' ', '0'))
        plt.savefig(file_name, dpi=300)


def fourth_stage_2_fish_beat(x, y, dir, ax, ax2, eodf1 = 1., eodf2 = .8, samplerate=30, duration=8):
    xx, yy = np.meshgrid(x, y)

    # fish params
    fish1 = ((-10, 0), (1, 0), 20.0, 0)
    poles1 = efish_monopoles(*fish1)
    pot1 = epotential_meshgrid(xx, yy, None, poles1)


    fish2 = ((10, 7.5), (-1, -0.5), 15.0, -45)
    poles2 = efish_monopoles(*fish2)
    pot2 = epotential_meshgrid(xx, yy, None, poles2)


    thresh = 0.65 * 3
    levels = np.linspace(-thresh, thresh, 16)
    zz1 = squareroot_transform(pot1/200, thresh)
    zz2 = squareroot_transform(pot2/200, thresh)

    wavefish1 = wavefish_eods('Alepto', eodf1, samplerate, duration, noise_std=0)
    wavefish2 = wavefish_eods('Alepto', eodf2, samplerate, duration, noise_std=0)

    wavefish = wavefish1 + wavefish2
    potential = zz1[100, 105] * wavefish1 + zz2[100, 105] * wavefish2
    potential *=-1

    duration_b = duration*5
    wavefish1_b = wavefish_eods('Alepto', eodf1, samplerate, duration_b, noise_std=0)
    wavefish2_b = wavefish_eods('Alepto', eodf2, samplerate, duration_b, noise_std=0)
    potential_b = zz1[100, 105] * wavefish1_b + zz2[100, 105] * wavefish2_b
    potential_b *=-1

    for i in tqdm(np.arange(len(wavefish)-1)):
        ax[0].clear()
        ax2.clear()

        ax[0].contourf(x, y, zz1 * wavefish1[i] + (zz2 * wavefish2[i]), levels, zorder=1, cmap='RdYlBu')
        ax[0].contour(x, y, zz1 * wavefish1[i] + (zz2 * wavefish2[i]), levels, zorder=2, colors='#707070', linewidths=0.1, linestyles='solid')

        ax[0].plot(xx[100, 100], yy[105, 105], 'o', color='firebrick', markersize=4, markeredgecolor='k', mew=.5)

        # fish shadow
        plot_fish(ax[0], "Alepto_top", pos=fish1[0], direction=fish1[1], size=fish1[2], bend=fish1[3],
                  bodykwargs=dict(lw=1, edgecolor='k', facecolor='k'),
                  finkwargs=dict(lw=1, edgecolor='k', facecolor='k'))
        plot_fish(ax[0], "Alepto_top", pos=fish2[0], direction=fish2[1], size=fish2[2], bend=fish2[3],
                  bodykwargs=dict(lw=1, edgecolor='k', facecolor='k'),
                  finkwargs=dict(lw=1, edgecolor='k', facecolor='k'))

        ax2.plot(np.arange(len(potential_b)) / samplerate, potential_b, color='k', lw=2)
        ylims = ax2.get_ylim()
        t = np.arange(len(potential_b)) / samplerate
        beat = np.cos(2 * np.pi * t * (eodf2 - eodf1)) * (np.max(potential_b)/2)*0.8 + np.max(potential_b)*1.1
        alpha = i / ((len(wavefish)-1)/5)
        alpha = 1 if alpha > 1 else alpha
        ax2.plot(t, beat, color='firebrick', lw=2, alpha =alpha, clip_on=False)
        ax2.set_ylim(ylims[0], ylims[1])

        ax2.set_xlim(0, duration_b)

        fig_cosmetics(ax, ax2)

        ax[0].text(0.01, 0.96, 'Speed:', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')
        ax[0].text(0.125, 0.96, '1/2000', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')

        ax[0].text(0.01, 0.89, 'EODf$_{1}$:', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')
        ax[0].text(0.125, 0.89, '1000 Hz', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')

        ax[0].text(0.01, 0.82, r'EODf$_{2}$:', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')
        ax[0].text(0.125, 0.82, '800 Hz', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')

        ax[0].text(0.01, 0.75, 'Beat:', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')
        ax[0].text(0.125, 0.75, '200 Hz', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')

        file_count = len(glob.glob(os.path.join(dir, '*.jpg')))
        file_name = os.path.join(dir, ('%4.f.jpg' % (file_count+1)).replace(' ', '0'))
        plt.savefig(file_name, dpi=300)

def fifth_stage_2_fish_beat(x, y, dir, ax, ax2, eodf1 = 1., eodf2 = .8, samplerate=30, duration=8):
    xx, yy = np.meshgrid(x, y)

    # fish params
    fish1 = ((-10, 0), (1, 0), 20.0, 0)
    poles1 = efish_monopoles(*fish1)
    pot1 = epotential_meshgrid(xx, yy, None, poles1)


    fish2 = ((10, 7.5), (-1, -0.5), 15.0, -45)
    poles2 = efish_monopoles(*fish2)
    pot2 = epotential_meshgrid(xx, yy, None, poles2)


    thresh = 0.65 * 3
    levels = np.linspace(-thresh, thresh, 16)
    zz1 = squareroot_transform(pot1/200, thresh)
    zz2 = squareroot_transform(pot2/200, thresh)

    wavefish1 = wavefish_eods('Alepto', eodf1, samplerate, duration, noise_std=0)
    wavefish2 = wavefish_eods('Alepto', eodf2, samplerate, duration, noise_std=0)

    wavefish = wavefish1 + wavefish2
    potential = zz1[100, 105] * wavefish1 + zz2[100, 105] * wavefish2
    potential *=-1

    duration_b = duration*5
    wavefish1_b = wavefish_eods('Alepto', eodf1, samplerate, duration_b, noise_std=0)
    wavefish2_b = wavefish_eods('Alepto', eodf2, samplerate, duration_b, noise_std=0)
    potential_b = zz1[100, 105] * wavefish1_b + zz2[100, 105] * wavefish2_b
    potential_b *=-1

    for i in tqdm(np.arange(len(wavefish)-1)):
        ax[0].clear()
        ax2.clear()

        ax[0].contourf(x, y, zz1 * wavefish1[i] + (zz2 * wavefish2[i]), levels, zorder=1, cmap='RdYlBu')
        ax[0].contour(x, y, zz1 * wavefish1[i] + (zz2 * wavefish2[i]), levels, zorder=2, colors='#707070', linewidths=0.1, linestyles='solid')

        ax[0].plot(xx[100, 100], yy[105, 105], 'o', color='firebrick', markersize=4, markeredgecolor='k', mew=.5)

        # fish shadow
        plot_fish(ax[0], "Alepto_top", pos=fish1[0], direction=fish1[1], size=fish1[2], bend=fish1[3],
                  bodykwargs=dict(lw=1, edgecolor='k', facecolor='k'),
                  finkwargs=dict(lw=1, edgecolor='k', facecolor='k'))
        plot_fish(ax[0], "Alepto_top", pos=fish2[0], direction=fish2[1], size=fish2[2], bend=fish2[3],
                  bodykwargs=dict(lw=1, edgecolor='k', facecolor='k'),
                  finkwargs=dict(lw=1, edgecolor='k', facecolor='k'))

        ax2.plot(np.arange(len(potential_b)) / samplerate, potential_b, color='k', lw=2)
        ylims = ax2.get_ylim()
        t = np.arange(len(potential_b)) / samplerate
        beat = np.cos(2 * np.pi * t * (eodf2 - eodf1)) * (np.max(potential_b)/2)*0.8 + np.max(potential_b)*1.1
        alpha=1
        ax2.plot(t, beat, color='firebrick', lw=2, alpha =alpha, clip_on=False)
        ax2.set_ylim(ylims[0], ylims[1])

        ax2.set_xlim(0, duration_b)

        fig_cosmetics(ax, ax2)

        ax[0].text(0.01, 0.96, 'Speed:', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')
        ax[0].text(0.125, 0.96, '1/2000', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')

        ax[0].text(0.01, 0.89, 'EODf$_{1}$:', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')
        ax[0].text(0.125, 0.89, '1000 Hz', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')

        ax[0].text(0.01, 0.82, r'EODf$_{2}$:', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')
        ax[0].text(0.125, 0.82, '800 Hz', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')

        ax[0].text(0.01, 0.75, 'Beat:', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')
        ax[0].text(0.125, 0.75, '200 Hz', transform=ax[0].transAxes, fontsize=12, va='center', ha='left')

        file_count = len(glob.glob(os.path.join(dir, '*.jpg')))
        file_name = os.path.join(dir, ('%4.f.jpg' % (file_count+1)).replace(' ', '0'))
        plt.savefig(file_name, dpi=300)


def create_figure(figsize=(15, 15*(9/16)), ax_lbrt=(0, 0, 1, 1)):
    fig = plt.figure(figsize=(figsize[0] / 2.54, figsize[1] / 2.54))
    gs = gridspec.GridSpec(1, 1, left=ax_lbrt[0], bottom=ax_lbrt[1], right=ax_lbrt[2], top=ax_lbrt[3])
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))

    gs2 = gridspec.GridSpec(1, 1, left=ax_lbrt[2] - .25, bottom=ax_lbrt[1]+.025, right=ax_lbrt[2]-.025, top=ax_lbrt[1] + .2)
    ax2 = fig.add_subplot(gs2[0, 0])

    maxx = figsize[0] * (ax_lbrt[2] - ax_lbrt[0]) * 2.5 #  fig width * axis size (size_hint = 0.8) * upscale
    maxy = figsize[1] * (ax_lbrt[3] - ax_lbrt[1])  * 2.5

    return fig, ax, ax2, maxx, maxy

def fig_cosmetics(ax, ax2):
    ax2.patch.set_alpha(0)
    ax2.set_axis_off()

    ax[0].set_xticks([])
    ax[0].set_yticks([])


def main():
    # grid params
    dir = './field_anim/'
    files_in_directory = os.listdir(dir)
    filtered_files = [file for file in files_in_directory if file.endswith(".jpg")]
    for file in filtered_files:
        path_to_file = os.path.join(dir, file)
        os.remove(path_to_file)

    fig, ax, ax2, maxx, maxy = create_figure()

    x = np.linspace(-maxx, maxx, 200)
    y = np.linspace(-maxy, maxy, 200)

    # first_stage_1_fish(x, y, dir, ax, ax2, eodf=0.5, duration=8)

    second_stage_2_fish(x, y, dir, ax, ax2, eodf1 = .5, eodf2 = .4, duration=10)

    thrid_stage_2_fish_zoom(x, y, dir, ax, ax2, eodf1 = .5, eodf2 = .4, duration=10)

    fourth_stage_2_fish_beat(x, y, dir, ax, ax2, eodf1 = .5, eodf2 = .4, duration=10)

    # fifth_stage_2_fish_beat(x, y, dir, ax, ax2, eodf1 = .5, eodf2 = .4, duration=10)
    pass


if __name__ == '__main__':
    main()