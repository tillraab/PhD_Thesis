import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as scp
from IPython import embed
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit
from plottools.tag import tag
from plottools.significance import significance_bar
# fig.tag(axes=[ax[1], ax[2]], fontsize=15, yoffs=2, xoffs=-3)

def non_nan_pearsonr(x, y, return_color=False):
    x, y = np.array(x), np.array(y)
    mask = np.intersect1d(np.arange(len(x))[~np.isnan(x)], np.arange(len(y))[~np.isnan(y)])
    # mask = np.unique(np.array(list(np.arange(len(x))[~np.isnan(x)]) + list(np.arange(len(y))[~np.isnan(y)])))
    r, p = scp.pearsonr(x[mask], y[mask])

    if not return_color:
        return r, p
    else:
        if p > .1:
            c = 'firebrick'
        elif p > 0.05:
            c = 'darkorange'
        else:
            c = 'limegreen'
        return r, p, c

def plot_dsize_rc(mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc,
                  mm_win_dsize, ff_win_dsize, mf_win_dsize, fm_win_dsize, mix_win_dsize, ax, select_i = None, alpha=0.8, markersize=8):

    select_i = np.arange(4) if not hasattr(select_i, '__len__') else select_i

    female_color, male_color= '#e74c3c', '#3498db'
    win_color = [male_color, female_color, male_color, female_color]
    lose_color = [male_color, female_color, female_color, male_color]

    mek= ['k', 'k', None, None]
    for enu, Lrc, Ldsize in zip(np.arange(4), [mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc],
                                [np.array(mm_win_dsize)*-1, np.array(ff_win_dsize)*-1, np.array(mf_win_dsize)*-1, np.array(fm_win_dsize)*-1]):
        if enu not in select_i:
            continue
        ax.plot(Ldsize, Lrc, 'p', color=win_color[enu], markeredgecolor=mek[enu], alpha=alpha, zorder=1, markersize=markersize)

    ax.set_xlim(-9, 5)
    ax.set_ylim(-10, 450)


def regression_line(mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc, mm_win_dsize, ff_win_dsize, mf_win_dsize,
                    fm_win_dsize, mix_win_dsize, ax, select_i = None, color='k', alpha = 1.):

    select_i = np.arange(4) if not hasattr(select_i, '__len__') else select_i

    all_dsize = np.array([mm_win_dsize, ff_win_dsize, mf_win_dsize, fm_win_dsize])
    all_rc = np.array([mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc])

    x_vals = np.hstack(all_dsize[select_i])*-1
    y_vals = np.hstack(all_rc[select_i])

    m, b, _, _, _ = scp.linregress(x_vals, y_vals)
    # XX = np.array([np.min(x_vals), np.max(x_vals)])
    XX = np.array([-9, 5])
    ax.plot(XX, m*XX+b, color=color, lw=2, alpha=alpha)


    print('\ndSize dependent rise count')
    r, p = non_nan_pearsonr(x_vals, y_vals)
    print('r=%.3f, p=%.4f\n' % (r, p))

def main():

    lose_rc = np.load('../lose_rc.npy', allow_pickle=True)
    dsize_win = np.load('../dsize_win.npy', allow_pickle=True)


    fs = 12
    female_color, male_color = '#e74c3c', '#3498db'
    embed()
    quit()
    fig = plt.figure(figsize=(12 / 2.54, 12 * (14 / 20) / 2.54))
    gs = gridspec.GridSpec(1, 1, left=0.16, bottom=0.2, top=0.85, right=0.95)
    ax = fig.add_subplot(gs[0, 0])

    plot_dsize_rc(*lose_rc, *dsize_win, ax, select_i=[3], markersize=7)
    plot_dsize_rc(*lose_rc, *dsize_win, ax, select_i=[0, 1, 2], alpha=0.2, markersize=7)
    # regression_line(*lose_rc, *dsize_win, ax, select_i=[0, 2], color=male_color)
    # regression_line(*lose_rc, *dsize_win, ax, select_i=[1, 3], color=female_color)
    regression_line(*lose_rc, *dsize_win, ax, select_i=[0, 2], color=male_color, alpha=0.2)

    ax.plot([0, 0], [-50, 500], 'grey', linestyle='dashed', lw=1.5)

    ax.set_xlabel(r'$\Delta$size [cm]', fontsize=fs+2)
    ax.set_ylabel('rises [n]', fontsize=fs+2)
    ax.set_xticks(np.arange(-7.5, 5.1, 2.5))

    ax.plot(-100, 2, 'p', markeredgecolor='k', color='k', label='Winner')
    ax.plot(-100, 2, 'o', color='grey', label='Loser')
    ax.plot(-100, 2, 's', color=male_color, label=u'\u2642')
    ax.plot(-100, 2, 's', color=female_color, label=u'\u2640')
    ax.plot(-100, 2, 's', color='lightgrey', markeredgecolor='k', label='same-sex')
    ax.plot(-100, 2, 's', color='lightgrey', label='mixed-sex')
    ax.legend(loc='upper center', frameon=False, bbox_to_anchor=(.5, 1.225), ncol=3, fontsize=9)
    ax.tick_params(labelsize=fs)

    plt.savefig('pres_rise_dsize3.jpg', dpi=300)
    plt.show()

    #######################################
    fs = 14
    fig = plt.figure(figsize=(10 / 2.54, 10 * (12 / 20) / 2.54))
    gs = gridspec.GridSpec(1, 1, left=0.175, bottom=0.25, top=0.85, right=0.95)
    ax = fig.add_subplot(gs[0, 0])

    plot_dsize_rc(*lose_rc, *dsize_win, ax, markersize=7)
    # regression_line(*lose_rc, *dsize_win, ax, select_i=[0, 2], color=male_color)
    # regression_line(*lose_rc, *dsize_win, ax, select_i=[1, 3], color=female_color)
    regression_line(*lose_rc, *dsize_win, ax, select_i=[0, 2], color=male_color)

    ax.plot([0, 0], [-50, 500], 'grey', linestyle='dashed', lw=1.5)

    ax.set_xlabel(r'$\Delta$size [cm]', fontsize=fs+2)
    ax.set_ylabel('rises [n]', fontsize=fs+2)

    ax.plot(-100, 2, 'p', markeredgecolor='k', color='k', label='Winner')
    ax.plot(-100, 2, 'o', color='grey', label='Loser')
    ax.plot(-100, 2, 's', color=male_color, label=u'\u2642')
    ax.plot(-100, 2, 's', color=female_color, label=u'\u2640')
    ax.plot(-100, 2, 's', color='lightgrey', markeredgecolor='k', label='same-sex')
    ax.plot(-100, 2, 's', color='lightgrey', label='mixed-sex')
    ax.legend(loc='upper center', frameon=False, bbox_to_anchor=(.5, 1.325), ncol=3, fontsize=9)
    ax.tick_params(labelsize=fs)

    plt.savefig('poster_rise_dsize.pdf')
    plt.show()
    ##############################

    #############################
    fig = plt.figure(figsize=(10 / 2.54, 10 * (12 / 20) / 2.54))
    gs = gridspec.GridSpec(1, 1, left=0.175, bottom=0.225, top=0.975, right=0.95)
    ax = fig.add_subplot(gs[0, 0])

    plot_dsize_rc(*lose_rc, *dsize_win, ax)
    # regression_line(*lose_rc, *dsize_win, ax, select_i=[0, 2], color=male_color)
    # regression_line(*lose_rc, *dsize_win, ax, select_i=[1, 3], color=female_color)
    regression_line(*lose_rc, *dsize_win, ax, select_i=[0, 1, 2])

    ax.set_xlabel(r'$\Delta$size [cm]', fontsize=fs+2)
    ax.set_ylabel('rises [n]', fontsize=fs+2)
    ax.tick_params(labelsize=fs)

    plt.savefig('dsize_rises_all.jpg', dpi=300)
    ##############################

    fig = plt.figure(figsize=(10 / 2.54, 10 * (12 / 20) / 2.54))
    gs = gridspec.GridSpec(1, 1, left=0.175, bottom=0.225, top=0.975, right=0.95)
    ax = fig.add_subplot(gs[0, 0])

    plot_dsize_rc(*lose_rc, *dsize_win, ax, select_i=[0, 1, 2])
    regression_line(*lose_rc, *dsize_win, ax, select_i=[0, 1, 2])

    ax.set_xlabel(r'$\Delta$size [cm]', fontsize=fs+2)
    ax.set_ylabel('rises [n]', fontsize=fs+2)
    ax.tick_params(labelsize=fs)

    plt.savefig('dsize_rises_fm_excl.jpg', dpi=300)
    ##############################

    fig = plt.figure(figsize=(10 / 2.54, 10 * (12 / 20) / 2.54))
    gs = gridspec.GridSpec(1, 1, left=0.175, bottom=0.225, top=0.975, right=0.95)
    ax = fig.add_subplot(gs[0, 0])

    plot_dsize_rc(*lose_rc, *dsize_win, ax, select_i=[0, 2])
    regression_line(*lose_rc, *dsize_win, ax, select_i=[0, 2])

    ax.set_xlabel(r'$\Delta$size [cm]', fontsize=fs+2)
    ax.set_ylabel('rises [n]', fontsize=fs+2)
    ax.tick_params(labelsize=fs)

    plt.savefig('dsize_rises_male_win.jpg', dpi=300)




    plt.show()


    pass

if __name__ == '__main__':
    main()

