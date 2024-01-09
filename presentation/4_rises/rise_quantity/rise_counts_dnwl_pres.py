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


def main():
    win_rc = np.load('../win_rc.npy', allow_pickle=True)
    lose_rc = np.load('../lose_rc.npy', allow_pickle=True)

    win_rc_day = np.load('../win_rc_day.npy', allow_pickle=True)
    lose_rc_day = np.load('../lose_rc_day.npy', allow_pickle=True)

    fs = 12
    fig = plt.figure(figsize=(8/2.54, 10/2.54))
    gs = gridspec.GridSpec(1, 1, bottom=0.1, left=0.225, right=0.975, top=0.975)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))

    female_color, male_color = '#e74c3c', '#3498db'

    win_color = [male_color, female_color, male_color, female_color]
    lose_color = [male_color, female_color, female_color, male_color]
    mek = ['k', 'k', None, None]
    for i in range(4):
        ax[0].plot(np.ones(len(win_rc_day[i]))* 4 + (np.random.rand(len(win_rc_day[i])) - 0.5) * 0.5, win_rc_day[i], 'p', color=win_color[i], zorder=2, markersize=5, markeredgecolor=mek[i])
        ax[0].plot(np.ones(len(lose_rc_day[i]))* 3 + (np.random.rand(len(lose_rc_day[i])) - 0.5) * 0.5, lose_rc_day[i], 'o', color=lose_color[i], zorder=2, markersize=5, markeredgecolor=mek[i])
        ax[0].plot(np.ones(len(win_rc[i]))* 2 + (np.random.rand(len(win_rc[i])) - 0.5) * 0.5, win_rc[i], 'p', color=win_color[i], zorder=2, markersize=5, markeredgecolor=mek[i])
        ax[0].plot(np.ones(len(lose_rc[i]))* 1 + (np.random.rand(len(lose_rc[i])) - 0.5) * 0.5, lose_rc[i], 'o', markeredgecolor=mek[i], color=lose_color[i], zorder=2, markersize=5)
    bp = ax[0].boxplot([np.hstack(lose_rc), np.hstack(win_rc), np.hstack(lose_rc_day), np.hstack(win_rc_day)], sym='', widths=0.75, zorder=3, patch_artist=True)
    plt.setp(bp['medians'], color='k')
    plt.setp(bp['boxes'], facecolor='none')

    ax[0].fill_between([0.5, 2.5], [-50, -50], [470, 470], color='grey', alpha=0.5, zorder=1)
    ax[0].set_ylim(-20, 470)
    ax[0].set_ylabel('# rises', fontsize=fs + 2)

    ax[0].set_xticklabels(['Lose', 'Win', 'Lose', 'Win'])

    ax[0].tick_params(labelsize=fs)

    plt.savefig('rise_counts_wldn.pres.jpg', dpi=300)

    plt.show()
    pass

if __name__ == '__main__':
    main()