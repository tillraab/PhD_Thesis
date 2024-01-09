import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as scp
from IPython import embed
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit
from plottools.tag import tag

def boltzmann(x, k, shift):
    return 1 / (1 + np.exp(-k * x + shift))

def main():
    def logistic_fit():
        df = pd.DataFrame()
        fit_x = np.hstack([d_param, np.array(d_param) * -1])
        fit_y = np.hstack([np.ones(len(d_param)), np.zeros(len(d_param))])

        df['win'] = fit_y
        df['d_param'] = fit_x
        df = sm.add_constant(df)
        try:
            logit_model = sm.Logit(df['win'][:, np.newaxis], df[['const', 'd_param']]).fit()
            fit_c = 'k'

            max_d_param = np.max(np.abs(dparam_range))
            test_df = pd.DataFrame()
            test_df['const'] = np.ones(len(np.linspace(-max_d_param, max_d_param, 100)))
            test_df['d_param'] = np.linspace(-max_d_param, max_d_param, 100)

            pred = logit_model.predict(test_df)

            x, y = test_df['d_param'], pred

        except:
            x = np.linspace(np.min(dparam_range), np.max(dparam_range), 100)
            if enu == 2:
                y = boltzmann(x, .5, 0)
            elif enu == 3:
                y = boltzmann(x, -.5, 0)
        return x, y

    fs = 12
    female_color, male_color = '#e74c3c', '#3498db'

    dsize_win = np.load('../dsize_win.npy', allow_pickle=True)
    df_win = np.load('../df_win.npy', allow_pickle=True)
    wl_sex_EODf_size = np.load('wl_sex_EODf_size.npy')
    sex_w, sex_l, EODf_w, EODf_l, size_w, size_l = wl_sex_EODf_size

    dparam_win_mm, dparam_win_ff, dparam_win_mf, dparam_win_fm = dsize_win[:4]

    dparam_range = np.hstack([dparam_win_mm, np.array(dparam_win_mm) * -1,
                              dparam_win_ff, np.array(dparam_win_ff) * -1,
                              dparam_win_mf, np.array(dparam_win_mf) * -1,
                              dparam_win_fm, np.array(dparam_win_fm) * -1])

    fig = plt.figure(figsize=(15/2.54, 15 * (12/20) / 2.54))
    gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.2, right=0.975, top=0.875)
    ax = fig.add_subplot(gs[0, 0])

    win_color = [male_color, female_color, male_color, female_color]
    lose_color = [male_color, female_color, female_color, male_color]
    mek= ['k', 'k', None, None]

    for enu, d_param in enumerate([dparam_win_mm, dparam_win_ff, dparam_win_mf, dparam_win_fm]):

        ax.plot(d_param, np.ones(len(d_param)) - 0.21 + 0.14*enu, 'p', color=win_color[enu], markeredgecolor=mek[enu], alpha=0.8, markersize=8)
        ax.plot(np.array(d_param) * -1, np.zeros(len(d_param)) - 0.21 + 0.14*enu, 'o', color=lose_color[enu], markeredgecolor=mek[enu], markersize=8)

    d_param = np.hstack([dparam_win_mm, dparam_win_ff, dparam_win_mf, dparam_win_fm])

    x, y = logistic_fit()
    ax.plot(x, y, linestyle='-', color='k', lw=2.5)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Lose', 'Win'])
    ax.set_xlabel(r'$\Delta$size [cm]', fontsize= fs+2)
    ax.tick_params(labelsize=fs)

    ylim = ax.get_ylim()

    ax.plot(0, 2, 'p', markeredgecolor='k', color='k', label='Winner')
    ax.plot(0, 2, 'o', color='grey', label='Loser')
    ax.plot(0, 2, 's', color=male_color, label=u'\u2642')
    ax.plot(0, 2, 's', color=female_color, label=u'\u2640')
    ax.plot(0, 2, 's', color='lightgrey', markeredgecolor='k', label='same-sex')
    ax.plot(0, 2, 's', color='lightgrey', label='mixed-sex')
    ax.legend(loc='upper center', frameon=False, bbox_to_anchor=(.5, 1.225), ncol=3, fontsize=fs-2)

    ax.set_ylim(ylim)

    plt.savefig('wl_size_pres.jpg', dpi=300)
    plt.show()
    pass

if __name__ == '__main__':
    main()