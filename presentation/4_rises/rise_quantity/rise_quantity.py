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

def gauss(t, shift, sigma, size, norm = False):
    if not hasattr(shift, '__len__'):
        g = np.exp(-((t - shift) / sigma) ** 2 / 2) * size
        if norm:
            g /= np.sum(g)
        return g
    else:
        t = np.array([t, ] * len(shift))
        res = np.exp(-((t.transpose() - shift).transpose() / sigma) ** 2 / 2) * size
        return res

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

def non_nan_mannwhitneyU(x, y):
    x, y = np.array(x), np.array(y)
    mask = np.intersect1d(np.arange(len(x))[~np.isnan(x)], np.arange(len(y))[~np.isnan(y)])
    # mask = np.unique(np.array(list(np.arange(len(x))[~np.isnan(x)]) + list(np.arange(len(y))[~np.isnan(y)])))
    U, p = scp.mannwhitneyu(x[mask], y[mask])

    return U, p

def boltzmann(x, k, shift):
    return 1 / (1 + np.exp(-k * x + shift))

def plot_drc(mm_win_rc, ff_win_rc, mf_win_rc, fm_win_rc, mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc, ax):

    mm_win_rd = np.array(mm_win_rc) - (mm_lose_rc)
    mm_lose_rd = mm_win_rd * -1

    ff_win_rd = np.array(ff_win_rc) - (ff_lose_rc)
    ff_lose_rd = ff_win_rd * -1

    mf_win_rd = np.array(mf_win_rc) - (mf_lose_rc)
    mf_lose_rd = mf_win_rd * -1

    fm_win_rd = np.array(fm_win_rc) - (fm_lose_rc)
    fm_lose_rd = fm_win_rd * -1

    max_rc = np.max(np.hstack([mm_win_rd, ff_win_rd, mf_win_rd, fm_win_rd, mm_lose_rd, ff_lose_rd, mf_lose_rd, fm_lose_rd]))
    female_color, male_color= '#e74c3c', '#3498db'
    win_color = [male_color, female_color, male_color, female_color]
    lose_color = [male_color, female_color, female_color, male_color]
    mek= ['k', 'k', None, None]

    for enu, win_rc, lose_rc in zip(np.arange(4), [mm_win_rd, ff_win_rd, mf_win_rd, fm_win_rd], [mm_lose_rd, ff_lose_rd, mf_lose_rd, fm_lose_rd]):
        ax.plot(win_rc, np.ones(len(win_rc)) - 0.15 + 0.10*enu, 'p', color=win_color[enu], markeredgecolor=mek[enu], alpha=0.8, zorder=1)
        ax.plot(lose_rc, np.zeros(len(lose_rc)) - 0.15 + 0.10*enu, 'o', color=lose_color[enu], markeredgecolor=mek[enu], alpha=0.8, zorder=2)


    df = pd.DataFrame()
    all_win_rd = np.hstack([mm_win_rd, ff_win_rd, mf_win_rd, fm_win_rd])
    all_lose_rd = np.hstack([mm_lose_rd, ff_lose_rd, mf_lose_rd, fm_lose_rd])
    fit_x = np.hstack([all_win_rd, all_lose_rd])
    fit_y = np.hstack([np.ones(len(all_win_rd)), np.zeros(len(all_lose_rd))])

    df['win'] = fit_y
    df['d_param'] = fit_x
    df = sm.add_constant(df)
    logit_model = sm.Logit(df['win'][:, np.newaxis], df[['const', 'd_param']]).fit()

    test_df = pd.DataFrame()
    test_df['const'] = np.ones(len(np.linspace(-max_rc, max_rc, 1000)))
    test_df['d_param'] = np.linspace(-max_rc, max_rc, 1000)
    pred = logit_model.predict(test_df)

    ax.plot(test_df['d_param'], pred, linestyle='-', color='k', lw=2, zorder=1)

    y0, y1 = ax.get_ylim()
    ax.plot([0, 0], [y0, y1], 'k', lw=1, linestyle='dotted')
    ax.set_ylim(y0, y1)

    ax.set_xlabel(r'$\Delta$EODf rises [n]', fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Lose', 'Win'])
    ax.tick_params(labelsize=10)

def plot_df_rc(mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc,
               mm_win_df, ff_win_df, mf_win_df, fm_win_df, mix_win_df, ax):

    female_color, male_color= '#e74c3c', '#3498db'
    win_color = [male_color, female_color, male_color, female_color]
    lose_color = [male_color, female_color, female_color, male_color]
    mek= ['k', 'k', None, None]

    for enu, Lrc, Ldf in zip(np.arange(4), [mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc],
                             [np.array(mm_win_df)*-1, np.array(ff_win_df)*-1, np.array(mf_win_df)*-1, np.array(fm_win_df)*-1]):
        ax.plot(Ldf, Lrc, 'p', color=win_color[enu], markeredgecolor=mek[enu], alpha=0.8, zorder=1)

    xx = np.hstack([np.array(mm_win_df)*-1, np.array(ff_win_df)*-1, np.array(mf_win_df)*-1, np.array(fm_win_df)*-1])
    yy = np.hstack([mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc])

    XX = np.array([np.min(xx), np.max(xx)])
    m, b, _, _, _ = scp.linregress(xx, yy)
    ax.plot(XX, m*XX+b, color='k', lw=2)

    ax.set_xlabel(r'$\Delta$EODf [Hz]', fontsize=12)
    ax.tick_params(labelsize=10)

    r, p = non_nan_pearsonr(xx, yy)
    print('\n dEODf dependente Rise count')
    print('r=%.3f, p=%.4f' % (r, p))
    print('')


def plot_rc(mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc, ax):

    female_color, male_color= '#e74c3c', '#3498db'
    lose_color = [male_color, male_color, female_color, female_color]
    mek= ['k', None, 'k', None]

    for enu, rc in enumerate([mm_lose_rc, mf_lose_rc, ff_lose_rc, fm_lose_rc]):
        ax.plot(np.ones(len(rc))* enu + (np.random.rand(len(rc)) - 0.5) * 0.25, rc, 'p', color=lose_color[enu], markeredgecolor=mek[enu], alpha=0.8)

    bp = ax.boxplot([mm_lose_rc, mf_lose_rc, ff_lose_rc, fm_lose_rc], positions=np.arange(4), sym='', patch_artist=True)
    plt.setp(bp['medians'], color='k')
    plt.setp(bp['boxes'], facecolor='none')

    print('\nRise count (pairing)')
    u, p = scp.mannwhitneyu(mm_lose_rc, mf_lose_rc)
    print('mm vs. mf: U= %.1f, p=%.4f' % (u, p))
    u, p = scp.mannwhitneyu(ff_lose_rc, fm_lose_rc)
    print('ff vs. fm: U= %.1f, p=%.4f' % (u, p))

    u, p = scp.mannwhitneyu(mm_lose_rc, fm_lose_rc)
    print('mm vs. fm: U= %.1f, p=%.4f' % (u, p))
    u, p = scp.mannwhitneyu(ff_lose_rc, mf_lose_rc)
    print('ff vs. mf: U= %.1f, p=%.4f' % (u, p))

    u, p = scp.mannwhitneyu(mm_lose_rc + mf_lose_rc, ff_lose_rc + fm_lose_rc)
    print('m-win vs f-win: U= %.1f, p=%.4f' % (u, p))
    u, p = scp.mannwhitneyu(mm_lose_rc + fm_lose_rc, ff_lose_rc + mf_lose_rc)
    print('m-lose vs f-lose: U= %.1f, p=%.4f' % (u, p))
    print('')
    ax.set_xticks(np.arange(4))
    labels = [u'\u2642\u2642', u'\u2642\u2640', u'\u2640\u2640', u'\u2640\u2642']
    ax.set_xticklabels(labels)

    ax.set_xlabel('pairing', fontsize=12)
    ax.tick_params(labelsize=10)

def plot_dsize_rc(mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc,
                  mm_win_dsize, ff_win_dsize, mf_win_dsize, fm_win_dsize, mix_win_dsize, ax):

    female_color, male_color= '#e74c3c', '#3498db'
    win_color = [male_color, female_color, male_color, female_color]
    lose_color = [male_color, female_color, female_color, male_color]

    mek= ['k', 'k', None, None]
    for enu, Lrc, Ldsize in zip(np.arange(4), [mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc],
                                [np.array(mm_win_dsize)*-1, np.array(ff_win_dsize)*-1, np.array(mf_win_dsize)*-1, np.array(fm_win_dsize)*-1]):
        ax.plot(Ldsize, Lrc, 'p', color=win_color[enu], markeredgecolor=mek[enu], alpha=0.8, zorder=1)


    x_vals = np.hstack([mm_win_dsize, ff_win_dsize, mf_win_dsize, fm_win_dsize])*-1
    m, b, _, _, _ = scp.linregress(np.hstack([mm_win_dsize, mf_win_dsize])*-1, np.hstack([mm_lose_rc, mf_lose_rc]))
    XX = np.array([np.min(x_vals), np.max(x_vals)])
    ax.plot(XX, m*XX+b, color=male_color, lw=2)

    m, b, _, _, _ = scp.linregress(np.hstack([ff_win_dsize, fm_win_dsize])*-1, np.hstack([ff_lose_rc, fm_lose_rc]))
    XX = np.array([np.min(x_vals), np.max(x_vals)])
    ax.plot(XX, m*XX+b, color=female_color, lw=2)

    ax.set_xlabel(r'$\Delta$size [cm]', fontsize=12)
    ax.tick_params(labelsize=10)

    print('dSize dependent rise count')
    r, p = non_nan_pearsonr(np.hstack([mm_win_dsize, mf_win_dsize])*-1, np.hstack([mm_lose_rc, mf_lose_rc]))
    print('m-win: r=%.3f, p=%.4f' % (r, p))
    r, p = non_nan_pearsonr(np.hstack([ff_win_dsize, fm_win_dsize])*-1, np.hstack([ff_lose_rc, fm_lose_rc]))
    print('f-win: r=%.3f, p=%.4f' % (r, p))
    print('')

def plot_exp_rc(mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc, mm_lose_exp, ff_lose_exp, mf_lose_exp, fm_lose_exp, ax):

    all_rc = np.hstack([mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc])
    all_exp = np.hstack([mm_lose_exp, ff_lose_exp, mf_lose_exp, fm_lose_exp])

    female_color, male_color= '#e74c3c', '#3498db'
    win_color = [male_color, female_color, male_color, female_color]
    lose_color = [male_color, female_color, female_color, male_color]

    mek= ['k', 'k', None, None]

    for enu, rc, exp in zip(np.arange(4), [mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc],
                            [mm_lose_exp, ff_lose_exp, mf_lose_exp, fm_lose_exp]):
        ax.plot(exp + (np.random.rand(len(exp)) - 0.5) * 0.25, rc, 'p', color=win_color[enu], markeredgecolor=mek[enu], alpha=0.8, zorder=1)

    rc_per_exp = []
    for e in np.sort(np.unique(all_exp)):
        rc_per_exp.append(all_rc[all_exp == e])

    bp = ax.boxplot(rc_per_exp, sym = '', patch_artist=True)
    plt.setp(bp['medians'], color='k')
    plt.setp(bp['boxes'], facecolor='none')

    ax.set_xlabel('loser experience [trial]', fontsize=12)
    # ax.set_ylabel('EODf rises [n]', fontsize=12)
    ax.tick_params(labelsize=10)

    r, p = non_nan_pearsonr(np.hstack([mm_lose_exp, ff_lose_exp, mf_lose_exp, fm_lose_exp]),
                            np.hstack([mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc]))
    print('Experience dependent Rise count')
    print('r=%.3f, p=%.4f' % (r, p))
    print('n.s. without experience == 1')

    r, p = non_nan_pearsonr(np.hstack([mm_lose_exp[mm_lose_exp>1], ff_lose_exp[ff_lose_exp>1],
                                       mf_lose_exp[mf_lose_exp>1], fm_lose_exp[fm_lose_exp>1]]),
                            np.hstack([np.array(mm_lose_rc)[mm_lose_exp>1], np.array(ff_lose_rc)[ff_lose_exp>1],
                                       np.array(mf_lose_rc)[mf_lose_exp>1], np.array(fm_lose_rc)[fm_lose_exp>1]]))


def significnace(ax, x1, x2, y, p, vertical=False, whisker_fac = 0.0035):
    x1, x2 = np.array([x1, x2])[np.argsort(np.array([x1, x2]))]

    if p < 0.001:
        ps = '***'
    elif p < 0.01:
        ps = '**'
    elif p < 0.05:
        ps = '*'
    else:
        'n.s.'
    if vertical:
        whisk_len = np.diff(ax.get_xlim())[0] * whisker_fac
        ax.plot([y, y], [x1, x2], lw=1, color='k')
        ax.plot([y-whisk_len, y], [x1, x1], lw=1, color='k')
        ax.plot([y-whisk_len, y], [x2, x2], lw=1, color='k')

        ax.text(y + whisk_len*1, x1 + (x2 - x1)/2, ps, fontsize=10, color='k', ha='left', va='center', rotation=90)
    else:
        whisk_len = np.diff(ax.get_ylim())[0] * whisker_fac
        ax.plot([x1, x2], [y, y], lw=1, color='k')
        ax.plot([x1, x1], [y - whisk_len, y], lw=1, color='k')
        ax.plot([x2, x2], [y - whisk_len, y], lw=1, color='k')

        ax.text( x1 + (x2 - x1) / 2, y + whisk_len*0.5, ps, fontsize=10, color='k', ha='center', va='bottom')

def plot_cont_rise_ratio_losers(win_rc_15b, lose_rc_15b, ax1):

    dRises_tw = [[] for i in range(12)]

    all_ratios = []
    all_abs = []
    for pairing in range(len(win_rc_15b)):
        for tw in range(len(win_rc_15b[0])):
            dRises_tw[tw].extend(np.array(lose_rc_15b[pairing][tw]) / (np.array(lose_rc_15b[pairing][tw]) + np.array(win_rc_15b[pairing][tw])))
            all_ratios.extend(np.array(lose_rc_15b[pairing][tw]) / (np.array(lose_rc_15b[pairing][tw]) + np.array(win_rc_15b[pairing][tw])))
            all_abs.extend(lose_rc_15b[pairing][tw])

    mean_Rise_ratio_tw = []
    std_Rise_ratio_tw = []
    for i in range(len(dRises_tw)):
        mean_Rise_ratio_tw.append(np.nanmean(dRises_tw[i]))
        std_Rise_ratio_tw.append(np.nanstd(dRises_tw[i]))
    mean_Rise_ratio_tw = np.array(mean_Rise_ratio_tw)
    std_Rise_ratio_tw = np.array(std_Rise_ratio_tw)


    for i in range(len(dRises_tw)):
        rr = np.array(dRises_tw)[:, i]


        for j in range(len(rr) - 1):
            ax1.plot([j * 15, (j + 1) * 15], [rr[j], rr[j]], color='grey', lw=.5)
            ax1.plot([(j + 1) * 15, (j + 1) * 15], [rr[j], rr[j + 1]], color='grey', lw=.5)

        ax1.plot([(len(rr)-1) * 15, (len(rr)) * 15], [rr[-1], rr[-1]], color='grey', lw=.5)


    for i in range(len(mean_Rise_ratio_tw)-1):
        ax1.plot([i*15, (i+1)*15], [mean_Rise_ratio_tw[i], mean_Rise_ratio_tw[i]], color='k', lw=2)
        ax1.plot([(i+1)*15, (i+1)*15], [mean_Rise_ratio_tw[i], mean_Rise_ratio_tw[i+1]], color='k', lw=2)
    ax1.plot([(len(mean_Rise_ratio_tw)-1) * 15, len(mean_Rise_ratio_tw) * 15], [mean_Rise_ratio_tw[-1], mean_Rise_ratio_tw[-1]], color='k', lw=2)

    #
    ax1.plot([0, 180], [0.5, 0.5], lw=1, color='k', linestyle='dotted')

def get_rise_rates(trial_lose_rise_time):
    rise_rates = []
    time_array = np.arange(0, 3*60*60, 0.1) # time in sex during night
    for i in tqdm(range(len(trial_lose_rise_time))):
        if len(trial_lose_rise_time[i]) == 0:
            continue

        rr_array = np.zeros(len(time_array))
        # e1 = trial_lose_rise_time[i][0]
        # rr_array[(time_array < e1)] = 1 / e1
        # for e0, e1 in zip(trial_lose_rise_time[i][:-1], trial_lose_rise_time[i][1:]):
        #     if e1 >= 3*60*60:
        #         break
        #     rr_array[(time_array >= e0) & (time_array < e1)] = 1/(e1 - e0)
        # e0 = trial_lose_rise_time[i][-1]
        # rr_array[(time_array >= e0)] = 1/ (3*60*60 - e0)
        for e in trial_lose_rise_time[i]:
            rr_array += gauss(time_array, e, 600, 1/600, norm=True)
        rise_rates.append(rr_array / np.mean(rr_array))

    return  rise_rates, time_array


def main():
    win_rc = np.load('../win_rc.npy', allow_pickle=True)
    lose_rc = np.load('../lose_rc.npy', allow_pickle=True)

    win_rc_day = np.load('../win_rc_day.npy', allow_pickle=True)
    lose_rc_day = np.load('../lose_rc_day.npy', allow_pickle=True)

    # win_rc30 = np.load('../win_rc30.npy', allow_pickle=True)
    # lose_rc30 = np.load('../lose_rc30.npy', allow_pickle=True)
    #
    # win_rc15 = np.load('../win_rc15.npy', allow_pickle=True)
    # lose_rc15 = np.load('../lose_rc15.npy', allow_pickle=True)

    size_win = np.load('../size_win.npy', allow_pickle=True)
    dsize_win = np.load('../dsize_win.npy', allow_pickle=True)
    df_win = np.load('../df_win.npy', allow_pickle=True)
    lose_exp = np.load('../lose_exp.npy', allow_pickle=True)
    win_exp = np.load('../win_exp.npy', allow_pickle=True)

    win_rc_1_30 = np.load('../win_rc_1_30.npy', allow_pickle=True)
    lose_rc_1_30 = np.load('../lose_rc_1_30.npy', allow_pickle=True)

    win_rc_15b = np.load('../win_rc_15b.npy', allow_pickle=True)
    lose_rc_15b = np.load('../lose_rc_15b.npy', allow_pickle=True)

    trial_lose_rise_time = np.load('../trial_lose_rise_times2.npy', allow_pickle=True)
    rise_rates = np.load('../rise_rates.npy', allow_pickle=True)
    time_array = np.load('../rise_rate_time_array.npy', allow_pickle=True)

    lose_win_hist = np.load('../lose_win_hist.npy', allow_pickle=True)
    lose_lose_hist = np.load('../lose_lose_hist.npy', allow_pickle=True)
    lose_lose = np.load('../lose_lose.npy', allow_pickle=True)

    ##########################
    fig, ax = plt.subplots()

    female_color, male_color = '#e74c3c', '#3498db'
    win_color = [male_color, female_color, male_color, female_color]
    lose_color = [male_color, female_color, female_color, male_color]
    mek = ['k', 'k', None, None]
    for i in range(4):
        ax.plot(np.zeros(len(lose_rc[i]))[np.array(lose_lose[i]) == 0] +
                (np.random.rand(len(np.array(lose_rc[i])[np.array(lose_lose[i]) == 0])) - 0.5) * 0.2,
                np.array(lose_rc[i])[np.array(lose_lose[i]) == 0], 'o', color=lose_color[i], markeredgecolor=mek[i], markersize=5)

        ax.plot(np.ones(len(lose_rc[i]))[np.array(lose_lose[i]) == 1] +
                (np.random.rand(len(np.array(lose_rc[i])[np.array(lose_lose[i]) == 1])) - 0.5) * 0.2,
                np.array(lose_rc[i])[np.array(lose_lose[i]) == 1], 'o', color=lose_color[i], markeredgecolor=mek[i], markersize=5)

    rc_all = np.hstack(lose_rc)
    ll_all = np.hstack(lose_lose)
    ax.boxplot([rc_all[ll_all == 0], rc_all[ll_all == 1]], positions = [0, 1])

    U, p = non_nan_mannwhitneyU(rc_all[ll_all == 0], rc_all[ll_all == 1])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['won\nlast', 'lost\nlast'])
    ax.set_ylabel('rises [n]', fontsize=12)
    print('rises whether or not last won: U=%.1f, p=%.3f' % (U,p))
    plt.show()
    # embed()
    # quit()
    ########################################################################################################

    dr_win_all = []
    for i in range(30):
        win_all = win_rc_1_30[0][i] + win_rc_1_30[1][i] + win_rc_1_30[2][i] + win_rc_1_30[3][i]
        lose_all = lose_rc_1_30[0][i] + lose_rc_1_30[1][i] + lose_rc_1_30[2][i] + lose_rc_1_30[3][i]

        dr_win_all.append(np.array(win_all) - np.array(lose_all))

    reliability = []
    for Cwin_rc in dr_win_all:
        lose_vals = Cwin_rc
        win_vals = Cwin_rc * -1
        true_neg = []
        false_neg = []
        for th in np.sort(list(win_vals) + list(lose_vals)):
            true_neg.append(len(lose_vals[lose_vals <= th]) / len(lose_vals))
            false_neg.append(len(win_vals[win_vals <= th]) / len(win_vals))

        true_neg = np.array(true_neg)
        false_neg = np.array(false_neg)

        AUC_values = []
        for i in np.unique(false_neg):
            AUC_values.append(np.max(true_neg[false_neg == i]))
        reliability.append(np.sum(AUC_values) / len(AUC_values))


    female_color, male_color = '#e74c3c', '#3498db'

    # embed()
    # quit()

    fig = plt.figure(figsize=(17.5/2.54, 12/2.54))
    gs = gridspec.GridSpec(2, 2, bottom=0.125, left=0.1, right=0.95, top=0.875, hspace=0.5, wspace=0.3)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[0, 1], sharey=ax[0]))
    ax.append(fig.add_subplot(gs[1, 0]))
    ax.append(fig.add_subplot(gs[1, 1], sharey=ax[2]))


    plot_rc(*lose_rc, ax[0])
    #significnace(ax[0], 0.5, 2.5, 450, 0.0147, whisker_fac=0.015)
    ax[0].plot([0.5, 2.5], [530, 530], lw=1, color='k')
    ax[0].plot([0.5, 0.5], [530, 490], lw=1, color='k')
    ax[0].plot([2.5, 2.5], [530, 490], lw=1, color='k')

    ax[0].plot([0, 1], [490, 490], lw=1, color='k')
    ax[0].plot([2, 3], [490, 490], lw=1, color='k')

    ax[0].text(1.5, 520, '*', fontsize=10, ha='center', va='bottom')

    ax[0].text(2.75, 480, '*', fontsize=10, ha='center', va='bottom')
    ax[0].text(0.25, 490, 'n.s.', fontsize=8, ha='center', va='bottom')


    # significance_bar(ax[1], 0.0147, 0.5, 2.5, 500)
    ax[0].set_ylim(top=600)

    plot_df_rc(*lose_rc, *df_win, ax[1])
    ax[1].text(ax[1].get_xlim()[1], 625, r'$r=0.32, p=0.049$', fontsize=9, ha='right', va='center')

    plot_dsize_rc(*lose_rc, *dsize_win, ax[2])
    ax[2].text(ax[2].get_xlim()[0], 470, r'$r=0.74, p<0.001$', fontsize=9, ha='left', va='center', color=male_color)
    ax[2].text(ax[2].get_xlim()[1], 470, r'$r=-0.75, p<0.001$', fontsize=9, ha='right', va='center', color=female_color)

    plot_exp_rc(*lose_rc, *win_exp, ax[3])
    ax[3].text(ax[3].get_xlim()[1], 470, r'$r=-0.39, p=0.018$', fontsize=9, ha='right', va='center')

    ax[0].set_ylabel('loser rises', fontsize=12)
    ax[2].set_ylabel('loser rises', fontsize=12)
    plt.setp(ax[1].get_yticklabels(), visible=False)
    plt.setp(ax[3].get_yticklabels(), visible=False)
    for a in ax:
        a.tick_params(labelsize=10)

    fig.tag(axes=[ax[0], ax[2]], labels=['A', 'C'], fontsize=15, yoffs=1.5, xoffs=-8)
    fig.tag(axes=[ax[1], ax[3]], labels=['B', 'D'], fontsize=15, yoffs=1.5, xoffs=-4)
    #fig.tag(axes=ax, fontsize=15, yoffs=1.5, xoffs=-4)

    female_color, male_color = '#e74c3c', '#3498db'
    x0, x1 = ax[0].get_xlim()
    ax[0].plot(-50, 2, 'p', color=male_color, label=u'\u2642' + ' Winner')
    ax[0].plot(-50, 2, 'p', color=female_color, label=u'\u2640' + ' Winner')
    ax[0].plot(-50, 2, 's', color='lightgrey', markeredgecolor='k', label='same-sex')
    ax[0].plot(-50, 2, 's', color='lightgrey', label='mixed-sex')
    ax[0].legend(loc='upper right', frameon=False, bbox_to_anchor=(1.65, 1.5), ncol=2, fontsize=9)
    ax[0].set_xlim(x0, x1)

    # plt.savefig('../../figures/EODf_rise_dependencies.pdf')
    # plt.savefig('EODf_rise_dependencies.pdf')
    #
    plt.show()
    #quit()

    ###################################################################################################################

    fig = plt.figure(figsize=(17.5/2.54, 14/2.54))
    gs = gridspec.GridSpec(3, 2, bottom=0.1, left=0.15, right=0.95, top=0.925, hspace=0.7, wspace=0.6, height_ratios=[1.5, 2, 2])
    ax = []
    ax.append(fig.add_subplot(gs[0, :]))
    ax.append(fig.add_subplot(gs[1, 0]))
    ax.append(fig.add_subplot(gs[2, 0]))
    ax.append(fig.add_subplot(gs[1, 1]))
    ax.append(fig.add_subplot(gs[2, 1]))


    win_color = [male_color, female_color, male_color, female_color]
    lose_color = [male_color, female_color, female_color, male_color]
    mek = ['k', 'k', None, None]
    for i in range(4):
        ax[0].plot(win_rc_day[i], np.ones(len(win_rc_day[i]))* 4 + (np.random.rand(len(win_rc_day[i])) - 0.5) * 0.5, 'p', color=win_color[i], zorder=2, markersize=5)
        ax[0].plot(lose_rc_day[i], np.ones(len(lose_rc_day[i]))* 3 + (np.random.rand(len(lose_rc_day[i])) - 0.5) * 0.5, 'o', color=lose_color[i], zorder=2, markersize=5)
        ax[0].plot(win_rc[i], np.ones(len(win_rc[i]))* 2 + (np.random.rand(len(win_rc[i])) - 0.5) * 0.5, 'p', color=win_color[i], zorder=2, markersize=5)
        ax[0].plot(lose_rc[i], np.ones(len(lose_rc[i]))* 1 + (np.random.rand(len(lose_rc[i])) - 0.5) * 0.5, 'o', markeredgecolor=mek[i], color=lose_color[i], zorder=2, markersize=5)
    bp = ax[0].boxplot([np.hstack(lose_rc), np.hstack(win_rc), np.hstack(lose_rc_day), np.hstack(win_rc_day)], sym='', widths=0.75, zorder=3, patch_artist=True, vert=False)
    plt.setp(bp['medians'], color='k')
    plt.setp(bp['boxes'], facecolor='none')

    #ax[0].fill_between([2.5, 4.5], [-50, -50], [470, 470], color='grey', alpha=0.5, zorder=1)
    ax[0].fill_between([-50, 470], [2.5, 2.5], [0.5, 0.5], color='grey', alpha=0.5, zorder=1)
    #ax[0].fill_between([0.5, 2.5], [-50, -50], [470, 470], color='grey', alpha=0.5, zorder=1)
    ax[0].set_xlim(-20, 470)
    ax[0].set_xlabel('rises', fontsize=12)

    ax[0].set_yticklabels(['Lose', 'Win', 'Lose', 'Win'])

    plot_drc(*win_rc, *lose_rc, ax[1])
    ax[1].set_xlabel(r'$\Delta$rises', fontsize=12)
    #plot_drc(*win_rc30, *lose_rc30, ax[2])
    #ax[2].set_xlabel(r'$\Delta$EODf rises [# / 30min]', fontsize=12)

    ax[1].text(ax[1].get_xlim()[1], 1.3, r'$t=9.5, p<0.001$', fontsize=9, ha='right', va='center')
    #ax[2].text(ax[2].get_xlim()[1], 1.3, 't=6.7, p<0.001', fontsize=9, ha='right', va='center')

    ax[3].plot(np.arange(len(reliability))+1, reliability, lw=2, color='k', clip_on=False)
    ax[3].plot([0, 30], [.94, .94], lw=1, color='k', linestyle='dotted')
    ax[3].text(1, .94, r'$94\,\%$', ha='left', va='top', fontsize=10)
    ax[3].set_ylim(0.5, 1)
    ax[3].set_yticks([0.5, 1])
    ax[3].set_yticklabels(['50%', '100%'])

    ax[3].set_xlim(0, 30)
    ax[3].set_ylabel('AUC', fontsize=12)
    ax[3].set_xlabel('time [min]', fontsize=12)


    plot_cont_rise_ratio_losers(win_rc_15b, lose_rc_15b, ax[4])
    ax[4].set_xlim(0, 180)
    ax[4].set_xticks([0, 60, 120, 180])
    ax[4].set_xticklabels([0, 1, 2, 3])
    ax[4].set_ylim(0, 1)
    ax[4].set_yticks([0, 0.5, 1])
    ax[4].set_yticklabels(['0%', '50%', '100%'])
    ax[4].set_ylabel('frac. loser rises', fontsize=12)
    ax[4].set_xlabel('time [h]', fontsize=12)

    ############################################################################################################

    #rise_rates, time_array = get_rise_rates(trial_lose_rise_time)
    #np.save('../rise_rates.npy', rise_rates)
    #np.save('../rise_rate_time_array.npy', time_array)

    # fig, ax = plt.subplots()
    f_rise_rates = []
    for i in range(len(lose_rc_15b)):
        rc = np.array(list(lose_rc_15b[i])).T
        for Crc in rc:
            f_rise_rates.append(Crc / np.mean(Crc))

    # fig, a = plt.subplots()
    # ax = [None, None, a]
    for id in range(len(f_rise_rates)):
        for i in range(len(f_rise_rates[id])):
            ax[2].plot([i * 15, (i+1) * 15], [f_rise_rates[i], f_rise_rates[i]], color='grey', lw=.5)
            if i != len(f_rise_rates[id])-1:
                ax[2].plot([(i+1) * 15, (i+1) * 15], [f_rise_rates[i], f_rise_rates[i+1]], color='grey', lw=.5)

    mean_rise_rates = np.mean(f_rise_rates, axis=0)
    for i in range(len(mean_rise_rates)):
        ax[2].plot([i * 15, (i + 1) * 15], [mean_rise_rates[i], mean_rise_rates[i]], color='k', lw=2)
        if i != len(mean_rise_rates)-1:
            ax[2].plot([(i + 1) * 15, (i + 1) * 15], [mean_rise_rates[i], mean_rise_rates[i+1]], color='k', lw=2)

    ax[2].plot([0, 3*60], [1, 1], linestyle='dotted', lw=1, color='k')
    ax[2].set_xticks([0, 60, 120, 180])
    ax[2].set_xticklabels([0, 1, 2, 3])
    ax[2].set_xlim(0, 180)
    ax[2].set_xlabel('time [h]', fontsize=12)
    #ax[2].set_ylabel('loser\nrise rate\n'+r'[$\mu_{rise\:rate}$]', fontsize=12)
    ax[2].set_ylabel('norm. loser rise rate', fontsize=12)
    ax[2].set_yticks([0, 1, 2, 3, 4])
    ax[2].set_ylim(bottom=0)

    fig.tag(axes=ax[:3], labels= ['A', 'B', 'D'], fontsize=15, yoffs=1, xoffs=-10)
    fig.tag(axes=ax[3:], labels= ['C', 'E'], fontsize=15, yoffs=1, xoffs=-11.5)
    #fig.tag(axes=ax[2:], labels=['C', 'D'], fontsize=15, yoffs=1.5, xoffs=-9)
    #fig.tag(axes=ax[:-1], fontsize=15, yoffs=1.5, xoffs=-9)
    #fig.tag(axes=ax[-1], fontsize=15, yoffs=1.5, xoffs=-2)

    #fig.align_ylabels([ax[2], ax[3]])

    female_color, male_color = '#e74c3c', '#3498db'
    ax[0].plot(-100, 2, 'p', markeredgecolor='k', color='k', label='Winner')
    ax[0].plot(-100, 2, 'o', color='grey', label='Loser')
    ax[0].plot(-100, 2, 's', color=male_color, label=u'\u2642')
    ax[0].plot(-100, 2, 's', color=female_color, label=u'\u2640')
    ax[0].plot(-100, 2, 's', color='lightgrey', markeredgecolor='k', label='same-sex')
    ax[0].plot(-100, 2, 's', color='lightgrey', label='mixed-sex')
    ax[0].legend(loc='upper center', frameon=False, bbox_to_anchor=(.5, 1.55), ncol=3, fontsize=9)


    # plt.savefig('../../figures/EODf_rise_n_d3h_auc_ratio2.pdf')
    # plt.savefig('EODf_rise_n_d3h_auc_ratio2.pdf')

    # embed()
    # quit()
    plt.show()

    ########################################################################################################

    fig = plt.figure(figsize=(9/2.54, 5/2.54))
    gs = gridspec.GridSpec(1, 1, left=0.2, bottom=0.3, right=0.9, top=0.9)
    ax = fig.add_subplot(gs[0, 0])

    plot_dsize_rc(*lose_rc, *dsize_win, ax)
    ax.set_xticks([-10, -5, 0, 5])
    ax.set_xlim(-10, 5)
    ax.text(ax.get_xlim()[0], 470, r'$r=0.74, p<0.001$', fontsize=9, ha='left', va='center', color=male_color)
    ax.text(ax.get_xlim()[1], 470, r'$r=-0.75, p<0.001$', fontsize=9, ha='right', va='center', color=female_color)
    ax.set_ylabel('rises', fontsize=12)

    # plt.savefig('poster_size_rise.pdf')
    plt.show()

if __name__ == '__main__':
    main()