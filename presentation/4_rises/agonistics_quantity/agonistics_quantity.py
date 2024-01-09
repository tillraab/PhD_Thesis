import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as scp
from IPython import embed
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit
from plottools.tag import tag
# fig.tag(axes=[ax[1], ax[2]], fontsize=15, yoffs=2, xoffs=-3)

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

def plot_df_rc(mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc,
               mm_win_df, ff_win_df, mf_win_df, fm_win_df, mix_win_df, ax):

    female_color, male_color= '#e74c3c', '#3498db'
    win_color = [male_color, female_color, male_color, female_color]
    lose_color = [male_color, female_color, female_color, male_color]
    mek= ['k', 'k', None, None]

    for enu, Lrc, Ldf in zip(np.arange(4), [mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc],
                             [np.array(mm_win_df)*-1, np.array(ff_win_df)*-1, np.array(mf_win_df)*-1, np.array(fm_win_df)*-1]):
        ax.plot(Ldf, Lrc, 'o', color=lose_color[enu], markeredgecolor=mek[enu], alpha=0.8, zorder=1)

    xx = np.hstack([np.array(mm_win_df)*-1, np.array(ff_win_df)*-1, np.array(mf_win_df)*-1, np.array(fm_win_df)*-1])
    yy = np.hstack([mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc])

    XX = np.array([np.min(xx), np.max(xx)])
    m, b, _, _, _ = scp.linregress(xx, yy)
    #print( non_nan_pearsonr(xx, yy))
    ax.plot(XX, m*XX+b, color='k', lw=2)

    ax.set_xlabel(r'$\Delta$EODf [Hz]', fontsize=12)
    ax.tick_params(labelsize=10)

    r, p = non_nan_pearsonr(xx, yy)
    print('\n dEODf dependente XXX')
    print('r=%.3f, p=%.4f' % (r, p))
    print('')


def plot_count(mm_lose_c, ff_lose_c, mf_lose_c, fm_lose_c, ax):

    female_color, male_color= '#e74c3c', '#3498db'
    lose_color = [male_color, female_color, female_color, male_color]
    win_color = [male_color, male_color, female_color, female_color]
    mek= ['k', None, 'k', None]

    for enu, rc in enumerate([mm_lose_c, mf_lose_c, ff_lose_c, fm_lose_c]):
        ax.plot(np.ones(len(rc))* enu + (np.random.rand(len(rc)) - 0.5) * 0.25, rc, 'p', color=win_color[enu], markeredgecolor=mek[enu], alpha=0.8)

    bp = ax.boxplot([mm_lose_c[~np.isnan(mm_lose_c)], mf_lose_c[~np.isnan(mf_lose_c)],
                ff_lose_c[~np.isnan(ff_lose_c)], fm_lose_c[~np.isnan(fm_lose_c)]], positions=np.arange(4), sym='', patch_artist=True)
    plt.setp(bp['medians'], color='k')
    plt.setp(bp['boxes'], facecolor='none')


    ax.set_xticks(np.arange(4))
    labels = [u'\u2642\u2642', u'\u2642\u2640', u'\u2640\u2640', u'\u2640\u2642']
    ax.set_xticklabels(labels)

    ax.set_xlabel('pairing', fontsize=12)
    ax.tick_params(labelsize=10)

    print('\nXXX count (pairing)')
    u, p = scp.mannwhitneyu(mm_lose_c, mf_lose_c)
    print('mm vs. mf: U= %.1f, p=%.4f' % (u, p))
    u, p = scp.mannwhitneyu(ff_lose_c, fm_lose_c)
    print('ff vs. fm: U= %.1f, p=%.4f' % (u, p))
    u, p = scp.mannwhitneyu(list(mm_lose_c) + list(mf_lose_c), list(ff_lose_c) + list(fm_lose_c))
    print('m-win vs f-win: U= %.1f, p=%.4f' % (u, p))
    u, p = scp.mannwhitneyu(list(mm_lose_c) + list(fm_lose_c), list(ff_lose_c) + list(mf_lose_c))
    print('m-lose vs f-lose: U= %.1f, p=%.4f' % (u, p))
    print('')

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
    x, y = np.hstack([mm_win_dsize, mf_win_dsize])*-1, np.hstack([mm_lose_rc, mf_lose_rc])
    x, y = x[~np.isnan(y)], y[~np.isnan(y)]
    m, b, _, _, _ = scp.linregress(x, y)
    XX = np.array([np.min(x_vals), np.max(x_vals)])
    ax.plot(XX, m*XX+b, color=male_color, lw=2)

    x, y = np.hstack([ff_win_dsize, fm_win_dsize])*-1, np.hstack([ff_lose_rc, fm_lose_rc])
    x, y = x[~np.isnan(y)], y[~np.isnan(y)]
    m, b, _, _, _ = scp.linregress(x, y)
    XX = np.array([np.min(x_vals), np.max(x_vals)])
    ax.plot(XX, m*XX+b, color=female_color, lw=2)

    ax.set_xlabel(r'$\Delta$size [cm]', fontsize=12)
    ax.tick_params(labelsize=10)

    print('dSize dependent XXX')
    r, p = non_nan_pearsonr(np.hstack([mm_win_dsize, mf_win_dsize])*-1, np.hstack([mm_lose_rc, mf_lose_rc]))
    print('m-win: r=%.3f, p=%.4f' % (r, p))
    r, p = non_nan_pearsonr(np.hstack([ff_win_dsize, fm_win_dsize])*-1, np.hstack([ff_lose_rc, fm_lose_rc]))
    print('f-win: r=%.3f, p=%.4f' % (r, p))
    r, p = non_nan_pearsonr(np.hstack([mm_win_dsize, fm_win_dsize])*-1, np.hstack([mm_lose_rc, fm_lose_rc]))
    print('m-lose: r=%.3f, p=%.4f' % (r, p))
    r, p = non_nan_pearsonr(np.hstack([ff_win_dsize, mf_win_dsize])*-1, np.hstack([ff_lose_rc, mf_lose_rc]))
    print('f-lose: r=%.3f, p=%.4f' % (r, p))
    r, p = non_nan_pearsonr(np.hstack([mm_win_dsize, ff_win_dsize, mf_win_dsize, fm_win_dsize]) * -1,
                            np.hstack([mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc]))
    print('all: r=%.3f, p=%.4f' % (r, p))
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
        exp_c = all_rc[all_exp == e]
        rc_per_exp.append(exp_c[~np.isnan(exp_c)])

    bp = ax.boxplot(rc_per_exp, sym = '', patch_artist=True)
    plt.setp(bp['medians'], color='k')
    plt.setp(bp['boxes'], facecolor='none')

    ax.set_xlabel('loser experience [trial]', fontsize=12)
    # ax.set_ylabel('EODf rises [n]', fontsize=12)
    ax.tick_params(labelsize=10)

    r, p = non_nan_pearsonr(np.hstack([mm_lose_exp, ff_lose_exp, mf_lose_exp, fm_lose_exp]),
                            np.hstack([mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc]))
    print('Experience dependent XXX')
    print('r=%.3f, p=%.4f' % (r, p))
    print('')

def plot_agon_rc(mm_agon, ff_agon, mf_agon, fm_agon, mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc, ax):

    female_color, male_color= '#e74c3c', '#3498db'
    win_color = [male_color, female_color, male_color, female_color]
    lose_color = [male_color, female_color, female_color, male_color]

    mek= ['k', 'k', None, None]

    for enu, rc, agons in zip(np.arange(4), [mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc],
                            [mm_agon, ff_agon, mf_agon, fm_agon]):
        ax.plot(rc, agons, 'p', color=win_color[enu], markeredgecolor=mek[enu],
                alpha=0.8, zorder=1)

    ax.set_xlabel('EODf rises [n]', fontsize=12)

    x_vals = np.hstack([mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc])
    x, y = np.hstack([mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc]), np.hstack([mm_agon, ff_agon, mf_agon, fm_agon])
    x, y = x[~np.isnan(y)], y[~np.isnan(y)]
    m, b, _, _, _ = scp.linregress(x, y)
    XX = np.array([np.min(x_vals), np.max(x_vals)])
    ax.plot(XX, m*XX+b, color='k', lw=2)


    r, p = non_nan_pearsonr(np.hstack([mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc]),
                            np.hstack([mm_agon, ff_agon, mf_agon, fm_agon]))
    print('rise count dependent XXX')
    print('r=%.3f, p=%.4f' % (r, p))
    print('')

def plot_agon_contact(mm_agon, ff_agon, mf_agon, fm_agon, mm_contact, ff_contact, mf_contact, fm_contact, ax):
    female_color, male_color= '#e74c3c', '#3498db'
    win_color = [male_color, female_color, male_color, female_color]
    lose_color = [male_color, female_color, female_color, male_color]

    mek= ['k', 'k', None, None]

    for enu, contact, agons in zip(np.arange(4), [mm_contact, ff_contact, mf_contact, fm_contact],
                            [mm_agon, ff_agon, mf_agon, fm_agon]):
        ax.plot(contact, agons, 'o', color=lose_color[enu], markeredgecolor=mek[enu],
                alpha=0.8, zorder=1)

    x_vals = np.hstack([mm_contact, ff_contact, mf_contact, fm_contact])
    x, y = np.hstack([mm_contact, ff_contact, mf_contact, fm_contact]), np.hstack([mm_agon, ff_agon, mf_agon, fm_agon])
    x, y = x[~np.isnan(y)], y[~np.isnan(y)]
    m, b, _, _, _ = scp.linregress(x, y)
    XX = np.array([np.nanmin(x_vals), np.nanmax(x_vals)])
    ax.plot(XX, m*XX+b, color='k', lw=2)

    r, p = non_nan_pearsonr(np.hstack([mm_contact, ff_contact, mf_contact, fm_contact]),
                            np.hstack([mm_agon, ff_agon, mf_agon, fm_agon]))
    print('agon dep. XXX')
    print('r=%.3f, p=%.4f' % (r, p))
    print('')

def hist_duration(mm_agon, ff_agon, mf_agon, fm_agon, med_mm_agon, med_ff_agon, med_mf_agon, med_fm_agon, ax0, ax1):
    all_dur = np.hstack(np.hstack([mm_agon, ff_agon, mf_agon, fm_agon]))
    bins = np.linspace(0, np.percentile(all_dur, 99), 50)
    n, bin_edge = np.histogram(all_dur, bins)
    n = n / np.sum(n) / (bin_edge[1] - bin_edge[0])

    female_color, male_color= '#e74c3c', '#3498db'

    dur_array = np.arange(0, np.max(all_dur), 0.1)
    mm_dur_conv, ff_dur_conv, mf_dur_conv, fm_dur_conv = np.zeros((4, len(dur_array)))
    for e in np.hstack(mm_agon):
        mm_dur_conv += gauss(dur_array, e, 2, 0.5, norm=True)
    for e in np.hstack(ff_agon):
        ff_dur_conv += gauss(dur_array, e, 2, 0.5, norm=True)
    for e in np.hstack(mf_agon):
        mf_dur_conv += gauss(dur_array, e, 2, 0.5, norm=True)
    for e in np.hstack(fm_agon):
        fm_dur_conv += gauss(dur_array, e, 2, 0.5, norm=True)

    mm_dur_conv /= len(np.hstack(mm_agon)) * 0.1
    ff_dur_conv /= len(np.hstack(ff_agon)) * 0.1
    mf_dur_conv /= len(np.hstack(mf_agon)) * 0.1
    fm_dur_conv /= len(np.hstack(fm_agon)) * 0.1

    # mm_dur_conv *= len(mm_agon)
    # mf_dur_conv *= len(mf_agon)
    # ff_dur_conv *= len(ff_agon)
    # fm_dur_conv *= len(fm_agon)

    ax0.plot(dur_array, mm_dur_conv, color=male_color, lw=2, label=u'\u2642\u2642')
    ax0.plot(dur_array, mf_dur_conv, '--', color=male_color, lw=2, label=u'\u2642\u2640')
    ax0.plot(dur_array, ff_dur_conv, color=female_color, lw=2, label=u'\u2640\u2640')
    ax0.plot(dur_array, fm_dur_conv, '--', color=female_color, lw=2, label=u'\u2640\u2642')
    ax0.legend(loc=1, frameon=False, fontsize=9)

    #ax0.bar(bin_edge[:-1] + (bin_edge[1] - bin_edge[0]) / 2, n, width=(bin_edge[1] - bin_edge[0])*0.75, color='grey')

    win_color = [male_color, male_color, female_color, female_color]
    lose_color = [male_color, female_color, female_color, male_color]
    mek= ['k', None, 'k', None]

    for enu, dur in enumerate([med_mm_agon, med_mf_agon, med_ff_agon, med_fm_agon]):
        ax1.plot(dur, np.ones(len(dur)) * (np.random.rand(len(dur)) - 0.5) * 0.25 + enu +1 , 'p', color=win_color[enu], markeredgecolor=mek[enu])

    bp = ax1.boxplot([med_mm_agon[~np.isnan(med_mm_agon)], med_mf_agon[~np.isnan(med_mf_agon)],
                      med_ff_agon[~np.isnan(med_ff_agon)], med_fm_agon[~np.isnan(med_fm_agon)]], widths=0.5, sym='', vert=False, patch_artist=True)

    plt.setp(bp['medians'], color='k')
    plt.setp(bp['boxes'], facecolor='none')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_visible(False)

    ax1.set_yticks([1, 2, 3, 4])
    ax1.set_yticklabels([u'\u2642\u2642', u'\u2642\u2640', u'\u2640\u2640', u'\u2640\u2642'])
    ax1.invert_yaxis()

    print('male-male dur vs rest')
    med_mm_agon = med_mm_agon[~np.isnan(med_mm_agon)]
    med_ff_agon = med_ff_agon[~np.isnan(med_ff_agon)]
    med_mf_agon = med_mf_agon[~np.isnan(med_mf_agon)]
    med_fm_agon = med_fm_agon[~np.isnan(med_fm_agon)]
    u, p = scp.mannwhitneyu(med_mm_agon, list(med_ff_agon)+ list(med_mf_agon)+list(med_fm_agon))
    print('U=%.1f, p=%.4f' % (u, p))
    # ax1.set_axis_off()

def main():
    win_rc = np.load('../win_rc.npy', allow_pickle=True)
    lose_rc = np.load('../lose_rc.npy', allow_pickle=True)

    agonistics = np.load('../agonistics.npy', allow_pickle=True)
    contact = np.load('../contact.npy', allow_pickle=True)
    agonistic_dur = np.load('../agonistic_dur.npy', allow_pickle=True)


    dsize_win = np.load('../dsize_win.npy', allow_pickle=True)
    df_win = np.load('../df_win.npy', allow_pickle=True)
    lose_exp = np.load('../lose_exp.npy', allow_pickle=True)
    win_exp = np.load('../win_exp.npy', allow_pickle=True)
    #embed()
    #quit()

    female_color, male_color = '#e74c3c', '#3498db'


    med_agonstic_dur = []
    for i in range(len(agonistic_dur)):
        med_agonstic_dur.append([])
        for j in range(len(agonistic_dur[i])):
            med_agonstic_dur[-1].append(np.nanmedian(agonistic_dur[i][j]))
        med_agonstic_dur[-1] = np.array(med_agonstic_dur[-1])
    med_agonstic_dur = np.array(med_agonstic_dur)


    # fig = plt.figure(figsize=(20/2.54, 10/2.54))
    # gs = gridspec.GridSpec(2, 2, bottom=0.15, left=0.1, right=0.95, top=0.95, wspace=0.4)
    # ax = []
    # ax.append(fig.add_subplot(gs[0, 0]))
    # ax.append(fig.add_subplot(gs[1, 0]))
    # ax.append(fig.add_subplot(gs[:, 1]))
    #
    # print('agon - cat')
    # plot_count(*agonistics, ax[0])
    # print('contact - cat')
    # plot_count(*contact, ax[1])
    # print('agon - contact')
    # plot_agon_contact(*agonistics, *contact, ax[2])
    # for a in ax:
    #     a.tick_params(labelsize=10)
    #
    # ax[0].set_xlabel('')
    # plt.setp(ax[0].get_xticklabels(), visible=False)
    # ax[0].set_ylabel('agonistics', fontsize=12)
    # ax[1].set_ylabel('contacts', fontsize=12)
    # ax[2].set_xlabel('agonistics', fontsize=12)
    # ax[2].set_ylabel('contacts', fontsize=12)
    #
    # plt.savefig('../../figures/agonistics_count.pdf')
    # plt.savefig('agonistics_count.pdf')

    ###

    # fig = plt.figure(figsize=(17.5/2.54, 11.5/2.54))
    # gs = gridspec.GridSpec(2, 2, bottom=0.11, left=0.125, right=0.95, top=0.95, hspace=0.5, wspace=0.3)
    # ax = []
    # ax.append(fig.add_subplot(gs[0, 0]))
    # ax.append(fig.add_subplot(gs[0, 1]))
    # ax.append(fig.add_subplot(gs[1, 0], sharey=ax[1]))
    # ax.append(fig.add_subplot(gs[1, 1], sharey=ax[2]))

    # embed()
    # quit()
    fig = plt.figure(figsize=(17.5/2.54, 12/2.54))
    gs = gridspec.GridSpec(2, 2, bottom=0.125, left=0.125, right=0.95, top=0.875, hspace=0.5, wspace=0.3)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[0, 1]))
    ax.append(fig.add_subplot(gs[1, 0], sharey=ax[1]))
    ax.append(fig.add_subplot(gs[1, 1], sharey=ax[2]))

    print('med dur - cat')
    plot_count(*med_agonstic_dur, ax[0])
    ax[0].set_ylim(1, 13)

    ax[0].plot([0, 1], [10, 10], lw=1, color='k')
    ax[0].text(0.5, 9.75, '*', fontsize=10, ha='center', va='bottom')

    ax[0].plot([0, 3], [11.5, 11.5], lw=1, color='k')
    ax[0].text(1.5, 11.25, '*', fontsize=10, ha='center', va='bottom')

    #ax[0].set_ylim(top=13)
    # significnace(ax[0], 0, 1, 10, 0.013, whisker_fac=0.025)
    # significnace(ax[0], 0, 3, 11.35, 0.013, whisker_fac=0.025)



    print('med dur - loser rises')
    plot_agon_rc(*med_agonstic_dur, *lose_rc, ax[1])
    ax[1].text(ax[1].get_xlim()[1], 9.9, 'r=0.64, p=0.003', fontsize=9, color='k', ha='right', va='center')
    print('med dur - winner size')
    plot_dsize_rc(*med_agonstic_dur, *dsize_win, ax[2])
    ax[2].text(ax[2].get_xlim()[0], 9.9, 'r=0.73, p=0.006', fontsize=9, ha='left', va='center', color=male_color)
    ax[2].text(ax[2].get_xlim()[1], 9.9, 'r=-0.65, p=0.084', fontsize=9, ha='right', va='center', color=female_color)
    print('med dur - exp')
    plot_exp_rc(*med_agonstic_dur, *lose_exp, ax[3])
    ax[3].set_xlim(0.5, 4.5)
    ax[3].text(ax[3].get_xlim()[1], 9.9, 'r=-0.50, p=0.029', fontsize=9, color='k', ha='right', va='center')

    ax[0].set_ylabel('chasing\nduration [s]', fontsize=12)
    ax[2].set_ylabel('chasing\nduration [s]', fontsize=12)
    fig.align_ylabels([ax[0], ax[2]])
    # ax[0].text(-1.1, -1, 'agonistic duration [s]', fontsize=12, ha='center', va='center', rotation=90)

    ax[1].set_ylim(1, 9.5)
    ax[0].set_yticks(np.arange(3, 12.1, 3))
    ax[1].set_yticks(np.arange(2, 8.1, 2))
    ax[3].set_yticks(np.arange(2, 8.1, 2))
    # plt.setp(ax[1].get_yticklabels(), visible=False)
    # plt.setp(ax[3].get_yticklabels(), visible=False)
    for a in ax:
        a.tick_params(labelsize=10)

    # fig.tag(axes=[ax[1], ax[3]], fontsize=15, xoffs=-5, labels=['B', 'D'])
    # fig.tag(axes=[ax[0], ax[2]], fontsize=15, xoffs=-10, labels=['A', 'C'])

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

    # plt.savefig('../../figures/agonistics_depend.pdf')
    # plt.savefig('agonistics_depend.pdf')

    ###########################################################################################################

    fig = plt.figure(figsize=(17.5/2.54, 9/2.54))
    # gs = gridspec.GridSpec(2, 2, bottom=0.15, left=0.1, right=0.95, top=0.95, wspace=0.4)
    gs = gridspec.GridSpec(2, 1, bottom=0.15, left=0.1, right=0.35, top=0.95)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[1, 0], sharex=ax[0]))
    # ax.append(fig.add_subplot(gs[:, 1]))

    gs = gridspec.GridSpec(2, 1, bottom=0.15, left=0.5, right=0.95, top=0.95, hspace=0, height_ratios=[2, 3])
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[1, 0], sharex=ax[2]))

    print('agon - cat')
    plot_count(*agonistics, ax[0])

    print('contact - cat')
    plot_count(*contact, ax[1])
    #plot_agon_contact(*agonistics, *contact, ax[2])
    print('agon_dur - cat')
    hist_duration(*agonistic_dur, *med_agonstic_dur, ax[3], ax[2])
    for a in ax:
        a.tick_params(labelsize=10)
    significnace(ax[2], 1, 2, 25, 0.013, vertical=True)
    significnace(ax[2], 1, 4, 27, 0.013, vertical=True)


    ax[0].set_xlabel('')
    plt.setp(ax[0].get_xticklabels(), visible=False)
    ax[0].set_ylim(bottom=0)
    ax[1].set_ylim(bottom=0)
    ax[0].set_ylabel('agonistics', fontsize=12)
    ax[1].set_ylabel('contacts', fontsize=12)

    ax[3].set_xlim([0, 30])
    ax[3].set_ylim(bottom=0)
    ax[3].set_yticks([0, 0.01])
    ax[3].set_xlabel('agonistic duration [s]', fontsize=12)
    ax[3].set_ylabel('PDF [1/s]', fontsize=12)

    fig.align_ylabels([ax[0], ax[1]])

    fig.tag(axes=[ax[0], ax[1], ax[2]], fontsize=15, labels=['A', 'B', 'C'])
    # plt.savefig('../../figures/agonistics_meta.pdf')
    # plt.savefig('agonistics_meta.pdf')

    #############################################################################################################

    fig = plt.figure(figsize=(17.5/2.54, 12.5/2.54))
    gs = gridspec.GridSpec(2, 1, bottom=0.5, left=0.1, right=0.35, top=0.9, hspace=0.3)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[1, 0], sharex=ax[0]))

    gs = gridspec.GridSpec(2, 1, bottom=0.5, left=0.5, right=0.95, top=0.9, hspace=0, height_ratios=[2, 3])
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[1, 0], sharex=ax[2]))

    gs = gridspec.GridSpec(1, 3, bottom=0.1, left=0.1, right=0.95, top=0.3, wspace=0.3)
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[0, 1], sharey=ax[4]))
    ax.append(fig.add_subplot(gs[0, 2], sharey=ax[4]))

    plot_count(*agonistics, ax[1])

    plot_count(*contact, ax[0])

    hist_duration(*agonistic_dur, *med_agonstic_dur, ax[3], ax[2])
    significnace(ax[2], 1, 2, 25, 0.013, vertical=True)
    significnace(ax[2], 1, 4, 27, 0.013, vertical=True)


    plot_dsize_rc(*med_agonstic_dur, *dsize_win, ax[4])
    ax[4].text(ax[4].get_xlim()[1], 9.9, r'$r=0.73, p=0.006$', fontsize=9, ha='right', va='center', color=male_color)
    ax[4].text(ax[4].get_xlim()[1], 10.9, r'$r=-0.65, p=0.084$', fontsize=9, ha='right', va='center', color=female_color)
    #ax[4].text(ax[4].get_xlim()[1], 9.9, 'r=-0.65, p=0.084', fontsize=9, ha='right', va='center', color=female_color)


    plot_exp_rc(*med_agonstic_dur, *lose_exp, ax[5])
    ax[5].set_xlim(0.5, 4.5)
    ax[5].text(ax[5].get_xlim()[1], 9.9, r'$r=-0.50, p=0.029$', fontsize=9, color='k', ha='right', va='center') # for lose
    #ax[5].text(ax[5].get_xlim()[1], 9.9, r'$r=-0.35, p=0.142$', fontsize=9, color='k', ha='right', va='center')

    plot_agon_rc(*med_agonstic_dur, *lose_rc, ax[6])
    ax[6].set_xlabel('loser rises', fontsize=12)
    ax[6].text(ax[6].get_xlim()[1], 9.9, r'$r=0.64, p=0.003$', fontsize=9, color='k', ha='right', va='center')

    ax[4].set_ylabel('duration [s]', fontsize=12)

    for a in ax:
        a.tick_params(labelsize=10)

    ax[0].set_xlabel('')
    plt.setp(ax[0].get_xticklabels(), visible=False)
    ax[1].set_ylim(bottom=0)
    ax[0].set_ylim(bottom=0)
    ax[1].set_ylabel('chasings', fontsize=12)
    ax[0].set_ylabel('contacts', fontsize=12)

    ax[3].set_xlim([0, 30])
    ax[3].set_ylim(bottom=0)
    ax[3].set_yticks([0, 0.1])
    ax[3].set_xlabel('chasing duration [s]', fontsize=12)
    ax[3].set_ylabel('PDF [1/s]', fontsize=12)

    fig.align_ylabels([ax[0], ax[1], ax[4]])

    plt.setp(ax[5].get_yticklabels(), visible=False)
    plt.setp(ax[6].get_yticklabels(), visible=False)

    fig.tag(axes=[ax[0], ax[1], ax[2]], fontsize=15, xoffs=-8, yoffs=1.5)
    fig.tag(axes=[ax[4]], labels=['D'], fontsize=15, xoffs=-8, yoffs=1.75)
    fig.tag(axes=[ax[5], ax[6]], labels=['E', 'F'], fontsize=15, xoffs=-3, yoffs=1.75)

    female_color, male_color = '#e74c3c', '#3498db'
    #x0, x1 = ax[0].get_xlim()
    ax[2].plot(-50, 2, 'p', color=male_color, label=u'\u2642' + ' Winner')
    ax[2].plot(-50, 2, 'p', color=female_color, label=u'\u2640' + ' Winner')
    ax[2].plot(-50, 2, 's', color='lightgrey', markeredgecolor='k', label='same-sex')
    ax[2].plot(-50, 2, 's', color='lightgrey', label='mixed-sex')
    ax[2].legend(loc='upper center', frameon=False, bbox_to_anchor=(.5, 1.7), ncol=2, fontsize=9)
    #ax[2].set_xlim(x0, x1)

    #plt.savefig('../../figures/agonistics_complete.pdf')
    #plt.savefig('agonistics_complete.pdf')

    r, p = non_nan_pearsonr(np.hstack(med_agonstic_dur), np.hstack(win_exp))

    r, p = non_nan_pearsonr(np.hstack(agonistics), np.hstack(lose_exp))
    r, p = non_nan_pearsonr(np.hstack(contact), np.hstack(lose_exp))


    # embed()
    # quit()

    ########################################################################################################################
    fig = plt.figure(figsize=(9/2.54, 5/2.54))
    gs = gridspec.GridSpec(1, 1, left=0.2, bottom=0.3, right=0.9, top=0.9)
    ax = fig.add_subplot(gs[0, 0])

    # plot_dsize_rc(*lose_rc, *dsize_win, ax)
    plot_dsize_rc(*med_agonstic_dur, *dsize_win, ax)
    ax.set_xticks([-10, -5, 0, 5])
    ax.set_xlim(-10, 5)
    ax.text(ax.get_xlim()[0], 9.9, r'$r=0.73, p=0.006$', fontsize=9, ha='left', va='center', color=male_color)
    ax.text(ax.get_xlim()[1], 9.9, r'$r=-0.65, p<0.084$', fontsize=9, ha='right', va='center', color=female_color)
    ax.set_ylabel('chasing\nduration [s]', fontsize=12)

    # plt.savefig('poster_size_duration.pdf')
    plt.show()

if __name__ == '__main__':
    main()