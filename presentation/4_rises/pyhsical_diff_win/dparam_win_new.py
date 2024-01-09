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


# def plot(df_win_mm, df_win_ff, df_win_mf, df_win_fm, df_win_mix,
#          dsize_win_mm, dsize_win_ff, dsize_win_mf, dsize_win_fm, dsize_win_mix,
#          size_win_mm, size_win_ff, size_win_mf, size_win_fm, size_win_mix,
#          EODf_win_mm, EODf_win_ff, EODf_win_mf, EODf_win_fm, EODf_win_mix):
#
#
#     alt_fig = plt.figure(figsize=(17.5/2.54, 14/2.54))
#
#     alt_df_ax=[]
#     # gs2 = gridspec.GridSpec(5, 2, left=0.1, bottom=0.1, right=0.475, top=0.9, height_ratios=[3, 3, 1, 2, 3], hspace=0.3)
#     gs2 = gridspec.GridSpec(5, 2, left=0.575, bottom=0.1, right=0.95, top=0.9, height_ratios=[3, 3, 1, 2, 2], hspace=0.3)
#     alt_df_ax.append(alt_fig.add_subplot(gs2[0, 0]))
#     alt_df_ax.append(alt_fig.add_subplot(gs2[0, 1], sharey=alt_df_ax[0]))
#     alt_df_ax.append(alt_fig.add_subplot(gs2[1, 0], sharey=alt_df_ax[0]))
#     alt_df_ax.append(alt_fig.add_subplot(gs2[1, 1], sharey=alt_df_ax[0]))
#
#     alt_ds_ax = []
#     # gs3 = gridspec.GridSpec(5, 2, left=0.575, bottom=0.1, right=0.95, top=0.9, height_ratios=[3, 3, 1, 2, 3], hspace=0.3)
#     gs3 = gridspec.GridSpec(5, 2, left=0.1, bottom=0.1, right=0.475, top=0.9, height_ratios=[3, 3, 1, 2, 2], hspace=0.3)
#     alt_ds_ax.append(alt_fig.add_subplot(gs3[0, 0], sharey=alt_df_ax[0]))
#     alt_ds_ax.append(alt_fig.add_subplot(gs3[0, 1], sharey=alt_df_ax[0]))
#     alt_ds_ax.append(alt_fig.add_subplot(gs3[1, 0], sharey=alt_df_ax[0]))
#     alt_ds_ax.append(alt_fig.add_subplot(gs3[1, 1], sharey=alt_df_ax[0]))
#
#     alt_ax = []
#     alt_ax.append(alt_fig.add_subplot(gs2[3:, :]))
#     alt_ax.append(alt_fig.add_subplot(gs3[3:, :]))
#
#
#     # fig = plt.figure(figsize=(20/2.54, 12/2.54))
#     # gs = gridspec.GridSpec(1, 2, left=0.1, bottom = 0.15, right=0.95, top=0.95)
#     # ax = []
#     # ax.append(fig.add_subplot(gs[0, 0]))
#     # ax.append(fig.add_subplot(gs[0, 1], sharey=ax[0]))
#
#     dsize_range = np.hstack([dsize_win_mm, np.array(dsize_win_mm) * -1, dsize_win_ff, np.array(dsize_win_ff) * -1,
#                              dsize_win_mf, np.array(dsize_win_mf) * -1, dsize_win_fm, np.array(dsize_win_fm) * -1])
#
#     df_range = np.hstack([df_win_mm, np.array(df_win_mm) * -1, df_win_ff, np.array(df_win_ff) * -1,
#                           df_win_mf, np.array(df_win_mf) * -1, df_win_fm, np.array(df_win_fm) * -1])
#
#     # female_color, male_color= '#ed665d', '#729ece'
#     female_color, male_color= '#e74c3c', '#3498db'
#     win_color = [male_color, female_color, male_color, female_color]
#     lose_color = [male_color, female_color, female_color, male_color]
#     mek= ['k', 'k', None, None]
#
#     for enu, d_param in enumerate([df_win_mm, df_win_ff, df_win_mf, df_win_fm]):
#
#         # ax[0].plot(d_param, np.ones(len(d_param)) - 0.12 + 0.08*enu, 'p', color=win_color[enu], markeredgecolor=mek[enu], alpha=0.8)
#         # ax[0].plot(np.array(d_param) * -1, np.zeros(len(d_param)) - 0.12 + 0.08*enu, 'o', color=lose_color[enu], markeredgecolor=mek[enu])
#
#         alt_ax[0].plot(d_param, np.ones(len(d_param)) - 0.12 + 0.08*enu, 'p', color=win_color[enu], markeredgecolor=mek[enu], alpha=0.8)
#         alt_ax[0].plot(np.array(d_param) * -1, np.zeros(len(d_param)) - 0.12 + 0.08*enu, 'o', color=lose_color[enu], markeredgecolor=mek[enu])
#
#         alt_df_ax[enu].plot(d_param, np.ones(len(d_param)) + (np.random.rand(len(d_param)) - 0.5) * 0.2, 'p', color=win_color[enu], markeredgecolor=mek[enu], alpha=0.8)
#         alt_df_ax[enu].plot(np.array(d_param) * -1, np.zeros(len(d_param)) + (np.random.rand(len(d_param)) - 0.5) * 0.2, 'o', color=lose_color[enu], markeredgecolor=mek[enu])
#
#         ###############################################################
#         df = pd.DataFrame()
#         fit_x = np.hstack([d_param, np.array(d_param) * -1])
#         fit_y = np.hstack([np.ones(len(d_param)), np.zeros(len(d_param))])
#
#         df['win'] = fit_y
#         df['d_param'] = fit_x
#         df = sm.add_constant(df)
#         try:
#             logit_model = sm.Logit(df['win'][:, np.newaxis], df[['const', 'd_param']]).fit()
#             fit_c = 'k'
#
#             max_d_param = np.max(np.abs(df_range))
#             test_df = pd.DataFrame()
#             test_df['const'] = np.ones(len(np.linspace(-max_d_param, max_d_param, 100)))
#             test_df['d_param'] = np.linspace(-max_d_param, max_d_param, 100)
#
#             pred = logit_model.predict(test_df)
#
#             x, y = test_df['d_param'], pred
#
#         except:
#             x = np.linspace(np.min(df_range), np.max(df_range), 100)
#             if enu == 2:
#                 y = boltzmann(x, .5, 0)
#             elif enu == 3:
#                 y = boltzmann(x, -.5, 0)
#
#         # ax[0].plot(x, y - 0.12 + 0.08*enu, '-', color='k', lw=2, alpha=0.2)
#         alt_ax[0].plot(x, y - 0.12 + 0.08*enu, '-', color='k', lw=2, alpha=0.15)
#         alt_df_ax[enu].plot(x, y, '-', color='k', lw=2, alpha=0.5)
#
#     df = pd.DataFrame()
#     all_df_win = np.hstack([df_win_mm, df_win_ff, df_win_mf, df_win_fm])
#     fit_x = np.hstack([all_df_win, all_df_win * -1])
#     fit_y = np.hstack([np.ones(len(all_df_win)), np.zeros(len(all_df_win))])
#
#     df['win'] = fit_y
#     df['d_param'] = fit_x
#     df = sm.add_constant(df)
#     logit_model = sm.Logit(df['win'][:, np.newaxis], df[['const', 'd_param']]).fit()
#
#     test_df = pd.DataFrame()
#     test_df['const'] = np.ones(len(np.linspace(-max_d_param, max_d_param, 100)))
#     test_df['d_param'] = np.linspace(-max_d_param, max_d_param, 100)
#     pred = logit_model.predict(test_df)
#
#     # ax[0].plot(test_df['d_param'], pred, linestyle='-', color='k', lw=3)
#     alt_ax[0].plot(test_df['d_param'], pred, linestyle='-', color='k', lw=3)
#
#
#     ###
#
#     for enu, d_param in enumerate([dsize_win_mm, dsize_win_ff, dsize_win_mf, dsize_win_fm]):
#
#         # ax[1].plot(d_param, np.ones(len(d_param)) - 0.12 + 0.08*enu, 'p', color=win_color[enu], markeredgecolor=mek[enu], alpha=0.8)
#         # ax[1].plot(np.array(d_param) * -1, np.zeros(len(d_param)) - 0.12 + 0.08*enu, 'o', color=lose_color[enu], markeredgecolor=mek[enu])
#
#         alt_ax[1].plot(d_param, np.ones(len(d_param)) - 0.12 + 0.08*enu, 'p', color=win_color[enu], markeredgecolor=mek[enu], alpha=0.8)
#         alt_ax[1].plot(np.array(d_param) * -1, np.zeros(len(d_param)) - 0.12 + 0.08*enu, 'o', color=lose_color[enu], markeredgecolor=mek[enu])
#
#         alt_ds_ax[enu].plot(d_param, np.ones(len(d_param)) + (np.random.rand(len(d_param)) - 0.5) * 0.2, 'p', color=win_color[enu], markeredgecolor=mek[enu], alpha=0.8)
#         alt_ds_ax[enu].plot(np.array(d_param) * -1, np.zeros(len(d_param)) + (np.random.rand(len(d_param)) - 0.5) * 0.2, 'o', color=lose_color[enu], markeredgecolor=mek[enu])
#
#         ###############################################################
#         df = pd.DataFrame()
#         fit_x = np.hstack([d_param, np.array(d_param) * -1])
#         fit_y = np.hstack([np.ones(len(d_param)), np.zeros(len(d_param))])
#
#         df['win'] = fit_y
#         df['d_param'] = fit_x
#         df = sm.add_constant(df)
#         try:
#             logit_model = sm.Logit(df['win'][:, np.newaxis], df[['const', 'd_param']]).fit()
#             fit_c = 'k'
#
#             max_d_param = np.max(np.abs(dsize_range))
#             test_df = pd.DataFrame()
#             test_df['const'] = np.ones(len(np.linspace(-max_d_param, max_d_param, 100)))
#             test_df['d_param'] = np.linspace(-max_d_param, max_d_param, 100)
#
#             pred = logit_model.predict(test_df)
#
#             x, y = test_df['d_param'], pred
#
#             # ax[0].plot(test_df['d_param'], pred, linestyle='--', color=win_color[enu], lw=2)
#         except:
#             x = np.linspace(np.min(df_range), np.max(df_range), 100)
#             if enu == 2:
#                 y = boltzmann(x, .5, 0)
#             elif enu == 3:
#                 y = boltzmann(x, -.5, 0)
#
#         # ax[1].plot(x, y - 0.12 + 0.08*enu, '-', color='k', lw=2, alpha=0.2)
#         alt_ax[1].plot(x, y - 0.12 + 0.08*enu, '-', color='k', lw=2, alpha=0.15)
#         alt_ds_ax[enu].plot(x, y, '-', color='k', lw=2, alpha=0.5)
#
#     df = pd.DataFrame()
#     all_dsize_win = np.hstack([dsize_win_mm, dsize_win_ff, dsize_win_mf, dsize_win_fm])
#     fit_x = np.hstack([all_dsize_win, all_dsize_win * -1])
#     fit_y = np.hstack([np.ones(len(all_dsize_win)), np.zeros(len(all_dsize_win))])
#
#     df['win'] = fit_y
#     df['d_param'] = fit_x
#     df = sm.add_constant(df)
#     logit_model = sm.Logit(df['win'][:, np.newaxis], df[['const', 'd_param']]).fit()
#
#     test_df = pd.DataFrame()
#     test_df['const'] = np.ones(len(np.linspace(-max_d_param, max_d_param, 100)))
#     test_df['d_param'] = np.linspace(-max_d_param, max_d_param, 100)
#     pred = logit_model.predict(test_df)
#
#     # ax[1].plot(test_df['d_param'], pred, linestyle='-', color='k', lw=3)
#     alt_ax[1].plot(test_df['d_param'], pred, linestyle='-', color='k', lw=3)
#
#     #################################################
#
#     # ax[0].set_yticks([0, 1])
#     # ax[0].set_yticklabels(['Lose', 'Win'])
#     # plt.setp(ax[1].get_yticklabels(), visible=False)
#
#     # ax[0].set_xlabel(r'$\Delta$EODf [Hz]', fontsize=12)
#     # ax[1].set_xlabel(r'$\Delta$size [cm]', fontsize=12)
#
#     alt_df_ax[0].set_ylim(-0.2, 1.2)
#
#     # fig.tag(axes=ax, fontsize=15, yoffs=3, xoffs=-4)
#
#     # for a in ax:
#     #     a.tick_params(labelsize=10)
#     #
#     # plt.savefig('../../figures/dparam_wl.pdf')
#     # plt.savefig('dparam_wl.pdf')
#
#
#     ##################################################
#     for enu, a in enumerate(alt_df_ax):
#         a.set_xticks([-200, 0, 200])
#         a.set_yticks([0, 1])
#         if enu in [0, 2]:
#             a.set_yticklabels(['Lose', 'Win'])
#         else:
#             plt.setp(a.get_yticklabels(), visible=False)
#
#         if enu in [0, 1]:
#             plt.setp(a.get_xticklabels(), visible=False)
#         else:
#             a.set_xlabel(r'$\Delta$EODf [Hz]', fontsize=12)
#
#     for enu, a in enumerate(alt_ds_ax):
#         a.set_yticks([0, 1])
#         if enu in [0, 2]:
#             a.set_yticklabels(['Lose', 'Win'])
#         else:
#             plt.setp(a.get_yticklabels(), visible=False)
#
#         if enu in [0, 1]:
#             plt.setp(a.get_xticklabels(), visible=False)
#         else:
#             a.set_xlabel(r'$\Delta$size [cm]', fontsize=12)
#
#     # legend
#     alt_df_ax[0].plot(0, 2, 'p', markeredgecolor='k', color='k', label='Winner')
#     alt_df_ax[0].plot(0, 2, 'o', markeredgecolor='k', color='k', label='Loser')
#     alt_df_ax[0].plot(0, 2, 's', color=male_color, label=u'\u2642')
#     alt_df_ax[0].plot(0, 2, 's', color=female_color, label=u'\u2640')
#     alt_df_ax[0].plot(0, 2, 's', color='grey', markeredgecolor='k', label='same-sex')
#     alt_df_ax[0].plot(0, 2, 's', color='grey', label='mixed-sex')
#     alt_df_ax[0].legend(loc='upper center', frameon=False, bbox_to_anchor=(-0.2, 1.7), ncol=3, fontsize=9)
#
#
#     alt_ax[0].set_yticks([0, 1])
#     alt_ax[0].set_yticklabels(['Lose', 'Win'])
#     alt_ax[1].set_yticks([0, 1])
#     alt_ax[1].set_yticklabels(['Lose', 'Win'])
#
#     alt_ax[0].set_xlabel(r'$\Delta$EODf [Hz]', fontsize=12)
#     alt_ax[1].set_xlabel(r'$\Delta$size [cm]', fontsize=12)
#
#     alt_fig.tag(axes=[alt_ds_ax[0], alt_ax[1], alt_df_ax[0], alt_ax[0]], fontsize=15, yoffs=1.5, xoffs=-4)
#
#     for a in alt_ax:
#         a.tick_params(labelsize=10)
#     for a in alt_df_ax:
#         a.tick_params(labelsize=10)
#     for a in alt_ds_ax:
#         a.tick_params(labelsize=10)
#
#
#     alt_df_ax[0].text(alt_df_ax[0].get_xlim()[1], 1.3, 't=0.79, p=0.465', fontsize=8, ha='right', va='center')
#     alt_df_ax[1].text(alt_df_ax[1].get_xlim()[1], 1.3, 't=1.41, p=0.192', fontsize=8, ha='right', va='center')
#     alt_df_ax[2].text(alt_df_ax[2].get_xlim()[1], 1.3, 't=10.77, p<0.001', fontsize=8, ha='right', va='center')
#     alt_df_ax[3].text(alt_df_ax[3].get_xlim()[1], 1.3, 't=-12.16, p<0.001', fontsize=8, ha='right', va='center')
#     #alt_ax[0].text(alt_ax[0].get_xlim()[1], 1.25, 't=2.14, p=0.040', fontsize=8, ha='right', va='center')
#
#     alt_ds_ax[0].text(alt_ds_ax[0].get_xlim()[1], 1.3, 't=2.86, p=0.036', fontsize=8, ha='right', va='center')
#     alt_ds_ax[1].text(alt_ds_ax[1].get_xlim()[1], 1.3, 't=2.09, p=0.066', fontsize=8, ha='right', va='center')
#     alt_ds_ax[2].text(alt_ds_ax[2].get_xlim()[1], 1.3, 't=3.85, p=0.002', fontsize=8, ha='right', va='center')
#     alt_ds_ax[3].text(alt_ds_ax[3].get_xlim()[1], 1.3, 't=2.01, p=0.091', fontsize=8, ha='right', va='center')
#
#     alt_ax[0].text(alt_ax[0].get_xlim()[1], 1.25, 't=2.14, p=0.040', fontsize=8, ha='right', va='center')
#     alt_ax[1].text(alt_ax[1].get_xlim()[1], 1.25, 't=5.30, p<0.001', fontsize=8, ha='right', va='center')
#
#     for a in alt_df_ax + alt_ds_ax + alt_ax:
#         y0, y1 = a.get_ylim()
#         a.plot([0, 0], [y0, y1], color='k', lw=1, linestyle='dotted')
#         a.set_ylim(y0, y1)
#
#     alt_fig.savefig('../../figures/dparam_wl2.pdf')
#     alt_fig.savefig('dparam_wl2.pdf')
#
#     ##################################################
#
#     if True:
#         print('\n### EODf ###')
#         for cat, par in zip(['mm', 'ff', 'mf', 'fm'], [df_win_mm, df_win_ff, df_win_mf, df_win_fm]):
#             u, p = scp.mannwhitneyu(par, np.hstack([par, np.array(par)*-1]))
#             print(cat + ' -- U: %.1f; p: %.4f' % (u, p))
#         u, p = scp.mannwhitneyu(all_df_win, np.hstack([all_df_win, all_df_win*-1]))
#         print('all -- U: %.1f; p: %.4f' % (u, p))
#
#         for cat, par, dpar in zip(['mm', 'ff', 'mf', 'fm'],
#                                   [EODf_win_mm, EODf_win_ff, EODf_win_mf, EODf_win_fm],
#                                   [df_win_mm, df_win_ff, df_win_mf, df_win_fm]):
#             t, p = scp.ttest_rel(par, np.array(par) - np.array(dpar))
#             print(cat + ' -- t: %.3f; p: %.4f' % (t, p))
#         t, p = scp.ttest_rel(np.hstack([EODf_win_mm, EODf_win_ff, EODf_win_mf, EODf_win_fm]),
#                              np.hstack([EODf_win_mm, EODf_win_ff, EODf_win_mf, EODf_win_fm]) - np.hstack([df_win_mm, df_win_ff, df_win_mf, df_win_fm]))
#         print('all -- t: %.3f; p: %.4f' % (t, p))
#         t, p = scp.ttest_rel(np.hstack([EODf_win_mm, EODf_win_ff]),
#                              np.hstack([EODf_win_mm, EODf_win_ff]) - np.hstack([df_win_mm, df_win_ff]))
#         print('same sex -- t: %.3f; p: %.4f' % (t, p))
#
#         print('\n### size ###')
#         for cat, par in zip(['mm', 'ff', 'mf', 'fm'], [dsize_win_mm, dsize_win_ff, dsize_win_mf, dsize_win_fm]):
#             u, p = scp.mannwhitneyu(par, np.hstack([par, np.array(par)*-1]))
#             print(cat + ' -- U: %.1f; p: %.4f' % (u, p))
#         u, p = scp.mannwhitneyu(all_dsize_win, np.hstack([all_dsize_win, all_dsize_win*-1]))
#         print('all -- U: %.1f; p: %.4f' % (u, p))
#
#         for cat, par, dpar in zip(['mm', 'ff', 'mf', 'fm'],
#                                   [size_win_mm, size_win_ff, size_win_mf, size_win_fm],
#                                   [dsize_win_mm, dsize_win_ff, dsize_win_mf, dsize_win_fm]):
#             t, p = scp.ttest_rel(par, np.array(par) - np.array(dpar))
#             print(cat + ' -- t: %.3f; p: %.4f' % (t, p))
#         t, p = scp.ttest_rel(np.hstack([size_win_mm, size_win_ff, size_win_mf, size_win_fm]),
#                              np.hstack([size_win_mm, size_win_ff, size_win_mf, size_win_fm]) - np.hstack([dsize_win_mm, dsize_win_ff, dsize_win_mf, dsize_win_fm]))
#         print('all -- t: %.3f; p: %.4f' % (t, p))
#
#         print('\n### male-female wins in mixed vs. sex ratio ###')
#         mixed_female_win, mixed_male_win = len(df_win_fm), len(df_win_mf)
#         dsize_win_mf, dsize_win_fm = np.array(dsize_win_mf), np.array(dsize_win_fm)
#         mixed_male_larger = len(dsize_win_mf[dsize_win_mf > 0]) #+ len(dsize_win_fm[dsize_win_fm < 0])
#         mixed_female_larger = len(dsize_win_fm[dsize_win_fm > 0]) #+ len(dsize_win_mf[dsize_win_mf < 0])
#         n_females, n_males = 12, 9
#
#         chi2, p = scp.chisquare([mixed_female_win, mixed_male_win], f_exp=[np.round((mixed_male_win + mixed_female_win)/2), np.round((mixed_male_win + mixed_female_win)/2)])
#         print('chi2: %.3f; p. %.3f\n' % (chi2, p))
#         chi2, p = scp.chisquare([mixed_female_win, mixed_male_win], f_exp=[np.round((mixed_female_win+mixed_male_win) * n_females / (n_females + n_males)),
#                                                                            np.round((mixed_female_win+mixed_male_win) * n_males / (n_females + n_males))])
#
#         print('chi2: %.3f; p. %.3f\n' % (chi2, p))
#
#         print('\n### male-female wins in mixed vs. sex male-female larger ###')
#         chi2, p = scp.chisquare([mixed_female_win, mixed_male_win], f_exp=[mixed_female_larger, mixed_male_larger])
#         print('chi2: %.3f; p. %.3f\n' % (chi2, p))

def boltzmann(x, k, shift):
    return 1 / (1 + np.exp(-k * x + shift))

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

def plot_d_param(ax, dparam_win_mm, dparam_win_ff, dparam_win_mf, dparam_win_fm, dparam_win_mix, gauss_sigma, gauss_size):
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

    female_color, male_color= '#e74c3c', '#3498db'

    dparam_range = np.hstack([dparam_win_mm, np.array(dparam_win_mm) * -1,
                              dparam_win_ff, np.array(dparam_win_ff) * -1,
                              dparam_win_mf, np.array(dparam_win_mf) * -1,
                              dparam_win_fm, np.array(dparam_win_fm) * -1])

    win_color = [male_color, female_color, male_color, female_color]
    lose_color = [male_color, female_color, female_color, male_color]
    mek= ['k', 'k', None, None]

    for enu, d_param in enumerate([dparam_win_mm, dparam_win_ff, dparam_win_mf, dparam_win_fm]):

        ax[0].plot(d_param, np.ones(len(d_param)) - 0.21 + 0.14*enu, 'p', color=win_color[enu], markeredgecolor=mek[enu], alpha=0.8, markersize=6)
        ax[0].plot(np.array(d_param) * -1, np.zeros(len(d_param)) - 0.21 + 0.14*enu, 'o', color=lose_color[enu], markeredgecolor=mek[enu], markersize=6)

        ax[enu+4].plot(d_param, np.ones(len(d_param)) + (np.random.rand(len(d_param)) - 0.5) * 0.2, 'p', color=win_color[enu], markeredgecolor=mek[enu], alpha=0.8, markersize=6)
        ax[enu+4].plot(np.array(d_param) * -1, np.zeros(len(d_param)) + (np.random.rand(len(d_param)) - 0.5) * 0.2, 'o', color=lose_color[enu], markeredgecolor=mek[enu], markersize=6)


        #   logistic fit
        x, y = logistic_fit()
        # ax[0].plot(x, y - 0.12 + 0.08*enu, '-', color='k', lw=2, alpha=0.2)
        # ax[0].plot(x, y - 0.12 + 0.08*enu, '-', color='k', lw=2, alpha=0.15)
        ax[enu+4].plot(x, y, '-', color='k', lw=2, alpha=0.5)

    d_param = np.hstack([dparam_win_mm, dparam_win_ff, dparam_win_mf, dparam_win_fm])

    x, y = logistic_fit()
    ax[0].plot(x, y, linestyle='-', color='k', lw=2)

    min_param, max_param = -1 * np.max(np.abs(d_param)), np.max(np.abs(d_param))
    param_range = np.linspace(min_param - (0.1 * (max_param - min_param)), max_param + (0.1 * (max_param - min_param)), 500)

    win_param_conv = np.zeros(len(param_range))
    for s in d_param:
        win_param_conv += gauss(param_range, s, gauss_sigma, gauss_size, norm=True)
    win_param_conv /= len(d_param)

    ax[2].plot(param_range, win_param_conv, lw=2, color='k')
    ax[2].plot(param_range, win_param_conv[::-1], lw=2, color='grey')

    ax[2].set_xlim(param_range[0], param_range[-1])
    ax[2].set_ylim(bottom=0)

def wl_meta_dist(ax, sex_w, sex_l, param_w, param_l, gauss_sigma, gauss_size):
    female_color, male_color = '#e74c3c', '#3498db'

    for i in range(len(sex_w)):
        w_color = male_color if sex_w[i] == 1 else female_color
        l_color = male_color if sex_l[i] == 1 else female_color
        mek = 'none' if sex_w[i] == sex_l[i] else 'k'

        rw = (np.random.rand() - 0.5) * 0.1
        rl = (np.random.rand() - 0.5) * 0.1
        ax[1].plot(param_w[i], 1 + rw, 'p', color=w_color, markeredgecolor=mek, markersize=6)
        ax[1].plot(param_l[i], 0 + rl, 'o', color=l_color, markeredgecolor=mek, markersize=6)

        ax[1].plot([param_w[i], param_l[i]], [1 + rw, 0 + rl], color='grey', lw=1, alpha=0.5)


    min_param, max_param = np.min(np.concatenate((param_w, param_l))), np.max(np.concatenate((param_w, param_l)))
    param_range = np.linspace(min_param - (0.1 * (max_param - min_param)), max_param + (0.1 * (max_param - min_param)), 500)

    win_param_conv = np.zeros(len(param_range))
    lose_param_conv = np.zeros(len(param_range))

    for s in param_w:
        win_param_conv += gauss(param_range, s, gauss_sigma, gauss_size, norm=True)
    win_param_conv /= len(param_w)

    for s in param_l:
        lose_param_conv += gauss(param_range, s, gauss_sigma, gauss_size, norm=True)
    lose_param_conv /= len(param_l)

    ax[3].plot(param_range, win_param_conv, lw=2, color='k')
    ax[3].plot(param_range, lose_param_conv, lw=2, color='grey')

    ax[3].set_ylim(bottom=0)
    ax[3].set_xlim(param_range[0], param_range[-1])
    #ax[3].set_xlim(EODf_range[0], EODf_range[-1])
    #ax[3].set_xlim(min_param, max_param)



def main():
    dsize_win = np.load('../dsize_win.npy', allow_pickle=True)
    df_win = np.load('../df_win.npy', allow_pickle=True)
    size_win = np.load('../size_win.npy', allow_pickle=True)
    EODf_win = np.load('../EODf_win.npy', allow_pickle=True)

    wl_sex_EODf_size = np.load('wl_sex_EODf_size.npy')
    sex_w, sex_l, EODf_w, EODf_l, size_w, size_l = wl_sex_EODf_size


    # fig = plt.figure(figsize=(17.5 / 2.54, 11.5 / 2.54))
    #
    # gs = gridspec.GridSpec(2, 2, left=0.075, bottom=0.55, right=0.975, top=0.925, hspace=0, height_ratios=[1, 3])
    # ax = []
    # ax.append(fig.add_subplot(gs[1, 0]))
    # ax.append(fig.add_subplot(gs[1, 1], sharey=ax[0]))
    # ax.append(fig.add_subplot(gs[0, 0], sharex=ax[0]))
    # ax.append(fig.add_subplot(gs[0, 1], sharex=ax[1]))
    #
    # gs2 = gridspec.GridSpec(1, 4, left=0.075, bottom=0.15, right=0.975, top=0.35, wspace=0.3)
    # ax.append(fig.add_subplot(gs2[0, 0]))
    # ax.append(fig.add_subplot(gs2[0, 1], sharey=ax[4]))
    # ax.append(fig.add_subplot(gs2[0, 2], sharey=ax[4]))
    # ax.append(fig.add_subplot(gs2[0, 3], sharey=ax[4]))
    #
    # ###########################################################################
    # embed()
    # quit()
    fig = plt.figure(figsize=(9 / 2.54, 18 / 2.54))

    ax = []
    gs = gridspec.GridSpec(2, 1, left=0.15, bottom=0.725, right=0.975, top=0.9, hspace=0, height_ratios=[1, 3])
    gs3 = gridspec.GridSpec(2, 2, left=0.15, bottom=0.35, right=0.975, top=0.6, wspace=0.3, hspace=0.35)
    gs2 = gridspec.GridSpec(2, 1, left=0.15, bottom=0.075, right=0.975, top=0.25, hspace=0, height_ratios=[1, 3])

    ax.append(fig.add_subplot(gs[1, 0]))
    ax.append(fig.add_subplot(gs2[1, 0]))
    ax.append(fig.add_subplot(gs[0, 0], sharex=ax[0]))
    ax.append(fig.add_subplot(gs2[0, 0], sharex=ax[1]))

    ax.append(fig.add_subplot(gs3[0, 0]))
    ax.append(fig.add_subplot(gs3[0, 1], sharey=ax[4]))
    ax.append(fig.add_subplot(gs3[1, 0]))
    ax.append(fig.add_subplot(gs3[1, 1], sharey=ax[6]))


    plot_d_param(ax, *dsize_win, gauss_sigma=0.5, gauss_size=2)

    wl_meta_dist(ax, sex_w, sex_l, size_w, size_l, gauss_sigma=0.5, gauss_size=2) # 10, 0.1 for EODf gauss

    # COSMETICS FIRST PART
    ax[4].set_ylim(-0.2, 1.2)
    for enu, a in enumerate(ax[4:]):
        a.set_xticks([-5, 0, 5])
        a.set_yticks([0, 1])
        if enu in [0, 2]:
            a.set_yticklabels(['Lose', 'Win'])
        else:
            plt.setp(a.get_yticklabels(), visible=False)

        if enu in [0, 1]:
            plt.setp(a.get_xticklabels(), visible=False)
        else:
            a.set_xlabel(r'$\Delta$size [cm]', fontsize=12)

    ax[0].set_xlim(-10, 10)
    ax[0].set_yticks([0, 1])
    ax[0].set_yticklabels(['Lose', 'Win'])
    ax[1].set_yticks([0, 1])
    ax[1].set_yticklabels(['Lose', 'Win'])

    ax[0].set_xlabel(r'$\Delta$size [cm]', fontsize=12)
    ax[1].set_xlabel('size [cm]', fontsize=12)

    for a in ax:
        a.tick_params(labelsize=10)
    ax[2].set_axis_off()
    ax[3].set_axis_off()

    # COSMETICS SECOND PART
    for a in [ax[0], ax[4], ax[5], ax[6], ax[7]]:
        y0, y1 = a.get_ylim()
        a.plot([0, 0], [y0, y1], color='k', lw=1, linestyle='dotted')
        a.set_ylim(y0, y1)

    for a in ax[:2]:
        a.set_yticks([0, 1])
        a.set_ylim(-0.3, 1.3)
        a.set_yticklabels(['Lose', 'Win'])
        a.tick_params(labelsize=10)
        ax[0].set_xlabel(r'$\Delta$size [cm]', fontsize=12)

    ax[0].text(ax[0].get_xlim()[0], 1.8, r'$t=5.30, p<0.001$', fontsize=8, ha='left', va='center')

    ax[4].text(ax[4].get_xlim()[1], 1.3, r'$t=2.86, p=0.036$', fontsize=8, ha='right', va='center')
    ax[5].text(ax[5].get_xlim()[1], 1.3, r'$t=2.09, p=0.066$', fontsize=8, ha='right', va='center')
    ax[6].text(ax[6].get_xlim()[1], 1.3, r'$t=3.85, p=0.002$', fontsize=8, ha='right', va='center')
    ax[7].text(ax[7].get_xlim()[1], 1.3, r'$t=2.01, p=0.091$', fontsize=8, ha='right', va='center')


    fig.tag(axes=[ax[0], ax[1]], labels=['A', 'F'], fontsize=15, yoffs=2, xoffs=-6)
    fig.tag(axes=[ax[4], ax[6]], labels=['B', 'D'], fontsize=15, yoffs=1.5, xoffs=-6)
    fig.tag(axes=[ax[5], ax[7]], labels=['C', 'E'], fontsize=15, yoffs=1.5, xoffs=-2.5)

    female_color, male_color = '#e74c3c', '#3498db'
    ax[0].plot(0, 2, 'p', markeredgecolor='k', color='k', label='Winner')
    ax[0].plot(0, 2, 'o', color='grey', label='Loser')
    ax[0].plot(0, 2, 's', color=male_color, label=u'\u2642')
    ax[0].plot(0, 2, 's', color=female_color, label=u'\u2640')
    ax[0].plot(0, 2, 's', color='lightgrey', markeredgecolor='k', label='same-sex')
    ax[0].plot(0, 2, 's', color='lightgrey', label='mixed-sex')
    ax[0].legend(loc='upper center', frameon=False, bbox_to_anchor=(.5, 2), ncol=3, fontsize=9)


    # plt.savefig('../../figures/size_wl_cat.pdf')
    # plt.savefig('size_wl_cat.pdf')

    ##############################################################################################################
    # fig = plt.figure(figsize=(17.5 / 2.54, 11 / 2.54))
    #
    # gs = gridspec.GridSpec(2, 2, left=0.075, bottom=0.55, right=0.975, top=0.975, hspace=0, height_ratios=[1, 3])
    # ax = []
    # ax.append(fig.add_subplot(gs[1, 0]))
    # ax.append(fig.add_subplot(gs[1, 1], sharey=ax[0]))
    # ax.append(fig.add_subplot(gs[0, 0], sharex=ax[0]))
    # ax.append(fig.add_subplot(gs[0, 1], sharex=ax[1]))
    #
    # gs2 = gridspec.GridSpec(1, 4, left=0.075, bottom=0.15, right=0.975, top=0.35, wspace=0.3)
    # ax.append(fig.add_subplot(gs2[0, 0]))
    # ax.append(fig.add_subplot(gs2[0, 1], sharey=ax[4]))
    # ax.append(fig.add_subplot(gs2[0, 2], sharey=ax[4]))
    # ax.append(fig.add_subplot(gs2[0, 3], sharey=ax[4]))

    fig = plt.figure(figsize=(9 / 2.54, 16.75 / 2.54))

    ax = []
    gs = gridspec.GridSpec(2, 1, left=0.15, bottom=0.775, right=0.975, top=0.975, hspace=0, height_ratios=[1, 3])
    gs2 = gridspec.GridSpec(2, 1, left=0.15, bottom=0.075, right=0.975, top=0.275, hspace=0, height_ratios=[1, 3])
    gs3 = gridspec.GridSpec(2, 2, left=0.15, bottom=0.375, right=0.975, top=0.65, wspace=0.3, hspace=0.35)

    ax.append(fig.add_subplot(gs[1, 0]))
    ax.append(fig.add_subplot(gs2[1, 0]))
    ax.append(fig.add_subplot(gs[0, 0], sharex=ax[0]))
    ax.append(fig.add_subplot(gs2[0, 0], sharex=ax[1]))

    ax.append(fig.add_subplot(gs3[0, 0]))
    ax.append(fig.add_subplot(gs3[0, 1], sharey=ax[4]))
    ax.append(fig.add_subplot(gs3[1, 0], sharey=ax[4]))
    ax.append(fig.add_subplot(gs3[1, 1], sharey=ax[4]))

    plot_d_param(ax, *df_win, gauss_sigma=10, gauss_size=0.1)

    wl_meta_dist(ax, sex_w, sex_l, EODf_w, EODf_l, gauss_sigma=10, gauss_size=0.1) # 10, 0.1 for EODf gauss

    # COSMETICS FIRST PART

    ax[4].set_ylim(-0.2, 1.2)
    for enu, a in enumerate(ax[4:]):
        a.set_xticks([-200, 0, 200])
        a.set_yticks([0, 1])
        if enu in [0, 2]:
            a.set_yticklabels(['Lose', 'Win'])
        else:
            plt.setp(a.get_yticklabels(), visible=False)

        if enu in [0, 1]:
            plt.setp(a.get_xticklabels(), visible=False)
        else:
            a.set_xlabel(r'$\Delta$EODf [Hz]', fontsize=12)

    ax[0].set_yticks([0, 1])
    ax[0].set_yticklabels(['Lose', 'Win'])
    ax[1].set_yticks([0, 1])
    ax[1].set_yticklabels(['Lose', 'Win'])

    ax[0].set_xlabel(r'$\Delta$EODf [Hz]', fontsize=12)
    ax[1].set_xlabel('EODf [Hz]', fontsize=12)

    for a in ax:
        a.tick_params(labelsize=10)
    ax[2].set_axis_off()
    ax[3].set_axis_off()

    # COSMETICS SECOND PART
    for a in [ax[0], ax[4], ax[5], ax[6], ax[7]]:
        y0, y1 = a.get_ylim()
        a.plot([0, 0], [y0, y1], color='k', lw=1, linestyle='dotted')
        a.set_ylim(y0, y1)

    for a in ax[:2]:
        a.set_yticks([0, 1])
        a.set_ylim(-0.3, 1.3)
        a.set_yticklabels(['Lose', 'Win'])
        a.tick_params(labelsize=10)
        ax[0].set_xlabel(r'$\Delta$EODf [Hz]', fontsize=12)


    ax[0].text(ax[0].get_xlim()[0], 1.9, r'$t=2.14, p=0.040$', fontsize=8, ha='left', va='center')

    ax[4].text(ax[4].get_xlim()[1], 1.3, r'$t=0.79, p=0.465$', fontsize=8, ha='right', va='center')
    ax[5].text(ax[5].get_xlim()[1], 1.3, r'$t=1.41, p=0.192$', fontsize=8, ha='right', va='center')
    ax[6].text(ax[6].get_xlim()[1], 1.3, r'$t=10.77, p<0.001$', fontsize=8, ha='right', va='center')
    ax[7].text(ax[7].get_xlim()[1], 1.3, r'$t=-12.16, p<0.001$', fontsize=8, ha='right', va='center')

    fig.tag(axes=[ax[0], ax[1]], labels=['A', 'F'], fontsize=15, yoffs=2, xoffs=-6)
    fig.tag(axes=[ax[4], ax[6]], labels=['B', 'D'], fontsize=15, yoffs=1.5, xoffs=-6)
    fig.tag(axes=[ax[5], ax[7]], labels=['C', 'E'], fontsize=15, yoffs=1.5, xoffs=-2.5)

    # plt.savefig('../../figures/EODf_wl_cat.pdf')
    # plt.savefig('EODf_wl_cat.pdf')
    plt.show()


    embed()
    quit()
if __name__ == '__main__':
    main()