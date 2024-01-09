import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as scp
from IPython import embed
from thunderfish.powerspectrum import decibel
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit
from plottools.axes import tag
# fig.tag(axes=[ax[1], ax[2]], fontsize=15, yoffs=2, xoffs=-3)
from Spec_figures import plot_rises

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

def main():
    win_rc = np.load('../win_rc.npy', allow_pickle=True)
    lose_rc = np.load('../lose_rc.npy', allow_pickle=True)

    win_rs = np.load('../win_rs.npy', allow_pickle=True)
    lose_rs = np.load('../lose_rs.npy', allow_pickle=True)

    win_rc_day = np.load('../win_rc_day.npy', allow_pickle=True)
    lose_rc_day = np.load('../lose_rc_day.npy', allow_pickle=True)

    win_rs_day = np.load('../win_rs_day.npy', allow_pickle=True)
    lose_rs_day = np.load('../lose_rs_day.npy', allow_pickle=True)


    all_rs = list(np.hstack(np.hstack(win_rs))) + list(np.hstack(np.hstack(lose_rs))) + \
             list(np.hstack(np.hstack(win_rs_day))) + list(np.hstack(np.hstack(lose_rs_day)))

    all_win_rs = np.hstack(np.hstack(win_rs))
    all_lose_rs = np.hstack(np.hstack(lose_rs))
    max_rs = np.max(list(all_win_rs) + list(all_lose_rs))
    bins = np.linspace(0, max_rs, 50)

    win_n = []
    lose_n = []
    for i in range(4):
        n, bin_edges = np.histogram(np.hstack(win_rs[i]), bins)
        win_n.append(n)

        n, bins = np.histogram(np.hstack(lose_rs[i]), bins)
        lose_n.append(n)

    win_n = np.array(win_n) / np.sum(win_n) / (bin_edges[1]- bin_edges[0])/2
    lose_n = np.array(lose_n) / np.sum(lose_n) / (bin_edges[1]- bin_edges[0])/2

    fig = plt.figure(figsize=(17.5/2.54, 10.5/2.54))
    gs = gridspec.GridSpec(2, 2, bottom=0.2, left=0.1, right=0.95, top=0.95, width_ratios=[1, 2], wspace=0.4)
    ax = []
    ax.append(fig.add_subplot(gs[0, 1]))
    ax.append(fig.add_subplot(gs[1, 1], sharex=ax[0]))
    ax.append(fig.add_subplot(gs[:, 0]))


    female_color, male_color= '#e74c3c', '#3498db'

    # embed()
    # quit()

    rs_array = np.arange(0, 70, 0.1)
    mm_rs_conv, ff_rs_conv, mf_rs_conv, fm_rs_conv = np.zeros((4, len(rs_array)))
    for e in np.hstack(np.hstack(lose_rs[0])):
        mm_rs_conv += gauss(rs_array, e, 2, 0.5, norm=True)
    for e in np.hstack(np.hstack(lose_rs[1])):
        ff_rs_conv += gauss(rs_array, e, 2, 0.5, norm=True)
    for e in np.hstack(np.hstack(lose_rs[2])):
        mf_rs_conv += gauss(rs_array, e, 2, 0.5, norm=True)
    for e in np.hstack(np.hstack(lose_rs[3])):
        fm_rs_conv += gauss(rs_array, e, 2, 0.5, norm=True)

    mm_rs_conv/= len(np.hstack(lose_rs[0])) * 0.1
    ff_rs_conv/= len(np.hstack(lose_rs[1])) * 0.1
    mf_rs_conv/= len(np.hstack(lose_rs[2])) * 0.1
    fm_rs_conv/= len(np.hstack(lose_rs[3])) * 0.1

    ax[0].plot(rs_array, mm_rs_conv, color=male_color, lw=2, label=u'\u2642\u2642')
    ax[0].plot(rs_array, ff_rs_conv, color=female_color, lw=2, label=u'\u2640\u2640')
    ax[0].plot(rs_array, mf_rs_conv, '--', color=female_color, lw=2, label=u'\u2642\u2640')
    ax[0].plot(rs_array, fm_rs_conv, '--', color=male_color, lw=2, label=u'\u2640\u2642')
    ax[0].legend(loc=1, frameon=False, fontsize=9)

    # ax[0].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, win_n[0], color=male_color, edgecolor='k', label=u'\u2642\u2642')
    # ax[0].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, win_n[1], bottom = win_n[0], color=female_color, edgecolor='k', label=u'\u2640\u2640')
    # ax[0].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, win_n[2], bottom = win_n[0] + win_n[1], color=male_color, label=u'\u2642\u2640')
    # ax[0].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, win_n[3], bottom = win_n[0] + win_n[1] + win_n[2], color=female_color, label=u'\u2640\u2642')
    # ax[0].legend(loc=1, frameon=False, fontsize=10)

    ax[1].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, lose_n[0], color=male_color, edgecolor='k')
    ax[1].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, lose_n[1], bottom = lose_n[0], color=female_color, edgecolor='k')
    ax[1].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, lose_n[2], bottom = lose_n[0] + lose_n[1], color=male_color)
    ax[1].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, lose_n[3], bottom = lose_n[0] + lose_n[1] + lose_n[2], color=female_color)


    win_color = [male_color, female_color, male_color, female_color]
    lose_color = [male_color, female_color, female_color, male_color]
    mek = ['k', 'k', None, None]
    for i in range(4):
        ax[2].plot(np.ones(len(win_rc_day[i])) + (np.random.rand(len(win_rc_day[i])) - 0.5) * 0.5, win_rc_day[i], 'p', markeredgecolor=mek[i], color=win_color[i])
        ax[2].plot(np.ones(len(lose_rc_day[i]))*2 + (np.random.rand(len(lose_rc_day[i])) - 0.5) * 0.5, lose_rc_day[i], 'o', markeredgecolor=mek[i], color=lose_color[i])
        ax[2].plot(np.ones(len(win_rc[i]))*3 + (np.random.rand(len(win_rc[i])) - 0.5) * 0.5, win_rc[i], 'p', markeredgecolor=mek[i], color=win_color[i])
        ax[2].plot(np.ones(len(lose_rc[i]))*4 + (np.random.rand(len(lose_rc[i])) - 0.5) * 0.5, lose_rc[i], 'o', markeredgecolor=mek[i], color=lose_color[i])

    ax[2].boxplot([np.hstack(win_rc_day), np.hstack(lose_rc_day), np.hstack(win_rc), np.hstack(lose_rc)], sym='', widths=0.75)

    ax[1].set_xlim(left=0)
    ax[1].set_ylim(bottom=0)
    ax[0].set_ylabel('probability', fontsize=12)
    ax[1].set_ylabel('probability', fontsize=12)
    ax[1].set_xlabel('rise size [Hz]', fontsize=12)
    plt.setp(ax[0].get_xticklabels(), visible=False)

    ax[2].set_ylabel('rises [n]', fontsize=12)
    ax[2].set_xticks([1, 2, 3, 4])
    ax[2].set_xticklabels(['dom.', 'sub.', 'dom.', 'sub.'], rotation=35)

    for a in ax:
        a.tick_params(labelsize=10)

    #############################################################################################
    fig = plt.figure(figsize=(17.5/2.54, 12/2.54))
    # gs = gridspec.GridSpec(1, 2, bottom=0.2, left=0.1, right=0.95, top=0.95, width_ratios=[1, 2], wspace=0.4)
    gs = gridspec.GridSpec(2, 6, bottom=0.1, left=0.1, right=0.95, top=0.95, wspace=2, hspace=0.4)
    ax = []
    ax.append(fig.add_subplot(gs[1, 2:]))
    ax.append(fig.add_subplot(gs[1, :2]))

    female_color, male_color= '#e74c3c', '#3498db'

    # ax[0].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, lose_n[0], color=male_color, edgecolor='k', label=u'\u2642\u2642')
    # ax[0].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, lose_n[1], bottom = lose_n[0], color=female_color, edgecolor='k', label=u'\u2640\u2640')
    # ax[0].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, lose_n[2], bottom = lose_n[0] + lose_n[1], color=male_color, label=u'\u2642\u2640')
    # ax[0].bar(bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2, lose_n[3], bottom = lose_n[0] + lose_n[1] + lose_n[2], color=female_color, label=u'\u2640\u2642')
    # ax[0].legend(loc=1, frameon=False, fontsize=10)

    ax[0].plot(rs_array, mm_rs_conv, color=male_color, lw=2, label=u'\u2642\u2642')
    ax[0].plot(rs_array, ff_rs_conv, color=female_color, lw=2, label=u'\u2640\u2640')
    ax[0].plot(rs_array, mf_rs_conv, '--', color=female_color, lw=2, label=u'\u2642\u2640')
    ax[0].plot(rs_array, fm_rs_conv, '--', color=male_color, lw=2, label=u'\u2640\u2642')
    ax[0].legend(loc=1, frameon=False, fontsize=9)
    ax[0].set_yticks([0, 0.002, 0.004, 0.006])

    win_color = [male_color, female_color, male_color, female_color]
    lose_color = [male_color, female_color, female_color, male_color]
    mek = ['k', 'k', None, None]
    for i in range(4):
        ax[1].plot(np.ones(len(win_rc_day[i])) + (np.random.rand(len(win_rc_day[i])) - 0.5) * 0.5, win_rc_day[i], '.', color=win_color[i], zorder=2)
        ax[1].plot(np.ones(len(lose_rc_day[i]))*2 + (np.random.rand(len(lose_rc_day[i])) - 0.5) * 0.5, lose_rc_day[i], '.', color=lose_color[i], zorder=2)
        ax[1].plot(np.ones(len(win_rc[i]))*3 + (np.random.rand(len(win_rc[i])) - 0.5) * 0.5, win_rc[i], '.', color=win_color[i], zorder=2)
        ax[1].plot(np.ones(len(lose_rc[i]))*4 + (np.random.rand(len(lose_rc[i])) - 0.5) * 0.5, lose_rc[i], 'o', markeredgecolor=mek[i], color=lose_color[i], zorder=2)
    ax[1].fill_between([2.5, 4.5], [-50, -50], [470, 470], color='grey', alpha=0.5, zorder=1)
    ax[1].boxplot([np.hstack(win_rc_day), np.hstack(lose_rc_day), np.hstack(win_rc), np.hstack(lose_rc)], sym='', widths=0.75, zorder=3)

    u, p = scp.mannwhitneyu(list(np.hstack(win_rc_day)) + list(np.hstack(lose_rc_day)), list(np.hstack(win_rc)) + list(np.hstack(lose_rc)))
    print('\nday vs. night rises')
    print('U=%.1f, p=%.3f' % (u,p))
    t, p = scp.ttest_rel(list(np.hstack(win_rc_day)) + list(np.hstack(lose_rc_day)), list(np.hstack(win_rc)) + list(np.hstack(lose_rc)))
    print('t=%.1f, p=%.3f' % (t, p))

    u, p = scp.mannwhitneyu(np.hstack(win_rc), np.hstack(lose_rc))
    print('\nnight dom vs sub')
    print('U=%.1f, p=%.3f' % (u, p))
    t, p = scp.ttest_rel(np.hstack(lose_rc), np.hstack(win_rc))
    print('t=%.1f, p=%.3f' % (t, p))

    ax.append(fig.add_subplot(gs[0, :3]))
    ax.append(fig.add_subplot(gs[0, 3:]))

    ax[2].plot([70, 90], [677, 677], lw=4, color='k', clip_on=False)
    ax[2].text(80, 671, '20 sec', fontsize=12, ha='center', va='center')

    ax[3].plot([170, 220], [695, 695], lw=4, color='k', clip_on=False)
    ax[3].text(195, 687, '50 sec', fontsize=12, ha='center', va='center')

    #plot_rises([ax[2], ax[3]])
    ax[2].plot(39., 733,  'v', color='k', markersize=8)
    X, Y = [49.5, 65, 91, 97, 118, 126, 148, 159, 168, 185], [743, 750, 739, 744, 746, 750, 747, 743, 739, 743]
    for x, y in zip(X, Y):
        ax[3].plot(x, y,  'v', color='k', markersize=8)

    ax[2].set_xticks([])
    ax[3].set_xticks([])

    ax[2].set_ylabel('EOD frequency [Hz]', fontsize=12)

    # ax[0].set_yticks([0, 0.01, 0.02])
    ax[0].set_xlim(left=0)
    ax[0].set_ylim(bottom=0)
    ax[0].set_ylabel('PDF', fontsize=12)
    ax[0].set_xlabel('EODf rise size [Hz]', fontsize=12)

    ax[1].set_ylabel('EODf rises [n]', fontsize=12)
    ax[1].set_xticks([1, 2, 3, 4])
    ax[1].set_xticklabels(['dom.', 'sub.', 'dom.', 'sub.'], rotation=35)
    ax[1].set_ylim(-20, 450)

    for a in ax:
        a.tick_params(labelsize=10)


    fig.tag(axes=[ax[2], ax[3], ax[1], ax[0]], fontsize=15, yoffs=2, xoffs=-5)

    # plt.savefig('../../figures/EODf_rises_n_size.pdf')
    # plt.savefig('EODf_rises_n_size.pdf')

    #####################
    plt.close('all')

    fig = plt.figure(figsize=(9/2.54, 12/2.54))
    gs = gridspec.GridSpec(3, 1, left=0.2, bottom = 0.1, right=0.95, top=0.95, hspace=0.4)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[1, 0]))
    ax.append(fig.add_subplot(gs[2, 0]))

    ax[0].plot([69, 89], [676, 676], lw=4, color='k', clip_on=False)
    ax[0].text(79, 669, '20 sec', fontsize=11, ha='center', va='center')

    ax[1].plot([168, 218], [695, 695], lw=4, color='k', clip_on=False)
    ax[1].text(193, 687, '50 sec', fontsize=11, ha='center', va='center')
    ax[0].set_xlim(right=220)

    # plot_rises([ax[0], ax[1]])

    rise_spec1 = np.load('rise_spec1.npy', allow_pickle=True)
    rise_spec_extent1 = np.load('rise_spec1_extent.npy', allow_pickle=True)

    rise_spec2 = np.load('rise_spec2.npy', allow_pickle=True)
    rise_spec_extent2 = np.load('rise_spec2_extent.npy', allow_pickle=True)

    for a, rise_spec, rise_spec_extent in zip(ax, [rise_spec1, rise_spec2], [rise_spec_extent1, rise_spec_extent2]):

        a.imshow(decibel(rise_spec[::-1]),extent=[rise_spec_extent[0] - rise_spec_extent[0],
                                                   rise_spec_extent[1] - rise_spec_extent[0], rise_spec_extent[2],
                                                   rise_spec_extent[3]],
                 aspect='auto', alpha=0.7, cmap='jet', interpolation='gaussian', vmin=-120, vmax=-50)
        a.set_xlim(rise_spec_extent[0] - rise_spec_extent[0], rise_spec_extent[1] - rise_spec_extent[0])
        a.set_ylim(rise_spec_extent[2], rise_spec_extent[3])

    ax[0].plot([35, 35], [700, 729], lw=2, color='k')
    ax[0].plot([34, 36], [729, 729], lw=2, color='k')
    ax[0].plot([34, 36], [700, 700], lw=2, color='k')
    ax[0].text(31, 714.5, 'size', fontsize=11, color='k', ha='center', va='center', rotation=90)


    # ax[1].imshow(decibel(rise_spec2[::-1]),extent=[rise_spec_extent2[0] - rise_spec_extent2[0],
    #                                            rise_spec_extent2[1] - rise_spec_extent2[0], rise_spec_extent2[2],
    #                                            rise_spec_extent2[3]],
    #          aspect='auto', alpha=0.7, cmap='jet', interpolation='gaussian', vmin=-120, vmax=-50)
    # ax[1].set_xlim(rise_spec_extent2[0] - rise_spec_extent2[0], rise_spec_extent2[1] - rise_spec_extent2[0])
    # ax[1].set_ylim(rise_spec_extent2[2], rise_spec_extent2[3])


    ax[0].plot(39., 733, 'v', color='k', markersize=6)
    X, Y = [49.5, 65, 91, 97, 118, 126, 148, 159, 168, 185], [743, 750, 739, 744, 746, 750, 747, 743, 739, 743]
    for x, y in zip(X, Y):
        ax[1].plot(x, y, 'v', color='k', markersize=6)

    ax[0].set_ylim(680, 740)
    ax[0].set_yticks(np.arange(680, 741, 20))
    #
    # ax[1].set_ylim(700, 775)
    ax[1].set_yticks(np.arange(700, 776, 25))

    ax[0].set_xticks([])
    ax[1].set_xticks([])

    ax[0].set_ylabel('EODf [Hz]', fontsize=12)
    ax[1].set_ylabel('EODf [Hz]', fontsize=12)

    #############################################
    ax[2].plot(rs_array, mm_rs_conv, color=male_color, lw=2, label=u'\u2642\u2642')
    ax[2].plot(rs_array, ff_rs_conv, color=female_color, lw=2, label=u'\u2640\u2640')
    ax[2].plot(rs_array, mf_rs_conv, '--', color=female_color, lw=2, label=u'\u2642\u2640')
    ax[2].plot(rs_array, fm_rs_conv, '--', color=male_color, lw=2, label=u'\u2640\u2642')
    ax[2].legend(loc=1, frameon=False, fontsize=9)

    ax[2].set_yticks([0, 0.02, 0.04, 0.06])
    ax[2].set_yticklabels([0, 20, 40, 60])
    ax[2].set_xlim(left=0)
    ax[2].set_ylim(bottom=0)
    ax[2].set_ylabel('PDF [1/kHz]', fontsize=12)
    ax[2].set_xlabel('rise size [Hz]', fontsize=12)

    fig.align_ylabels()

    for a in ax:
        a.tick_params(labelsize=10)

    fig.tag(axes=ax, fontsize=15, yoffs=2, xoffs=-8)

    # plt.savefig('../../figures/EODf_rises_n_size.pdf')
    # plt.savefig('EODf_rises_n_size.pdf')
    plt.show()

    ####
    fig = plt.figure(figsize=(9/2.54, 5/2.54))
    gs = gridspec.GridSpec(1, 1, left=0.2, bottom = 0.3, right=0.95, top=0.95)
    ax = fig.add_subplot(gs[0, 0])


    ax.imshow(decibel(rise_spec2[::-1]),extent=[rise_spec_extent2[0] - rise_spec_extent2[0],
                                               rise_spec_extent2[1] - rise_spec_extent2[0], rise_spec_extent2[2],
                                               rise_spec_extent2[3]],
             aspect='auto', alpha=0.7, cmap='jet', interpolation='gaussian', vmin=-120, vmax=-50)

    X, Y = [49.5, 65, 91, 97, 118, 126, 148, 159, 168, 185], [743, 750, 739, 744, 746, 750, 747, 743, 739, 743]
    for x, y in zip(X, Y):
        ax.plot(x, y, 'v', color='k', markersize=6)


    ax.set_xlim(rise_spec_extent2[0] - rise_spec_extent2[0], rise_spec_extent2[1] - rise_spec_extent2[0])
    ax.set_ylim(rise_spec_extent2[2], rise_spec_extent2[3])

    ax.set_ylabel('EODf [Hz]', fontsize=12)
    ax.set_xticks([])

    ax.plot([168, 218], [690, 690], lw=4, color='k', clip_on=False)
    ax.text(193, 680, '50 sec', fontsize=11, ha='center', va='center')

    plt.savefig('Poster_rises.pdf')



    embed()
    quit()

if __name__ == '__main__':
    main()
