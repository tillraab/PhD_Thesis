import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from IPython import embed
from plottools.tag import tag

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

def meta_dist(ax, bins, comp_freq_hist, female_EODf, female_size, female_weight, male_EODf, male_size, male_weight):

    female_color, male_color = '#e74c3c', '#3498db'

    f_checked_mask = np.zeros(len(female_EODf), dtype=bool)
    f_checked_mask[np.array([7, 9, 11])] = True
    m_checked_mask = np.zeros(len(male_EODf), dtype=bool)
    m_checked_mask[np.array([2, 3, 6, 7, 8])] = True

    ax[0].bar(bins[:-1][bins[:-1] < 740] + (bins[1] - bins[0]) / 2, comp_freq_hist[bins[:-1] < 740], width=8, color=female_color, edgecolor='k', lw=.5)
    ax[0].bar(bins[:-1][bins[:-1] >= 740] + (bins[1] - bins[0]) / 2, comp_freq_hist[bins[:-1] >= 740], width=8, color=male_color, edgecolor='k', lw=.5)

    ax[1].plot(female_EODf[~f_checked_mask], female_size[~f_checked_mask], 'D', color=female_color, markeredgecolor=None)
    ax[1].plot(male_EODf[~m_checked_mask], male_size[~m_checked_mask], 'D', color=male_color, markeredgecolor=None)
    ax[1].plot(female_EODf[f_checked_mask], female_size[f_checked_mask], 'D', color=female_color, markeredgecolor='k')
    ax[1].plot(male_EODf[m_checked_mask], male_size[m_checked_mask], 'D', color=male_color, markeredgecolor='k')

    ax[2].plot(female_weight[~f_checked_mask], female_size[~f_checked_mask], 'D', color=female_color, markeredgecolor=None)
    ax[2].plot(male_weight[~m_checked_mask], male_size[~m_checked_mask], 'D', color=male_color, markeredgecolor=None)
    ax[2].plot(female_weight[f_checked_mask], female_size[f_checked_mask], 'D', color=female_color, markeredgecolor='k', label=u'\u2640')
    ax[2].plot(male_weight[m_checked_mask], male_size[m_checked_mask], 'D', color=male_color, markeredgecolor='k', label=u'\u2642')
    ax[2].legend(loc='upper center', fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.3), ncol=2, columnspacing=0.8)

    ax[1].set_xlabel(r'EOD$f_{25}$ [Hz]', fontsize = 12)
    ax[1].set_ylabel('size [cm]', fontsize = 12)
    ax[2].set_xlabel('weight [g]', fontsize = 12)
    ax[1].tick_params(labelsize=10)
    ax[2].tick_params(labelsize=10)
    ax[0].set_axis_off()

    ax[1].set_xlim(580, 950)
    ax[1].set_yticks([10, 14, 18])
    ax[2].set_yticks([10, 14, 18])
    ax[2].text(ax[2].get_xlim()[1], 20, r'$r=0.94$, $p<0.001$', fontsize=9, color='k', ha='right', va='center')

def wl_meta_dist(ax, sex_w, sex_l, EODf_w, EODf_l, size_w, size_l):
    female_color, male_color = '#e74c3c', '#3498db'

    for i in range(len(sex_w)):
        w_color = male_color if sex_w[i] == 1 else female_color
        l_color = male_color if sex_l[i] == 1 else female_color
        mek = 'none' if sex_w[i] == sex_l[i] else 'k'

        rw = (np.random.rand() - 0.5) * 0.1
        rl = (np.random.rand() - 0.5) * 0.1
        ax[0].plot(size_w[i], 1 + rw, 'p', color=w_color, markeredgecolor=mek, markersize=7)
        ax[0].plot(size_l[i], 0 + rl, 'o', color=l_color, markeredgecolor=mek, markersize=7)

        ax[0].plot([size_w[i], size_l[i]], [1 + rw, 0 + rl], color='grey', lw=1, alpha=0.5)

        ax[1].plot(EODf_w[i], 1 + rw, 'p', color=w_color, markeredgecolor=mek, markersize=7)
        ax[1].plot(EODf_l[i], 0 + rl, 'o', color=l_color, markeredgecolor=mek, markersize=7)

        ax[1].plot([EODf_w[i], EODf_l[i]], [1 + rw, 0 + rl], color='grey', lw=1, alpha=0.5)


    min_size, max_size = np.min(np.concatenate((size_w, size_l))), np.max(np.concatenate((size_w, size_l)))
    size_range = np.linspace(min_size - (0.1 * (max_size - min_size)), max_size + (0.1 * (max_size - min_size)), 500)

    min_EODf, max_EODf = np.min(np.concatenate((EODf_w, EODf_l))), np.max(np.concatenate((EODf_w, EODf_l)))
    EODf_range = np.linspace(min_EODf - (0.1 * (max_EODf - min_EODf)), max_EODf + (0.1 * (max_EODf - min_EODf)), 500)

    win_size_conv = np.zeros(len(size_range))
    win_EODf_conv = np.zeros(len(EODf_range))
    lose_size_conv = np.zeros(len(size_range))
    lose_EODf_conv = np.zeros(len(EODf_range))

    for s in size_w:
        win_size_conv += gauss(size_range, s, 0.5, 2, norm=True)
    win_size_conv /= len(size_w)

    for s in size_l:
        lose_size_conv += gauss(size_range, s, 0.5, 2, norm=True)
    lose_size_conv /= len(size_l)

    for f in EODf_w:
        win_EODf_conv += gauss(EODf_range, f, 10, 0.1, norm=True)
    win_EODf_conv /= len(EODf_w)

    for f in EODf_l:
        lose_EODf_conv += gauss(EODf_range, f, 10, 0.1, norm=True)
    lose_EODf_conv /= len(EODf_l)

    ax[2].plot(size_range, win_size_conv, lw=2, color='k')
    ax[2].plot(size_range, lose_size_conv, lw=2, color='grey')

    ax[3].plot(EODf_range, win_EODf_conv, lw=2, color='k')
    ax[3].plot(EODf_range, lose_EODf_conv, lw=2, color='grey')

    for a in ax[:2]:
        a.set_yticks([0, 1])
        a.set_yticklabels(['Lose', 'Win'])
        a.tick_params(labelsize=10)
    ax[0].set_xlabel('size [cm]', fontsize=12)
    ax[1].set_xlabel('EODf [Hz]', fontsize=12)

    ax[2].set_ylim(bottom=0)
    ax[3].set_ylim(bottom=0)
    ax[2].set_xlim(size_range[0], size_range[-1])
    ax[3].set_xlim(EODf_range[0], EODf_range[-1])
    ax[2].set_axis_off()
    ax[3].set_axis_off()

    plt.setp(ax[1].get_yticklabels(), visible=False)

def main():
    wl_sex_EODf_size = np.load('wl_sex_EODf_size.npy')
    sex_w, sex_l, EODf_w, EODf_l, size_w, size_l = wl_sex_EODf_size

    hist_bins= np.load('panel_a__hist_bins.npy')
    hist_n= np.load('panel_a__hist_n.npy')

    female_EODf = np.load('panel_a__female_EODf.npy')
    female_size = np.load('panel_a__female_size.npy')
    female_weight = np.load('panel_a__female_weight.npy')

    male_EODf= np.load('panel_a__male_EODf.npy')
    male_size= np.load('panel_a__male_size.npy')
    male_weight= np.load('panel_a__male_weight.npy')

    fig = plt.figure(figsize=(17.5 / 2.54, 8 / 2.54))

    gs = gridspec.GridSpec(2, 2, left=0.075, bottom=0.2, right=0.45, top=0.975, height_ratios=[1, 3], hspace=0, wspace=0.2)
    ax_tl = []
    ax_tl.append(fig.add_subplot(gs[0, 0]))
    ax_fake = fig.add_subplot(gs[0, 1])
    ax_tl.append(fig.add_subplot(gs[1, 0], sharex=ax_tl[0]))
    ax_tl.append(fig.add_subplot(gs[1, 1], sharey=ax_tl[1]))

    meta_dist(ax_tl, hist_bins, hist_n, female_EODf, female_size, female_weight, male_EODf, male_size, male_weight)

    gs_pic = gridspec.GridSpec(1, 1, bottom = 0.05, left = 0.5, top = 0.975, right=1.)
    ax_pic = fig.add_subplot(gs_pic[0, 0])

    setup = mpimg.imread('./setup.png')
    # ax_pic.imshow(setup)

    ax_pic.set_axis_off()
    ax_fake.set_axis_off()

    ax_tl[2].set_xlim(left=0)
    ax_tl[2].set_xticks([0, 10, 20])
    ax_tl[1].set_xticks([600, 750, 900])

    for a in ax_tl:
        a.tick_params(labelsize=10)
    plt.setp(ax_tl[2].get_yticklabels(), visible=False)

    fig.tag(axes=[ax_tl[0], ax_fake], labels = ['A', 'B'], fontsize=15, yoffs=3, xoffs=-3)
    fig.tag(axes=ax_pic, labels='C', fontsize=15, yoffs=2, xoffs=0)

    # plt.savefig('/setup_fish_meta_blanc.pdf')
    plt.savefig('setup_fish_meta.pdf')
    plt.show()
    quit()


    fig = plt.figure(figsize=(17.5 / 2.54, 7 / 2.54))
    gs = gridspec.GridSpec(2, 2, left=0.1, bottom=0.2, right=0.95, top=0.85, height_ratios=[1, 3], hspace=0, wspace=0.3)
    ax = []
    ax.append(fig.add_subplot(gs[1, 0]))
    ax.append(fig.add_subplot(gs[1, 1]))

    ax.append(fig.add_subplot(gs[0, 0], sharex=ax[0]))
    ax.append(fig.add_subplot(gs[0, 1], sharex=ax[1]))

    wl_meta_dist(ax, sex_w, sex_l, EODf_w, EODf_l, size_w, size_l)

    for a in ax:
        a.tick_params(labelsize=10)

    fig.tag(axes=ax[2:], labels = ['A', 'B'], fontsize=15, yoffs=0, xoffs=-3)

    female_color, male_color = '#e74c3c', '#3498db'
    y0, y1 = ax[1].get_ylim()
    ax[1].plot(700, 2, 'p', markeredgecolor='k', color='k', label='Winner')
    ax[1].plot(700, 2, 'o', color='grey', label='Loser')
    ax[1].plot(700, 2, 's', color=male_color, label=u'\u2642')
    ax[1].plot(700, 2, 's', color=female_color, label=u'\u2640')
    ax[1].plot(700, 2, 's', color='lightgrey', markeredgecolor='k', label='same-sex')
    ax[1].plot(700, 2, 's', color='lightgrey', label='mixed-sex')
    ax[1].legend(loc='upper center', frameon=False, bbox_to_anchor=(-0.125, 1.7), ncol=3, fontsize=9)
    ax[1].set_ylim(y0, y1)

    plt.savefig('wl_size_EODf.pdf')
    plt.savefig('wl_size_EODf.pdf')

    plt.show()
    embed()
    quit()


if __name__ == '__main__':
    main()
