import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import scipy.stats as scp
from matplotlib.lines import Line2D
from IPython import embed

def main():
    fs = 12
    colors = ['#BA2D22', '#53379B', '#F47F17', '#3673A4', '#AAB71B', '#DC143C', '#1E90FF', 'k']

    _, day_mean, day_std, night_mean, night_std = np.array(np.load('fish_transition_combo.npy')).T
    freq = np.load('freqs.npy', allow_pickle=True)

    day_mean /= 1000
    day_std /= 1000
    night_mean /= 1000
    night_std /= 1000

    female_color, male_color = '#e74c3c', '#3498db'

    ####################################################################################################################

    fig = plt.figure(figsize=(15/2.54, 15 * (12/20)/2.54))
    gs = gridspec.GridSpec(1, 1, left=0.125, bottom=0.175, right=0.95, top=0.925)
    ax = fig.add_subplot(gs[0, 0])

    # ax.errorbar(day_mean[:6], night_mean[:6], xerr=day_std[:6], yerr=night_std[:6], color='k',
    #             marker='o', markersize=5, fmt='o')

    # slope, intercept, _, _, _ = scp.linregress(day_mean[:6], night_mean[:6])
    # ax.plot(np.array([0.5, 2.5]), slope * np.array([0.5, 2.5]) + intercept, lw=2, color=male_color)

    if True:
        ax.errorbar(day_mean[:6], night_mean[:6], xerr=day_std[:6], yerr=night_std[:6], color=male_color,
                    marker='o', markersize=5, fmt='o', alpha=0.25)

        ax.annotate("", xy=(0.5, 12.5), xytext=(2.5, 5.5), arrowprops=dict(arrowstyle='-|>',
                                                                           color=male_color,
                                                                           lw=3,
                                                                           mutation_scale=20))
        ax.text(1.6, 9.25, 'EODf', fontsize=fs, rotation=-35, ha='center', va='center', color=male_color, fontweight='bold')

    # ax.errorbar(day_mean[6:], night_mean[6:], xerr=day_std[6:], yerr=night_std[6:], color='k', marker='o', markersize=5, fmt='o')

    # slope, intercept, _, _, _ = scp.linregress(day_mean[6:], night_mean[6:])
    # ax.plot(np.array([0.5, 3.5]), slope * np.array([0.5, 3.75]) + intercept, lw=2, color=female_color)

    if True:
        ax.errorbar(day_mean[6:], night_mean[6:], xerr=day_std[6:], yerr=night_std[6:], color=female_color,
                    marker='o', markersize=5, fmt='o', alpha=0.25)

        ax.annotate("", xy=(3.5, 10), xytext=(0.75, 1.5), arrowprops=dict(arrowstyle='-|>',
                                                                           color=female_color,
                                                                           lw=3,
                                                                           mutation_scale=20))
        ax.text(1.9, 5.75, 'EODf', fontsize=fs, rotation=33, ha='center', va='center', color=female_color, fontweight='bold')


    ax.plot([0, 5], [0, 5], '--', lw=1, color='k')
    ax.text(-0.05, 1.075, r'$\times$1000', fontsize=fs-2, transform=ax.transAxes, ha='center', va='center')
    ax.text(.99, -.15, r'$\times$1000', fontsize=fs-2, transform=ax.transAxes, ha='center', va='center')

    ax.set_xlabel(r'transitions$_{day}$', fontsize=fs+2)
    ax.set_ylabel(r'transitions$_{night}$', fontsize=fs+2)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 14)

    ax.tick_params(labelsize=fs)
    legend_elements = [Patch(facecolor=male_color, edgecolor='none', label=u'\u2642'),
                       Patch(facecolor=female_color, edgecolor='none'  , label=u'\u2640')]

    ax.legend(handles=legend_elements, loc=1, frameon=False, fontsize=10, ncol=2, bbox_to_anchor=(1, 1.125))

    plt.savefig('./transitions_dn_arrow.pdf')

    # plt.show()
    # quit()
    plt.close()


    ####################################################################################################################

    fig = plt.figure(figsize=(15/2.54, 15 * (12/20)/2.54))
    gs = gridspec.GridSpec(2, 1, left=0.125 , bottom=0.15, right=0.975, top=0.925)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[1, 0]))

    ax[0].errorbar(freq[:6], night_mean[:6], yerr=night_std[:6], color=male_color, fmt='o')
    ax[0].errorbar(freq[6:], night_mean[6:], yerr=night_std[6:], color=female_color, fmt='o')

    slope, intercept, _, _, _ = scp.linregress(freq[:6], night_mean[:6])
    ax[0].plot(freq[:6], slope * freq[:6] + intercept, lw=2, color=male_color)
    slope, intercept, _, _, _ = scp.linregress(freq[6:], night_mean[6:])
    ax[0].plot(freq[6:], slope * freq[6:] + intercept, lw=2, color=female_color)

    ax[1].errorbar(freq[:6], day_mean[:6], yerr=day_std[:6], color=male_color, fmt='o')
    ax[1].errorbar(freq[6:], day_mean[6:], yerr=day_std[6:], color=female_color, fmt='o')

    slope, intercept, _, _, _ = scp.linregress(freq[:6], day_mean[:6])
    ax[1].plot(freq[:6], slope * freq[:6] + intercept, lw=2, color=male_color)
    slope, intercept, _, _, _ = scp.linregress(freq[6:], day_mean[6:])
    ax[1].plot(freq[6:], slope * freq[6:] + intercept, lw=2, color=female_color)

    ax[0].set_yticks(np.arange(0, 13, 4))
    ax[1].set_xticks(np.arange(700, 851, 50))
    ax[0].set_xticks(np.arange(700, 851, 50))
    ax[0].set_xticklabels([])
    ax[0].text(-0.03, 1.1, r'$\times$1000', fontsize=fs-2, transform=ax[0].transAxes, ha='center', va='center')

    ax[1].set_xlabel('EODf [Hz]', fontsize = fs+2)
    ax[0].set_ylabel(r'transitions$_{night}$', fontsize = fs+2)
    ax[1].set_ylabel(r'transitions$_{day}$', fontsize = fs+2)
    ax[0].set_ylim(bottom=0)
    ax[1].set_ylim(bottom=0)
    fig.align_ylabels(ax)

    # ax[1].text(0.99, 0.95, 'day', color='cornflowerblue', fontsize=fs+2, ha='right', va='top', transform=ax[1].transAxes)
    # ax[0].text(0.99, 0.05, 'night', color='#888888', fontsize=fs+2, ha='right', va='bottom', transform=ax[0].transAxes)

    ax[0].tick_params(labelsize=fs)
    ax[1].tick_params(labelsize=fs)

    legend_elements = [Patch(facecolor=male_color, edgecolor='none', label=u'\u2642'),
                       Patch(facecolor=female_color, edgecolor='none'  , label=u'\u2640')]
    ax[0].legend(handles=legend_elements, loc=1, frameon=False, fontsize=10, ncol=2, bbox_to_anchor=(1, 1.25))


    plt.savefig('./transitions_freq.pdf')
    plt.show()


if __name__ == '__main__':
    main()