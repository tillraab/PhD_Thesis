import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import embed

def load_files():
    day_x = np.load('day_x.npy', allow_pickle=True)
    day_y = np.load('day_y.npy', allow_pickle=True)
    night_x = np.load('night_x.npy', allow_pickle=True)
    night_y= np.load('night_y.npy', allow_pickle=True)

    return day_x, day_y, night_x, night_y

def main():
    day_x, day_y, night_x, night_y = load_files()
    hab_colors = ['k', 'grey', 'green', 'yellow', 'lightblue']
    last_t = 876186.1349 # this is so dirty...
    fs = 12

    fig = plt.figure(figsize=(20/2.54, 12/2.54))
    gs = gridspec.GridSpec(2, 1, left=0.1, bottom=0.1, right=0.95, top=0.95, hspace=0)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[1, 0]))

    for i in range(len(day_y)-1):
        ax[0].fill_between(day_x, day_y[i], day_y[i+1], color=hab_colors[i])
        ax[1].fill_between(night_x, night_y[i], night_y[i+1], color=hab_colors[i])

    time_ticks = np.arange(110 * 60 + 18 * 60 * 60, last_t, 24*60*60)

    ax[0].plot([time_ticks[0] - 18 * 60 * 60, time_ticks[-1] + 6 * 60 * 60], [0, 0], color='white', lw = .5, clip_on=False, zorder = 2)
    ax[1].plot([time_ticks[0] - 18 * 60 * 60, time_ticks[-1] + 6 * 60 * 60], [0, 0], color='white', lw = .5, clip_on=False, zorder = 2)

    ax[0].text(time_ticks[0] - 12 * 60 * 60, .5, 'day', fontsize=fs, va='center', ha='center', rotation=90)
    ax[1].text(time_ticks[-1], .5, 'night', fontsize=fs, va='center', ha='center', rotation=90)

    ax[0].text(time_ticks[1], 0.1, 'stacked stones', fontsize=fs-2, color='white', ha='center', va='center')
    ax[0].text(time_ticks[1], 0.6, 'plants', fontsize=fs-2, color='k', ha='center', va='center')
    ax[1].text(time_ticks[1], 0.825, 'open water', fontsize=fs-2, color='k', ha='center', va='center')
    ax[1].text(time_ticks[6], 0.75, 'gravel', fontsize=fs-2, color='k', ha='center', va='center')
    ax[1].text(time_ticks[6], 0.3, 'isolated stones', fontsize=fs-2, color='k', ha='center', va='center')

    ax[0].set_xticks([])

    ax[1].set_xticks(time_ticks)
    ax[1].set_xticklabels(['day 1', 'day 2', 'day 3', 'day 4', 'day 5', 'day 6', 'day 7', 'day 8', 'day 9', 'day 10'])

    ax[0].set_yticks([0, .5, 1])
    ax[0].set_yticklabels(['0', '.5', '1'])
    ax[0].set_ylim([0, 1])

    ax[1].set_yticks([.5, 1])
    ax[1].set_yticklabels(['.5', '1'])
    ax[1].set_ylim([0, 1])

    ax[0].spines['bottom'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].invert_yaxis()

    ax[0].set_xlim(night_x[0], day_x[-1])
    ax[1].set_xlim(night_x[0], day_x[-1])

    plt.savefig('./habitat_occupation_1E.pdf')
    plt.show()

    ######################

    fig = plt.figure(figsize=(17/2.54, 17 * (10/20) /2.54))
    gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.175, right=0.995, top=0.975)
    ax = fig.add_subplot(gs[0, 0])

    for i in range(len(day_y)-1):
        ax.fill_between(day_x, day_y[i], day_y[i+1], color=hab_colors[i])

    time_ticks = np.arange(110 * 60 + 18 * 60 * 60, last_t, 24*60*60)

    # ax[0].text(time_ticks[0] - 12 * 60 * 60, .5, 'day', fontsize=fs, va='center', ha='center', rotation=90)
    ax.text(time_ticks[1], 0.1, 'stacked stones', fontsize=fs-2, color='white', ha='center', va='center')
    ax.text(time_ticks[1], 0.65, 'plants', fontsize=fs-2, color='k', ha='center', va='center')
    ax.text(time_ticks[1], 0.95, 'open water', fontsize=fs-2, color='k', ha='center', va='center')
    ax.text(time_ticks[6], 0.9, 'gravel', fontsize=fs-2, color='k', ha='center', va='center')
    ax.text(time_ticks[6], 0.35, 'isolated stones', fontsize=fs-2, color='k', ha='center', va='center')

    ax.set_xticks(time_ticks)
    ax.set_xticklabels(['day 1', 'day 2', 'day 3', 'day 4', 'day 5', 'day 6', 'day 7', 'day 8', 'day 9', 'day 10'], rotation=45, ha='right')

    ax.set_yticks([0, .5, 1])
    ax.set_ylim([0, 1])

    ax.set_xlim(day_x[0], day_x[-1])

    ax.set_ylabel('rel. fish density', fontsize=fs+2)
    ax.tick_params(labelsize=fs)

    plt.savefig('rel_fish_density.jpg', dpi=300)

    plt.show()
if __name__ == '__main__':
    main()