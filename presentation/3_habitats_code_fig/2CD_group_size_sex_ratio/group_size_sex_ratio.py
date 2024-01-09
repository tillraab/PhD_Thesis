import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import embed

def load_files():
    day_ratio = np.load('day_ratio.npy', allow_pickle=True)
    night_ratio = np.load('night_ratio.npy', allow_pickle=True)

    male_in_group = np.load('male_in_gr.npy', allow_pickle=True)
    female_in_group = np.load('female_in_gr.npy', allow_pickle=True)

    return day_ratio, night_ratio, male_in_group, female_in_group

def main():
    fs = 12

    day_ratio, night_ratio, male_in_group, female_in_group = load_files()

    fig = plt.figure(figsize=(14/2.54, 14 * (12/20)/2.54))
    gs = gridspec.GridSpec(1, 1, left=0.125, bottom=0.25, right=0.975, top=0.975)
    ax = fig.add_subplot(gs[0, 0])

    ax.errorbar(np.arange(len(male_in_group[0])) - .1, male_in_group[0], yerr=male_in_group[1], fmt='none', ecolor='k')
    ax.bar(np.arange(len(male_in_group[0])) - .1, male_in_group[0], width=.2, color='firebrick', label=u'\u2642')

    ax.errorbar(np.arange(len(female_in_group[0])) + .1, female_in_group[0], yerr=female_in_group[1], fmt='none', ecolor='k')
    ax.bar(np.arange(len(female_in_group[0])) + .1, female_in_group[0], width=.2, color='#F47F17', label=u'\u2642')

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(['st. stones', 'iso. stones', 'grass', 'gravel', 'water'], rotation = 45)
    ax.tick_params(labelsize=fs)
    ax.set_ylabel('group size', fontsize=fs+2)
    ax.legend(loc=1, fontsize=fs-2, frameon=False)

    plt.savefig('./group_sizes.png', dpi=300)


    ####################################################################################################################

    fig = plt.figure(figsize=(14/2.54, 14 * (12/20)/2.54))
    gs = gridspec.GridSpec(1, 1, left=0.125, bottom=0.25, right=0.975, top=0.975)
    ax = fig.add_subplot(gs[0, 0])

    ax.errorbar(np.arange(len(day_ratio[0])) - .1, day_ratio[0], yerr=day_ratio[1], fmt='none', ecolor='k')
    ax.bar(np.arange(len(day_ratio[0])) - .1, day_ratio[0], width=.2, color='cornflowerblue', label='day')

    ax.errorbar(np.arange(len(night_ratio[0])) + .1, night_ratio[0], yerr=night_ratio[1], fmt='none', ecolor='k')
    ax.bar(np.arange(len(night_ratio[0])) + .1, night_ratio[0], width=.2, color='#888888', label='night')

    ax.plot([-0.5, 4.5], [6/14, 6/14], '--', lw=1, color='k')

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(['st. stones', 'iso. stones', 'grass', 'gravel', 'water'], rotation = 45)
    ax.tick_params(labelsize=fs)
    ax.set_ylabel('male ratio', fontsize=fs+2)
    ax.legend(loc=1, fontsize=fs-2, frameon=False)

    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, 4.5)
    plt.savefig('./sex_ratio.png', dpi=300)


    plt.show()
    # embed()
    # quit()

if __name__ == '__main__':
    main()