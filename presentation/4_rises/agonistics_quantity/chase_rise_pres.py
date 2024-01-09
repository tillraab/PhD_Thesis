import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as scp
from IPython import embed

def main():
    def plot_agon_rc(mm_agon, ff_agon, mf_agon, fm_agon, mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc, ax):

        female_color, male_color = '#e74c3c', '#3498db'
        win_color = [male_color, female_color, male_color, female_color]
        lose_color = [male_color, female_color, female_color, male_color]

        mek = ['k', 'k', None, None]

        for enu, rc, agons in zip(np.arange(4), [mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc],
                                  [mm_agon, ff_agon, mf_agon, fm_agon]):
            ax.plot(rc, agons, 'p', color=win_color[enu], markeredgecolor=mek[enu], markersize=8,
                    alpha=0.8, zorder=1)

        x_vals = np.hstack([mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc])
        x, y = np.hstack([mm_lose_rc, ff_lose_rc, mf_lose_rc, fm_lose_rc]), np.hstack(
            [mm_agon, ff_agon, mf_agon, fm_agon])
        x, y = x[~np.isnan(y)], y[~np.isnan(y)]
        m, b, _, _, _ = scp.linregress(x, y)
        XX = np.array([np.min(x_vals), np.max(x_vals)])
        ax.plot(XX, m * XX + b, color='k', lw=2)

    win_rc = np.load('../win_rc.npy', allow_pickle=True)
    lose_rc = np.load('../lose_rc.npy', allow_pickle=True)

    agonistics = np.load('../agonistics.npy', allow_pickle=True)
    contact = np.load('../contact.npy', allow_pickle=True)
    agonistic_dur = np.load('../agonistic_dur.npy', allow_pickle=True)

    dsize_win = np.load('../dsize_win.npy', allow_pickle=True)
    df_win = np.load('../df_win.npy', allow_pickle=True)
    lose_exp = np.load('../lose_exp.npy', allow_pickle=True)
    win_exp = np.load('../win_exp.npy', allow_pickle=True)
    # embed()
    # quit()

    female_color, male_color = '#e74c3c', '#3498db'

    med_agonstic_dur = []
    for i in range(len(agonistic_dur)):
        med_agonstic_dur.append([])
        for j in range(len(agonistic_dur[i])):
            med_agonstic_dur[-1].append(np.nanmedian(agonistic_dur[i][j]))
        med_agonstic_dur[-1] = np.array(med_agonstic_dur[-1])
    med_agonstic_dur = np.array(med_agonstic_dur)

    fs = 12
    fig = plt.figure(figsize=(10 / 2.54, 10 * (12 / 20) / 2.54))
    gs = gridspec.GridSpec(1, 1, left=0.175, bottom=0.225, top=0.975, right=0.975)
    ax = fig.add_subplot(gs[0, 0])

    plot_agon_rc(*med_agonstic_dur, *lose_rc, ax)

    ax.set_xlabel('EODf rises [n]', fontsize=fs+2)
    ax.set_ylabel('chase\nduration [s]', fontsize=fs+2)
    ax.tick_params(labelsize=fs)

    ax.set_xlim(left=0)
    ax.set_ylim(1, 9.5)
    ax.set_yticks(np.arange(2, 8.1, 2))

    plt.savefig('chasedur_rises_pres.jpg', dpi=300)
    plt.show()
    pass

if __name__ == '__main__':
    main()