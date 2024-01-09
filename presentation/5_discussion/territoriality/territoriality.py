import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import embed
import glob



def main():
    files = ['./fish_combo_data_0.npy']
    # files = glob.glob('/home/raab/code/dfg_grand/fish_combo_data*.npy')
    # spo = [0, 3, 1, 4, 2, 5]

    dates = ['09.04', '10.04.', '10.04.', '11.04.', '11.04.', '12.04.', '12.04.', '13.04.', '13.04.', '14.04.',
             '14.04.', '15.04.', '15.04.', '16.04.', '16.04.', '17.04.', '17.04.']

    day_night_switch_m = np.array([     0.,    455.,   1175.,   1895.,   2615.,   3335.,   4055., 4775.,   5495.,
                                        6215.,   6935.,   7655.,   8375.,   9095., 9815.,  10535.,  11255.,  11975.,
                                        12695.])

    for file in files:
        time, freq, x_pos, y_pos = np.load(file, allow_pickle=True)
        time = np.array(time)
        freq = np.array(freq)
        x_pos = np.array(x_pos)
        y_pos = np.array(y_pos)

        panels = len(day_night_switch_m[day_night_switch_m < time[-1]/60])

        fig = plt.figure(figsize=(20/2.54, 12/2.54))
        gs = gridspec.GridSpec(2, (panels+1)//2, left=0.1, bottom=0.1, right=0.95, top=0.95)
        ax = []
        for i in range((panels+1)//2):
            ax.append(fig.add_subplot(gs[0, i]))
            ax.append(fig.add_subplot(gs[1, i]))

        for enu in range(panels):
            c_xpos = x_pos[(time / 60 >= day_night_switch_m[enu]) & (time / 60 < day_night_switch_m[enu + 1])]
            c_ypos = y_pos[(time / 60 >= day_night_switch_m[enu]) & (time / 60 < day_night_switch_m[enu + 1])]
            # c_time = time[(time / 60 >= day_night_switch_m[enu]) & (time / 60 < day_night_switch_m[enu + 1])]

            H, xedges, yedges = np.histogram2d(c_xpos, c_ypos, bins=(np.arange(9) - 0.5, np.arange(9) - 0.5))
            H_turned = H.T

            ax[enu].imshow(H_turned[::-1] / np.max(H_turned), interpolation='gaussian', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=0, vmax=1)
            ax[enu].set_xlim([0, 7])
            ax[enu].set_ylim([0, 7])

            X, Y = np.meshgrid(xedges[:-1] + (xedges[1] - xedges[0]) / 2, yedges[:-1] + (yedges[1] - yedges[0]) / 2)

            ax[enu].contour(X, Y, H_turned / np.max(H_turned), levels=[.25, .5], colors=['orange', 'firebrick'], alpha=0.7)
            ax[enu].set_xticks([0, 2, 4, 6])
            if enu % 2 != 0:
                ax[enu].set_xticklabels([0, 1, 2, 3])
            else:
                ax[enu].set_xticklabels([])
            ax[enu].set_yticks([0, 2, 4, 6])
            ax[enu].set_yticklabels([0, 1, 2, 3])

            p = len(np.hstack(H_turned)[np.hstack(H_turned >= np.max(H_turned) * 0.1)]) * ((xedges[1] - xedges[0]) / 2) ** 2
            ax[enu].text(0.25, 0.25, r'%.2f$m^2$' % p, color='orange')

        if len(ax) > panels:
            ax[-1].set_visible(False)

        ax[0].set_ylabel(r'$\bf{night}$' + '\ny [m]', fontsize=10)
        ax[1].set_ylabel(r'$\bf{day}$' + '\ny [m]', fontsize=10)
        ax[1].set_xlabel('x [m]', fontsize=10)
        ax[3].set_xlabel('x [m]', fontsize=10)
        ax[5].set_xlabel('x [m]', fontsize=10)

        ax[0].set_title(r'$\bf{%s}$' % dates[1], fontsize=10)
        ax[2].set_title(r'$\bf{%s}$' % dates[3], fontsize=10)
        ax[4].set_title(r'$\bf{%s}$' % dates[5], fontsize=10)

        plt.savefig('territoriality.jpg', dpi=300)
        plt.show()

    embed()
    quit()
if __name__ == '__main__':
    main()
