import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import scipy.stats as scp

def main():
    def contact_rises():
        fs = 12
        fig = plt.figure(figsize=(10 / 2.54, 10 * (12 / 20) / 2.54))
        gs = gridspec.GridSpec(1, 1, left=0.15, bottom=0.225, top=0.975, right=0.975)
        ax = fig.add_subplot(gs[0, 0])
        ax_m = ax.twinx()

        b_mean, b_std, b_pct1, b_pct99, s_mean, s_std, s_pct1, s_pct99 = contact_m_s_pct
        b_mean, b_std, b_pct1, b_pct99, s_mean, s_std, s_pct1, s_pct99 = \
            b_mean * 19 * 60, b_std * 19 * 60, b_pct1 * 19 * 60, b_pct99 * 19 * 60, s_mean * 19 * 60, s_std * 19 * 60, s_pct1 * 19 * 60, s_pct99 * 19 * 60

        ax.plot(conv_t, b_mean, color='cornflowerblue', alpha=0.75)
        ax.fill_between(conv_t, b_pct1, b_pct99, color='cornflowerblue', alpha=0.5)
        ax.plot(conv_t, dt_contact_mass_conv * 19 * 60, color='firebrick', lw=2)
        for file_i in range(len(all_Ldt_contact)):
            ax_m.plot(all_Ldt_contact[file_i], np.ones(len(all_Ldt_contact[file_i])) * file_i + 0.5, '|', color='k',
                      alpha=0.2, markersize=6)
        ax.fill_between(conv_t, s_pct1, s_pct99, color='firebrick', alpha=0.5)

        ax.set_ylabel('rise rate [1/min]', fontsize=fs + 2)
        ax.set_xlabel(r'$\Delta$time [s]', fontsize=fs + 2)
        ax.tick_params(labelsize=fs)

        ax_m.set_yticks([])
        ax.set_ylim(0, 3.5)
        ax.set_yticks(np.arange(0, 3.1))

        ax.set_xlim(-30, 30)
        ax.set_xticks(np.arange(-30, 31, 15))
        ax.set_xticklabels(np.arange(-30, 31, 15))
        plt.savefig('contact_rises.jpg', dpi=300)


    def chontact_chase():
        fs = 12
        fig = plt.figure(figsize=(10 / 2.54, 10 * (12 / 20) / 2.54))
        gs = gridspec.GridSpec(1, 1, left=0.15, bottom=0.225, top=0.975, right=0.975)
        ax = fig.add_subplot(gs[0, 0])
        ax_m = ax.twinx()

        b_mean, b_std, b_pct1, b_pct99, s_mean, s_std, s_pct1, s_pct99 = ag_on_m_s_pct
        b_mean, b_std, b_pct1, b_pct99, s_mean, s_std, s_pct1, s_pct99 = \
            b_mean * 19 * 60, b_std * 19 * 60, b_pct1 * 19 * 60, b_pct99 * 19 * 60, s_mean * 19 * 60, s_std * 19 * 60, s_pct1 * 19 * 60, s_pct99 * 19 * 60

        ax.plot(conv_t, b_mean, color='cornflowerblue', alpha=0.75)
        ax.fill_between(conv_t, b_mean + b_std, b_mean - b_std, color='cornflowerblue', alpha=0.5)
        ax.plot(conv_t, dt_ag_on_mass_conv * 19 * 60, color='darkorange', lw=2)
        for file_i in range(len(all_Ldt_ag_on)):
            ax_m.plot(all_Ldt_ag_on[file_i], np.ones(len(all_Ldt_ag_on[file_i])) * file_i + 0.5, '|', color='k',
                      alpha=0.2, markersize=4)

        ax.fill_between(conv_t, s_mean + s_std, s_mean - s_std, color='darkorange', alpha=0.5)

        ax.set_ylabel('rise rate [1/min]', fontsize=fs + 2)
        ax.set_xlabel(r'$\Delta$time [s]', fontsize=fs + 2)
        ax.tick_params(labelsize=fs)

        ax_m.set_yticks([])
        ax.set_ylim(0, 2)
        ax.set_yticks(np.arange(0, 2.1))

        ax.set_xlim(-30, 30)
        ax.set_xticks(np.arange(-30, 31, 15))
        ax.set_xticklabels(np.arange(-30, 31, 15))
        plt.savefig('chase_rises.jpg', dpi=300)


    contact_m_s_pct = np.load('./contact_m_s_pct.npy', allow_pickle=True)
    ag_on_m_s_pct = np.load('./ag_on_m_s_pct.npy', allow_pickle=True)
    all_Ldt_contact = np.load('./all_Ldt_contact.npy', allow_pickle=True)
    all_Ldt_ag_on = np.load('./all_Ldt_ag_on.npy', allow_pickle=True)
    max_dt = np.load('./max_dt.npy', allow_pickle=True)
    dt_contact_mass_conv = np.load('./dt_contact_mass_conv.npy', allow_pickle=True)
    dt_ag_on_mass_conv = np.load('./dt_ag_on_mass_conv.npy', allow_pickle=True)
    conv_t = np.load('./conv_t.npy', allow_pickle=True)

    event_counts = np.load('./event_counts.npy', allow_pickle=True)
    event_time_counts = np.load('./event_time_counts.npy', allow_pickle=True)

    contact_rises()

    chontact_chase()

    plt.show()
if __name__ == '__main__':
    main()