import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

def loaddata(datafile):
    fund_v=np.load(datafile+"/fund_v.npy")
    ident_v=np.load(datafile+"/ident_v.npy")
    idx_v=np.load(datafile+"/idx_v.npy")
    times=np.load(datafile+"/times.npy")
    sign_v=np.load(datafile+"/sign_v.npy")
    times_v=times[idx_v]
    return fund_v,ident_v,idx_v,times_v,sign_v

def load_files():
    datafiles = np.load('datafiles.npy', allow_pickle=True)
    file_time_shifts = np.load('file_time_shift.npy', allow_pickle=True)
    dn_borders = np.load('dn_borders.npy', allow_pickle=True)
    fish_colos= np.load('fish_colors.npy', allow_pickle=True)
    fish_nr_in_rec = np.load('fish_nr_in_rec.npy', allow_pickle=True)
    return datafiles, file_time_shifts, fish_colos, dn_borders, fish_nr_in_rec

def main():
    datafiles, file_time_shifts, fish_colos, dn_borders, fish_nr_in_rec = load_files()

    fs = 12
    fig = plt.figure(figsize=(20/2.54, 12/2.54))
    gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.135, right=0.975, top=0.975)
    ax = fig.add_subplot(gs[0, 0])
    last_t = 0

    for datei_nr in tqdm(np.arange(len(datafiles))):
        fund_v, ident_v, idx_v, times_v, sign_v = loaddata(datafiles[datei_nr])
        times_v += file_time_shifts[datei_nr]
        last_t = times_v[-1]

        for fish_nr in range(len(fish_nr_in_rec)):
            if np.isnan(fish_nr_in_rec[fish_nr][datei_nr]):
                continue
            p = sign_v[ident_v == fish_nr_in_rec[fish_nr][datei_nr]][:, 0]
            f = fund_v[ident_v == fish_nr_in_rec[fish_nr][datei_nr]]
            t = times_v[ident_v == fish_nr_in_rec[fish_nr][datei_nr]]

            ax.plot(t[~np.isnan(p)], f[~np.isnan(p)], color=fish_colos[fish_nr])

    for ns, ne in zip(dn_borders[::2], dn_borders[1::2]):
        ax.fill_between([ns, ne], [650, 650], [970, 970], color='#888888')
    ax.set_yticks([700, 800, 900])

    # ax.set_xlim([dn_borders[0], dn_borders[dn_borders < last_t][-1]])
    ax.set_xlim([dn_borders[0], last_t])
    ax.set_ylim([650, 970])

    ax.set_ylabel('frequency [Hz]', fontsize = fs + 2)
    # ax.set_xlabel('Datum')

    time_ticks = np.arange(110 * 60 + 18 * 60 * 60, last_t, 24 * 60 * 60)
    ax.set_xticklabels([])
    ax.set_xticks(time_ticks)
    ax.set_xticklabels(['day 1', 'day 2', 'day 3', 'day 4', 'day 5', 'day 6', 'day 7', 'day 8', 'day 9', 'day 10'], rotation=45, ha='right')
    ax.tick_params(labelsize=fs)

    plt.savefig('./habitat_traces_1C.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()