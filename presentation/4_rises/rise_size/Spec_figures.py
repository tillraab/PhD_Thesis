from thunderfish.dataloader import open_data
from thunderfish.powerspectrum import decibel
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from IPython import embed
import matplotlib.gridspec as gridspec

def load(folder):
    fund_v = np.load(os.path.join(folder, 'fund_v.npy'))
    ident_v = np.load(os.path.join(folder, 'ident_v.npy'))
    idx_v = np.load(os.path.join(folder, 'idx_v.npy'))
    times = np.load(os.path.join(folder, 'times.npy'))
    spec = np.load(os.path.join(folder, 'spec.npy'))

    return fund_v, ident_v, idx_v, times, spec

def get_fine_spec(folder):
    # np.memmap(string, dtype='float', mode='r', shape=(len freq, len time), order='F')
    spec_shape = np.load(os.path.join(folder, 'fill_spec_shape.npy'))
    fine_times = np.load(os.path.join(folder, 'fill_times.npy'))
    fine_times = fine_times[:spec_shape[1]]
    fine_freqs = np.load(os.path.join(folder, 'fill_freqs.npy'))
    fine_spec = np.memmap(os.path.join(folder, 'fill_spec.npy'), dtype='float', mode='r+', shape=(spec_shape[0], spec_shape[1]), order='F')

    return fine_spec, fine_freqs, fine_times

def plot_rises(ax):
    folder = '/home/raab/data/2019_tube_competition/2020-05-28-10_00'
    fine_spec, fine_freq, fine_t = get_fine_spec(folder)

    for enu, a in enumerate(ax):
        if enu == 0:
            min_t = (3*60 + 11) * 60
            dt = 90
            min_f = 679
            max_f = 740
        else:
            min_t = (1*60 + 33) * 60
            dt = 220
            min_f = 700
            max_f = 775

        i00 = np.arange(len(fine_freq))[fine_freq >= min_f][0]
        i01 = np.arange(len(fine_freq))[fine_freq <= max_f][-1]
        i10 = np.arange(len(fine_t))[fine_t >= min_t][0]
        i11 = np.arange(len(fine_t))[fine_t <= min_t + dt][-1]

        if enu == 0:
            np.save('rise_spec1.npy', fine_spec[i00: i01, i10:i11])
            np.save('rise_spec1_extent.npy', np.array([fine_t[i10] - fine_t[i10], fine_t[i11] - fine_t[i10], fine_freq[i00], fine_freq[i01]]))
        else:
            np.save('rise_spec2.npy', fine_spec[i00: i01, i10:i11])
            np.save('rise_spec2_extent.npy', np.array([fine_t[i10] - fine_t[i10], fine_t[i11] - fine_t[i10], fine_freq[i00], fine_freq[i01]]))

        a.imshow(decibel(fine_spec[i00: i01, i10:i11][::-1]),
                 extent=[fine_t[i10] - fine_t[i10], fine_t[i11] - fine_t[i10], fine_freq[i00], fine_freq[i01]],
                 aspect='auto', alpha=0.7, cmap='jet', interpolation='gaussian', vmin=-120, vmax=-50)
        a.set_xlim([fine_t[i10] - fine_t[i10], fine_t[i11] - fine_t[i10]])
        a.set_ylim(fine_freq[i00], fine_freq[i01])
def main():
    # ToDo: adapt the path after you mounted the colombia2016 data using sshfs !!!
    # folder = '/home/raab/data/2019_tube_competition/2020-05-28-10_00'
    # folder = '/home/raab/data/2016-colombia/2016-04-14-19_12'
    # folder2 = '/home/raab/data/2016-colombia/2016-04-09-22_25/'

    # for enu, f in enumerate([folder, folder2]):
    #     fig, ax = plt.subplots(figsize=(20/2.54, 12/2.54), facecolor='white')
    #
    #     fund_v, ident_v, idx_v, times, spec = load(f)
    # fine_spec, fine_freq, fine_t = get_fine_spec(folder)

    fig = plt.figure(figsize=(20/2.54, 8/2.54))
    gs = gridspec.GridSpec(1, 2, bottom = 0.2, left=.1, right=0.9, top=0.9)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[0, 1]))

    plot_rises(ax)
    # for enu, a in enumerate(ax):
    #     if enu == 0:
    #         min_t = (3*60 + 11) * 60
    #         dt = 90
    #         min_f = 680
    #         max_f = 740
    #     else:
    #         min_t = (1*60 + 33) * 60
    #         dt = 220
    #         min_f = 700
    #         max_f = 775
    #
    #     i00 = np.arange(len(fine_freq))[fine_freq >= min_f][0]
    #     i01 = np.arange(len(fine_freq))[fine_freq <= max_f][-1]
    #     i10 = np.arange(len(fine_t))[fine_t >= min_t][0]
    #     i11 = np.arange(len(fine_t))[fine_t <= min_t + dt][-1]
    #
    #     a.imshow(decibel(fine_spec[i00: i01, i10:i11][::-1]), extent=[fine_t[i10] - fine_t[i10], fine_t[i11] - fine_t[i10], fine_freq[i00], fine_freq[i01]], aspect='auto', alpha=0.7, cmap='jet', interpolation='gaussian')

    ax[0].set_xlabel('time [s]', fontsize=12)
    ax[1].set_xlabel('time [s]', fontsize=12)
    ax[0].set_ylabel('EOD frequency [Hz]', fontsize = 12)

    for a in ax:
        a.tick_params(labelsize=10)

    plt.show()


if __name__ == '__main__':
    main()