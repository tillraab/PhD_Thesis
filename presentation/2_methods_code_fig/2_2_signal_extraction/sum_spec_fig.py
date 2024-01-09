import sys
import os
import time
import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
from thunderfish.version import __version__
from thunderfish.powerspectrum import decibel, next_power_of_two, spectrogram
from thunderfish.dataloader import open_data, fishgrid_grids, fishgrid_spacings
from thunderfish.harmonics import harmonic_groups, fundamental_freqs
# from signal_tracker import freq_tracking_v5, plot_tracked_traces, Emit_progress
from thunderfish.eventdetection import hist_threshold
from plottools.tag import tag
import multiprocessing
from functools import partial
from IPython import embed

def load(filename):
    data = open_data(filename, -1, 60, 10)
    samplerate = data.samplerate
    channels = data.channels
    return data, samplerate, channels

def calc_specs(data, samplerate, idx0):

    core_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(core_count - 1)

    func = partial(spectrogram, ratetime=samplerate, freq_resolution=0.5, overlap_frac=0.9)
    a = pool.map(func, [data[idx0: idx0 + 180*samplerate, ch] for ch in range(64)])

    spectra = [a[channel][0] for channel in range(len(a))]
    spec_freqs = a[0][1]
    spec_times = a[0][2]
    pool.terminate()

    comb_spectra = np.sum(spectra, axis=0)

    return comb_spectra, spec_freqs, spec_times

    ####################

def main():
    data, samplerate, channels = load('/home/raab/data/2016-colombia/2016-04-10-11_12/traces-grid1.raw')
    idx0 = int(285 * 60 * samplerate) # min 285

    # comb_spectra, spec_freqs, spec_times = calc_specs(data, samplerate, idx0)
    comb_spectra = np.load('comb_spectra.npy', allow_pickle=True)
    spec_freqs = np.load('spec_freqs.npy', allow_pickle=True)
    spec_times = np.load('spec_times.npy', allow_pickle=True)


    # embed()
    # quit()
    fs = 12
    fig = plt.figure(figsize=(15/2.54, 15 * (12/20) / 2.54))
    gs = gridspec.GridSpec(1, 1, left=0.15, bottom=0.15, right=0.975, top=0.975)
    ax_main = fig.add_subplot(gs[0, 0])

    ax_main.imshow(decibel(comb_spectra)[::-1], extent=[0, 180, spec_freqs[0], spec_freqs[-1]],
              aspect='auto', alpha=0.7, cmap='jet', vmax = -50, vmin = -110, interpolation='gaussian', zorder=1)
    # ax_main.plot([50, 50], [890, 930], linestyle='dotted', lw=2, color='k')
    # ax_main.set_ylim(875, 950)
    ax_main.set_ylim(890, 930)
    ax_main.set_yticks(np.arange(890, 931, 10))
    ax_main.set_xticks(np.arange(0, 181, 30))
    # ax_main.set_xticklabels([])

    ax_main.tick_params(labelsize=fs)
    ax_main.set_ylabel('frequency [Hz]', fontsize=fs+2)
    ax_main.set_xlabel('time [s]', fontsize=fs+2)

    plt.savefig('sum_spec.jpg', dpi=300)

    plt.show()
if __name__ == '__main__':
    main()