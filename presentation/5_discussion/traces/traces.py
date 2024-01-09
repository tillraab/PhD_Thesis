import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import embed

def load_data():
    time = np.load('col_times.npy', allow_pickle=True)
    freq = np.load('col_freq.npy', allow_pickle=True)
    color = np.load('col_color.npy', allow_pickle=True)

    nt_time = np.load('col_nt_time.npy', allow_pickle=True)
    nt_freq = np.load('col_nt_freq.npy', allow_pickle=True)

    dates = ['10.04.', '11.04.', '12.04.', '13.04.', '14.04.', '15.04.', '16.04.', '17.04.', '18.04.']
    return time, freq, color, nt_time, nt_freq, dates

def main():
    time, freq, color, nt_time, nt_freq, dates = load_data()

    fs=12
    fig = plt.figure(figsize=(20/2.54, 20 * (12/20) /2.54))
    gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.15, right=0.995, top=0.995)
    ax = fig.add_subplot(gs[0, 0])

    night_end = np.arange((455 + 24 * 60) / 60, 9 * 24, 24) # in h

    ax.fill_between([0, 455 * 60], [400, 400], [950, 950], color='#666666')
    ax.fill_between([455 * 60, 455 * 60 + 12 * 60 * 60], [400, 400], [950, 950], color='#dddddd')

    for ne in night_end:
        ax.fill_between([(ne - 12) * 3600, ne * 3600], [400, 400], [950, 950], color='#666666', edgecolor=None)
        ax.fill_between([ne* 3600, (ne + 12) * 3600], [400, 400], [950, 950], color='#dddddd', edgecolor=None)

    ax.plot(nt_time[::20], nt_freq[::20], '.', color='white', markersize=1)

    for i in range(len(time)):
        ax.plot(time[i], freq[i], color=color[i], marker='.', markersize=1)

    ax.set_ylim(500, 950)
    ax.set_yticks(np.arange(500, 901, 100))
    ax.set_xlim(0, 731500)

    ax.set_xticks(np.array((night_end - 18) * 60 * 60))
    ax.set_xticklabels(np.array(dates)[:-1], rotation=45, ha='right')
    ax.set_ylabel('frequency [Hz]', fontsize = fs+2)

    ax.tick_params(labelsize=fs)

    ax.set_rasterized(True)
    plt.savefig('traces_colobia2016.jpg', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()