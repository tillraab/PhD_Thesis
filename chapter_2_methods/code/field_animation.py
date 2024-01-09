import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tqdm import tqdm
from IPython import embed

def load():
    folder = "/home/raab/writing/2021_tracking/data/2016-04-10-11_12/"

    part_spec = np.load(os.path.join(folder, 'part_spec.npy'))
    part_times = np.load(os.path.join(folder, 'part_times.npy'))
    part_freqs = np.load(os.path.join(folder, 'part_freqs.npy'))

    fund_v = np.load(os.path.join(folder, 'fund_v.npy'))
    sign_v = np.load(os.path.join(folder, 'sign_v.npy'))
    idx_v = np.load(os.path.join(folder, 'idx_v.npy'))
    ident_v = np.load(os.path.join(folder, 'ident_v.npy'))
    times = np.load(os.path.join(folder, 'times.npy'))
    spec = np.load(os.path.join(folder, 'spec.npy'))
    a_error_dist = np.load(os.path.join(folder, 'a_error_dist.npy'))
    start_time, end_time = np.load(os.path.join(folder, 'meta.npy'))

    return fund_v, sign_v, ident_v, idx_v, times

def main():
    fund_v, sign_v, ident_v, idx_v, times = load()
    times = times - times[0]

    ids_plus_900 = []
    for id in np.unique(ident_v[~np.isnan(ident_v)]):
        if 900 < np.mean(fund_v[ident_v == id]):
            ids_plus_900.append(id)

    sign_oi = []
    fund_oi = []
    for id in ids_plus_900:
        s = sign_v[ident_v == id]
        f = fund_v[ident_v == id]
        t = times[idx_v[ident_v == id]]

        f_interp = np.interp(times, t, f)
        s_interp = np.zeros((len(times), np.shape(s)[1]))
        for i in range(np.shape(s)[1]):
            s_interp[:, i] = np.interp(times, t, s[:, i])

        sign_oi.append(s_interp)
        fund_oi.append(f_interp)


    fig = plt.figure(figsize=(20/2.54, 14/2.54))
    gs = gridspec.GridSpec(2, 6, left=0.1, bottom = 0.1, right=0.9, top=0.95, hspace=0.4)
    ax = []
    ax.append(fig.add_subplot(gs[1, 0:2]))
    ax.append(fig.add_subplot(gs[1, 2:4]))
    ax.append(fig.add_subplot(gs[1, 4:]))

    color = ['firebrick', 'darkorange', 'forestgreen']
    for a, c in zip(ax, color):
        a.set_xticks([])
        a.set_yticks([])

        plt.setp(a.spines.values(), color=c)
        for key in ['top', 'bottom', 'left', 'right']:
            a.spines[key].set_linewidth(3)

    ax_trace = fig.add_subplot(gs[0, 1:-1])

    ax_trace.set_ylabel('frequency [Hz]', fontsize=10)
    ax_trace.set_xlabel('time [s]', fontsize=10)
    ax_trace.tick_params(labelsize=9)

    for i in range(len(ids_plus_900)):
        ax_trace.plot(times, fund_oi[i], marker='.', color=color[i], markersize=2)
    ax_trace.set_ylim(905, 925)
    ax_trace.set_xlim(times[0], times[-1])

    img_handle = [None, None, None]
    dot_handle = [None, None, None]
    line_handle = None


    counte = 0
    for i in tqdm(np.arange(len(times))):
    #for i in range(10):
        if line_handle != None:
            line_handle.remove()
            line_handle = None
        line_handle, = ax_trace.plot([times[i], times[i]], [905, 925], lw=2, color='k')

        for enu, s, f, in zip(np.arange(3), sign_oi, fund_oi):
            if img_handle[enu] != None:
                img_handle[enu].remove()
                dot_handle[enu].remove()
                img_handle[enu] = None
                dot_handle[enu] = None

            img_handle[enu] = ax[enu].imshow(s[i].reshape(8, 8)[::-1], cmap='jet', interpolation = 'gaussian')
            dot_handle[enu], = ax_trace.plot(times[i], f[i], color=color[enu], marker='o')

        counter_str = ('%3.0f' % counte).replace(' ', '0')
        plt.savefig('./anim_pics/' + counter_str + '.jpg', dpi=300)
        #plt.pause(0.03)
        counte+=1


    plt.show()



if __name__ == '__main__':
    main()

    # def life_tmp_ident_init(self, min_i0, max_i1):
    #
    #     self.fig, self.ax = plt.subplots()
    #     self.ax.imshow(decibel(self.spec)[::-1], extent=[self.times[0], self.times[-1], 0, 2000],
    #               aspect='auto', alpha=0.7, cmap='jet', vmax=-50, vmin=-110, interpolation='gaussian', zorder=1)
    #
    #     self.ax.set_xlim(self.times[self.idx_v[min_i0]], self.times[self.idx_v[max_i1]])
    #     self.ax.set_ylim(880, 950)
    #     # self.fig.canvas.draw()
    #     plt.pause(0.05)
    #
    #
    # def life_tmp_ident_update(self, tmp_indet_v, new=None, update=None, delete=None):
    #     if new:
    #         self.handles[new], = self.ax.plot(self.times[self.idx_v[tmp_indet_v == new]], self.fund_v[tmp_indet_v == new], marker='.')
    #     if update:
    #         self.handles[update].set_data(self.times[self.idx_v[tmp_indet_v == update]], self.fund_v[tmp_indet_v == update])
    #     if delete:
    #         self.handles[delete].remove()
    #         del self.handles[delete]
    #
    #     # self.fig.canvas.draw()
    #     plt.pause(0.05)