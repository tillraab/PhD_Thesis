import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import embed
import pandas as pd
import matplotlib.cm as cm
import matplotlib as mpl
import scipy.stats as scp
from mpl_toolkits import mplot3d
from plottools.tag import tag

def gauss(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp( - 0.5* ((x-mu)/sigma)**2 )

def create_population_matrix(n):
    #RHP = np.random.rand(n)
    mu, sigma = 5, 3
    RHP = np.random.normal(mu, sigma, n) + 2*mu

    # help_param = np.max(np.abs(RHP))
    help_param = np.max(np.abs(RHP))

    #count, bins = np.histogram(RHP, bins=np.linspace(-help_param, help_param, 101))
    count, bins = np.histogram(RHP, bins=np.linspace(0, help_param, 51))
    fig = plt.figure(figsize=(9/2.54, 9/2.54))
    gs = gridspec.GridSpec(1, 1, left=0.15, bottom=0.15, right=0.8, top=0.9)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(ax[0].twinx())
    ax[0].bar(bins[:-1] + (bins[1] - bins[0]) / 2, count / np.sum(count) / (bins[1] - bins[0]), width=0.8 * (bins[1] - bins[0]), color='grey')
    x_range = np.arange(0, 30, 0.1)
    g = gauss(x_range, mu*3, sigma)
    ax[0].plot(x_range, g, color='k', lw=2)

    #ax.set_xticks([0])
    #ax.set_xticklabels([r'$\overline{RHP}_{population}$'])
    # ax[0].set_ylabel('n', fontsize=10)
    ax[0].set_xlabel('RHP', fontsize=10)
    ax[0].set_xlim(0, 30)
    ax[0].set_yticks([0, 0.1])

    # ax[1].plot(np.arange(0, 30, 0.1), own_costs(np.arange(0, 30, 0.1), RHP))
    ax[1].plot(np.arange(0, 30, 0.1), inflicted_costs(np.arange(0, 30, 0.1), RHP), lw=2, color='firebrick', label=r'c$_i$(RHP) = 0.02 $\cdot$ RHP')
    ax[1].set_ylabel(r'inflicted costs [RHP / s]', fontsize=10)
    ax[1].legend(loc=1, frameon = False, bbox_to_anchor=(1, 1.1), fontsize=9)
    # ax[1].set_ylim(0, 0.35)
    # ax[1].set_yticks([0, 0.001 * np.mean(RHP), 0.1, 0.2, 0.3])
    # ax[1].set_yticklabels([0, r'c$_i$', 0.1, 0.2, 0.3])


    # ax[1].plot(np.sort(RHP), np.ones(len(RHP)) * 0.01 * np.mean(RHP), color='darkorange', lw=2)
    # ax[1].plot(np.sort(RHP), np.ones(len(RHP)) * 0.01 * np.mean(RHP) + 0.01 * np.mean(RHP) * np.linspace(0, 1, len(RHP)), color='firebrick', lw=2)

    # ax[1].set_ylim(0, 0.1)

    ids = np.arange(len(RHP))

    x, y = np.meshgrid(ids, ids)
    x = np.hstack(x)
    y = np.hstack(y)

    help_bool = x>y

    id0 = x[help_bool]
    id1 = y[help_bool]

    RHP_dict = {}
    RHP_dict['id0'] = id0
    RHP_dict['id1'] = id1
    RHP_dict['RHP0'] = RHP[id0]
    RHP_dict['RHP1'] = RHP[id1]
    RHP_dict['RHP_l'] = np.min(np.array([RHP_dict['RHP0'], RHP_dict['RHP1']]), axis=0)
    RHP_dict['RHP_w'] = np.max(np.array([RHP_dict['RHP0'], RHP_dict['RHP1']]), axis=0)
    RHP_dict['RHP_d'] = np.abs(RHP_dict['RHP0'] - RHP_dict['RHP1'])

    RHP_dict = pd.DataFrame(RHP_dict)
    return RHP_dict, RHP

def predictions(RHP_dict, RHP):
    sparse_fact = int(len(RHP_dict) / 200)
    fig = plt.figure(figsize=(17.5/2.54, 11/2.54))
    gs = gridspec.GridSpec(3, 4, left=0.1, bottom = 0.15, right=0.95, top=0.95, width_ratios=[2, 2, 2, 1], hspace=0.4, wspace=0.4)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[0, 1], sharey=ax[0]))
    ax.append(fig.add_subplot(gs[0, 2], sharey=ax[0]))

    handles = []

    # self assessment
    dur_self = sim_self_assessment(RHP_dict['RHP_l'], RHP_dict['RHP_w'], RHP)
    handles.append(ax[0].plot(RHP_dict['RHP_w'][::sparse_fact], dur_self[::sparse_fact], '.', markersize=4, alpha=0.5, color='grey')[0])
    handles.append(ax[1].plot(RHP_dict['RHP_l'][::sparse_fact], dur_self[::sparse_fact], '.', markersize=4, alpha=0.5, color='grey')[0])
    handles.append(ax[2].plot(RHP_dict['RHP_d'][::sparse_fact], dur_self[::sparse_fact], '.', markersize=4, alpha=0.5, color='grey')[0])

    ax.append(fig.add_subplot(gs[1, 0]))
    ax.append(fig.add_subplot(gs[1, 1], sharey=ax[3]))
    ax.append(fig.add_subplot(gs[1, 2], sharey=ax[3]))

    # cummulative assessment
    dur_cum = sim_cumulative_assessment(RHP_dict['RHP_l'], RHP_dict['RHP_w'], RHP)
    handles.append(ax[3].plot(RHP_dict['RHP_w'][::sparse_fact], dur_cum[::sparse_fact], '.', markersize=4, alpha=0.5, color='grey')[0])
    handles.append(ax[4].plot(RHP_dict['RHP_l'][::sparse_fact], dur_cum[::sparse_fact], '.', markersize=4, alpha=0.5, color='grey')[0])
    handles.append(ax[5].plot(RHP_dict['RHP_d'][::sparse_fact], dur_cum[::sparse_fact], '.', markersize=4, alpha=0.5, color='grey')[0])

    ax.append(fig.add_subplot(gs[2, 0]))
    ax.append(fig.add_subplot(gs[2, 1], sharey=ax[6]))
    ax.append(fig.add_subplot(gs[2, 2], sharey=ax[6]))

    # mutual assessment assessment
    dur_mut = sim_mutual_assessment(RHP_dict['RHP_l'], RHP_dict['RHP_w'], RHP)
    handles.append(ax[6].plot(RHP_dict['RHP_w'][::sparse_fact], dur_mut[::sparse_fact], '.', markersize=4, alpha=0.5, color='grey')[0])
    handles.append(ax[7].plot(RHP_dict['RHP_l'][::sparse_fact], dur_mut[::sparse_fact], '.', markersize=4, alpha=0.5, color='grey')[0])
    handles.append(ax[8].plot(RHP_dict['RHP_d'][::sparse_fact], dur_mut[::sparse_fact], '.', markersize=4, alpha=0.5, color='grey')[0])

    ax[0].set_ylabel('fight duration', labelpad=15, fontsize=10)
    ax[3].set_ylabel('fight duration', labelpad=15, fontsize=10)
    ax[6].set_ylabel('fight duration', labelpad=15, fontsize=10)

    ax[6].set_xlabel(r'RHP$_{win}$', fontsize=10)
    ax[7].set_xlabel(r'RHP$_{lose}$', fontsize=10)
    ax[8].set_xlabel(r'$\Delta$RHP', fontsize=10)

    for i in range(9):
        plt.setp(ax[i].get_yticklabels(), visible=False)
        plt.setp(ax[i].get_yticklabels(), visible=False)
        ax[i].set_yticks([])
        if i not in [6, 7, 8]:
            plt.setp(ax[i].get_xticklabels(), visible=False)

    text_ax = []
    for i in range(3):
        text_ax.append(fig.add_subplot(gs[i, 3]))
        text_ax[i].set_axis_off()
        text_ax[i].set_xlim(0, 1)
        text_ax[i].set_ylim(0, 1)
    text_ax[0].text(.5, .5, 'self\nassessment', fontsize=10, va='center', ha='center')
    text_ax[1].text(.5, .5, 'cumulative\nassessment', fontsize=10, va='center', ha='center')
    text_ax[2].text(.5, .5, 'mutual\nassessment', fontsize=10, va='center', ha='center')

    counter = 0
    for dur in [dur_self, dur_cum, dur_mut]:
        for CRHP in [RHP_dict['RHP_w'], RHP_dict['RHP_l'], RHP_dict['RHP_d']]:
            slope, intercept, _, _, _  = scp.linregress(CRHP, dur)

            all_x = handles[counter].get_xdata()
            min_max = np.array([np.min(all_x), np.max(all_x)])
            ax[counter].plot(min_max, min_max * slope + intercept, color='k')

            r, p = scp.pearsonr(CRHP, dur)
            x_lims = ax[counter].get_xlim()
            y_lims = ax[counter].get_ylim()
            ax[counter].text(x_lims[1], y_lims[1] + 0.05 * (y_lims[1] - y_lims[0]), 'r=%.2f' % (r), fontsize=9, clip_on=False, ha='right', va='center')
            # ax[counter].text(x_lims[1], y_lims[1] + 0.05 * (y_lims[1] - y_lims[0]), 'p = %.2f, r=%.2f' % (p, r), fontsize=9, clip_on=False, ha='right', va='center')
            ax[counter].set_xlim(x_lims)
            ax[counter].set_ylim(y_lims)

            counter += 1
    fig.tag(axes=[ax[0], ax[3], ax[6]], labels=['A', 'B', 'C'], fontsize=15, yoffs=1, xoffs=-8)
    plt.savefig('assessment_models_correlations.pdf')

def own_costs(x, RHP):
    return 0.001 * np.mean(RHP) * np.ones(len(x))

def inflicted_costs(y, RHP):
    # embed()
    # quit()
    # return 0.001 * np.mean(RHP) + ((y - np.min(RHP)) / (np.max(RHP) - np.min(RHP))) * 0.005 * np.mean(RHP)
    #return 0.001 * np.mean(RHP) + y * 0.01
    return y * 0.02
    # return 0.01 * np.mean(RHP) + 0.01 * np.mean(RHP) * y

def sim_cumulative_assessment(x, y, RHP, self_weight = 0.01, other_weight = 0.01):
    # ret = x / (self_weight * np.mean(RHP) + other_weight * np.mean(RHP) * y)
    ret = x / (own_costs(x, RHP) + inflicted_costs(y, RHP))
    # ret = x / (inflicted_costs(y, RHP))
    ret[x >= y] = np.nan
    return ret

def sim_self_assessment(x, y, RHP):
    # ret = x / (0.01 * np.mean(RHP))
    ret = x / (own_costs(x, RHP))
    ret[x >= y] = np.nan
    return ret

def sim_mutual_assessment(x, y, RHP):
    # ret = x - y
    ret = np.max(RHP) - (y - x)
    ret[x >= y] = np.nan
    return ret

def competition_simulations(RHP):
    RHPl = np.arange(5, 26, 1.25, dtype=float)
    RHPw = np.arange(5, 26, 1.25, dtype=float)

    X, Y = np.meshgrid(RHPl, RHPw)

    Z0 = sim_self_assessment(X, Y, RHP)
    Z1 = sim_cumulative_assessment(X, Y, RHP)
    Z2 = sim_mutual_assessment(X, Y, RHP)

    #fig = plt.figure(figsize=(15/2.54, 20/2.54))
    fig = plt.figure(figsize=(12/2.54, 16/2.54))
    #gs = gridspec.GridSpec(3, 2, left=0.0, bottom=0.2, right=0.95, top=0.95, width_ratios=[2, 1], wspace=0.1)
    gs = gridspec.GridSpec(3, 1, left=0.0, bottom=0.2, right=0.6, top=1)
    ax = []
    ax.append(fig.add_subplot(gs[0, 0], projection='3d'))
    ax.append(fig.add_subplot(gs[1, 0], projection='3d'))
    ax.append(fig.add_subplot(gs[2, 0], projection='3d'))

    ax[0].plot_wireframe(X, Y, Z0, color='grey', lw=1, alpha=0.5)
    ax[1].plot_wireframe(X, Y, Z1, color='grey', lw=1, alpha=0.5)
    ax[2].plot_wireframe(X, Y, Z2, color='grey', lw=1, alpha=0.5)

    ax[0].view_init(elev=15, azim=150)
    ax[1].view_init(elev=15, azim=150)
    ax[2].view_init(elev=15, azim=150)

    gs2 = gridspec.GridSpec(3, 1, left=0.65, bottom=0.225, right=0.9, top=0.975, hspace=0.5)
    ax.append(fig.add_subplot(gs2[0, 0]))
    ax.append(fig.add_subplot(gs2[1, 0]))
    ax.append(fig.add_subplot(gs2[2, 0]))


    x = np.hstack(X)
    y = np.hstack(Y)
    z0 = np.hstack(Z0)
    z1 = np.hstack(Z1)
    z2 = np.hstack(Z2)

    for dRHP in RHPl - 5:
        c = cm.Greys(1 - dRHP / (np.max(RHPl)-5))
        mask = np.arange(len(x))[y-x == dRHP]
        ax[3].plot(x[mask], z0[mask], color=c)
        ax[4].plot(x[mask], z1[mask], color=c)
        ax[5].plot(x[mask], z2[mask], color=c)

        ax[0].plot3D(x[mask], y[mask], z0[mask], color=c)
        ax[1].plot3D(x[mask], y[mask], z1[mask], color=c)
        ax[2].plot3D(x[mask], y[mask], z2[mask], color=c)

    for a in ax[:3]:
        a.set_xlabel(r'RHP$_{lose}$')
        a.set_ylabel(r'RHP$_{win}$')
        a.set_zticks([])

    for a in ax[3:]:
        a.set_ylabel('fight duration', fontsize=10, labelpad=15)
        a.set_yticks([])
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        a.yaxis.set_ticks_position('left')
        a.xaxis.set_ticks_position('bottom')

    plt.setp(ax[3].get_xticklabels(), visible=False)
    plt.setp(ax[4].get_xticklabels(), visible=False)

    ax[-1].set_xlabel(r'RHP$_{lose}$')



    gs_cb = gridspec.GridSpec(1,1, bottom = 0.075, top=0.1, left=0.15, right=0.90)
    ax_cb = fig.add_subplot(gs_cb[0, 0])

    cmap = mpl.cm.Greys
    norm = mpl.colors.Normalize(vmin=0, vmax=np.max(RHPl))

    cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation='horizontal')
    cb1.set_label(r'$\Delta$RHP', fontsize=10)

    fig.tag(axes=[ax[3:]], labels=['A', 'B', 'C'], fontsize=15, yoffs=0, xoffs=-42)

    xtl = np.array(ax_cb.get_xticks(), dtype=int)
    ax_cb.set_xticklabels(xtl[::-1])

    plt.savefig('assessment_model_simulation.pdf')


def main():

    RHP_dict, RHP = create_population_matrix(n=1000)

    predictions(RHP_dict, RHP)

    competition_simulations(RHP)

    plt.show()
if __name__ == '__main__':
    main()