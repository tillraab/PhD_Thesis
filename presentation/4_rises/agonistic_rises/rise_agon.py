import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import scipy.stats as scp
from IPython import embed
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit
from plottools.tag import tag
# fig.tag(axes=[ax[1], ax[2]], fontsize=15, yoffs=2, xoffs=-3)

def significnace(ax, x1, x2, y, p, vertical=False, whisker_fac = 0.0035):
    x1, x2 = np.array([x1, x2])[np.argsort(np.array([x1, x2]))]

    if p < 0.001:
        ps = '***'
    elif p < 0.01:
        ps = '**'
    elif p < 0.05:
        ps = '*'
    else:
        'n.s.'
    if vertical:
        whisk_len = np.diff(ax.get_xlim())[0] * whisker_fac
        ax.plot([y, y], [x1, x2], lw=1, color='k')
        ax.plot([y-whisk_len, y], [x1, x1], lw=1, color='k')
        ax.plot([y-whisk_len, y], [x2, x2], lw=1, color='k')

        ax.text(y + whisk_len*1, x1 + (x2 - x1)/2, ps, fontsize=10, color='k', ha='left', va='center', rotation=90)
    else:
        whisk_len = np.diff(ax.get_ylim())[0] * whisker_fac
        ax.plot([x1, x2], [y, y], lw=1, color='k')
        ax.plot([x1, x1], [y - whisk_len, y], lw=1, color='k')
        ax.plot([x2, x2], [y - whisk_len, y], lw=1, color='k')

        ax.text( x1 + (x2 - x1) / 2, y + whisk_len*0.5, ps, fontsize=10, color='k', ha='center', va='bottom')

def main():
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

    fig = plt.figure(figsize=(17.5 / 2.54, 10 / 2.54))
    gs = gridspec.GridSpec(2, 3, left=0.125, bottom=0.15, top=0.975, right=0.95, hspace=.3, wspace=0.6, width_ratios=[3, 1, 1.25])
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[1, 0], sharex=ax[0]))
    ax_mirror = []
    for Cax in ax:
        ax_mirror.append(Cax.twinx())
    ax.append(fig.add_subplot(gs[0, 1]))
    ax.append(fig.add_subplot(gs[1, 1], sharey=ax[2], sharex=ax[2]))
    gs = gridspec.GridSpec(1, 1, left = 0.75, right=1, top=1, bottom=0)
    ax.append(fig.add_subplot(gs[0, 0]))
    # ax.append(fig.add_subplot(gs[:, 2]))
    #ax.append(fig.add_subplot(gs[1, 2], sharey=ax[4]))

    b_mean, b_std, b_pct1, b_pct99, s_mean, s_std, s_pct1, s_pct99 = contact_m_s_pct

    b_mean, b_std, b_pct1, b_pct99, s_mean, s_std, s_pct1, s_pct99 = \
        b_mean * 19 * 60, b_std * 19 * 60, b_pct1 * 19 * 60, b_pct99 * 19 * 60, s_mean * 19 * 60, s_std * 19 * 60, s_pct1 * 19 * 60, s_pct99 * 19 * 60

    ax[0].plot(conv_t, b_mean, color='cornflowerblue', alpha=0.75)
    ax[0].fill_between(conv_t, b_pct1, b_pct99, color='cornflowerblue', alpha=0.5)
    ax[0].plot(conv_t, dt_contact_mass_conv * 19 * 60, color='firebrick', lw=2)
    for file_i in range(len(all_Ldt_contact)):
        ax_mirror[0].plot(all_Ldt_contact[file_i], np.ones(len(all_Ldt_contact[file_i])) * file_i + 0.5, '|', color='k',
                          alpha=0.2, markersize=4)
    ax[0].fill_between(conv_t, s_pct1, s_pct99, color='firebrick', alpha=0.5)
    ax[0].set_xlim([- max_dt / 2, max_dt / 2])

    b_mean, b_std, b_pct1, b_pct99, s_mean, s_std, s_pct1, s_pct99 = ag_on_m_s_pct
    b_mean, b_std, b_pct1, b_pct99, s_mean, s_std, s_pct1, s_pct99 = \
        b_mean * 19 * 60, b_std * 19 * 60, b_pct1 * 19 * 60, b_pct99 * 19 * 60, s_mean * 19 * 60, s_std * 19 * 60, s_pct1 * 19 * 60, s_pct99 * 19 * 60

    ax[1].plot(conv_t, b_mean, color='cornflowerblue', alpha=0.75)
    ax[1].fill_between(conv_t, b_mean + b_std, b_mean - b_std, color='cornflowerblue', alpha=0.5)
    ax[1].plot(conv_t, dt_ag_on_mass_conv * 19 * 60, color='darkorange', lw=2)
    for file_i in range(len(all_Ldt_ag_on)):
        ax_mirror[1].plot(all_Ldt_ag_on[file_i], np.ones(len(all_Ldt_ag_on[file_i])) * file_i + 0.5, '|', color='k',
                          alpha=0.2, markersize=4)

    ax[1].fill_between(conv_t, s_mean + s_std, s_mean - s_std, color='darkorange', alpha=0.5)
    ax[1].set_xlim([- max_dt / 2, max_dt / 2])

    for Cax, Cax_m in zip(ax, ax_mirror):
        Cax_m.set_xlim([-30, 30])
        Cax_m.set_ylim([-0.5, 20.5])

        Cax.set_xlim([-30, 30])
        #Cax.set_ylim(bottom=0)

        #Cax_m.set_ylabel('trial', fontsize=12)
        #Cax.set_ylabel('rise rate [mHz]', fontsize=12)

    ax[0].text(-42.5, -0.5, 'rise rate [1/min]', fontsize=12, ha='center', va='center', clip_on=False, rotation=90)
    #ax[0].text(-42.5, -0.00025, 'EODf rise probability [1/1000]', fontsize=12, ha='center', va='center', clip_on=False, rotation=90)

    ax[0].set_ylim(0, 3.5)
    ax[0].set_yticks(np.arange(0, 3.1))
    #ax[0].set_yticklabels(np.arange(0, 51, 10))

    ax[1].set_ylim(0, 2)
    ax[1].set_yticks(np.arange(0, 2.1, 1))
    #ax[1].set_yticklabels(np.arange(0, 31, 10))
    #
    # ax[1].set_ylim(0.0004, 0.0016)
    # ax[1].set_yticks([0.0004, 0.001, 0.0016])
    # ax[1].set_yticklabels([0.4, 1, 1.6])


    ax[1].set_xticks(np.arange(-30, 31, 10))
    ax[1].set_xlabel(r'$\Delta$time [s]', fontsize=12)

    plt.setp(ax[0].get_xticklabels(), visible=False)

    for a in ax_mirror:
        a.set_yticks([])
        # a.tick_params(labelsize=10)




    ####no

    contact_l, ag_on_l, ag_l, no_l = event_counts
    contact_t_l, ag_on_t_l, ag_t_l, no_t_l = event_time_counts
    #
    # ###################
    # plt.close('all')
    # fig, ax = plt.subplots(2, 1)
    # pairings = np.array([2, 3, 2, 3, 0, 3, 3, 1, 0, 3, 2, 2, 2, 0, 1, 0, np.nan, 3, 0])
    # for i in range(4):
    #     contact = np.sum(np.array(contact_l)[pairings == i])
    #     ag_on = np.sum(np.array(ag_on_l)[pairings == i])
    #     ag = np.sum(np.array(ag_l)[pairings == i])
    #     no = np.sum(np.array(no_l)[pairings == i])
    #
    #     contact_t = np.sum(np.array(contact_t_l)[pairings == i])
    #     ag_on_t = np.sum(np.array(ag_on_t_l)[pairings == i])
    #     ag_t = np.sum(np.array(ag_t_l)[pairings == i])
    #     no_t = np.sum(np.array(no_t_l)[pairings == i])
    #
    #     c_box = np.array(contact_l)[pairings == i] / (np.array(contact_l)[pairings == i] + np.array(ag_on_l)[pairings == i] + np.array(ag_l)[pairings == i] + np.array(no_l)[pairings == i])
    #     ct_box = np.array(contact_t_l)[pairings == i] / (
    #                 np.array(contact_t_l)[pairings == i] + np.array(ag_on_t_l)[pairings == i] + np.array(ag_t_l)[pairings == i] + np.array(no_t_l)[pairings == i])
    #
    #     ao_box = np.array(ag_on_l)[pairings == i] / (np.array(contact_l)[pairings == i] + np.array(ag_on_l)[pairings == i] + np.array(ag_l)[pairings == i] +
    #                                                  np.array(no_l)[pairings == i])
    #     aot_box = np.array(ag_on_t_l)[pairings == i] / (
    #                 np.array(contact_t_l)[pairings == i] + np.array(ag_on_t_l)[pairings == i] + np.array(ag_t_l)[pairings == i] + np.array(no_t_l)[pairings == i])
    #
    #     a_box = np.array(ag_l)[pairings == i] / (np.array(contact_l)[pairings == i] + np.array(ag_on_l)[pairings == i] + np.array(ag_l)[pairings == i] +
    #                                              np.array(no_l)[pairings == i])
    #     at_box = np.array(ag_t_l)[pairings == i] / (np.array(contact_t_l)[pairings == i] + np.array(ag_on_t_l)[pairings == i] +
    #                                                 np.array(ag_t_l)[pairings == i] + np.array(no_t_l)[pairings == i])
    #
    #     bp1 = ax[0].boxplot([c_box, ct_box], positions = np.array([1, 2]) + i*3, sym='', widths=0.5, patch_artist=True)
    #     bp2 = ax[1].boxplot([ao_box, aot_box], positions = np.array([1, 2]) + i*3, sym='', widths=0.5, patch_artist=True)

    ###############################

    contact = np.sum(contact_l)
    ag_on = np.sum(ag_on_l)
    ag = np.sum(ag_l)
    no = np.sum(no_l)

    contact_t = np.sum(contact_t_l)
    ag_on_t = np.sum(ag_on_t_l)
    ag_t = np.sum(ag_t_l)
    no_t = np.sum(no_t_l)

    c_box = np.array(contact_l) / (np.array(contact_l) + np.array(ag_on_l) + np.array(ag_l) + np.array(no_l))
    ct_box = np.array(contact_t_l) / (np.array(contact_t_l) + np.array(ag_on_t_l) + np.array(ag_t_l) + np.array(no_t_l))

    ao_box = np.array(ag_on_l) / (np.array(contact_l) + np.array(ag_on_l) + np.array(ag_l) + np.array(no_l))
    aot_box = np.array(ag_on_t_l) / (np.array(contact_t_l) + np.array(ag_on_t_l) + np.array(ag_t_l) + np.array(no_t_l))

    a_box = np.array(ag_l) / (np.array(contact_l) + np.array(ag_on_l) + np.array(ag_l) + np.array(no_l))
    at_box = np.array(ag_t_l) / (np.array(contact_t_l) + np.array(ag_on_t_l) + np.array(ag_t_l) + np.array(no_t_l))

    bp1 = ax[2].boxplot([c_box, ct_box], sym='', widths=0.5, patch_artist=True)
    bp2 = ax[3].boxplot([ao_box, aot_box], sym='', widths=0.5, patch_artist=True)

    for enu, box in enumerate(bp1['boxes']):
        box.set_color('firebrick')
        box.set_alpha(1 - enu * 0.4)
    for med in bp1['medians']:
        med.set_color('k')

    for enu, box in enumerate(bp2['boxes']):
        box.set_color('darkorange')
        box.set_alpha(1 - enu * 0.4)
    for med in bp2['medians']:
        med.set_color('k')

    size = 0.3
    outer_colors = ['firebrick', 'darkorange', 'forestgreen', 'grey']
    inner_colors = ['firebrick', 'darkorange', 'forestgreen', 'grey']

    significnace(ax[2], 1, 2, 0.22, 0.015, whisker_fac=0.025)
    significnace(ax[3], 1, 2, 0.22, 0.007, whisker_fac=0.025)

    #ax[4].pie([contact, ag_on, ag, no], radius=1, colors=outer_colors, wedgeprops=dict(width=size, edgecolor='w'), startangle=90)
    #ax[4].pie([contact_t, ag_on_t, ag_t, no_t], radius=1-size, colors=inner_colors, wedgeprops=dict(width=size, edgecolor='w', alpha=0.7), startangle=90)

    ac = contact + ag_on + ag + no
    contact_r, ag_on_r, ag_r, no_r = contact / ac, ag_on / ac, ag / ac, no / ac

    ac = contact_t + ag_on_t + ag_t + no_t
    contact_r_t, ag_on_r_t, ag_r_t, no_r_t = contact_t / ac, ag_on_t / ac, ag_t / ac, no_t / ac

    ax[4].pie([contact_r, ag_on_r, ag_r, no_r], radius=1, colors=outer_colors, wedgeprops=dict(width=size, edgecolor='w'), startangle=90)
    ax[4].pie([contact_r_t, ag_on_r_t, ag_r_t, no_r_t], radius=1-size, colors=inner_colors, wedgeprops=dict(width=size, edgecolor='w', alpha=0.6), startangle=90)



    for a in [ax[2], ax[3]]:
        a.set_yticks([0, .1, .2])
        a.set_yticklabels([r'0$\,$%', r'10$\,$%', r'20$\,$%'])
        # a.set_yticklabels([0, 10, 20])
        # a.set_ylabel('%', fontsize=12)
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.yaxis.set_ticks_position('left')
        a.xaxis.set_ticks_position('bottom')
    ax[3].set_xticks([1, 2])
    ax[3].set_xticklabels(['rises', 'time'], rotation=70)
    plt.setp(ax[2].get_xticklabels(), visible=False)
    ax[3].set_ylim(bottom=0)

    for a in ax:
        a.tick_params(labelsize=10)


    legend_elements = [Patch(facecolor='firebrick', edgecolor='w', label='%.1f' % (contact_r * 100) + '%'),
                       Patch(facecolor='darkorange', edgecolor='w', label='%.1f' % (ag_on_r * 100) + '%'),
                       Patch(facecolor='forestgreen', edgecolor='w', label='%.1f' % (ag_r * 100) + '%'),
                       Patch(facecolor='firebrick', alpha=0.6, edgecolor='w', label='%.1f' % (contact_r_t * 100) + '%'),
                       Patch(facecolor='darkorange', alpha=0.6, edgecolor='w', label='%.1f' % (ag_on_r_t * 100) + '%'),
                       Patch(facecolor='forestgreen', alpha=0.6, edgecolor='w', label='%.1f' % (ag_r_t * 100) + '%')]

    ax[4].text(-0.65, -1.4, 'rises', fontsize=10, va='center', ha='center')
    ax[4].text(0.75, -1.4, 'time', fontsize=10, va='center', ha='center')
    ax[4].legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.5), frameon=False, fontsize=9)

    legend_elements = [Patch(facecolor='firebrick', edgecolor='w', label='pre-contact'),
                       Patch(facecolor='darkorange', edgecolor='w', label='pre-chasing'),
                       Patch(facecolor='forestgreen', edgecolor='w', label='chasing'),
                       Patch(facecolor='grey', edgecolor='w', label='no interaction')]
    ax[2].legend(handles=legend_elements, loc=1, ncol=1, bbox_to_anchor=(3.6, 1.1), frameon=False, fontsize=9)

    for a in ax:
        a.tick_params(labelsize=10)
    for a in ax_mirror:
        a.tick_params(labelsize=10)

    fig.tag(axes=[ax[0],ax[2]], fontsize=15, yoffs=2, xoffs=-8)
    fig.tag(axes=[ax[4]], fontsize=15, yoffs=2, xoffs=-4)
    # plt.savefig('../../figures/event_rises.pdf')
    # plt.savefig('event_rises.pdf')


    print('ratio rises vs. time')
    t, p = scp.ttest_rel(c_box, ct_box)
    print('contact: t=%.3f, p=%.4f' % (t, p))
    t, p = scp.ttest_rel(a_box, at_box)
    print('ag on: t=%.3f, p=%.4f' % (t, p))
    t, p = scp.ttest_rel(ao_box, aot_box)
    print('ag rest: t=%.3f, p=%.4f' % (t, p))

    plt.show()

    ####
    embed()
    quit()

    fs = 16
    fig = plt.figure(figsize=(10/2.54, 12/2.54))
    gs = gridspec.GridSpec(2, 1, left=0.175, bottom = 0.135, right=0.95, top=1  )
    ax = []
    ax.append(fig.add_subplot(gs[0, 0]))
    ax.append(fig.add_subplot(gs[1, 0]))
    ax_mirror = []
    for Cax in ax:
        ax_mirror.append(Cax.twinx())

    b_mean, b_std, b_pct1, b_pct99, s_mean, s_std, s_pct1, s_pct99 = contact_m_s_pct

    b_mean, b_std, b_pct1, b_pct99, s_mean, s_std, s_pct1, s_pct99 = \
        b_mean * 19 * 60, b_std * 19 * 60, b_pct1 * 19 * 60, b_pct99 * 19 * 60, s_mean * 19 * 60, s_std * 19 * 60, s_pct1 * 19 * 60, s_pct99 * 19 * 60

    ax[0].plot(conv_t, b_mean, color='cornflowerblue', alpha=0.75)
    ax[0].fill_between(conv_t, b_pct1, b_pct99, color='cornflowerblue', alpha=0.5)
    ax[0].plot(conv_t, dt_contact_mass_conv * 19 * 60, color='firebrick', lw=2)
    for file_i in range(len(all_Ldt_contact)):
        ax_mirror[0].plot(all_Ldt_contact[file_i], np.ones(len(all_Ldt_contact[file_i])) * file_i + 0.5, '|', color='k',
                          alpha=0.2, markersize=6)
    ax[0].fill_between(conv_t, s_pct1, s_pct99, color='firebrick', alpha=0.5)
    ax[0].set_xlim([- max_dt / 2, max_dt / 2])

    b_mean, b_std, b_pct1, b_pct99, s_mean, s_std, s_pct1, s_pct99 = ag_on_m_s_pct
    b_mean, b_std, b_pct1, b_pct99, s_mean, s_std, s_pct1, s_pct99 = \
        b_mean * 19 * 60, b_std * 19 * 60, b_pct1 * 19 * 60, b_pct99 * 19 * 60, s_mean * 19 * 60, s_std * 19 * 60, s_pct1 * 19 * 60, s_pct99 * 19 * 60

    ax[1].plot(conv_t, b_mean, color='cornflowerblue', alpha=0.75)
    ax[1].fill_between(conv_t, b_mean + b_std, b_mean - b_std, color='cornflowerblue', alpha=0.5)
    ax[1].plot(conv_t, dt_ag_on_mass_conv * 19 * 60, color='darkorange', lw=2)
    for file_i in range(len(all_Ldt_ag_on)):
        ax_mirror[1].plot(all_Ldt_ag_on[file_i], np.ones(len(all_Ldt_ag_on[file_i])) * file_i + 0.5, '|', color='k',
                          alpha=0.2, markersize=6)

    ax[1].fill_between(conv_t, s_mean + s_std, s_mean - s_std, color='darkorange', alpha=0.5)
    ax[1].set_xlim([- max_dt / 2, max_dt / 2])

    for Cax, Cax_m in zip(ax, ax_mirror):
        Cax_m.set_xlim([-30, 30])
        Cax_m.set_ylim([-0.5, 20.5])

        Cax.set_xlim([-30, 30])

    ax[0].text(-40, -0.25, 'rise rate [1/min]', fontsize=fs+2, ha='center', va='center', clip_on=False, rotation=90)
    #ax[0].text(-42.5, -0.00025, 'EODf rise probability [1/1000]', fontsize=12, ha='center', va='center', clip_on=False, rotation=90)

    ax[0].set_ylim(0, 3.5)
    ax[0].set_yticks(np.arange(0, 3.1))
    ax[0].plot([0, 0], [0, 3.5], color='grey', lw=2, linestyle='dashed')
    #ax[0].set_yticklabels(np.arange(0, 51, 10))

    ax[1].set_ylim(0, 2)
    ax[1].plot([0, 0], [0, 2], color='grey', lw=1.5, linestyle='dashed')
    ax[1].set_yticks(np.arange(0, 2.1, 1))

    ax[1].set_xticks(np.arange(-30, 31, 15))
    ax[1].set_xlabel(r'$\Delta$time [s]', fontsize=fs+2)
    # plt.gcf().supylabel('rise rate [1/min]', fontsize=fs+2)
    plt.setp(ax[0].get_xticklabels(), visible=False)

    for a in ax_mirror:
        a.set_yticks([])
    for a in ax:
        a.tick_params(labelsize=fs)
    plt.savefig('pres_agon_rises.jpg', dpi=300)
    plt.show()
    # embed()
    # quit()
if __name__ == '__main__':
    main()