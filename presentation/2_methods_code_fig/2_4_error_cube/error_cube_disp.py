import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import embed

fig = plt.figure(figsize=(17.5/2.54, 12/2.54))
gs = gridspec.GridSpec(1, 1, left=0, bottom=0, right=1, top=1)

ax = fig.add_subplot(gs[0, 0])

for i in np.arange(9):
    x0, x1 = 0+i*0.025, 0.5+i*0.025

    c0, c1 = ('k', 'white') if 3 <= i < 6 else ('grey', 'white')
    ax.fill_between([x0, x1], [x0, x0], [x1, x1], color=c0, zorder=10-i)
    ax.plot([x0, x1], [x0, x0], lw=2, color=c1, zorder=10-i)
    ax.plot([x0, x1], [x1, x1], lw=2, color=c1, zorder=10-i)
    ax.plot([x0, x0], [x0, x1], lw=2, color=c1, zorder=10-i)
    ax.plot([x1, x1], [x0, x1], lw=2, color=c1, zorder=10-i)

    if i == 0:
        ax.text(x0, x1 + 0.025, r'$\alpha_i$', fontsize=14, va='bottom', ha='right')
        ax.plot([x0, x0], [x0, x1 + 0.025], '--', lw=1, color='lightgrey', zorder=100)

        ax.plot([x0, x1], [x0, x0], '--', lw=1, color='lightgrey', zorder=100)
        ax.text(x0 + 0.025, x0, r'$\beta_{j} = \beta_{i+1}$, ... , $\beta_{i+\Delta I}$', fontsize=14, va='top', ha='left', zorder=100)

    if i == 3:
        ax.text(x0, x1 + 0.025, r'$\alpha_{i + \Delta I}$', fontsize=14, va='bottom', ha='right')
        ax.plot([x0, x0], [x0, x1 + 0.025], '--', lw=1, color='lightgrey', zorder=100)

        ax.plot([x0, x1], [x0, x0], '--', lw=1, color='lightgrey', zorder=100)
        ax.text(x0 + 0.025, x0, r'$\beta_{j + \Delta I} = \beta_{i+\Delta I+1}$, ... , $\beta_{i+ 2 \Delta I}$', fontsize=14, va='top', ha='left', zorder=100)

    if i == 6:
        ax.text(x0, x1 + 0.025, r'$\alpha_{i + 2  \Delta I}$', fontsize=14, va='bottom', ha='right')
        ax.plot([x0, x0], [x0, x1 + 0.025], '--', lw=1, color='lightgrey', zorder=100)

        ax.plot([x0, x1], [x0, x0], '--', lw=1, color='lightgrey', zorder=100)
        ax.text(x0 + 0.025, x0, r'$\beta_{j + 2  \Delta I} = \beta_{i+ 2  \Delta I+1}$, ... , $\beta_{i+ 3  \Delta I}$', fontsize=14, va='top', ha='left', zorder=100)


x0, x1 = 0+8*0.025, 0.5+8*0.025
# ax.text(x0, x1 + 0.025, r'$\alpha_{i + 3  \Delta I}$', fontsize=14, va='bottom', ha='right')
ax.plot([x0, x0], [x0, x1], '--', lw=1, color='lightgrey', zorder=100)
ax.plot([x0, x1], [x0, x0], '--', lw=1, color='lightgrey', zorder=100)

ax.plot([0, 0 + 8*0.025], [0, 0 + 8*0.025], '--', lw=1, color='lightgrey', zorder=100)

ax.text(0.35, 0.35, r'$\varepsilon_{\alpha, \beta}$', fontsize=30, ha='center', va='center', zorder=100)
# ax.plot([0.5, 0.5 + 9*0.025], [0, 0 + 9*0.025], '--', lw=1, color='grey', zorder=100)
# ax.plot([0.0 + 9*0.025, 0.5 + 8*0.025], [0 + 9*0.025, 0+ 9*0.025], '--', lw=1, color='lightgrey', zorder=100)


ax.set_xlim(-0.075, .75)
ax.set_ylim(-0.1, .775)

ax.set_axis_off()

plt.savefig('cube_error.pdf')
plt.show()