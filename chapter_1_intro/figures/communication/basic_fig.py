import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plottools.tag import tag

fig = plt.figure(figsize=(9/2.54, 13/2.54))
gs = gridspec.GridSpec(3, 1, left=0.1, bottom = 0, right=1, top=0.95) 

ax = []
ax.append(fig.add_subplot(gs[0, 0]))
ax.append(fig.add_subplot(gs[1, 0]))
ax.append(fig.add_subplot(gs[2, 0]))

for a in ax:
    a.set_axis_off()
    
fig.tag(axes= ax, labels= ['A', 'B', 'C'])
    
plt.savefig('basic_fig.pdf')
