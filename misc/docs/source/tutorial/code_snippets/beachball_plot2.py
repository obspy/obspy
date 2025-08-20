import numpy as np
import matplotlib.pyplot as plt
from obspy.imaging.beachball import beach

files = ['beachballs_ipts1_iref1_lune_psmeca','beachballs_ipts1_iref2_lune_psmeca','beachballs_ipts1_iref3_lune_psmeca','beachballs_ipts1_iref4_lune_psmeca','beachballs_ipts1_iref5_lune_psmeca']
fig1, ax1 = plt.subplots(nrows=1,ncols=5,figsize=(12,6),sharey=True)
xticks = np.arange(-50,51,25)
yticks = np.arange(-100,101,20)
ax1[0].set_yticks(ticks=yticks,labels=yticks)
n = 0

# Read in the datasets, obtained from: https://github.com/carltape/mtbeach/tree/master/gmt/dfiles courtesy of C. Tape
for file in files:
    cols1 = [3, 4, 5, 6, 7, 8]
    cols2 = [0, 1, 2]
    
    mt_data = np.loadtxt(file, usecols=cols1)
    xyz_data = np.loadtxt(file, usecols=cols2)

    ax1[n].axis("equal")
    for mt, (x, y, depth) in zip(mt_data, xyz_data):
        coll = beach(mt, xy=(x, y), width=10, size=13, plot_zerotrace=False)
        ax1[n].add_collection(coll)

    ax1[n].set_xticks(ticks=xticks,labels=xticks)
    n += 1

fig1.suptitle("Datasets courtesy of C. Tape")
plt.tight_layout()
plt.show()
