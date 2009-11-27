# -*- coding: utf-8 -*-
# 2009-11-27 Moritz

import numpy as np
import matplotlib.pyplot as plt

def stringToBool(s):
    if s == 'S':
        return True
    return False


# do the fourier transformation
data = np.loadtxt("china8b.asc",usecols=[1])
data -= data.mean()
df = 0.1
fdat = np.fft.rfft(data)
fdat /= fdat.max() #normalize to 1

# get the eigenmodes
eigen = np.loadtxt("eiglst", usecols=[0,1,2,3], converters={1:stringToBool})
# only the S part
import pdb; pdb.set_trace()
ind1 = eigen[:,1].astype(bool)
ind2 = eigen[:,0]
ind = ((ind2 == 0) & ind1)
modes = eigen[ind,3]/1000 #normalize

# plot the first 1000 points results
freq = np.linspace(0,df/2,len(data)/2)
freq = freq[1:1001]
fdat = fdat[0:1000]
plt.clf()
plt.plot(freq,abs(fdat))
plt.vlines(modes[0:len(modes)/2],0,1)
plt.show()
