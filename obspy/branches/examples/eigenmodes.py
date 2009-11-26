import numpy as np
import matplotlib.pyplot as plt

def tors(x):
    if x == 'S':
        return True
    return False

# do the fourier transformation
data = np.loadtxt("china8b.asc",usecols=[1])
data -= data.mean()
df = 0.1
fdat = np.fft.rfft(data)
fdat /= fdat.max() #normalize to 1

# get the eigenmodes
eigen = np.loadtxt("eiglst", usecols=[1,3], converters={1:tors})
# only the S part
ind = eigen[:,0].astype(bool)
modes = eigen[ind,1]/1000 #normalize

# plot the first 1000 points results
freq = np.linspace(0,df/2,len(data)/2)
freq = freq[1:1001]
fdat = fdat[0:1000]
plt.clf()
plt.plot(freq,abs(fdat))
plt.vlines(modes[0:len(modes)/2],0,1)
plt.show()
