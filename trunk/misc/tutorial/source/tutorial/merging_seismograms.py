from obspy.core import read
import matplotlib.pyplot as plt
import numpy as np

# Read in all files starting with dis.
st = read("http://examples.obspy.org/dis.G.SCZ.__.BHE")
st += read("http://examples.obspy.org/dis.G.SCZ.__.BHE.1")
st += read("http://examples.obspy.org/dis.G.SCZ.__.BHE.2")

# Go through the stream object, determine time range in julian seconds
# and plot the data with a shared x axis
ax = plt.subplot(4, 1, 1)  # dummy for tying axis
for i in range(3):
    plt.subplot(4, 1, i + 1, sharex=ax)
    t = np.linspace(st[i].stats.starttime.timestamp,
                    st[i].stats.endtime.timestamp,
                    st[i].stats.npts)
    plt.plot(t, st[i].data)

# Merge the data together and show plot in a similar way
st.merge(method=1)
plt.subplot(4, 1, 4, sharex=ax)
t = np.linspace(st[0].stats.starttime.timestamp,
                st[0].stats.endtime.timestamp,
                st[0].stats.npts)
plt.plot(t, st[0].data, 'r')
plt.show()
