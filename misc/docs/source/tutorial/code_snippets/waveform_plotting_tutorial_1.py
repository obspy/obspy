import obspy
import matplotlib.pyplot as plt


singlechannel = obspy.read('https://examples.obspy.org/COP.BHZ.DK.2009.050')
singlechannel.plot()
plt.show()
