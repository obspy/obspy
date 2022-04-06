import obspy

singlechannel = obspy.read('https://examples.obspy.org/COP.BHZ.DK.2009.050')
fig = singlechannel.plot(handle=True)
