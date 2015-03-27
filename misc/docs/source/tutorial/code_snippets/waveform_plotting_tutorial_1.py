import obspy


singlechannel = obspy.read('http://examples.obspy.org/COP.BHZ.DK.2009.050')
singlechannel.plot()
