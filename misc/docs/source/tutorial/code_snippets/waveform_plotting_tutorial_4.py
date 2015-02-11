from obspy.core import read


singlechannel = read('http://examples.obspy.org/COP.BHZ.DK.2009.050')
singlechannel.plot(type='dayplot')
