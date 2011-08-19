from obspy.core import read

singlechannel = read('http://examples.obspy.org/COP.BHE.DK.2009.050')
singlechannel.plot(type='dayplot')
