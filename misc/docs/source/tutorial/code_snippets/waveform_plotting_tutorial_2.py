import obspy


singlechannel = obspy.read('https://examples.obspy.org/COP.BHZ.DK.2009.050')
dt = singlechannel[0].stats.starttime
singlechannel.plot(color='red', number_of_ticks=7, tick_rotation=5,
                   tick_format='%I:%M %p', starttime=dt + 60 * 60,
                   endtime=dt + 60 * 60 + 120)
