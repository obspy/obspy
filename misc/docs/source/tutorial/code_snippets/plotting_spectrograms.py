import obspy

st = obspy.read("https://examples.obspy.org/RJOB_061005_072159.ehz.new")
st.spectrogram(log=True, title='BW.RJOB ' + str(st[0].stats.starttime))
