import obspy


st = obspy.read('http://examples.obspy.org/RJOB_061005_072159.ehz.new')
st.plot()
