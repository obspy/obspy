import obspy


st = obspy.read('https://examples.obspy.org/RJOB_061005_072159.ehz.new')
st.plot()
