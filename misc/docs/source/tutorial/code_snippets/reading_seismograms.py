from obspy.core import read


st = read('http://examples.obspy.org/RJOB_061005_072159.ehz.new')
st.plot()
