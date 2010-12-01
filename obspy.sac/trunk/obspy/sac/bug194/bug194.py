from obspy.core import read

st = read("TEMP1", format='SACXY')
st.write("TEMP1.obspy", format='SACXY')
#st = read("testxy.sac", format='SACXY')
#st.write("testxy.sac.obspy", format='SACXY')
