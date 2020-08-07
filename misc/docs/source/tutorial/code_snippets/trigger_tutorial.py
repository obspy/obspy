import obspy

st = obspy.read("https://examples.obspy.org/ev0_6.a01.gse2")
st = st.select(component="Z")
tr = st[0]
tr.plot(type="relative")
