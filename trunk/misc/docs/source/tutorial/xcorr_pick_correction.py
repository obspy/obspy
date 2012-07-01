from obspy.core import read, UTCDateTime
from obspy.signal.cross_correlation import xcorrPickCorrection

st1 = read("http://examples.obspy.org/BW.UH1..EHZ.D.2010.147.a.slist.gz")
st2 = read("http://examples.obspy.org/BW.UH1..EHZ.D.2010.147.b.slist.gz")
tr1 = st1.select(component="Z")[0]
tr2 = st2.select(component="Z")[0]
t1 = UTCDateTime("2010-05-27T16:24:33.315000Z")
t2 = UTCDateTime("2010-05-27T16:27:30.585000Z")

print xcorrPickCorrection(t1, tr1, t2, tr2, 0.05, 0.2, 0.1, plot=True)
print xcorrPickCorrection(t1, tr1, t2, tr2, 0.05, 0.2, 0.1, filter="bandpass",
        filter_options={'freqmin': 1, 'freqmax': 10}, plot=True)
