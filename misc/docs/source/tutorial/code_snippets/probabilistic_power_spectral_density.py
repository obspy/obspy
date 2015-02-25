from obspy.core import read
from obspy.signal import PPSD
from obspy.xseed import Parser


st = read("http://examples.obspy.org/BW.KW1..EHZ.D.2011.037")
tr = st.select(id="BW.KW1..EHZ")[0]
parser = Parser("http://examples.obspy.org/dataless.seed.BW_KW1")
paz = parser.getPAZ(tr.id)
ppsd = PPSD(tr.stats, paz)
ppsd.add(st)

st = read("http://examples.obspy.org/BW.KW1..EHZ.D.2011.038")
ppsd.add(st)

ppsd.plot()
