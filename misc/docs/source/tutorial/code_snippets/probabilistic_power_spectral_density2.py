from obspy import read
from obspy.signal import PPSD
from obspy.imaging.cm import pqlx
from obspy.io.xseed import Parser


st = read("http://examples.obspy.org/BW.KW1..EHZ.D.2011.037")
parser = Parser("http://examples.obspy.org/dataless.seed.BW_KW1")
ppsd = PPSD(st[0].stats, metadata=parser)
ppsd.add(st)

st = read("http://examples.obspy.org/BW.KW1..EHZ.D.2011.038")
ppsd.add(st)

ppsd.plot(cmap=pqlx)
