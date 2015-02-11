from obspy import read
from obspy.xseed import Parser


st = read("http://examples.obspy.org/BW.BGLD..EH.D.2010.037")
parser = Parser("http://examples.obspy.org/dataless.seed.BW_BGLD")
st.simulate(seedresp={'filename': parser, 'units': "DIS"})
