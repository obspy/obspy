import obspy
from obspy.io.xseed import Parser


st = obspy.read("https://examples.obspy.org/BW.BGLD..EH.D.2010.037")
parser = Parser("https://examples.obspy.org/dataless.seed.BW_BGLD")
st.simulate(seedresp={'filename': parser, 'units': "DIS"})
