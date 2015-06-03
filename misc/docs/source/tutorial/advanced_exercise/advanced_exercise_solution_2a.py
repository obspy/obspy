from __future__ import print_function

from math import log10

from obspy import read


st = read("../data/LKBD_WA_CUT.MSEED")

tr_n = st.select(component="N")[0]
ampl_n = max(abs(tr_n.data))

tr_e = st.select(component="E")[0]
ampl_e = max(abs(tr_e.data))

ampl = max(ampl_n, ampl_e)

epi_dist = 20

a = 0.018
b = 2.17
ml = log10(ampl * 1000) + a * epi_dist + b
print(ml)
