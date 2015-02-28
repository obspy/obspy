from scipy.io import savemat

from obspy.core import read


st = read("http://examples.obspy.org/BW.BGLD..EH.D.2010.037")
for i, tr in enumerate(st):
    mdict = dict([[j, str(k)] for j, k in tr.stats.iteritems()])
    mdict['data'] = tr.data
    savemat("data-%d.mat" % i, mdict)
