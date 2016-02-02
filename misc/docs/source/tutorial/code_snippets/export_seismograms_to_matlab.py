from scipy.io import savemat

import obspy


st = obspy.read("https://examples.obspy.org/BW.BGLD..EH.D.2010.037")
for i, tr in enumerate(st):
    mdict = {k: str(v) for k, v in tr.stats.iteritems()}
    mdict['data'] = tr.data
    savemat("data-%d.mat" % i, mdict)
