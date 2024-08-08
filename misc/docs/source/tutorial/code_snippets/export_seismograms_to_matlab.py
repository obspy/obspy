from scipy.io import savemat

import obspy


st = obspy.read("https://examples.obspy.org/BW.BGLD..EH.D.2010.037")
for i, tr in enumerate(st):
    mdict = {k: str(v) for k, v in tr.stats.items()}
    mdict['data'] = tr.data
    savemat(f"data-{i}.mat", mdict)
