# -*- coding: utf-8 -*-

import os

from obspy.seed import SEEDParser

for root,dir,files in os.walk(os.path.join("data", "orfeus")):
    filelist = [ os.path.join(root,fi) for fi in files]
    for f in filelist: 
        try:
            sp = SEEDParser(f, strict=True)
        except Exception, e:
            sp = SEEDParser(f, debug=True, strict=True)
