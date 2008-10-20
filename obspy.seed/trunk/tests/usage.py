# -*- coding: utf-8 -*-

from obspy.seed import SEEDParser


sp = SEEDParser(verify=True)
sp.parseSEEDFile('data/dataless.seed.BW_ZUGS')
fp = open('output/dataless.seed.BW_ZUGS.xml','w')
fp.write(sp.getXML())
fp.close()

sp = SEEDParser()
sp.parseSEEDFile('data/dataless.seed')
fp = open('output/dataless.seed.xml','w')
fp.write(sp.getXML())
fp.close()
