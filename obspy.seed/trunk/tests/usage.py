# -*- coding: utf-8 -*-

from lxml.etree import tostring

from obspy.seed import SEEDParser


sp = SEEDParser('data/dataless.seed.BW_ZUGS', verify=True)
fp = open('output/dataless.seed.BW_ZUGS.xml','w')
fp.write(tostring(sp.getXML(), pretty_print=True))
fp.close()

sp = SEEDParser('data/dataless.seed')
fp = open('output/dataless.seed.xml','w')
fp.write(tostring(sp.getXML(), pretty_print=True))
fp.close()
