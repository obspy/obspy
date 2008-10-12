# -*- coding: utf-8 -*-

from lxml.etree import tostring

from obspy.seed import SEEDParser


sp = SEEDParser('data/dataless.seed', verify=True)
fp = open('output/dataless.seed.xml','w')
fp.write(tostring(sp.doc, pretty_print=True))
fp.close()
print sp.blockettes.get(10)[0].__dict__
#
#sp = SEEDParser('data/dataless-odc.GR_WET')
#fp = open('output/dataless-odc.GR_WET.xml','w')
#fp.write(tostring(sp.doc, pretty_print=True))
#fp.close()