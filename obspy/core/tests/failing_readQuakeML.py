#!/usr/bin/env python
# -*- coding: utf8 -*-
from obspy import readEvents

cat = readEvents('data/neries_events.xml')
print 'returned resource_id:', cat.resource_id
