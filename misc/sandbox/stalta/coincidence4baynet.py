#!/usr/bin/env python
"""
USAGE: ./coincidence4baynet.py file1.pick file2.pick

Build Coincidence for Baynet. file1.pick, file2.pick ...
are the output of ./stalta4baynet.py and look for station RMOA like:

2009-05-01T00:11:13.499999Z,2009-05-01T00:11:43.499999Z,RMOA
2009-05-01T15:33:26.299999Z,2009-05-01T15:33:33.515000Z,RMOA
...
"""

import os, glob, sys
from obspy.core import UTCDateTime
from pprint import pprint


MIN_STATIONS = 3


if len(sys.argv) == 1:
    print __doc__
    sys.exit(1)


trigger_list = []
for file in sys.argv[1:]:
    try:
        f = open(file)
    except IOError:
        continue
    for line in f:
        begin, end, station = line.strip().split(",")
        trigger_list.append((UTCDateTime(begin).timestamp,
                             UTCDateTime(end).timestamp,
                             station))

trigger_list.sort()
#pprint(trigger_list)


while len(trigger_list) > 1:
    coinc = set()
    on, off, sta = trigger_list[0]
    coinc.add(sta)
    for i in range(1, len(trigger_list)):
        tmp_on, tmp_off, tmp_sta = trigger_list[i]
        if tmp_on < off:
            #import ipdb; ipdb.set_trace()
            coinc.add(tmp_sta)
            off = max(off, tmp_off)
        else:
            break
    if len(coinc) >= MIN_STATIONS:
        print "%s %04.1f %s" % (UTCDateTime(on), off - on, coinc)
    # index i marks the index of the next non-matching pick
    trigger_list = trigger_list[i:]

