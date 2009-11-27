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

if len(sys.argv) == 1:
    print __doc__
    sys.exit(1)


trigger_list = []
for file in sys.argv[1:]:
    try:
        f = open(file)
    except:
        continue
    for line in f:
        begin, end, station = line.strip().split(",")
        trigger_list.append([UTCDateTime(begin).timestamp,
                             UTCDateTime(end).timestamp,
                             station])

trigger_list.sort()
#pprint(trigger_list)

count = set()
on, of, nix = trigger_list[0]
while len(trigger_list) > 2:
    begin, end, station = trigger_list.pop(0)
    if begin < of:
        count.add(station)
        if end > of:
            of = end
        continue
    else:
        if len(count) >= 4:
            print "%s %04.1f %s" % (UTCDateTime(on), of - on, count)
        count = set()
        on, of, nix = trigger_list[1]
#
if len(count) >= 4:
    print "%s %04.1f %s" % (UTCDateTime(on), of - on, count)
