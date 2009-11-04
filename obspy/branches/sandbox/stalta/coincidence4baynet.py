#!/usr/bin/env python
"""
Build Coincidence for Baynet
"""

import os, glob, sys
from obspy.core import UTCDateTime
from pprint import pprint

trigger_list = []
for file in sys.argv[1:]:
    for line in open(file):
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
