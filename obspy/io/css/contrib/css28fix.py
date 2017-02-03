#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick and dirty conversion routine from CSS 2.8 to Seismic Handler ASCII format

- expects wfdisc index file as parameter
- processes only first line of wfdisc (no support for multiple streams)
- output written to stdout
- shows plot for inspection (if matplotlib installed)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import struct
import sys
from datetime import datetime


# read header (assume CSS2.8)
try:
    # only process first line
    head = list(map(str.strip, open(sys.argv[1]).readlines()[0].split()))
except Exception:
    sys.exit("cannot read wfdisc file (arg)")

input = head[15]  # input file name
# 0 -> timestamp, 1-> milliseconds
timedata = list(map(int, head[1].split(".")))

# headers for SH ASCII file
SH = {
    "STATION": head[2],
    "COMP": head[3][-1].upper(),  # guess from channel naming
    "START": ".".join((
        datetime.fromtimestamp(timedata[0]).strftime("%d-%b-%Y_%H:%M:%S"),
        "%03d" % timedata[1])),
    "DELTA": 1.0 / float(head[5]),
    "LENGTH": int(head[4]),
    "CALIB": float(head[6])
}

# binary format (big endian integers)
fmt = ">" + "i" * SH["LENGTH"]

# convert binary data
data = struct.unpack(fmt, open(input, "rb").read(struct.calcsize(fmt)))

# echo headers
for header in SH:
    print(": ".join((header, str(SH[header]))))

# echo data
for dat in data:
    # CALIB factor
    print("%e" % (dat * SH["CALIB"],))

# inspection plot
try:
    import pylab
    pylab.plot([x * SH["DELTA"] for x in range(SH["LENGTH"])],
               [d * SH["CALIB"] for d in data])
    pylab.show()
except Exception:
    sys.exit("cannot show plot!")
