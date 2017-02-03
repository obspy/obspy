#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import locale
import os
import struct
import sys
import time


locale.setlocale(locale.LC_ALL, 'C')

formats = (str, str, float, int, int, int, float, int, float, float, float,
           str, str, str, str, str, str, int, int, str)


def main(wfdisc):
    # read wfdisc file line by line
    for line in wfdisc:
        # split line into separate fields
        parts = line.split()

        i = 0
        for data in parts:
            # convert each field to desired format (string, float, int)
            parts[i] = formats[i](data)
            i += 1

        # build destination name
        destname = "%s-%s-%s-%s.ASC" % \
            (os.path.splitext(parts[16])[0], parts[0], parts[1],
             time.strftime("%Y%m%d-%H%M%S", time.gmtime(parts[2])))
        print("station %s, component %s, %u samples" % (parts[0], parts[1],
                                                        parts[7]))
        print("=> %s ..." % destname)

        # check if already there
        if os.path.exists(destname):
            print("I won't overwrite existing file \"%s\", skipping..." %
                  os.path.split(destname)[1])
            continue

        # read unnormalized data
        datatonorm = convert(parts)

        # normalize data
        normalized = []
        for i in datatonorm:
            normalized.append(i * parts[9])

        # write ASCII file
        out = open(destname, "w")
        # write headers
        for header in buildheader(parts):
            out.write("%s\n" % header)
        # write data
        for value in normalized:
            out.write("%e\n" % value)
        out.close()


def convert(parts):
    # open binary data file
    datafile = open(parts[16], "rb")

    fmt, size = calcfmt(format=parts[13], samples=parts[7])

    try:
        datafile.seek(parts[17])
        values = struct.unpack(fmt, datafile.read(size))
    except Exception:
        print("error reading binary packed data from \"%s\"" %
              os.path.split(parts[16])[1])
        return False

    datafile.close()

    # if its 4 byte format, we are done
    if parts[13].lower() in ["t4", "s4"]:
        return values
    # 3 byte format

    return False


def calcfmt(format, samples):
    # four byte floats
    if format.lower() == "s4":
        fmt = ">" + "i" * samples
        return (fmt, struct.calcsize(fmt))
    # 4 byte integer
    elif format.lower() == "t4":
        fmt = ">" + "f" * samples
        return (fmt, struct.calcsize(fmt))
    # 3 byte floats
    elif format.lower() == "s3":
        return (False, False)
    else:
        return (False, False)


def buildheader(parts):
    headers = []

    headers.append("DELTA: %e" % (1.0 / parts[8]))
    headers.append("LENGTH: %u" % parts[7])
    headers.append("STATION: %s" % parts[0])
    if len(parts[1]) == 3:
        comp = parts[1][2]
    else:
        comp = parts[1]
    headers.append("COMP: %s" % comp.upper())
    headers.append("START: %s" % time.strftime("%Y-%b-%d_%H:%M:%S",
                                               time.gmtime(parts[2])))

    return headers


if __name__ == '__main__':
    try:
        wfdisc = open(sys.argv[1])
    except IndexError:
        print("""
        Usage: css2asc wfdisc-file

        All traces referenced by the given wfdisc file will be converted
        to ASCII
        """)
    except IOError:
        print("Cannot access file \"%s\"!" % sys.argv[1])

    main(wfdisc)

    # close file
    try:
        wfdisc.close()
    except Exception:
        pass
