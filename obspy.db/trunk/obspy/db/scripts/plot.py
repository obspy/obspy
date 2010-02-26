#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
from obspy.core import read
from obspy.db import __version__
from optparse import OptionParser


def main():
    usage = "USAGE: %prog [options]\n\n" + \
            "\n".join(__doc__.split("\n")[3:])
    parser = OptionParser(usage.strip(), version="%prog " + __version__)
    parser.add_option("-f", default=None, type="string",
                      dest="format", help="Waveform format.")

    (options, args) = parser.parse_args()
    st = read(args[0], format=options.format)
    st.plot()


if __name__ == "__main__":
    main()
