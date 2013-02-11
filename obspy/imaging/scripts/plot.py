#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
USAGE: obspy-plot [ -f format ] file

Wiggle plot of the data in file
"""
from obspy import read
from obspy import __version__
from optparse import OptionParser


def main():
    parser = OptionParser(__doc__.strip(), version="%prog " + __version__)
    parser.add_option("-f", default=None, type="string",
                      dest="format", help="Waveform format.")

    (options, args) = parser.parse_args()
    st = read(args[0], format=options.format)
    st.plot()


if __name__ == "__main__":
    main()
