#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
USAGE: obspy-print [ -f format ] file1 file2 ...

Print stream information for waveform data in local files
"""
from obspy import read, Stream
from obspy import __version__
from optparse import OptionParser


def main():
    parser = OptionParser(__doc__.strip(), version="%prog " + __version__)
    parser.add_option("-f", default=None, type="string",
                      dest="format", help="Waveform format.")
    parser.add_option("-n", "--nomerge", default=True, action="store_false",
                      dest="merge", help="Switch off cleanup merge.")
    parser.add_option("-g", "--print-gaps", default=False, action="store_true",
                      dest="print_gaps",
                      help="Switch on printing of gap information.")

    (options, args) = parser.parse_args()
    st = Stream()
    for arg in args:
        st += read(arg, format=options.format)
    if options.merge:
        st.merge(-1)
    print st
    if options.print_gaps:
        print
        st.printGaps()


if __name__ == "__main__":
    main()
