#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Filename: reftekrescue.py
#  Purpose: Restore REFTEK data from raw binary data dumps
#   Author: Tobias Megies
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2011 Tobias Megies
#------------------------------------------------------------------------------
"""
Restore REFTEK data from raw binary data dumps.

This program is intended for restoring REFTEK 130-01 packets from raw binary
dumped disk images, e.g. from formatted but not yet (completely) overwritten
storage media. The raw dumped data is searched for a header pattern consisting
of experiment number, year and REFTEK DAS ID.
Found packets are written to one file per recording event like in normal
acquisition. The output filenames consist of (separated by dots):

- REFTEK DAS ID
- recording event number
- packet information (number of found EH-ET-DT packets)
- 'ok' or 'bad' depending on the number of different packet types found
- 'reftek' file suffix

The restored REFTEK data can then be converted to other formats using available
conversion tools.

.. seealso::
    For details on the data format specifications of the REFTEK packets refer
    to http://support.reftek.com/support/130-01/doc/130_record.pdf.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from optparse import OptionParser
import warnings
import mmap
import contextlib
import os
from binascii import b2a_hex, a2b_hex

# The REFTEK documentation defines other packets too, but these seem to be the
# only ones appearing in normal acquisition.
# see http://support.reftek.com/support/130-01/doc/130_record.pdf
# The longer the search patterns, the safer the identification of a packet
# header starting point. The search could probably be improved using the
# regular expressions module.
PACKET_TYPES = ('DT', 'EH', 'ET')


def reftek_rescue(input_file, output_folder, reftek_id, year,
                  experiment_number):
    """
    """
    # Make binary representation of search pattern
    pattern = experiment_number + year + reftek_id
    pattern = a2b_hex(pattern)

    # In REFTEK nomenclature an 'event' in normal acquisition seems to be a
    # piece of continuous registration that gets written to a single file
    # consisting of one EH (event header) packet, many DT (data) packets and
    # one ET (event trailer) packet.
    # The event number is coded in the EH/DT/ET packets directly after the
    # header fields common to all packet types. Counting the different packet
    # types found for an event number gives some kind of indication if it
    # seems like the event can be reconstructed OK.
    event_info = {}

    # memory map the file
    with open(input_file, 'r') as f:
        fno = f.fileno()
        access = mmap.ACCESS_READ
        with contextlib.closing(mmap.mmap(fno, 0, access=access)) as m:

            # pos marks the current position for the pattern search
            # (searched pattern starts 2 bytes after packet start)
            pos = m.find(pattern, 2)
            # abort when no new occurrence of pattern is found
            while pos > -1:
                # ind marks the actual packet start 2 bytes left of pos
                ind = pos - 2
                # if it seems we have found a packet, process it
                pt = m[ind:(ind + 2)]
                if pt in PACKET_TYPES:
                    # all packet types have the same 16 byte header
                    header = m[ind:(ind + 16)]
                    # from byte 3 onward information is stored in packed BCD
                    # format
                    header = header[:2] + b2a_hex(header[2:])
                    header = header.upper()
                    # get event number, encoded in 2 bytes after the header
                    # at least for packet types 'DT', 'EH' and 'ET'
                    try:
                        event_no = int(b2a_hex(m[(ind + 16):(ind + 18)]))
                    except:
                        msg = "Could not decode event number. Dropping " + \
                              "possibly corrupted packet at byte position" + \
                              " %d in input file."
                        msg = msg % ind
                        warnings.warn(msg)
                        pos = m.find(pattern, pos + 1)
                        continue
                    # add event/packettype information to dictionary
                    d = event_info.setdefault(event_no,
                                              {'EH': 0, 'ET': 0, 'DT': 0})
                    d[pt] += 1
                    # all packets consist of 1024 bytes
                    packet = m[ind:(ind + 1024)]
                    # write to output folder, one file per recording event
                    filename = "%s.%04d" % (reftek_id, event_no)
                    filename = os.path.join(output_folder, filename)
                    open(filename, "ab").write(packet)
                # search for pattern in memory map starting right of last
                # position
                pos = m.find(pattern, pos + 1)

    # rename event files with packet information included
    for ev_no, ev_info in event_info.iteritems():
        filename_old = "%s.%04d" % (reftek_id, ev_no)
        filename_new = filename_old + \
                ".%d-%d-%05d" % (ev_info['EH'], ev_info['ET'], ev_info['DT'])
        if ev_info['EH'] != 1 or ev_info['ET'] != 1 or ev_info['DT'] < 1:
            filename_new += ".bad"
        else:
            filename_new += ".ok"
        filename_new += ".reftek"
        filename_old = os.path.join(output_folder, filename_old)
        filename_new = os.path.join(output_folder, filename_new)
        os.rename(filename_old, filename_new)


def main():
    warnings.simplefilter('always')
    # would make more sense to use argparse module but it is not part of the
    # Python Standard Library for versions < 2.7
    usage = "USAGE: %prog [options]\n\n" + \
            "\n".join(__doc__.split("\n")[3:])
    parser = OptionParser(usage.strip())
    parser.add_option("-i", "--input-file", default="/export/data/A03F.IMG",
                      type="string", dest="input_file",
                      help="Path and filename of input file.")
    parser.add_option("-e", "--experiment-number", default="00",
                      type="string", dest="experiment_number",
                      help="Experiment number set during acquisition " + \
                           "(2 decimal characters)")
    parser.add_option("-r", "--reftek-id", default="A03F",
                      type="string", dest="reftek_id",
                      help="REFTEK DAS ID of unit used for acquisition " + \
                           "(4 hex characters)")
    parser.add_option("-y", "--year", default="11",
                      type="string", dest="year",
                      help="Year of acquisition (last 2 characters)")
    parser.add_option("-o", "--output-folder", default="/export/data/rescue",
                      type="string", dest="output_folder",
                      help="Folder for output of reconstructed data. " + \
                           "An empty folder has to be specified.")
    (options, _) = parser.parse_args()
    # be friendly, do some checks.
    msg = "Invalid length for "
    if len(options.experiment_number) != 2:
        msg += "experiment number."
        raise ValueError(msg)
    if len(options.year) != 2:
        msg += "year."
        raise ValueError(msg)
    if len(options.reftek_id) != 4:
        msg += "REFTEK DAS ID."
        raise ValueError(msg)
    # check if output folder is empty (and implicitly if it is there at all)
    if os.listdir(options.output_folder) != []:
        msg = "Output directory must be empty as data might get appended " + \
              "to existing files otherwise."
        raise Exception(msg)

    reftek_rescue(options.input_file, options.output_folder, options.reftek_id,
                  options.year, options.experiment_number)


if __name__ == "__main__":
    # It is not possible to add the code of main directly to here.
    # This script is automatically installed with name obspy-runtests by
    # setup.py to the Scripts or bin directory of your Python distribution
    # setup.py needs a function to which it's scripts can be linked.
    main()
