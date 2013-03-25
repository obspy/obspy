#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from lxml.builder import E

import obspy


# Define some constants for writing StationXML files.
SOFTWARE_MODULE = "ObsPy %s" % obspy.__version__
SOFTWARE_URI = "http://www.obspy.org"
SCHEME_VERSION = "1.0"


def write_StationXML(inventory, buffer):
    """
    Writes an inventory object to a buffer.

    :type inventory: :class:`~obspy.station.inventory.SeismicInventory`
    :param inventory: The inventory instance to be written.
    :type buffer: Open file or file-like object.
    :param buffer: The file buffer object the StationXML will be written to.
    """

    # Use the etree factory to create the very basic structure.
    xml_doc = E.FSDNStationXML(
        E.Source(str(inventory.source),

