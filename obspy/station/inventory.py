#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides the Inventory class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import obspy
from obspy.station import stationxml


# Manual "plug-in" system. Likely sufficient for the limited number of possible
# formats.
FORMAT_FCTS = {
    "stationxml": {
        "is_fct": stationxml.is_StationXML,
        "read_fct": stationxml.read_StationXML}
}


def readInventory(path_or_file_object, format=None):
    """
    Function to read inventory files.

    :param path_or_file_object: Filename or file like object.
    """
    if format:
        fileformat = format.lower()
        if fileformat not in FORMAT_FCTS.keys():
            msg = "Unsupported format '%s'.\nSupported formats: %s" % (
                format, ", ".join(sorted(FORMAT_FCTS.keys())))
            raise ValueError(msg)
    else:
        found_format = False
        for fileformat, value in FORMAT_FCTS.iteritems():
            if value["is_fct"](path_or_file_object):
                found_format = True
                break
        if found_format is not True:
            msg = "Unsupported fileformat.\nSupported formats: %s" % (
                ", ".join(sorted(FORMAT_FCTS.keys())))
            raise ValueError(msg)
    return FORMAT_FCTS[fileformat]["read_fct"](path_or_file_object)


class SeismicInventory(object):
    """
    The root object of the Inventory->Network->Station->Channel hierarchy.

    In essence just a container for one or more networks.
    """
    def __init__(self, networks, source, sender=None, created=None):
        """
        :type networks: List of :class:`~obspy.station.network.SeismicNetwork`
        :param networks: A list of networks part of this inventory.
        :type source: String
        :param source: Network ID of the institution sending the message.
        :type sender: String
        :param sender: Name of the institution sending this message. Optional.
        :type created: :class:`~obspy.core.utcddatetime.UTCDateTime`
        :param creatd: The time when the document was created. Will be set to
            the current time if not given. Optional.
        """
        self.networks = networks
        # Set the created field to the current time if not given otherwise.
        if created is None:
            self.created = obspy.UTCDateTime()
        else:
            self.created = created

    @property
    def networks(self):
        return self.__networks

    @networks.setter
    def networks(self, value):
        if not hasattr(value, "__iter__"):
            msg = "networks needs to be iterable, e.g. a list."
            raise ValueError(msg)
        self.__networks = value
