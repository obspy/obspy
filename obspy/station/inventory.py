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


class SeismicInventory(object):
    """
    The root object of the Inventory->Network->Station->Channel hierarchy.

    In essence just a container for one or more networks.
    """
    def __init__(self, source, sender=None, created=None, networks=[]):
        """
        :type source: String
        :param source: Network ID of the institution sending the message.
        :type sender: String
        :param sender: Name of the institution sending this message. Optional.
        :type created: :class:`~obspy.core.utcddatetime.UTCDateTime`
        :param creatd: The time when the document was created. Will be set to
            the current time if not given. Optional.
        :type networks: List of :class:`~obspy.station.network.SeismicNetwork`
        :param networks: A list of networks part of this inventory.
        """
        self.networks = []
        # The module should correspond to the software module that created the
        # document (in StationXML).
        self.module = "ObsPy %s" % obspy.__version__
        self.module_uri = "http://www.obspy.org"
        # The created field is, by
        self.created = obspy.UTCDateTime()

    @property
    def networks(self):
        return self.__networks

    @networks.setter
    def networks(self, value):
        if not hasattr(value, "__iter__"):
            msg = "networks needs to be iterable, e.g. a list."
            raise ValueError(value)
        self.__networks = value
