#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides the SeismicNetwork class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from obspy.station.util import BaseNode


class SeismicNetwork(BaseNode):
    """
    From the StationXML definition:
        This type represents the Network layer, all station metadata is
        contained within this element. The official name of the network or
        other descriptive information can be included in the Description
        element. The Network can contain 0 or more Stations.
    """
    def __init__(self, code, *args, **kwargs):
        """
        :type code: String
        :type code: The network code.
        :type total_number_of_stations: int
        :param  total_number_of_stations: The total number of stations
            contained in this networkork, including inactive or terminated
            stations.
        :param selected_number_of_stations: The total number of stations in
            this network that were selected by the query that produced this
            document, even if the stations do not appear in the document. (This
            might happen if the user only wants a document that goes contains
            only information at the Network level.)
        """
        self.total_number_of_stations = kwargs.get("total_number_of_stations",
            None)
        self.selected_number_of_stations = \
            kwargs.get("selected_number_of_stations", None)

        super(SeismicNetwork, self).__init__(code, *args, **kwargs)

    def __short_str__(self):
        return "%s" % self.code
