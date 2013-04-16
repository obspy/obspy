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
    def __init__(self, code, stations=[], total_number_of_stations=None,
            selected_number_of_stations=None, description=None, comments=[],
            start_date=None, end_date=None, restricted_status=None,
            alternate_code=None, historical_code=None):
        """
        :type code: String
        :type code: The SEED network code.
        :type total_number_of_stations: int
        :param  total_number_of_stations: The total number of stations
            contained in this networkork, including inactive or terminated
            stations.
        :param selected_number_of_stations: The total number of stations in
            this network that were selected by the query that produced this
            document, even if the stations do not appear in the document. (This
            might happen if the user only wants a document that goes contains
            only information at the Network level.)
        :type description: String, optional
        :param description: A description of the resource
        :type comments: List of :class:`~obspy.station.util.Comment`, optional
        :param comments: An arbitrary number of comments to the resource
        :type start_date: :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param start_date: The start date of the resource
        :type end_date: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param end_date: The end date of the resource
        :type restricted_status: String, optional
        :param restricted_status: The restriction status
        :type alternate_code: String, optional
        :param alternate_code: A code used for display or association,
            alternate to the SEED-compliant code.
        :type historical_code: String, optional
        :param historical_code: A previously used code if different from the
            current code.
        """
        self.stations = stations
        self.total_number_of_stations = total_number_of_stations
        self.selected_number_of_stations = selected_number_of_stations

        super(SeismicNetwork, self).__init__(code=code,
            description=description, comments=comments, start_date=start_date,
            end_date=end_date, restricted_status=restricted_status,
            alternate_code=alternate_code, historical_code=historical_code)

    def get_contents(self):
        """
        Returns a dictionary containing the contents of the object.

        Example
        >>> network_object.get_contents()  # doctest: +SKIP
        {"networks": ["BW (Bavarian Network)"],
         "stations": ["BW.A", "BW.B"],
         "channels": ["BW.A..EHE", "BW.A..EHN", ...]}
        """
        desc = "%s%s" % (self.code, " (%s)" % (self.description if
            self.description else ""))
        content_dict = {"networks": [desc], "stations": [],
            "channels": []}

        for station in self.stations:
            contents = station.get_contents()
            content_dict["stations"].extend(
                "%s.%s" % (self.code, _i) for _i in contents["stations"])
            content_dict["channels"].extend(
                "%s.%s" % (self.code, _i) for _i in contents["channels"])
        return content_dict

    @property
    def stations(self):
        return self.__stations

    @stations.setter
    def stations(self, values):
        if not hasattr(values, "__iter__"):
            msg = "stations needs to be iterable, e.g. a list."
            raise ValueError(msg)
        self.__stations = values

    def __short_str__(self):
        return "%s" % self.code
