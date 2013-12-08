#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides the Network class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from obspy.station.util import BaseNode
from obspy.station.station import Station
import textwrap
import warnings


class Network(BaseNode):
    """
    From the StationXML definition:
        This type represents the Network layer, all station metadata is
        contained within this element. The official name of the network or
        other descriptive information can be included in the Description
        element. The Network can contain 0 or more Stations.
    """
    def __init__(self, code, stations=None, total_number_of_stations=None,
                 selected_number_of_stations=None, description=None,
                 comments=None, start_date=None, end_date=None,
                 restricted_status=None, alternate_code=None,
                 historical_code=None):
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
        self.stations = stations or []
        self.total_number_of_stations = total_number_of_stations
        self.selected_number_of_stations = selected_number_of_stations

        super(Network, self).__init__(
            code=code, description=description, comments=comments,
            start_date=start_date, end_date=end_date,
            restricted_status=restricted_status, alternate_code=alternate_code,
            historical_code=historical_code)

    def __getitem__(self, index):
        return self.stations[index]

    def __str__(self):
        ret = ("Network {id} {description}\n"
               "\tStation Count: {selected}/{total} (Selected/Total)\n"
               "\t{start_date} - {end_date}\n"
               "\tAccess: {restricted} {alternate_code}{historical_code}\n")
        ret = ret.format(
            id=self.code,
            description="(%s)" % self.description if self.description else "",
            selected=self.selected_number_of_stations,
            total=self.total_number_of_stations,
            start_date=str(self.start_date),
            end_date=str(self.end_date) if self.end_date else "",
            restricted=self.restricted_status,
            alternate_code="Alternate Code: %s " % self.alternate_code if
            self.alternate_code else "",
            historical_code="historical Code: %s " % self.historical_code if
            self.historical_code else "")
        contents = self.get_contents()
        ret += "\tContains:\n"
        ret += "\t\tStations (%i):\n" % len(contents["stations"])
        ret += "\n".join(["\t\t\t%s" % _i for _i in contents["stations"]])
        ret += "\n"
        ret += "\t\tChannels (%i):\n" % len(contents["channels"])
        ret += "\n".join(textwrap.wrap(", ".join(
            contents["channels"]), initial_indent="\t\t\t",
            subsequent_indent="\t\t\t", expand_tabs=False))
        return ret

    def get_contents(self):
        """
        Returns a dictionary containing the contents of the object.

        Example
        >>> from obspy import read_inventory
        >>> example_filename = "/path/to/IRIS_single_channel_with_response.xml"
        >>> inventory = read_inventory(example_filename)
        >>> network = inventory.networks[0]
        >>> network.get_contents()  # doctest: +NORMALIZE_WHITESPACE
        {'channels': ['IU.ANMO.10.BHZ'],
         'stations': [u'IU.ANMO (Albuquerque, New Mexico, USA)']}
        """
        content_dict = {"stations": [], "channels": []}

        for station in self.stations:
            contents = station.get_contents()
            content_dict["stations"].extend(
                "%s.%s" % (self.code, _i) for _i in contents["stations"])
            content_dict["channels"].extend(
                "%s.%s" % (self.code, _i) for _i in contents["channels"])
        return content_dict

    @property
    def stations(self):
        return self._stations

    @stations.setter
    def stations(self, values):
        if not hasattr(values, "__iter__"):
            msg = "stations needs to be iterable, e.g. a list."
            raise ValueError(msg)
        if any([not isinstance(x, Station) for x in values]):
            msg = "stations can only contain Station objects."
            raise ValueError(msg)
        self._stations = values

    def __short_str__(self):
        return "%s" % self.code

    def get_response(self, seed_id, datetime):
        """
        Find response for a given channel at given time.

        :type seed_id: str
        :param seed_id: SEED ID string of channel to get response for.
        :type datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param datetime: Time to get response for.
        :rtype: :class:`~obspy.station.response.Response`
        :returns: Response for timeseries specified by input arguments.
        """
        network, station, location, channel = seed_id.split(".")
        if self.code != network:
            responses = []
        else:
            channels = [cha for sta in self.stations for cha in sta.channels
                        if cha.code == channel
                        and cha.location_code == location
                        and (cha.start_date is None
                             or cha.start_date <= datetime)
                        and (cha.end_date is None or cha.end_date >= datetime)]
            responses = [cha.response for cha in channels
                         if cha.response is not None]
        if len(responses) > 1:
            msg = "Found more than one matching response. Returning first."
            warnings.warn(msg)
        elif len(responses) < 1:
            msg = "No matching response information found."
            raise Exception(msg)
        return responses[0]


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
