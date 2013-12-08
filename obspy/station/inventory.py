#d!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides the Inventory class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from pkg_resources import load_entry_point
import obspy
from obspy.core.util.base import ComparingObject
from obspy.core.util import getExampleFile
from obspy.core.util.base import ENTRY_POINTS, _readFromPlugin
from obspy.station.stationxml import SOFTWARE_MODULE, SOFTWARE_URI
from obspy.station.network import Network
import textwrap
import warnings
from copy import deepcopy


def read_inventory(path_or_file_object, format=None):
    """
    Function to read inventory files.

    :param path_or_file_object: Filename or file like object.
    """
    # if pathname starts with /path/to/ try to search in examples
    if isinstance(path_or_file_object, basestring) and \
       path_or_file_object.startswith('/path/to/'):
        try:
            path_or_file_object = getExampleFile(path_or_file_object[9:])
        except:
            # otherwise just try to read the given /path/to folder
            pass
    return _readFromPlugin("inventory", path_or_file_object, format=format)[0]


class Inventory(ComparingObject):
    """
    The root object of the Inventory->Network->Station->Channel hierarchy.

    In essence just a container for one or more networks.
    """
    def __init__(self, networks, source, sender=None, created=None,
                 module=SOFTWARE_MODULE, module_uri=SOFTWARE_URI):
        """
        :type networks: List of :class:`~obspy.station.network.Network`
        :param networks: A list of networks part of this inventory.
        :type source: String
        :param source: Network ID of the institution sending the message.
        :type sender: String
        :param sender: Name of the institution sending this message. Optional.
        :type created: :class:`~obspy.core.utcddatetime.UTCDateTime`
        :param created: The time when the document was created. Will be set to
            the current time if not given. Optional.
        :type module: String
        :param module: Name of the software module that generated this
            document, defaults to ObsPy related information.
        :type module_uri: String
        :param module_uri: This is the address of the query that generated the
            document, or, if applicable, the address of the software that
            generated this document, defaults to ObsPy related information.
        """
        self.networks = networks
        self.source = source
        self.sender = sender
        self.module = module
        self.module_uri = module_uri
        # Set the created field to the current time if not given otherwise.
        if created is None:
            self.created = obspy.UTCDateTime()
        else:
            self.created = created

    def __add__(self, other):
        new = deepcopy(self)
        if isinstance(other, Inventory):
            new.networks.extend(other.networks)
        elif isinstance(other, Network):
            new.networks.append(other)
        else:
            msg = ("Only Inventory and Network objects can be added to "
                   "an Inventory.")
            raise TypeError(msg)
        return new

    def __iadd__(self, other):
        if isinstance(other, Inventory):
            self.networks.extend(other.networks)
        elif isinstance(other, Network):
            self.networks.append(other)
        else:
            msg = ("Only Inventory and Network objects can be added to "
                   "an Inventory.")
            raise TypeError(msg)
        return self

    def __getitem__(self, index):
        return self.networks[index]

    def get_contents(self):
        """
        Returns a dictionary containing the contents of the object.

        Example
        >>> example_filename = "/path/to/IRIS_single_channel_with_response.xml"
        >>> inventory = read_inventory(example_filename)
        >>> inventory.get_contents()  # doctest: +NORMALIZE_WHITESPACE
        {'channels': ['IU.ANMO.10.BHZ'],
         'networks': ['IU'],
         'stations': [u'IU.ANMO (Albuquerque, New Mexico, USA)']}
        """
        content_dict = {
            "networks": [],
            "stations": [],
            "channels": []}
        for network in self.networks:
            content_dict['networks'].append(network.code)
            for key, value in network.get_contents().iteritems():
                content_dict.setdefault(key, [])
                content_dict[key].extend(value)
                content_dict[key].sort()
        return content_dict

    def __str__(self):
        ret_str = "Inventory created at %s\n" % str(self.created)
        if self.module:
            module_uri = self.module_uri
            if module_uri and len(module_uri) > 70:
                module_uri = textwrap.wrap(module_uri, width=67)[0] + "..."
            ret_str += ("\tCreated by: %s%s\n" % (
                self.module,
                "\n\t\t    %s" % (module_uri if module_uri else "")))
        ret_str += "\tSending institution: %s%s\n" % (
            self.source, " (%s)" % self.sender if self.sender else "")
        contents = self.get_contents()
        ret_str += "\tContains:\n"
        ret_str += "\t\tNetworks (%i):\n" % len(contents["networks"])
        ret_str += "\n".join(["\t\t\t%s" % _i for _i in contents["networks"]])
        ret_str += "\n"
        ret_str += "\t\tStations (%i):\n" % len(contents["stations"])
        ret_str += "\n".join(["\t\t\t%s" % _i for _i in contents["stations"]])
        ret_str += "\n"
        ret_str += "\t\tChannels (%i):\n" % len(contents["channels"])
        ret_str += "\n".join(textwrap.wrap(
            ", ".join(contents["channels"]), initial_indent="\t\t\t",
            subsequent_indent="\t\t\t", expand_tabs=False))
        return ret_str

    def write(self, path_or_file_object, format, **kwargs):
        """
        Writes the inventory object to a file or file-like object in
        the specified format.

        :param path_or_file_object: Filename or file-like object to be written
            to.
        :param format: The format of the written file.
        """
        format = format.upper()
        try:
            # get format specific entry point
            format_ep = ENTRY_POINTS['inventory_write'][format]
            # search writeFormat method for given entry point
            writeFormat = load_entry_point(
                format_ep.dist.key,
                'obspy.plugin.inventory.%s' % (format_ep.name), 'writeFormat')
        except (IndexError, ImportError, KeyError):
            msg = "Writing format \"%s\" is not supported. Supported types: %s"
            raise TypeError(msg % (format,
                                   ', '.join(ENTRY_POINTS['inventory_write'])))
        return writeFormat(self, path_or_file_object, **kwargs)

    @property
    def networks(self):
        return self._networks

    @networks.setter
    def networks(self, value):
        if not hasattr(value, "__iter__"):
            msg = "networks needs to be iterable, e.g. a list."
            raise ValueError(msg)
        if any([not isinstance(x, Network) for x in value]):
            msg = "networks can only contain Network objects."
            raise ValueError(msg)
        self._networks = value

    def get_response(self, seed_id, datetime):
        """
        Find response for a given channel at given time.

        >>> from obspy import read_inventory, UTCDateTime
        >>> inventory = read_inventory("/path/to/BW_RJOB.xml")
        >>> datetime = UTCDateTime("2009-08-24T00:20:00")
        >>> response = inventory.get_response("BW.RJOB..EHZ", datetime)
        >>> print response  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Channel Response
           From M/S (Velocity in Meters Per Second) to COUNTS (Digital Counts)
           Overall Sensitivity: 2.5168e+09 defined at 0.020 Hz
           4 stages:
              Stage 1: PolesZerosResponseStage from M/S to V, gain: 1500.00
              Stage 2: CoefficientsTypeResponseStage from V to COUNTS, ...
              Stage 3: FIRResponseStage from COUNTS to COUNTS, gain: 1.00
              Stage 4: FIRResponseStage from COUNTS to COUNTS, gain: 1.00

        :type seed_id: str
        :param seed_id: SEED ID string of channel to get response for.
        :type datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param datetime: Time to get response for.
        :rtype: :class:`~obspy.station.response.Response`
        :returns: Response for timeseries specified by input arguments.
        """
        network, station, location, channel = seed_id.split(".")
        networks = [net for net in self.networks if net.code == network]
        stations = [sta for sta in net.stations if sta.code == station
                    for net in networks]
        channels = [cha for sta in stations for cha in sta.channels
                    if cha.code == channel
                    and cha.location_code == location
                    and (cha.start_date is None or cha.start_date <= datetime)
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
