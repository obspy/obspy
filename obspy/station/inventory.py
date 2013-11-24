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
import obspy
from obspy.core.util.base import ComparingObject
from obspy.station import stationxml
from obspy.station.network import Network
import textwrap


# Manual "plug-in" system. Likely sufficient for the limited number of
# available formats.
FORMAT_FCTS = {
    "stationxml": {
        "is_fct": stationxml.is_StationXML,
        "read_fct": stationxml.read_StationXML,
        "write_fct": stationxml.write_StationXML}
}


def read_inventory(path_or_file_object, format=None):
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


class Inventory(ComparingObject):
    """
    The root object of the Inventory->Network->Station->Channel hierarchy.

    In essence just a container for one or more networks.
    """
    def __init__(self, networks, source, sender=None, created=None,
                 module=None, module_uri=None):
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
            document.
        :type module_uri: String
        :param module_uri: This is the address of the query that generated the
            document, or, if applicable, the address of the software that
            generated this document.
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
         'stations': ['IU.ANMO (Albuquerque, New Mexico, USA)']}
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
        available_write_formats = [key for key, value in
            FORMAT_FCTS.iteritems() if "write_fct" in value]

        fileformat = format.lower()
        if fileformat not in available_write_formats:
            msg = "Unsupported format '%s'.\nSupported formats: %s" % (
                format, ", ".join(sorted(available_write_formats)))
            raise ValueError(msg)
        return FORMAT_FCTS[fileformat]["write_fct"](self, path_or_file_object,
            **kwargs)

    @property
    def networks(self):
        return self.__networks

    @networks.setter
    def networks(self, value):
        if not hasattr(value, "__iter__"):
            msg = "networks needs to be iterable, e.g. a list."
            raise ValueError(msg)
        if any([not isinstance(x, Network) for x in value]):
            msg = "networks can only contain Network objects."
            raise ValueError(msg)
        self.__networks = value


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
