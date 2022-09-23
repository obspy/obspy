#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides the Inventory class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import copy
import fnmatch
import textwrap
import warnings

import obspy
from obspy.core.util.base import (ENTRY_POINTS, ComparingObject,
                                  _read_from_plugin, _generic_reader)
from obspy.core.util.decorator import map_example_filename, uncompress_file
from obspy.core.util.misc import buffered_load_entry_point
from obspy.core.util.obspy_types import ObsPyException, ZeroSamplingRate

from .network import Network
from .util import _unified_content_strings, _textwrap, _response_plot_label

# Make sure this is consistent with obspy.io.stationxml! Importing it
# from there results in hard to resolve cyclic imports.
SOFTWARE_MODULE = "ObsPy %s" % obspy.__version__
SOFTWARE_URI = "https://www.obspy.org"


def _create_example_inventory():
    """
    Create an example inventory.
    """
    return read_inventory('/path/to/BW_GR_misc.xml', format="STATIONXML")


@map_example_filename("path_or_file_object")
def read_inventory(path_or_file_object=None, format=None, level='response',
                   *args, **kwargs):
    """
    Function to read inventory files.

    :type path_or_file_object: str, pathlib.Path, or file-like object, optional
    :param path_or_file_object: String containing a file name or a URL, a Path
        object, or a open file-like object. Wildcards are allowed for a file
        name. If this attribute is omitted, an example
        :class:`~obspy.core.inventory.inventory.Inventory` object will be
        returned.
    :type format: str
    :param format: Format of the file to read (e.g. ``"STATIONXML"``). See the
        `Supported Formats`_ section below for a list of supported formats.
    :type level: str
    :param level: Level of detail to read from file. One of ``'response'``,
        ``'channel'``, ``'station'`` or ``'network'``. Lower level of detail
        can result in much shorter reading times for some file formats.
    :rtype: :class:`~obspy.core.inventory.inventory.Inventory`
    :return: An ObsPy :class:`~obspy.core.inventory.inventory.Inventory`
        object.

    Additional args and kwargs are passed on to the underlying ``_read_X()``
    methods of the inventory plugins.

    .. rubric:: _`Supported Formats`

    Additional ObsPy modules extend the functionality of the
    :func:`~obspy.core.inventory.inventory.read_inventory` function. The
    following table summarizes all known file formats currently supported by
    ObsPy.

    Please refer to the `Linked Function Call`_ of each module for any extra
    options available at the import stage.

    %s

    .. note::

        For handling additional information not covered by the
        StationXML standard and how to output it to StationXML
        see the :ref:`ObsPy Tutorial <stationxml-extra>`.
    """
    # add default parameters to kwargs so sub-modules may handle them
    kwargs['level'] = level

    if path_or_file_object is None:
        # if no pathname or URL specified, return example catalog
        return _create_example_inventory()
    else:
        return _generic_reader(path_or_file_object, _read, format=format,
                               **kwargs)


@uncompress_file
def _read(filename, format=None, **kwargs):
    """
    Reads a single event file into a ObsPy Catalog object.
    """
    inventory, format = _read_from_plugin('inventory', filename, format=format,
                                          **kwargs)
    return inventory


class Inventory(ComparingObject):
    """
    The root object of the
    :class:`~obspy.core.inventory.network.Network`->
    :class:`~obspy.core.inventory.station.Station`->
    :class:`~obspy.core.inventory.channel.Channel` hierarchy.

    In essence just a container for one or more networks.
    """
    def __init__(self, networks=None, source=SOFTWARE_MODULE, sender=None,
                 created=None, module=SOFTWARE_MODULE,
                 module_uri=SOFTWARE_URI):
        """
        :type networks: list of
            :class:`~obspy.core.inventory.network.Network`
        :param networks: A list of networks part of this inventory.
        :type source: str
        :param source: Network ID of the institution sending the message.
        :type sender: str, optional
        :param sender: Name of the institution sending this message.
        :type created: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param created: The time when the document was created. Will be set to
            the current time if not given.
        :type module: str
        :param module: Name of the software module that generated this
            document, defaults to ObsPy related information.
        :type module_uri: str
        :param module_uri: This is the address of the query that generated the
            document, or, if applicable, the address of the software that
            generated this document, defaults to ObsPy related information.

        .. note::

            For handling additional information not covered by the
            StationXML standard and how to output it to StationXML
            see the :ref:`ObsPy Tutorial <stationxml-extra>`.
        """
        self.networks = networks if networks is not None else []
        self.source = source
        self.sender = sender
        self.module = module
        self.module_uri = module_uri
        # Set the created field to the current time if not given otherwise.
        if created is None:
            self.created = obspy.UTCDateTime()
        else:
            self.created = created

    def __eq__(self, other):
        """
        __eq__ method of the Inventory object.

        :type other: :class:`~obspy.core.inventory.inventory.Inventory`
        :param other: Inventory object for comparison.
        :rtype: bool
        :return: ``True`` if both Inventories are equal.

        .. rubric:: Example

        >>> from obspy.core.inventory import read_inventory
        >>> inv = read_inventory()
        >>> inv2 = inv.copy()
        >>> inv is inv2
        False
        >>> inv == inv2
        True
        """
        if not isinstance(other, Inventory):
            return False
        return self.networks == other.networks

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        new = copy.copy(self)
        new += other
        return new

    def __iadd__(self, other):
        if isinstance(other, Inventory):
            self.networks.extend(other.networks)
            # This is a straight inventory merge.
            self.__copy_inventory_metadata(other)
        elif isinstance(other, Network):
            self.networks.append(other)
        else:
            msg = ("Only Inventory and Network objects can be added to "
                   "an Inventory.")
            raise TypeError(msg)
        return self

    def __len__(self):
        return len(self.networks)

    def __getitem__(self, index):
        return self.networks[index]

    def __copy_inventory_metadata(self, other):
        """
        Will be called after two inventory objects have been merged. It
        attempts to assure that inventory meta information is somewhat
        correct after the merging.

        The networks in other will have been moved to self.
        """
        # The creation time is naturally adjusted to the current time.
        self.created = obspy.UTCDateTime()

        # Merge the source.
        srcs = [self.source, other.source]
        srcs = [_i for _i in srcs if _i]
        all_srcs = []
        for src in srcs:
            all_srcs.extend(src.split(","))
        if all_srcs:
            src = sorted(list(set(all_srcs)))
            self.source = ",".join(src)
        else:
            self.source = None

        # Do the same with the sender.
        sndrs = [self.sender, other.sender]
        sndrs = [_i for _i in sndrs if _i]
        all_sndrs = []
        for sndr in sndrs:
            all_sndrs.extend(sndr.split(","))
        if all_sndrs:
            sndr = sorted(list(set(all_sndrs)))
            self.sender = ",".join(sndr)
        else:
            self.sender = None

        # The module and URI strings will be changed to ObsPy as it did the
        # modification.
        self.module = SOFTWARE_MODULE
        self.module_uri = SOFTWARE_URI

    def get_contents(self):
        """
        Returns a dictionary containing the contents of the object.

        .. rubric:: Example

        >>> example_filename = "/path/to/IRIS_single_channel_with_response.xml"
        >>> inventory = read_inventory(example_filename)
        >>> inventory.get_contents()  \
                # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        {...}
        >>> for k, v in sorted(inventory.get_contents().items()):  \
                    # doctest: +NORMALIZE_WHITESPACE
        ...     print(k, v[0])
        channels IU.ANMO.10.BHZ
        networks IU
        stations IU.ANMO (Albuquerque, New Mexico, USA)
        """
        content_dict = {
            "networks": [],
            "stations": [],
            "channels": []}
        for network in self.networks:
            content_dict['networks'].append(network.code)
            for key, value in network.get_contents().items():
                content_dict.setdefault(key, [])
                content_dict[key].extend(value)
                content_dict[key].sort()
        content_dict['networks'].sort()
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
        ret_str += "\n".join(_textwrap(
            ", ".join(_unified_content_strings(contents["networks"])),
            initial_indent="\t\t\t", subsequent_indent="\t\t\t",
            expand_tabs=False))
        ret_str += "\n"
        ret_str += "\t\tStations (%i):\n" % len(contents["stations"])
        ret_str += "\n".join([
            "\t\t\t%s" % _i
            for _i in _unified_content_strings(contents["stations"])])
        ret_str += "\n"
        ret_str += "\t\tChannels (%i):\n" % len(contents["channels"])
        ret_str += "\n".join(_textwrap(
            ", ".join(_unified_content_strings(contents["channels"])),
            initial_indent="\t\t\t", subsequent_indent="\t\t\t",
            expand_tabs=False))
        return ret_str

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def extend(self, network_list):
        """
        Extends the current Inventory object with another Inventory or a list
        of Network objects.
        """
        if isinstance(network_list, list):
            for _i in network_list:
                # Make sure each item in the list is a Network.
                if not isinstance(_i, Network):
                    msg = 'Extend only accepts a list of Network objects.'
                    raise TypeError(msg)
            self.networks.extend(network_list)
        elif isinstance(network_list, Inventory):
            self.networks.extend(network_list.networks)
            self.__copy_inventory_metadata(network_list)
        else:
            msg = 'Extend only supports a list of Network objects as argument.'
            raise TypeError(msg)

    def write(self, path_or_file_object, format, **kwargs):
        """
        Writes the inventory object to a file or file-like object in
        the specified format.

        :param path_or_file_object: File name or file-like object to be written
            to.
        :type format: str
        :param format: The file format to use (e.g. ``"STATIONXML"``). See the
            `Supported Formats`_ section below for a list of supported formats.
        :param kwargs: Additional keyword arguments passed to the underlying
            plugin's writer method.

        .. rubric:: Example

        >>> from obspy import read_inventory
        >>> inventory = read_inventory()
        >>> inventory.write("example.xml",
        ...                 format="STATIONXML")  # doctest: +SKIP

        .. rubric:: _`Supported Formats`

        Additional ObsPy modules extend the parameters of the
        :meth:`~obspy.core.inventory.inventory.Inventory.write()` method. The
        following table summarizes all known formats with write capability
        currently available for ObsPy.

        Please refer to the `Linked Function Call`_ of each module for any
        extra options available.

        %s
        """
        format = format.upper()
        try:
            # get format specific entry point
            format_ep = ENTRY_POINTS['inventory_write'][format]
            # search writeFormat method for given entry point
            write_format = buffered_load_entry_point(
                format_ep.dist.key,
                'obspy.plugin.inventory.%s' % (format_ep.name), 'writeFormat')
        except (IndexError, ImportError, KeyError):
            msg = "Writing format '{}' is not supported. Supported types: {}"
            msg = msg.format(format,
                             ', '.join(ENTRY_POINTS['inventory_write']))
            raise ValueError(msg)
        return write_format(self, path_or_file_object, **kwargs)

    def copy(self):
        """
        Return a deepcopy of the Inventory object.

        :rtype: :class:`~obspy.core.inventory.inventory.Inventory`
        :return: Copy of current inventory.

        .. rubric:: Examples

        1. Create an Inventory and copy it

            >>> from obspy import read_inventory
            >>> inv = read_inventory()
            >>> inv2 = inv.copy()

           The two objects are not the same:

            >>> inv is inv2
            False

           But they are (currently) equal:

            >>> inv == inv2
            True
        """
        return copy.deepcopy(self)

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
        >>> inventory = read_inventory("/path/to/IU_ANMO_BH.xml")
        >>> datetime = UTCDateTime("2012-08-24T00:00:00")
        >>> response = inventory.get_response("IU.ANMO.00.BHZ", datetime)
        >>> print(response)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Channel Response
          From M/S (Velocity in Meters Per Second) to COUNTS (Digital Counts)
          Overall Sensitivity: 3.27508e+09 defined at 0.020 Hz
          3 stages:
            Stage 1: PolesZerosResponseStage from M/S to V, gain: 1952.1
            Stage 2: CoefficientsTypeResponseStage from V to COUNTS, gain: ...
            Stage 3: CoefficientsTypeResponseStage from COUNTS to COUNTS, ...

        :type seed_id: str
        :param seed_id: SEED ID string of channel to get response for.
        :type datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param datetime: Time to get response for.
        :rtype: :class:`~obspy.core.inventory.response.Response`
        :returns: Response for time series specified by input arguments.
        """
        network, _, _, _ = seed_id.split(".")

        responses = []
        for net in self.networks:
            if net.code != network:
                continue
            try:
                responses.append(net.get_response(seed_id, datetime))
            except Exception:
                pass
        if len(responses) > 1:
            msg = "Found more than one matching response. Returning first."
            warnings.warn(msg)
        elif len(responses) < 1:
            msg = "No matching response information found."
            raise Exception(msg)
        return responses[0]

    def get_channel_metadata(self, seed_id, datetime=None):
        """
        Return basic metadata for a given channel.

        :type seed_id: str
        :param seed_id: SEED ID string of channel to get metadata for.
        :type datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param datetime: Time to get metadata for.
        :rtype: dict
        :return: Dictionary containing coordinates and orientation (latitude,
            longitude, elevation, azimuth, dip)
        """
        network, _, _, _ = seed_id.split(".")

        metadata = []
        for net in self.networks:
            if net.code != network:
                continue
            try:
                metadata.append(net.get_channel_metadata(seed_id, datetime))
            except Exception:
                pass
        if len(metadata) > 1:
            msg = ("Found more than one matching channel metadata. "
                   "Returning first.")
            warnings.warn(msg)
        elif len(metadata) < 1:
            msg = "No matching channel metadata found."
            raise Exception(msg)
        return metadata[0]

    def get_coordinates(self, seed_id, datetime=None):
        """
        Return coordinates for a given channel.

        >>> from obspy import read_inventory, UTCDateTime
        >>> inv = read_inventory()
        >>> t = UTCDateTime("2015-01-01")
        >>> inv.get_coordinates("GR.FUR..LHE", t)  # doctest: +SKIP
        {'elevation': 565.0,
         'latitude': 48.162899,
         'local_depth': 0.0,
         'longitude': 11.2752}

        :type seed_id: str
        :param seed_id: SEED ID string of channel to get coordinates for.
        :type datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param datetime: Time to get coordinates for.
        :rtype: dict
        :return: Dictionary containing coordinates (latitude, longitude,
            elevation, local_depth)
        """
        metadata = self.get_channel_metadata(seed_id, datetime)
        coordinates = {}
        for key in ['latitude', 'longitude', 'elevation', 'local_depth']:
            coordinates[key] = metadata[key]
        return coordinates

    def get_orientation(self, seed_id, datetime=None):
        """
        Return orientation for a given channel.

        >>> from obspy import read_inventory, UTCDateTime
        >>> inv = read_inventory()
        >>> t = UTCDateTime("2015-01-01")
        >>> inv.get_orientation("GR.FUR..LHE", t)  # doctest: +SKIP
        {'azimuth': 90.0,
         'dip': 0.0}

        :type seed_id: str
        :param seed_id: SEED ID string of channel to get orientation for.
        :type datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param datetime: Time to get orientation for.
        :rtype: dict
        :return: Dictionary containing orientation (azimuth, dip).
        """
        metadata = self.get_channel_metadata(seed_id, datetime)
        orientation = {}
        for key in ['azimuth', 'dip']:
            orientation[key] = metadata[key]
        return orientation

    def select(self, network=None, station=None, location=None, channel=None,
               time=None, starttime=None, endtime=None, sampling_rate=None,
               keep_empty=False, minlatitude=None, maxlatitude=None,
               minlongitude=None, maxlongitude=None, latitude=None,
               longitude=None, minradius=None, maxradius=None):
        r"""
        Return a copy of the inventory filtered on various parameters.

        Returns the :class:`Inventory` object with only the
        :class:`~obspy.core.inventory.network.Network`\ s /
        :class:`~obspy.core.inventory.station.Station`\ s /
        :class:`~obspy.core.inventory.channel.Channel`\ s that match the given
        criteria (e.g. all channels with ``channel="EHZ"``).

        .. warning::
            The returned object is based on a shallow copy of the original
            object. That means that modifying any mutable child elements will
            also modify the original object
            (see https://docs.python.org/2/library/copy.html).
            Use :meth:`copy()` afterwards to make a new copy of the data in
            memory.

        .. rubric:: Example

        >>> from obspy import read_inventory, UTCDateTime
        >>> inv = read_inventory()
        >>> t = UTCDateTime(2007, 7, 1, 12)
        >>> inv = inv.select(channel="*Z", station="[RW]*", time=t)
        >>> print(inv)  # doctest: +NORMALIZE_WHITESPACE
        Inventory created at 2014-03-03T11:07:06.198000Z
            Created by: fdsn-stationxml-converter/1.0.0
                    http://www.iris.edu/fdsnstationconverter
            Sending institution: Erdbebendienst Bayern
            Contains:
                Networks (2):
                    BW, GR
                Stations (2):
                    BW.RJOB (Jochberg, Bavaria, BW-Net)
                    GR.WET (Wettzell, Bavaria, GR-Net)
                Channels (4):
                    BW.RJOB..EHZ, GR.WET..BHZ, GR.WET..HHZ, GR.WET..LHZ

        The `network`, `station`, `location` and `channel` selection criteria
        may also contain UNIX style wildcards (e.g. ``*``, ``?``, ...; see
        :func:`~fnmatch.fnmatch`).

        :type network: str
        :param network: Potentially wildcarded network code. If not given,
            all network codes will be accepted.
        :type station: str
        :param station: Potentially wildcarded station code. If not given,
            all station codes will be accepted.
        :type location: str
        :param location: Potentially wildcarded location code. If not given,
            all location codes will be accepted.
        :type channel: str
        :param channel: Potentially wildcarded channel code. If not given,
            all channel codes will be accepted.
        :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param time: Only include networks/stations/channels active at given
            point in time.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Only include networks/stations/channels active at or
            after given point in time (i.e. channels ending before given time
            will not be shown).
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Only include networks/stations/channels active before
            or at given point in time (i.e. channels starting after given time
            will not be shown).
        :type sampling_rate: float
        :param sampling_rate: Only include channels whose sampling rate
            matches the given sampling rate, in Hz (within absolute tolerance
            of 1E-8 Hz and relative tolerance of 1E-5)
        :type keep_empty: bool
        :param keep_empty: If set to `True`, networks/stations that match
            themselves but have no matching child elements (stations/channels)
            will be included in the result.
        :type minlatitude: float
        :param minlatitude: Only include stations/channels with a latitude
            larger than the specified minimum.
        :type maxlatitude: float
        :param maxlatitude: Only include stations/channels with a latitude
            smaller than the specified maximum.
        :type minlongitude: float
        :param minlongitude: Only include stations/channels with a longitude
            larger than the specified minimum.
        :type maxlongitude: float
        :param maxlongitude: Only include stations/channels with a longitude
            smaller than the specified maximum.
        :type latitude: float
        :param latitude: Specify the latitude to be used for a radius
            selection.
        :type longitude: float
        :param longitude: Specify the longitude to be used for a radius
            selection.
        :type minradius: float
        :param minradius: Only include stations/channels within the specified
            minimum number of degrees from the geographic point defined by the
            latitude and longitude parameters.
        :type maxradius: float
        :param maxradius: Only include stations/channels within the specified
            maximum number of degrees from the geographic point defined by the
            latitude and longitude parameters.
        """
        networks = []
        for net in self.networks:
            # skip if any given criterion is not matched
            if network is not None:
                if not fnmatch.fnmatch(net.code.upper(),
                                       network.upper()):
                    continue
            if any([t is not None for t in (time, starttime, endtime)]):
                if not net.is_active(time=time, starttime=starttime,
                                     endtime=endtime):
                    continue

            has_stations = bool(net.stations)

            net_ = net.select(
                station=station, location=location, channel=channel, time=time,
                starttime=starttime, endtime=endtime,
                sampling_rate=sampling_rate, keep_empty=keep_empty,
                minlatitude=minlatitude, maxlatitude=maxlatitude,
                minlongitude=minlongitude, maxlongitude=maxlongitude,
                latitude=latitude, longitude=longitude,
                minradius=minradius, maxradius=maxradius)

            # If the network previously had stations but no longer has any
            # and keep_empty is False: Skip the network.
            if has_stations and not keep_empty and not net_.stations:
                continue
            networks.append(net_)
        inv = copy.copy(self)
        inv.networks = networks
        return inv

    def remove(self, network='*', station='*', location='*', channel='*',
               keep_empty=False):
        r"""
        Return a copy of the inventory with selected elements removed.

        Returns the :class:`Inventory` object but excluding the
        :class:`~obspy.core.inventory.network.Network`\ s /
        :class:`~obspy.core.inventory.station.Station`\ s /
        :class:`~obspy.core.inventory.channel.Channel`\ s that match the given
        criteria (e.g. remove all ``EHZ`` channels with ``channel="EHZ"``).

        .. warning::
            The returned object is based on a shallow copy of the original
            object. That means that modifying any mutable child elements will
            also modify the original object
            (see https://docs.python.org/2/library/copy.html).
            Use :meth:`copy()` afterwards to make a new copy of the data in
            memory.

        .. rubric:: Example

        >>> from obspy import read_inventory, UTCDateTime
        >>> inv = read_inventory()
        >>> inv_new = inv.remove(network='BW')
        >>> print(inv_new)  # doctest: +NORMALIZE_WHITESPACE
        Inventory created at 2014-03-03T11:07:06.198000Z
            Created by: fdsn-stationxml-converter/1.0.0
                    http://www.iris.edu/fdsnstationconverter
            Sending institution: Erdbebendienst Bayern
            Contains:
                Networks (1):
                    GR
                Stations (2):
                    GR.FUR (Fuerstenfeldbruck, Bavaria, GR-Net)
                    GR.WET (Wettzell, Bavaria, GR-Net)
                Channels (21):
                    GR.FUR..BHZ, GR.FUR..BHN, GR.FUR..BHE, GR.FUR..HHZ,
                    GR.FUR..HHN, GR.FUR..HHE, GR.FUR..LHZ, GR.FUR..LHN,
                    GR.FUR..LHE, GR.FUR..VHZ, GR.FUR..VHN, GR.FUR..VHE,
                    GR.WET..BHZ, GR.WET..BHN, GR.WET..BHE, GR.WET..HHZ,
                    GR.WET..HHN, GR.WET..HHE, GR.WET..LHZ, GR.WET..LHN,
                    GR.WET..LHE
        >>> inv_new = inv.remove(network='BW', channel="[EH]*")
        >>> print(inv_new)  # doctest: +NORMALIZE_WHITESPACE
        Inventory created at 2014-03-03T11:07:06.198000Z
            Created by: fdsn-stationxml-converter/1.0.0
                    http://www.iris.edu/fdsnstationconverter
            Sending institution: Erdbebendienst Bayern
            Contains:
                Networks (1):
                    GR
                Stations (2):
                    GR.FUR (Fuerstenfeldbruck, Bavaria, GR-Net)
                    GR.WET (Wettzell, Bavaria, GR-Net)
                Channels (21):
                    GR.FUR..BHZ, GR.FUR..BHN, GR.FUR..BHE, GR.FUR..HHZ,
                    GR.FUR..HHN, GR.FUR..HHE, GR.FUR..LHZ, GR.FUR..LHN,
                    GR.FUR..LHE, GR.FUR..VHZ, GR.FUR..VHN, GR.FUR..VHE,
                    GR.WET..BHZ, GR.WET..BHN, GR.WET..BHE, GR.WET..HHZ,
                    GR.WET..HHN, GR.WET..HHE, GR.WET..LHZ, GR.WET..LHN,
                    GR.WET..LHE
        >>> inv_new = inv.remove(network='BW', channel="[EH]*",
        ...                      keep_empty=True)
        >>> print(inv_new)  # doctest: +NORMALIZE_WHITESPACE
        Inventory created at 2014-03-03T11:07:06.198000Z
            Created by: fdsn-stationxml-converter/1.0.0
                    http://www.iris.edu/fdsnstationconverter
            Sending institution: Erdbebendienst Bayern
            Contains:
                Networks (2):
                    BW, GR
                Stations (5):
                    BW.RJOB (Jochberg, Bavaria, BW-Net) (3x)
                    GR.FUR (Fuerstenfeldbruck, Bavaria, GR-Net)
                    GR.WET (Wettzell, Bavaria, GR-Net)
                Channels (21):
                    GR.FUR..BHZ, GR.FUR..BHN, GR.FUR..BHE, GR.FUR..HHZ,
                    GR.FUR..HHN, GR.FUR..HHE, GR.FUR..LHZ, GR.FUR..LHN,
                    GR.FUR..LHE, GR.FUR..VHZ, GR.FUR..VHN, GR.FUR..VHE,
                    GR.WET..BHZ, GR.WET..BHN, GR.WET..BHE, GR.WET..HHZ,
                    GR.WET..HHN, GR.WET..HHE, GR.WET..LHZ, GR.WET..LHN,
                    GR.WET..LHE

        The `network`, `station`, `location` and `channel` selection criteria
        may also contain UNIX style wildcards (e.g. ``*``, ``?``, ...; see
        :func:`~fnmatch.fnmatch`).

        :type network: str
        :param network: Potentially wildcarded network code. If not specified,
            then all network codes will be matched for removal (combined with
            other options).
        :type station: str
        :param station: Potentially wildcarded station code. If not specified,
            then all station codes will be matched for removal (combined with
            other options).
        :type location: str
        :param location: Potentially wildcarded location code. If not
            specified, then all location codes will be matched for removal
            (combined with other options).
        :type channel: str
        :param channel: Potentially wildcarded channel code. If not specified,
            then all channel codes will be matched for removal (combined with
            other options).
        :type keep_empty: bool
        :param keep_empty: If set to `True`, networks/stations that are left
            without child elements (stations/channels) will still be included
            in the result.
        """
        # Select all objects that are to be removed.
        selected = self.select(network=network, station=station,
                               location=location, channel=channel)
        selected_networks = [net for net in selected]
        selected_stations = [sta for net in selected_networks for sta in net]
        selected_channels = [cha for net in selected_networks
                             for sta in net for cha in sta]
        # Iterate inventory tree and rebuild it excluding selected components.
        networks = []
        for net in self:
            if net in selected_networks and station == '*' and \
                    location == '*' and channel == '*':
                continue
            stations = []
            for sta in net:
                if sta in selected_stations and location == '*' \
                        and channel == '*':
                    continue
                channels = []
                for cha in sta:
                    if cha in selected_channels:
                        continue
                    channels.append(cha)
                channels_were_empty = not bool(sta.channels)
                if not channels and not (keep_empty or channels_were_empty):
                    continue
                sta = copy.copy(sta)
                sta.channels = channels
                stations.append(sta)
            stations_were_empty = not bool(net.stations)
            if not stations and not (keep_empty or stations_were_empty):
                continue
            net = copy.copy(net)
            net.stations = stations
            networks.append(net)
        inv = copy.copy(self)
        inv.networks = networks
        return inv

    def plot(self, projection='global', resolution='l',
             continent_fill_color='0.9', water_fill_color='1.0', marker="v",
             size=15**2, label=True, color='#b15928', color_per_network=False,
             colormap="Paired", legend="upper left", time=None, show=True,
             outfile=None, method=None, fig=None, **kwargs):  # @UnusedVariable
        """
        Creates a preview map of all networks/stations in current inventory
        object.

        :type projection: str, optional
        :param projection: The map projection. Currently supported are:

            * ``"global"`` (Will plot the whole world.)
            * ``"ortho"`` (Will center around the mean lat/long.)
            * ``"local"`` (Will plot around local events)

            Defaults to ``"global"``
        :type resolution: str, optional
        :param resolution: Resolution of the boundary database to use.
            Possible values are:

            * ``"c"`` (crude)
            * ``"l"`` (low)
            * ``"i"`` (intermediate)
            * ``"h"`` (high)
            * ``"f"`` (full)

            Defaults to ``"l"``
        :type continent_fill_color: valid matplotlib color, optional
        :param continent_fill_color:  Color of the continents. Defaults to
            ``"0.9"`` which is a light gray.
        :type water_fill_color: valid matplotlib color, optional
        :param water_fill_color: Color of all water bodies.
            Defaults to ``"white"``.
        :type marker: str
        :param marker: Marker symbol (see :func:`matplotlib.pyplot.scatter`).
        :type size: float
        :param size: Marker size (see :func:`matplotlib.pyplot.scatter`).
        :type label: bool
        :param label: Whether to label stations with "network.station" or not.
        :type color: str
        :param color: Face color of marker symbol (see
            :func:`matplotlib.pyplot.scatter`). Defaults to the first color
            from the single-element "Paired" color map.
        :type color_per_network: bool or dict
        :param color_per_network: If set to ``True``, each network will be
            drawn in a different color. A dictionary can be provided that maps
            network codes to color values (e.g.
            ``color_per_network={"GR": "black", "II": "green"}``).
        :type colormap: str, valid matplotlib colormap, optional
        :param colormap: Only used if ``color_per_network=True``. Specifies
            which colormap is used to draw the colors for the individual
            networks. Defaults to the "Paired" color map.
        :type legend: str or None
        :param legend: Location string for legend, if networks are plotted in
            different colors (i.e. option ``color_per_network`` in use). See
            :func:`matplotlib.pyplot.legend` for possible values for
            legend location string. To disable legend set to ``None``.
        :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param time: Only plot stations available at given point in time.
        :type show: bool
        :param show: Whether to show the figure after plotting or not. Can be
            used to do further customization of the plot before showing it.
        :type outfile: str
        :param outfile: Output file path to directly save the resulting image
            (e.g. ``"/tmp/image.png"``). Overrides the ``show`` option, image
            will not be displayed interactively. The given path/file name is
            also used to automatically determine the output format. Supported
            file formats depend on your matplotlib backend.  Most backends
            support png, pdf, ps, eps and svg. Defaults to ``None``.
        :type method: str
        :param method: Method to use for plotting. Possible values are:

            * ``'cartopy'`` to use the Cartopy library
            * ``None`` to use the best available library

            Defaults to ``None``.
        :type fig: :class:`matplotlib.figure.Figure`
        :param fig: Figure instance to reuse, returned from a previous
            inventory/catalog plot call with `method=cartopy`.
            If a previous cartopy plot is reused, any kwargs regarding the
            cartopy plot setup will be ignored (i.e.  `projection`,
            `resolution`, `continent_fill_color`, `water_fill_color`). Note
            that multiple plots using colorbars likely are problematic, but
            e.g. one station plot (without colorbar) and one event plot (with
            colorbar) together should work well.
        :returns: Figure instance with the plot.

        .. rubric:: Example

        Mollweide projection for global overview:

        >>> from obspy import read_inventory
        >>> inv = read_inventory()
        >>> inv.plot(label=False)  # doctest:+SKIP

        .. plot::

            from obspy import read_inventory
            inv = read_inventory()
            inv.plot(label=False)

        Orthographic projection, automatic colors per network:

        >>> inv.plot(projection="ortho", label=False,
        ...          color_per_network=True)  # doctest:+SKIP

        .. plot::

            from obspy import read_inventory
            inv = read_inventory()
            inv.plot(projection="ortho", label=False, color_per_network=True)

        Local (Albers equal area) projection, with custom colors:

        >>> colors = {'GR': 'blue', 'BW': 'green'}
        >>> inv.plot(projection="local",
        ...          color_per_network=colors)  # doctest:+SKIP

        .. plot::

            from obspy import read_inventory
            inv = read_inventory()
            inv.plot(projection="local",
                     color_per_network={'GR': 'blue',
                                        'BW': 'green'})

        Combining a station and event plot:

        >>> from obspy import read_inventory, read_events
        >>> inv = read_inventory()
        >>> cat = read_events()
        >>> fig = inv.plot(show=False)  # doctest:+SKIP
        >>> cat.plot(fig=fig)  # doctest:+SKIP

        .. plot::

            from obspy import read_inventory, read_events
            inv = read_inventory()
            cat = read_events()
            fig = inv.plot(show=False)
            cat.plot(fig=fig)
        """
        from obspy.imaging.maps import plot_map
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        # The empty ones must be kept as otherwise inventory files without
        # channels will end up with nothing.
        inv = self.select(time=time, keep_empty=True)

        # lat/lon coordinates, magnitudes, dates
        lats = []
        lons = []
        labels = []
        colors = []

        if color_per_network and not isinstance(color_per_network, dict):
            codes = set([n.code for n in inv])
            cmap = plt.get_cmap(name=colormap, lut=len(codes))
            color_per_network = dict([(code, cmap(i))
                                      for i, code in enumerate(sorted(codes))])

        for net in inv:
            for sta in net:
                if sta.latitude is None or sta.longitude is None:
                    msg = ("Station '%s' does not have latitude/longitude "
                           "information and will not be plotted." % label)
                    warnings.warn(msg)
                    continue
                if color_per_network:
                    label_ = "   %s" % sta.code
                    color_ = color_per_network.get(net.code, "k")
                else:
                    label_ = "   " + ".".join((net.code, sta.code))
                    color_ = color
                lats.append(sta.latitude)
                lons.append(sta.longitude)
                labels.append(label_)
                colors.append(color_)

        if not label:
            labels = None

        fig = plot_map(method, lons, lats, size, colors, labels,
                       projection=projection, resolution=resolution,
                       continent_fill_color=continent_fill_color,
                       water_fill_color=water_fill_color,
                       colormap=None, colorbar=False, marker=marker,
                       title=None, show=False, fig=fig, **kwargs)

        if legend is not None and color_per_network:
            ax = fig.axes[0]
            legend_elements = [
                Line2D([0], [0], marker=marker, color=color, label=code,
                       markersize=size ** 0.5, ls='')
                for code, color in sorted(color_per_network.items())]
            ax.legend(handles=legend_elements, loc=legend, fancybox=True,
                      scatterpoints=1, fontsize="medium", markerscale=0.8,
                      handletextpad=0.1)

        if outfile:
            fig.savefig(outfile)
        else:
            if show:
                plt.show()

        return fig

    def plot_response(self, min_freq, output="VEL", network="*", station="*",
                      location="*", channel="*", time=None, starttime=None,
                      endtime=None, axes=None, unwrap_phase=False,
                      plot_degrees=False, show=True, outfile=None,
                      label_epoch_dates=False):
        """
        Show bode plot of instrument response of all (or a subset of) the
        inventory's channels.

        :type min_freq: float
        :param min_freq: Lowest frequency to plot.
        :type output: str
        :param output: Output units. One of:

            ``"DISP"``
                displacement, output unit is meters
            ``"VEL"``
                velocity, output unit is meters/second
            ``"ACC"``
                acceleration, output unit is meters/second**2
            ``"DEF"``
                default units, the response is calculated in
                output units/input units (last stage/first stage).
                Useful if the units for a particular type of sensor (e.g., a
                pressure sensor) cannot be converted to displacement, velocity
                or acceleration.

        :type network: str
        :param network: Only plot matching networks. Accepts UNIX style
            patterns and wildcards (e.g. ``"G*"``, ``"*[ER]"``; see
            :func:`~fnmatch.fnmatch`)
        :type station: str
        :param station: Only plot matching stations. Accepts UNIX style
            patterns and wildcards (e.g. ``"L44*"``, ``"L4?A"``,
            ``"[LM]44A"``; see :func:`~fnmatch.fnmatch`)
        :type location: str
        :param location: Only plot matching channels. Accepts UNIX style
            patterns and wildcards (e.g. ``"BH*"``, ``"BH?"``, ``"*Z"``,
            ``"[LB]HZ"``; see :func:`~fnmatch.fnmatch`)
        :type channel: str
        :param channel: Only plot matching channels. Accepts UNIX style
            patterns and wildcards (e.g. ``"BH*"``, ``"BH?"``, ``"*Z"``,
            ``"[LB]HZ"``; see :func:`~fnmatch.fnmatch`)
        :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param time: Only regard networks/stations/channels active at given
            point in time.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Only regard networks/stations/channels active at or
            after given point in time (i.e. networks ending before given time
            will not be shown).
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Only regard networks/stations/channels active before or
            at given point in time (i.e. networks starting after given time
            will not be shown).
        :type axes: list of 2 :class:`matplotlib.axes.Axes`
        :param axes: List/tuple of two axes instances to plot the
            amplitude/phase spectrum into. If not specified, a new figure is
            opened.
        :type unwrap_phase: bool
        :param unwrap_phase: Set optional phase unwrapping using NumPy.
        :type plot_degrees: bool
        :param plot_degrees: if ``True`` plot bode in degrees
        :type show: bool
        :param show: Whether to show the figure after plotting or not. Can be
            used to do further customization of the plot before showing it.
        :type outfile: str
        :param outfile: Output file path to directly save the resulting image
            (e.g. ``"/tmp/image.png"``). Overrides the ``show`` option, image
            will not be displayed interactively. The given path/file name is
            also used to automatically determine the output format. Supported
            file formats depend on your matplotlib backend.  Most backends
            support png, pdf, ps, eps and svg. Defaults to ``None``.
        :type label_epoch_dates: bool
        :param label_epoch_dates: Whether to add channel epoch dates in the
            plot's legend labels.

        .. rubric:: Basic Usage

        >>> from obspy import read_inventory
        >>> inv = read_inventory()
        >>> inv.plot_response(0.001, station="RJOB")  # doctest: +SKIP

        .. plot::

            from obspy import read_inventory
            inv = read_inventory()
            inv.plot_response(0.001, station="RJOB")
        """
        import matplotlib.pyplot as plt

        if axes is not None:
            ax1, ax2 = axes
            fig = ax1.figure
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax1)

        matching = self.select(network=network, station=station,
                               location=location, channel=channel, time=time,
                               starttime=starttime, endtime=endtime)

        for net in matching.networks:
            for sta in net.stations:
                for cha in sta.channels:
                    label = _response_plot_label(
                        net, sta, cha, label_epoch_dates=label_epoch_dates)
                    try:
                        cha.plot(min_freq=min_freq, output=output,
                                 axes=(ax1, ax2), label=label,
                                 unwrap_phase=unwrap_phase,
                                 plot_degrees=plot_degrees, show=False,
                                 outfile=None)
                    except ZeroSamplingRate:
                        msg = ("Skipping plot of channel with zero "
                               "sampling rate:\n%s")
                        warnings.warn(msg % str(cha), UserWarning)
                    except ObsPyException as e:
                        msg = "Skipping plot of channel (%s):\n%s"
                        warnings.warn(msg % (str(e), str(cha)), UserWarning)
        # final adjustments to plot if we created the figure in here
        if axes is None:
            from obspy.core.inventory.response import _adjust_bode_plot_figure
            _adjust_bode_plot_figure(fig, plot_degrees, show=False)
        if outfile:
            fig.savefig(outfile)
        else:
            if show:
                plt.show()

        return fig


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
