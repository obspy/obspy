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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import python_2_unicode_compatible, native_str

import copy
import fnmatch
import os
import textwrap
import warnings

import obspy
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util.base import (ENTRY_POINTS, ComparingObject,
                                  _read_from_plugin, NamedTemporaryFile,
                                  download_to_file, sanitize_filename)
from obspy.core.util.decorator import map_example_filename
from obspy.core.util.misc import buffered_load_entry_point
from obspy.core.util.obspy_types import ObsPyException, ZeroSamplingRate

from .network import Network
from .util import (_unified_content_strings, _textwrap, plot_inventory_epochs,
                   _merge_plottable_structs)

# Make sure this is consistent with obspy.io.stationxml! Importing it
# from there results in hard to resolve cyclic imports.
SOFTWARE_MODULE = "ObsPy %s" % obspy.__version__
SOFTWARE_URI = "https://www.obspy.org"


def _create_example_inventory():
    """
    Create an example inventory.
    """
    data_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data")
    path = os.path.join(data_dir, "BW_GR_misc.xml")
    return read_inventory(path, format="STATIONXML")


@map_example_filename("path_or_file_object")
def read_inventory(path_or_file_object=None, format=None, *args, **kwargs):
    """
    Function to read inventory files.

    :param path_or_file_object: File name or file like object. If this
        attribute is omitted, an example :class:`Inventory`
        object will be returned.
    :type format: str
    :param format: Format of the file to read (e.g. ``"STATIONXML"``). See the
        `Supported Formats`_ section below for a list of supported formats.
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
    if path_or_file_object is None:
        # if no pathname or URL specified, return example catalog
        return _create_example_inventory()
    elif isinstance(path_or_file_object, (str, native_str)) and \
            "://" in path_or_file_object:
        # some URL
        # extract extension if any
        suffix = \
            os.path.basename(path_or_file_object).partition('.')[2] or '.tmp'
        with NamedTemporaryFile(suffix=sanitize_filename(suffix)) as fh:
            download_to_file(url=path_or_file_object, filename_or_buffer=fh)
            return read_inventory(fh.name, format=format)
    return _read_from_plugin("inventory", path_or_file_object,
                             format=format, *args, **kwargs)[0]


@python_2_unicode_compatible
class Inventory(ComparingObject):
    """
    The root object of the Inventory->Network->Station->Channel hierarchy.

    In essence just a container for one or more networks.
    """
    def __init__(self, networks, source, sender=None, created=None,
                 module=SOFTWARE_MODULE, module_uri=SOFTWARE_URI):
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
        new = copy.deepcopy(self)
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
               keep_empty=False):
        """
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
        :type keep_empty: bool
        :param keep_empty: If set to `True`, networks/stations that match
            themselves but have no matching child elements (stations/channels)
            will be included in the result.
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
                sampling_rate=sampling_rate, keep_empty=keep_empty)

            # If the network previously had stations but no longer has any
            # and keep_empty is False: Skip the network.
            if has_stations and not keep_empty and not net_.stations:
                continue
            networks.append(net_)
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
        :param resolution: Resolution of the boundary database to use. Will be
            based directly to the basemap module. Possible values are:

            * ``"c"`` (crude)
            * ``"l"`` (low)
            * ``"i"`` (intermediate)
            * ``"h"`` (high)
            * ``"f"`` (full)

            Defaults to ``"l"``
        :type continent_fill_color: Valid matplotlib color, optional
        :param continent_fill_color:  Color of the continents. Defaults to
            ``"0.9"`` which is a light gray.
        :type water_fill_color: Valid matplotlib color, optional
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
        :type colormap: str, any matplotlib colormap, optional
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

            * ``'basemap'`` to use the Basemap library
            * ``'cartopy'`` to use the Cartopy library
            * ``None`` to use the best available library

            Defaults to ``None``.
        :type fig: :class:`matplotlib.figure.Figure`
        :param fig: Figure instance to reuse, returned from a previous
            inventory/catalog plot call with `method=basemap`.
            If a previous basemap plot is reused, any kwargs regarding the
            basemap plot setup will be ignored (i.e.  `projection`,
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

        Combining a station and event plot (uses basemap):

        >>> from obspy import read_inventory, read_events
        >>> inv = read_inventory()
        >>> cat = read_events()
        >>> fig = inv.plot(method="basemap", show=False)  # doctest:+SKIP
        >>> cat.plot(method="basemap", fig=fig)  # doctest:+SKIP

        .. plot::

            from obspy import read_inventory, read_events
            inv = read_inventory()
            cat = read_events()
            fig = inv.plot(show=False)
            cat.plot(fig=fig)
        """
        from obspy.imaging.maps import plot_map
        import matplotlib.pyplot as plt

        # The empty ones must be kept as otherwise inventory files without
        # channels will end up with nothing.
        inv = self.select(time=time, keep_empty=True)

        # lat/lon coordinates, magnitudes, dates
        lats = []
        lons = []
        labels = []
        colors = []

        if color_per_network and not isinstance(color_per_network, dict):
            from matplotlib.cm import get_cmap
            codes = set([n.code for n in inv])
            cmap = get_cmap(name=colormap, lut=len(codes))
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
            count = len(ax.collections)
            for code, color in sorted(color_per_network.items()):
                ax.scatter([0], [0], size, color, label=code, marker=marker)
            # workaround for older matplotlib versions
            try:
                ax.legend(loc=legend, fancybox=True, scatterpoints=1,
                          fontsize="medium", markerscale=0.8,
                          handletextpad=0.1)
            except TypeError:
                leg_ = ax.legend(loc=legend, fancybox=True, scatterpoints=1,
                                 markerscale=0.8, handletextpad=0.1)
                leg_.prop.set_size("medium")
            # remove collections again solely created for legend handles
            ax.collections = ax.collections[:count]

        if outfile:
            fig.savefig(outfile)
        else:
            if show:
                plt.show()

        return fig

    def plot_response(self, min_freq, output="VEL", network="*", station="*",
                      location="*", channel="*", time=None, starttime=None,
                      endtime=None, axes=None, unwrap_phase=False,
                      plot_degrees=False, show=True, outfile=None):
        """
        Show bode plot of instrument response of all (or a subset of) the
        inventory's channels.

        :type min_freq: float
        :param min_freq: Lowest frequency to plot.
        :type output: str
        :param output: Output units. One of:

                * ``"DISP"`` -- displacement, output unit is meters;
                * ``"VEL"`` -- velocity, output unit is meters/second; or,
                * ``"ACC"`` -- acceleration, output unit is meters/second**2.

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
                    try:
                        cha.plot(min_freq=min_freq, output=output,
                                 axes=(ax1, ax2),
                                 label=".".join((net.code, sta.code,
                                                 cha.location_code, cha.code)),
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

    def _get_epoch_plottable_struct(self):
        # get structure of inventory epoch's data in format
        # {inventory: ([(start, end)...], sample_rate, subdictionary)}
        # where subdictionary is recursively the same for netwk, stn, ch data
        # note that sample rate is only defined for the station level
        # and subdictionary of inventory is network objects
        # the conistent formatting at each level means that each level can be
        # concatenated together -- i.e., each network epoch is a diff. object
        # in obspy, but here they have their times merged together by name
        plot_dict = {}
        sub_dict = {}
        for network in self.networks:
            eps = network._get_epoch_plottable_struct()
            sub_dict = _merge_plottable_structs(sub_dict, eps)
        if hasattr(self, 'start_date'):
            start = self.start_date
            if self.end_time is None:
                end = UTCDateTime.now()
            else:
                end = self.end_time
        else:
            start = UTCDateTime(0)
            end = UTCDateTime(0)
        time_tuple = (start, end)
        plot_dict[str('')] = ([time_tuple], 0, sub_dict)
        return plot_dict

    def plot_epochs(self, outfile=None, colormap=None, show=True,
                    combine=True):
        """
        Plot the epochs of this given inventory object.
        Returns a pyplot figure which can be saved to file.
        :param outfile: If included, the plot will be saved to a file with the
            given filename. (Otherwise it will be displayed in a window)
        :type outfile: str
        :param colormap: If this parameter is included, the plot will use the
            given colorspace for inventory plotting
        :type colormap: matplotlib.colors.LinearSegmentedColormap
        :param show: If set as true, will display the plot in a window
        :type show: boolean
        :param combine: If set as true, channels with matching epochs will be
            merged onto the same y-axis values
        :type combine: boolean

        .. rubric:: Example

        >>> inv = read_inventory()
        >>> inv.plot_epochs(show=True) # doctest: +SKIP

        .. plot::
            inv = read_inventory()
            inv.plot_epochs(show=True)

        """
        plot_dict = self._get_epoch_plottable_struct()
        fig = plot_inventory_epochs(plot_dict, outfile, colormap, show,
                                    combine)
        return fig


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
