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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from pkg_resources import load_entry_point
import obspy
from obspy.core.util.base import ComparingObject
from obspy.core.util.decorator import map_example_filename
from obspy.core.util.base import ENTRY_POINTS, _readFromPlugin
from obspy.station.stationxml import SOFTWARE_MODULE, SOFTWARE_URI
from obspy.station.network import Network
import textwrap
import warnings
import copy
import fnmatch
import numpy as np


def _createExampleInventory():
    """
    Create an example inventory.
    """
    return read_inventory('/path/to/BW_GR_misc.xml.gz', format="STATIONXML")


@map_example_filename("path_or_file_object")
def read_inventory(path_or_file_object=None, format=None):
    """
    Function to read inventory files.

    :param path_or_file_object: File name or file like object. If this
        attribute is omitted, an example :class:`Inventory`
        object will be returned.
    :type format: str, optional
    :param format: Format of the file to read (e.g. ``"STATIONXML"``).
    """
    if path_or_file_object is None:
        # if no pathname or URL specified, return example catalog
        return _createExampleInventory()
    return _readFromPlugin("inventory", path_or_file_object, format=format)[0]


class Inventory(ComparingObject):
    """
    The root object of the Inventory->Network->Station->Channel hierarchy.

    In essence just a container for one or more networks.
    """
    def __init__(self, networks, source, sender=None, created=None,
                 module=SOFTWARE_MODULE, module_uri=SOFTWARE_URI):
        """
        :type networks: list of :class:`~obspy.station.network.Network`
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

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def write(self, path_or_file_object, format, **kwargs):
        """
        Writes the inventory object to a file or file-like object in
        the specified format.

        :param path_or_file_object: File name or file-like object to be written
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
        >>> print(response)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        Channel Response
           From M/S (Velocity in Meters Per Second) to COUNTS (Digital Counts)
           Overall Sensitivity: 2.5168e+09 defined at 0.020 Hz
           4 stages:
              Stage 1: PolesZerosResponseStage from M/S to V, gain: 1500
              Stage 2: CoefficientsTypeResponseStage from V to COUNTS, ...
              Stage 3: FIRResponseStage from COUNTS to COUNTS, gain: 1
              Stage 4: FIRResponseStage from COUNTS to COUNTS, gain: 1

        :type seed_id: str
        :param seed_id: SEED ID string of channel to get response for.
        :type datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param datetime: Time to get response for.
        :rtype: :class:`~obspy.station.response.Response`
        :returns: Response for time series specified by input arguments.
        """
        network, _, _, _ = seed_id.split(".")

        responses = []
        for net in self.networks:
            if net.code != network:
                continue
            try:
                responses.append(net.get_response(seed_id, datetime))
            except:
                pass
        if len(responses) > 1:
            msg = "Found more than one matching response. Returning first."
            warnings.warn(msg)
        elif len(responses) < 1:
            msg = "No matching response information found."
            raise Exception(msg)
        return responses[0]

    def get_coordinates(self, seed_id, datetime=None):
        """
        Return coordinates for a given channel.

        :type seed_id: str
        :param seed_id: SEED ID string of channel to get coordinates for.
        :type datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param datetime: Time to get coordinates for.
        :rtype: dict
        :return: Dictionary containing coordinates (latitude, longitude,
            elevation)
        """
        network, _, _, _ = seed_id.split(".")

        coordinates = []
        for net in self.networks:
            if net.code != network:
                continue
            try:
                coordinates.append(net.get_coordinates(seed_id, datetime))
            except:
                pass
        if len(coordinates) > 1:
            msg = "Found more than one matching coordinates. Returning first."
            warnings.warn(msg)
        elif len(coordinates) < 1:
            msg = "No matching coordinates found."
            raise Exception(msg)
        return coordinates[0]

    def select(self, network=None, station=None, location=None, channel=None,
               time=None, starttime=None, endtime=None, sampling_rate=None,
               keep_empty=False):
        """
        Returns the :class:`Inventory` object with only the
        :class:`~obspy.station.network.Network`\ s /
        :class:`~obspy.station.station.Station`\ s /
        :class:`~obspy.station.channel.Channel`\ s that match the given
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
                    GR
                    BW
                Stations (2):
                    BW.RJOB (Jochberg, Bavaria, BW-Net)
                    GR.WET (Wettzell, Bavaria, GR-Net)
                Channels (4):
                    BW.RJOB..EHZ, GR.WET..BHZ, GR.WET..HHZ, GR.WET..LHZ

        The `network`, `station`, `location` and `channel` selection criteria
        may also contain UNIX style wildcards (e.g. ``*``, ``?``, ...; see
        :func:`~fnmatch.fnmatch`).

        :type network: str
        :type station: str
        :type location: str
        :type channel: str
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

            net_ = net.select(
                station=station, location=location, channel=channel, time=time,
                starttime=starttime, endtime=endtime,
                sampling_rate=sampling_rate, keep_empty=keep_empty)
            if not keep_empty and not net_.stations:
                continue
            networks.append(net_)
        inv = copy.copy(self)
        inv.networks = networks
        return inv

    def plot(self, projection='cyl', resolution='l',
             continent_fill_color='0.9', water_fill_color='1.0', marker="v",
             size=15**2, label=True, color='blue', color_per_network=False,
             colormap="jet", legend="upper left", time=None, show=True,
             outfile=None, **kwargs):  # @UnusedVariable
        """
        Creates a preview map of all networks/stations in current inventory
        object.

        :type projection: str, optional
        :param projection: The map projection. Currently supported are:

            * ``"cyl"`` (Will plot the whole world.)
            * ``"ortho"`` (Will center around the mean lat/long.)
            * ``"local"`` (Will plot around local events)

            Defaults to ``"cyl"``
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
            :func:`matplotlib.pyplot.scatter`).
        :type color_per_network: bool or dict
        :param color_per_network: If set to ``True``, each network will be
            drawn in a different color. A dictionary can be provided that maps
            network codes to color values (e.g.
            ``color_per_network={"GR": "black", "II": "green"}``).
        :type colormap: str, any matplotlib colormap, optional
        :param colormap: Only used if ``color_per_network=True``. Specifies
            which colormap is used to draw the colors for the individual
            networks.
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

        .. rubric:: Example

        Cylindrical projection for global overview:

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

        Local (azimuthal equidistant) projection, with custom colors:

        >>> colors = {'GR': 'blue', 'BW': 'green'}
        >>> inv.plot(projection="local",
        ...          color_per_network=colors)  # doctest:+SKIP

        .. plot::

            from obspy import read_inventory
            inv = read_inventory()
            inv.plot(projection="local",
                     color_per_network={'GR': 'blue',
                                        'BW': 'green'})
        """
        from obspy.imaging.maps import plot_basemap
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
            cmap = get_cmap(name=colormap)
            codes = set([n.code for n in inv])
            nums = np.linspace(0, 1, endpoint=False, num=len(codes))
            color_per_network = dict([(code, cmap(n))
                                      for code, n in zip(sorted(codes), nums)])

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

        fig = plot_basemap(lons, lats, size, colors, labels,
                           projection=projection, resolution=resolution,
                           continent_fill_color=continent_fill_color,
                           water_fill_color=water_fill_color,
                           colormap=None, colorbar=False, marker=marker,
                           title=None, show=False, **kwargs)

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
                      endtime=None, axes=None, unwrap_phase=False, show=True,
                      outfile=None):
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

        if axes:
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
                    cha.plot(min_freq=min_freq, output=output, axes=(ax1, ax2),
                             label=".".join((net.code, sta.code,
                                             cha.location_code, cha.code)),
                             unwrap_phase=unwrap_phase, show=False,
                             outfile=None)

        # final adjustments to plot if we created the figure in here
        if not axes:
            from obspy.station.response import _adjust_bode_plot_figure
            _adjust_bode_plot_figure(fig, show=False)

        if outfile:
            fig.savefig(outfile)
        else:
            if show:
                plt.show()

        return fig


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
