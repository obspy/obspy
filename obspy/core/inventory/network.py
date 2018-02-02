#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides the Network class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import python_2_unicode_compatible

import copy
import fnmatch
import warnings

from obspy.core.util.obspy_types import ObsPyException, ZeroSamplingRate

from .station import Station
from .util import BaseNode, _unified_content_strings, _textwrap


@python_2_unicode_compatible
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
                 historical_code=None, data_availability=None):
        """
        :type code: str
        :param code: The SEED network code.
        :type total_number_of_stations: int
        :param total_number_of_stations: The total number of stations
            contained in this network, including inactive or terminated
            stations.
        :param selected_number_of_stations: The total number of stations in
            this network that were selected by the query that produced this
            document, even if the stations do not appear in the document. (This
            might happen if the user only wants a document that goes contains
            only information at the Network level.)
        :type description: str, optional
        :param description: A description of the resource
        :type comments: list of :class:`~obspy.core.inventory.util.Comment`
        :param comments: An arbitrary number of comments to the resource
        :type start_date: :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param start_date: The start date of the resource
        :type end_date: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param end_date: The end date of the resource
        :type restricted_status: str, optional
        :param restricted_status: The restriction status
        :type alternate_code: str, optional
        :param alternate_code: A code used for display or association,
            alternate to the SEED-compliant code.
        :type historical_code: str, optional
        :param historical_code: A previously used code if different from the
            current code.
        :type data_availability: :class:`~obspy.station.util.DataAvailability`
        :param data_availability: Information about time series availability
            for the network.
        """
        self.stations = stations or []
        self.total_number_of_stations = total_number_of_stations
        self.selected_number_of_stations = selected_number_of_stations

        super(Network, self).__init__(
            code=code, description=description, comments=comments,
            start_date=start_date, end_date=end_date,
            restricted_status=restricted_status, alternate_code=alternate_code,
            historical_code=historical_code,
            data_availability=data_availability)

    @property
    def total_number_of_stations(self):
        return self._total_number_of_stations

    @total_number_of_stations.setter
    def total_number_of_stations(self, value):
        if value is not None and value < 0:
            msg = "total_number_of_stations cannot be negative."
            raise ValueError(msg)
        self._total_number_of_stations = value

    @property
    def selected_number_of_stations(self):
        return self._selected_number_of_stations

    @selected_number_of_stations.setter
    def selected_number_of_stations(self, value):
        if value is not None and value < 0:
            msg = "selected_number_of_stations cannot be negative."
            raise ValueError(msg)
        self._selected_number_of_stations = value

    def __len__(self):
        return len(self.stations)

    def __getitem__(self, index):
        return self.stations[index]

    def __str__(self):
        ret = ("Network {id} {description}\n"
               "\tStation Count: {selected}/{total} (Selected/Total)\n"
               "\t{start_date} - {end_date}\n"
               "\tAccess: {restricted}\n"
               "{alternate_code}"
               "{historical_code}")
        ret = ret.format(
            id=self.code,
            description="(%s)" % self.description if self.description else "",
            selected=self.selected_number_of_stations,
            total=self.total_number_of_stations,
            start_date=str(self.start_date) if self.start_date else "--",
            end_date=str(self.end_date) if self.end_date else "--",
            restricted=self.restricted_status or "UNKNOWN",
            alternate_code=("\tAlternate Code: %s\n" % self.alternate_code
                            if self.alternate_code else ""),
            historical_code=("\tHistorical Code: %s\n" % self.historical_code
                             if self.historical_code else ""))
        contents = self.get_contents()
        ret += "\tContains:\n"
        ret += "\t\tStations (%i):\n" % len(contents["stations"])
        ret += "\n".join([
            "\t\t\t%s" % _i
            for _i in _unified_content_strings(contents["stations"])])
        ret += "\n"
        ret += "\t\tChannels (%i):\n" % len(contents["channels"])
        ret += "\n".join(_textwrap(", ".join(
            _unified_content_strings(contents["channels"])),
            initial_indent="\t\t\t", subsequent_indent="\t\t\t",
            expand_tabs=False))
        return ret

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def get_contents(self):
        """
        Returns a dictionary containing the contents of the object.

        .. rubric:: Example

        >>> from obspy import read_inventory
        >>> example_filename = "/path/to/IRIS_single_channel_with_response.xml"
        >>> inventory = read_inventory(example_filename)
        >>> network = inventory.networks[0]
        >>> network.get_contents()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        {...}
        >>> for k, v in sorted(network.get_contents().items()):
        ...     print(k, v[0])
        channels IU.ANMO.10.BHZ
        stations IU.ANMO (Albuquerque, New Mexico, USA)
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
        :rtype: :class:`~obspy.core.inventory.response.Response`
        :returns: Response for time series specified by input arguments.
        """
        network, station, location, channel = seed_id.split(".")
        if self.code != network:
            responses = []
        else:
            channels = [cha for sta in self.stations for cha in sta.channels
                        if sta.code == station and
                        cha.code == channel and
                        cha.location_code == location and
                        (cha.start_date is None or
                         cha.start_date <= datetime) and
                        (cha.end_date is None or cha.end_date >= datetime)]
            responses = [cha.response for cha in channels
                         if cha.response is not None]
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
        :type datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param datetime: Time to get metadata for.
        :rtype: dict
        :return: Dictionary containing coordinates and orientation (latitude,
            longitude, elevation, azimuth, dip)
        """
        network, station, location, channel = seed_id.split(".")
        metadata = []
        if self.code != network:
            pass
        elif self.start_date and self.start_date > datetime:
            pass
        elif self.end_date and self.end_date < datetime:
            pass
        else:
            for sta in self.stations:
                # skip wrong station
                if sta.code != station:
                    continue
                # check datetime only if given
                if datetime:
                    # skip if start date before given datetime
                    if sta.start_date and sta.start_date > datetime:
                        continue
                    # skip if end date before given datetime
                    if sta.end_date and sta.end_date < datetime:
                        continue
                for cha in sta.channels:
                    # skip wrong channel
                    if cha.code != channel:
                        continue
                    # skip wrong location
                    if cha.location_code != location:
                        continue
                    # check datetime only if given
                    if datetime:
                        # skip if start date before given datetime
                        if cha.start_date and cha.start_date > datetime:
                            continue
                        # skip if end date before given datetime
                        if cha.end_date and cha.end_date < datetime:
                            continue
                    # prepare coordinates
                    data = {}
                    for key in ('latitude', 'longitude', 'elevation'):
                        value = getattr(cha, key, None)
                        # if channel latitude/longitude/elevation is not given
                        # use station information
                        if value is None:
                            value = getattr(sta, key, None)
                        data[key] = value
                    data['local_depth'] = cha.depth
                    data['azimuth'] = cha.azimuth
                    data['dip'] = cha.dip
                    metadata.append(data)
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
        Return coordinates and orientation for a given channel.

        :type seed_id: str
        :param seed_id: SEED ID string of channel to get coordinates and
            orientation for.
        :type datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`
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

        :type seed_id: str
        :param seed_id: SEED ID string of channel to get orientation for.
        :type datetime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param datetime: Time to get orientation for.
        :rtype: dict
        :return: Dictionary containing orientation (azimuth, dip).
        """
        metadata = self.get_channel_metadata(seed_id, datetime)
        orientation = {}
        for key in ['azimuth', 'dip']:
            orientation[key] = metadata[key]
        return orientation

    def select(self, station=None, location=None, channel=None, time=None,
               starttime=None, endtime=None, sampling_rate=None,
               keep_empty=False):
        """
        Returns the :class:`Network` object with only the
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
        >>> net = read_inventory()[0]
        >>> t = UTCDateTime(2008, 7, 1, 12)
        >>> net = net.select(channel="[LB]HZ", time=t)
        >>> print(net)  # doctest: +NORMALIZE_WHITESPACE
        Network GR (GRSN)
            Station Count: None/None (Selected/Total)
            -- - --
            Access: UNKNOWN
            Contains:
                Stations (2):
                    GR.FUR (Fuerstenfeldbruck, Bavaria, GR-Net)
                    GR.WET (Wettzell, Bavaria, GR-Net)
                Channels (4):
                    GR.FUR..BHZ, GR.FUR..LHZ, GR.WET..BHZ, GR.WET..LHZ

        The `station`, `location` and `channel` selection criteria  may also
        contain UNIX style wildcards (e.g. ``*``, ``?``, ...; see
        :func:`~fnmatch.fnmatch`).

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
        :param time: Only include stations/channels active at given point in
            time.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Only include stations/channels active at or after
            given point in time (i.e. channels ending before given time will
            not be shown).
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Only include stations/channels active before or at
            given point in time (i.e. channels starting after given time will
            not be shown).
        :type sampling_rate: float
        :type keep_empty: bool
        :param keep_empty: If set to `True`, stations that match
            themselves but have no matching child elements (channels)
            will be included in the result. This flag has no effect for
            initially empty stations which will always be retained if they
            are matched by the other parameters.
        """
        stations = []
        for sta in self.stations:
            # skip if any given criterion is not matched
            if station is not None:
                if not fnmatch.fnmatch(sta.code.upper(),
                                       station.upper()):
                    continue
            if any([t is not None for t in (time, starttime, endtime)]):
                if not sta.is_active(time=time, starttime=starttime,
                                     endtime=endtime):
                    continue

            has_channels = bool(sta.channels)

            sta_ = sta.select(
                location=location, channel=channel, time=time,
                starttime=starttime, endtime=endtime,
                sampling_rate=sampling_rate)

            # If the station previously had channels but no longer has any
            # and keep_empty is False: Skip the station.
            if has_channels and not keep_empty and not sta_.channels:
                continue
            stations.append(sta_)
        net = copy.copy(self)
        net.stations = stations
        return net

    def plot(self, projection='global', resolution='l',
             continent_fill_color='0.9', water_fill_color='1.0', marker="v",
             size=15**2, label=True, color='#b15928', time=None, show=True,
             outfile=None, method=None, fig=None, **kwargs):  # @UnusedVariable
        """
        Creates a preview map of all stations in current network object.

        :type projection: str, optional
        :param projection: The map projection. Currently supported are:

            * ``"global"`` (Will plot the whole world.)
            * ``"ortho"`` (Will center around the mean lat/long.)
            * ``"local"`` (Will plot around local events)

            Defaults to "global"
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
        :type label: bool
        :param label: Whether to label stations with "network.station" or not.
        :type color: str
        :param color: Face color of marker symbol (see
            :func:`matplotlib.pyplot.scatter`). Defaults to the first color
            from the single-element "Paired" color map.
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
        >>> net = read_inventory()[0]
        >>> net.plot(label=False)  # doctest:+SKIP

        .. plot::

            from obspy import read_inventory
            net = read_inventory()[0]
            net.plot(label=False)

        Orthographic projection:

        >>> net.plot(projection="ortho")  # doctest:+SKIP

        .. plot::

            from obspy import read_inventory
            net = read_inventory()[0]
            net.plot(projection="ortho")

        Local (Albers equal area) projection:

        >>> net.plot(projection="local")  # doctest:+SKIP

        .. plot::

            from obspy import read_inventory
            net = read_inventory()[0]
            net.plot(projection="local")
        """
        from obspy.imaging.maps import plot_map
        import matplotlib.pyplot as plt

        # lat/lon coordinates, magnitudes, dates
        lats = []
        lons = []
        labels = []
        for sta in self.select(time=time).stations:
            label_ = "   " + ".".join((self.code, sta.code))
            if sta.latitude is None or sta.longitude is None:
                msg = ("Station '%s' does not have latitude/longitude "
                       "information and will not be plotted." % label)
                warnings.warn(msg)
                continue
            lats.append(sta.latitude)
            lons.append(sta.longitude)
            labels.append(label_)

        if not label:
            labels = None

        fig = plot_map(method, lons, lats, size, color, labels,
                       projection=projection, resolution=resolution,
                       continent_fill_color=continent_fill_color,
                       water_fill_color=water_fill_color,
                       colormap=None, marker=marker, title=None,
                       show=False, **kwargs)

        if outfile:
            fig.savefig(outfile)
        else:
            if show:
                plt.show()

        return fig

    def plot_response(self, min_freq, output="VEL", station="*", location="*",
                      channel="*", time=None, starttime=None, endtime=None,
                      axes=None, unwrap_phase=False, show=True, outfile=None):
        """
        Show bode plot of instrument response of all (or a subset of) the
        network's channels.

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

        :type station: str
        :param station: Only plot matching stations. Accepts UNIX style
            patterns and wildcards (e.g. ``"L44*"``, ``"L4?A"``,
            ``"[LM]44A``"; see :func:`~fnmatch.fnmatch`)
        :type location: str
        :param location: Only plot matching channels. Accepts UNIX style
            patterns and wildcards (e.g. ``"BH*"``, ``"BH?"``, ``"*Z"``,
            ``"[LB]HZ"``; see :func:`~fnmatch.fnmatch`)
        :type channel: str
        :param channel: Only plot matching channels. Accepts UNIX style
            patterns and wildcards (e.g. ``"BH*"``, ``"BH?"``, ``"*Z"``,
            ``"[LB]HZ"``; see :func:`~fnmatch.fnmatch`)
        :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param time: Only regard stations active at given point in time.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Only regard stations active at or after given point
            in time (i.e. stations ending before given time will not be shown).
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Only regard stations active before or at given point in
            time (i.e. stations starting after given time will not be shown).
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
        >>> net = read_inventory()[0]
        >>> net.plot_response(0.001, station="FUR")  # doctest: +SKIP

        .. plot::

            from obspy import read_inventory
            net = read_inventory()[0]
            net.plot_response(0.001, station="FUR")
        """
        import matplotlib.pyplot as plt

        if axes:
            ax1, ax2 = axes
            fig = ax1.figure
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax1)

        matching = self.select(station=station, location=location,
                               channel=channel, time=time,
                               starttime=starttime, endtime=endtime)

        for sta in matching.stations:
            for cha in sta.channels:
                try:
                    cha.plot(min_freq=min_freq, output=output, axes=(ax1, ax2),
                             label=".".join((self.code, sta.code,
                                             cha.location_code, cha.code)),
                             unwrap_phase=unwrap_phase, show=False,
                             outfile=None)
                except ZeroSamplingRate:
                    msg = ("Skipping plot of channel with zero "
                           "sampling rate:\n%s")
                    warnings.warn(msg % str(cha), UserWarning)
                except ObsPyException as e:
                    msg = "Skipping plot of channel (%s):\n%s"
                    warnings.warn(msg % (str(e), str(cha)), UserWarning)

        # final adjustments to plot if we created the figure in here
        if not axes:
            from obspy.core.inventory.response import _adjust_bode_plot_figure
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
