#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides the Station class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import copy
import fnmatch
import warnings

import numpy as np

from obspy import UTCDateTime
from obspy.core.util.obspy_types import (ObsPyException, ZeroSamplingRate,
                                         FloatWithUncertaintiesAndUnit)
from obspy.geodetics import inside_geobounds

from .util import (BaseNode, Equipment, Operator, Distance, Latitude,
                   Longitude, Site, _unified_content_strings_expanded)


class Station(BaseNode):
    """
    From the StationXML definition:
        This type represents a Station epoch. It is common to only have a
        single station epoch with the station's creation and termination dates
        as the epoch start and end dates.
    """
    def __init__(self, code, latitude, longitude, elevation, channels=None,
                 site=None, vault=None, geology=None, equipments=None,
                 operators=None, creation_date=None, termination_date=None,
                 total_number_of_channels=None,
                 selected_number_of_channels=None, description=None,
                 comments=None, start_date=None, end_date=None,
                 restricted_status=None, alternate_code=None,
                 historical_code=None, data_availability=None,
                 identifiers=None, water_level=None, source_id=None):
        """
        :type channels: list of :class:`~obspy.core.inventory.channel.Channel`
        :param channels: All channels belonging to this station.
        :type site: :class:`~obspy.core.inventory.util.Site`
        :param site: The lexical description of the site
        :type latitude: :class:`~obspy.core.inventory.util.Latitude`
        :param latitude: The latitude of the station
        :type longitude: :class:`~obspy.core.inventory.util.Longitude`
        :param longitude: The longitude of the station
        :param elevation: The elevation of the station in meter.
        :param site: These fields describe the location of the station using
            geopolitical entities (country, city, etc.).
        :param vault: Type of vault, e.g. WWSSN, tunnel, transportable array,
            etc
        :param geology: Type of rock and/or geologic formation.
        :type equipments: list of :class:`~obspy.core.inventory.util.Equipment`
        :param equipments: Equipment used by all channels at a station.
        :type operators: list of :class:`~obspy.core.inventory.util.Operator`
        :param operators: An operating agency and associated contact persons.
        :type creation_date: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param creation_date: Date and time (UTC) when the station was first
            installed
        :type termination_date: :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param termination_date: Date and time (UTC) when the station was
            terminated or will be terminated. A blank value should be assumed
            to mean that the station is still active.
        :type total_number_of_channels: int
        :param total_number_of_channels: Total number of channels recorded at
            this station.
        :type selected_number_of_channels: int
        :param selected_number_of_channels: Number of channels recorded at this
            station and selected by the query that produced this document.
        :type external_references: list of
            :class:`~obspy.core.inventory.util.ExternalReference`
        :param external_references: URI of any type of external report, such as
            IRIS data reports or dataless SEED volumes.
        :type description: str
        :param description: A description of the resource
        :type comments: list of :class:`~obspy.core.inventory.util.Comment`
        :param comments: An arbitrary number of comments to the resource
        :type start_date: :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param start_date: The start date of the resource
        :type end_date: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param end_date: The end date of the resource
        :type restricted_status: str
        :param restricted_status: The restriction status
        :type alternate_code: str
        :param alternate_code: A code used for display or association,
            alternate to the SEED-compliant code.
        :type historical_code: str
        :param historical_code: A previously used code if different from the
            current code.
        :type data_availability:
            :class:`~obspy.core.inventory.util.DataAvailability`
        :param data_availability: Information about time series availability
            for the station.
        :type identifiers: list[str], optional
        :param identifiers: Persistent identifiers for network/station/channel
            (schema version >=1.1). URIs are in general composed of a 'scheme'
            and a 'path' (optionally with additional components), the two of
            which separated by a colon.
        :type water_level: float, optional
        :param water_level: Elevation of the water surface in meters for
            underwater sites, where 0 is sea level. (schema version >=1.1)
        :type source_id: str, optional
        :param source_id: A data source identifier in URI form
            (schema version >=1.1). URIs are in general composed of a 'scheme'
            and a 'path' (optionally with additional components), the two of
            which separated by a colon.
        """
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.channels = channels or []
        self.site = site if site is not None else Site()
        self.vault = vault
        self.geology = geology
        self.equipments = equipments or []
        self.operators = operators or []
        self.creation_date = creation_date
        self.termination_date = termination_date
        self.total_number_of_channels = total_number_of_channels
        self.selected_number_of_channels = selected_number_of_channels
        self.external_references = []
        self.water_level = water_level
        super(Station, self).__init__(
            code=code, description=description, comments=comments,
            start_date=start_date, end_date=end_date,
            restricted_status=restricted_status, alternate_code=alternate_code,
            historical_code=historical_code,
            data_availability=data_availability, identifiers=identifiers,
            source_id=source_id)

    @property
    def total_number_of_channels(self):
        return self._total_number_of_channels

    @total_number_of_channels.setter
    def total_number_of_channels(self, value):
        if value is not None and value < 0:
            msg = "total_number_of_channels cannot be negative."
            raise ValueError(msg)
        self._total_number_of_channels = value

    @property
    def selected_number_of_channels(self):
        return self._selected_number_of_channels

    @selected_number_of_channels.setter
    def selected_number_of_channels(self, value):
        if value is not None and value < 0:
            msg = "selected_number_of_channels cannot be negative."
            raise ValueError(msg)
        self._selected_number_of_channels = value

    def __str__(self):
        contents = self.get_contents()
        ret = ("Station {station_name}\n"
               "\tStation Code: {station_code}\n"
               "\tChannel Count: {selected}/{total} (Selected/Total)\n"
               "\t{start_date} - {end_date}\n"
               "\tAccess: {restricted} {alternate_code}{historical_code}\n"
               "\tLatitude: {lat:.4f}, Longitude: {lng:.4f}, "
               "Elevation: {elevation:.1f} m\n")
        ret = ret.format(
            station_name=contents["stations"][0],
            station_code=self.code,
            selected=self.selected_number_of_channels,
            total=self.total_number_of_channels,
            start_date=str(self.start_date),
            end_date=str(self.end_date) if self.end_date else "",
            restricted=self.restricted_status,
            lat=self.latitude, lng=self.longitude, elevation=self.elevation,
            alternate_code="Alternate Code: %s " % self.alternate_code if
            self.alternate_code else "",
            historical_code="historical Code: %s " % self.historical_code if
            self.historical_code else "")
        ret += "\tAvailable Channels:\n"
        for ele in _unified_content_strings_expanded(self.channels):
            ret += "\t%s\n" % ele
        return ret

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def __getitem__(self, index):
        return self.channels[index]

    def __len__(self):
        return len(self.channels)

    def get_contents(self):
        """
        Returns a dictionary containing the contents of the object.

        .. rubric:: Example

        >>> from obspy import read_inventory
        >>> example_filename = "/path/to/IRIS_single_channel_with_response.xml"
        >>> inventory = read_inventory(example_filename)
        >>> station = inventory.networks[0].stations[0]
        >>> station.get_contents()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        {...}
        >>> for (k, v) in sorted(station.get_contents().items()):
        ...     print(k, v[0])
        channels ANMO.10.BHZ
        stations ANMO (Albuquerque, New Mexico, USA)
        """
        site_name = None
        if self.site and self.site.name:
            site_name = self.site.name
        desc = "%s%s" % (self.code, " (%s)" % (site_name if site_name else ""))
        content_dict = {"stations": [desc], "channels": []}

        for channel in self.channels:
            content_dict["channels"].append(
                "%s.%s.%s" % (self.code, channel.location_code, channel.code))
        return content_dict

    @property
    def operators(self):
        return self._operators

    @operators.setter
    def operators(self, value):
        if not hasattr(value, "__iter__"):
            msg = "Operators needs to be an iterable, e.g. a list."
            raise ValueError(msg)
        # make sure to unwind actual iterators, or the just might get exhausted
        # at some point
        operators = [operator for operator in value]
        if any([not isinstance(x, Operator) for x in operators]):
            msg = "Operators can only contain Operator objects."
            raise ValueError(msg)
        self._operators = operators

    @property
    def equipments(self):
        return self._equipments

    @equipments.setter
    def equipments(self, value):
        if not hasattr(value, "__iter__"):
            msg = "equipments needs to be an iterable, e.g. a list."
            raise ValueError(msg)
        # make sure to unwind actual iterators, or the just might get exhausted
        # at some point
        equipments = [equipment for equipment in value]
        if any([not isinstance(x, Equipment) for x in equipments]):
            msg = "equipments can only contain Equipment objects."
            raise ValueError(msg)
        self._equipments = equipments
        # if value is None or isinstance(value, Equipment):
        #    self._equipment = value
        # elif isinstance(value, dict):
        #    self._equipment = Equipment(**value)
        # else:
        #    msg = ("equipment needs to be be of type
        # obspy.core.inventory.Equipment "
        #        "or contain a dictionary with values suitable for "
        #        "initialization.")
        #    raise ValueError(msg)

    @property
    def creation_date(self):
        return self._creation_date

    @creation_date.setter
    def creation_date(self, value):
        if value is None:
            self._creation_date = None
            return
        if not isinstance(value, UTCDateTime):
            value = UTCDateTime(value)
        self._creation_date = value

    @property
    def termination_date(self):
        return self._termination_date

    @termination_date.setter
    def termination_date(self, value):
        if value is not None and not isinstance(value, UTCDateTime):
            value = UTCDateTime(value)
        self._termination_date = value

    @property
    def external_references(self):
        return self._external_references

    @external_references.setter
    def external_references(self, value):
        if not hasattr(value, "__iter__"):
            msg = "external_references needs to be iterable, e.g. a list."
            raise ValueError(msg)
        self._external_references = value

    @property
    def longitude(self):
        return self._longitude

    @longitude.setter
    def longitude(self, value):
        if isinstance(value, Longitude):
            self._longitude = value
        else:
            self._longitude = Longitude(value)

    @property
    def latitude(self):
        return self._latitude

    @latitude.setter
    def latitude(self, value):
        if isinstance(value, Latitude):
            self._latitude = value
        else:
            self._latitude = Latitude(value)

    @property
    def elevation(self):
        return self._elevation

    @elevation.setter
    def elevation(self, value):
        if isinstance(value, Distance):
            self._elevation = value
        else:
            self._elevation = Distance(value)

    @property
    def water_level(self):
        return self._water_level

    @water_level.setter
    def water_level(self, value):
        if value is None:
            self._water_level = None
        elif isinstance(value, FloatWithUncertaintiesAndUnit):
            self._water_level = value
        else:
            self._water_level = FloatWithUncertaintiesAndUnit(value)

    def select(self, location=None, channel=None, time=None, starttime=None,
               endtime=None, sampling_rate=None, minlatitude=None,
               maxlatitude=None, minlongitude=None, maxlongitude=None,
               latitude=None, longitude=None, minradius=None, maxradius=None):
        r"""
        Returns the :class:`Station` object with only the
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
        >>> sta = read_inventory()[0][0]
        >>> t = UTCDateTime(2008, 7, 1, 12)
        >>> sta = sta.select(channel="[LB]HZ", time=t)
        >>> print(sta)  # doctest: +NORMALIZE_WHITESPACE
        Station FUR (Fuerstenfeldbruck, Bavaria, GR-Net)
            Station Code: FUR
            Channel Count: None/None (Selected/Total)
            2006-12-16T00:00:00.000000Z -
            Access: None
            Latitude: 48.1629, Longitude: 11.2752, Elevation: 565.0 m
            Available Channels:
             ..BHZ        20.0 Hz  2006-12-16 to None
             ..LHZ         1.0 Hz  2006-12-16 to None

        The `location` and `channel` selection criteria  may also contain UNIX
        style wildcards (e.g. ``*``, ``?``, ...; see
        :func:`~fnmatch.fnmatch`).

        :type location: str
        :param location: Potentially wildcarded location code. If not given,
            all location codes will be accepted.
        :type channel: str
        :param channel: Potentially wildcarded channel code. If not given,
            all channel codes will be accepted.
        :type time: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param time: Only include channels active at given point in time.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Only include channels active at or after given point
            in time (i.e. channels ending before given time will not be shown).
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Only include channels active before or at given point
            in time (i.e. channels starting after given time will not be
            shown).
        :type sampling_rate: float
        :param sampling_rate: Only include channels whose sampling rate
            matches the given sampling rate, in Hz (within absolute tolerance
            of 1E-8 Hz and relative tolerance of 1E-5)
        :type minlatitude: float
        :param minlatitude: Only include channels with a latitude larger than
            the specified minimum.
        :type maxlatitude: float
        :param maxlatitude: Only include channels with a latitude smaller than
            the specified maximum.
        :type minlongitude: float
        :param minlongitude: Only include channels with a longitude larger than
            the specified minimum.
        :type maxlongitude: float
        :param maxlongitude: Only include channels with a longitude smaller
            than the specified maximum.
        :type latitude: float
        :param latitude: Specify the latitude to be used for a radius
            selection.
        :type longitude: float
        :param longitude: Specify the longitude to be used for a radius
            selection.
        :type minradius: float
        :param minradius: Only include channels within the specified
            minimum number of degrees from the geographic point defined by the
            latitude and longitude parameters.
        :type maxradius: float
        :param maxradius: Only include channels within the specified
            maximum number of degrees from the geographic point defined by the
            latitude and longitude parameters.
        """
        channels = []
        for cha in self.channels:
            # skip if any given criterion is not matched
            if location is not None:
                if not fnmatch.fnmatch(cha.location_code.upper(),
                                       location.upper()):
                    continue
            if channel is not None:
                if not fnmatch.fnmatch(cha.code.upper(),
                                       channel.upper()):
                    continue
            if sampling_rate is not None:
                if cha.sample_rate is None:
                    msg = ("Omitting channel that has no sampling rate "
                           "specified.")
                    warnings.warn(msg)
                    continue
                if not np.allclose(float(sampling_rate), cha.sample_rate,
                                   rtol=1E-5, atol=1E-8):
                    continue
            if any([t is not None for t in (time, starttime, endtime)]):
                if not cha.is_active(time=time, starttime=starttime,
                                     endtime=endtime):
                    continue
            geo_filters = dict(
                minlatitude=minlatitude, maxlatitude=maxlatitude,
                minlongitude=minlongitude, maxlongitude=maxlongitude,
                latitude=latitude, longitude=longitude, minradius=minradius,
                maxradius=maxradius)
            if any(value is not None for value in geo_filters.values()):
                if not inside_geobounds(cha, **geo_filters):
                    continue

            channels.append(cha)
        sta = copy.copy(self)
        sta.channels = channels
        return sta

    def plot(self, min_freq, output="VEL", location="*", channel="*",
             time=None, starttime=None, endtime=None, axes=None,
             unwrap_phase=False, plot_degrees=False, show=True, outfile=None):
        """
        Show bode plot of instrument response of all (or a subset of) the
        station's channels.

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

        :type location: str
        :param location: Only plot matching channels. Accepts UNIX style
            patterns and wildcards (e.g. ``"BH*"``, ``"BH?"``, ``"*Z"``,
            ``"[LB]HZ"``; see :func:`~fnmatch.fnmatch`)
        :type channel: str
        :param channel: Only plot matching channels. Accepts UNIX style
            patterns and wildcards (e.g. ``"BH*"``, ``"BH?"``, ``"*Z"``,
            ``"[LB]HZ"``; see :func:`~fnmatch.fnmatch`)
        :param time: Only show channels active at given point in time.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Only show channels active at or after given point in
            time (i.e. channels ending before given time will not be shown).
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: Only show channels active before or at given point in
            time (i.e. channels starting after given time will not be shown).
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
        >>> sta = read_inventory()[0][0]
        >>> sta.plot(0.001, output="VEL", channel="*Z")  # doctest: +SKIP

        .. plot::

            from obspy import read_inventory
            sta = read_inventory()[0][0]
            sta.plot(0.001, output="VEL", channel="*Z")
        """
        import matplotlib.pyplot as plt

        if axes is not None:
            ax1, ax2 = axes
            fig = ax1.figure
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax1)

        matching = self.select(location=location, channel=channel, time=time,
                               starttime=starttime, endtime=endtime)

        for cha in matching.channels:
            try:
                cha.plot(min_freq=min_freq, output=output, axes=(ax1, ax2),
                         label=".".join((self.code, cha.location_code,
                                         cha.code)),
                         unwrap_phase=unwrap_phase, plot_degrees=plot_degrees,
                         show=False, outfile=None)
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
            _adjust_bode_plot_figure(fig, plot_degrees=plot_degrees,
                                     show=False)

        if outfile:
            fig.savefig(outfile)
        else:
            if show:
                plt.show()

        return fig


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
