# -*- coding: utf-8 -*-
"""
Module for basemap related plotting in ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future.utils import native_str

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import Colorbar
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, Formatter, \
    MaxNLocator
from matplotlib.dates import date2num, AutoDateLocator, \
    AutoDateFormatter
import datetime
import numpy as np
import warnings
from obspy import UTCDateTime
from obspy.core.util.base import getMatplotlibVersion, getBasemapVersion


MATPLOTLIB_VERSION = getMatplotlibVersion()

if MATPLOTLIB_VERSION < [1, 0, 0]:
    path_effect_kwargs = {}
else:
    import matplotlib.patheffects as PathEffects
    path_effect_kwargs = dict(
        path_effects=[PathEffects.withStroke(linewidth=3, foreground="white")])


BASEMAP_VERSION = getBasemapVersion()
if BASEMAP_VERSION:
    from mpl_toolkits.basemap import Basemap
    HAS_BASEMAP = True
    if BASEMAP_VERSION < [1, 0, 4]:
        warnings.warn("All basemap version < 1.0.4 contain a serious bug "
                      "when rendering countries and continents. ObsPy will "
                      "still work but the maps might be wrong. Please update "
                      "your basemap installation.")
else:
    warnings.warn("basemap not installed.")
    HAS_BASEMAP = False


def plot_basemap(lons, lats, size, color, labels=None,
                 projection='cyl', resolution='l', continent_fill_color='0.8',
                 water_fill_color='1.0', colormap=None, colorbar=None,
                 marker="o", title=None, colorbar_ticklabel_format=None,
                 show=True, **kwargs):  # @UnusedVariable
    """
    Creates a basemap plot with a data point scatter plot.

    :type lons: list/tuple of floats
    :param lons: Longitudes of the data points.
    :type lats: list/tuple of floats
    :param lats: Latitudes of the data points.
    :type size: float or list/tuple of floats
    :param size: Size of the individual points in the scatter plot.
    :type color: list/tuple of floats (or objects that can be
        converted to floats, like e.g.
        :class:`~obspy.core.utcdatetime.UTCDateTime`)
    :param color: Color information of the individual data points to be
        used in the specified color map (e.g. origin depths,
        origin times).
    :type labels: list/tuple of str
    :param labels: Annotations for the individual data points.
    :type projection: str, optional
    :param projection: The map projection. Currently supported are
        * ``"cyl"`` (Will plot the whole world.)
        * ``"ortho"`` (Will center around the mean lat/long.)
        * ``"local"`` (Will plot around local events)
        Defaults to "cyl"
    :type resolution: str, optional
    :param resolution: Resolution of the boundary database to use. Will be
        based directly to the basemap module. Possible values are
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
    :type colormap: str, any matplotlib colormap, optional
    :param colormap: The colormap for color-coding the events as provided
        in `color` kwarg.
        The event with the smallest `color` property will have the
        color of one end of the colormap and the event with the highest
        `color` property the color of the other end with all other events
        in between.
        Defaults to None which will use the default matplotlib colormap.
    :type colorbar: bool, optional
    :param colorbar: When left `None`, a colorbar is plotted if more than one
        object is plotted. Using `True`/`False` the colorbar can be forced
        on/off.
    :type title: str
    :param title: Title above plot.
    :type colorbar_ticklabel_format: str or function or
        subclass of :class:`matplotlib.ticker.Formatter`
    :param colorbar_ticklabel_format: Format string or Formatter used to format
        colorbar tick labels.
    :type show: bool
    :param show: Whether to show the figure after plotting or not. Can be used
        to do further customization of the plot before showing it.
    """
    min_color = min(color)
    max_color = max(color)

    if isinstance(color[0], (datetime.datetime, UTCDateTime)):
        datetimeplot = True
        color = [date2num(t) for t in color]
    else:
        datetimeplot = False

    scal_map = ScalarMappable(norm=Normalize(min_color, max_color),
                              cmap=colormap)
    scal_map.set_array(np.linspace(0, 1, 1))

    fig = plt.figure()
    # The colorbar should only be plotted if more then one event is
    # present.

    if colorbar is not None:
        show_colorbar = colorbar
    else:
        if len(lons) > 1 and hasattr(color, "__len__") and \
                not isinstance(color, (str, native_str)):
            show_colorbar = True
        else:
            show_colorbar = False

    if projection == "local":
        ax_x0, ax_width = 0.10, 0.80
    else:
        ax_x0, ax_width = 0.05, 0.90

    if show_colorbar:
        map_ax = fig.add_axes([ax_x0, 0.13, ax_width, 0.77])
        cm_ax = fig.add_axes([ax_x0, 0.05, ax_width, 0.05])
    else:
        ax_y0, ax_height = 0.05, 0.85
        if projection == "local":
            ax_y0 += 0.05
            ax_height -= 0.05
        map_ax = fig.add_axes([ax_x0, ax_y0, ax_width, ax_height])

    if projection == 'cyl':
        bmap = Basemap(resolution=resolution, ax=map_ax)
    elif projection == 'ortho':
        bmap = Basemap(projection='ortho', resolution=resolution,
                       area_thresh=1000.0, lat_0=np.mean(lats),
                       lon_0=np.mean(lons), ax=map_ax)
    elif projection == 'local':
        if min(lons) < -150 and max(lons) > 150:
            max_lons = max(np.array(lons) % 360)
            min_lons = min(np.array(lons) % 360)
        else:
            max_lons = max(lons)
            min_lons = min(lons)
        lat_0 = max(lats) / 2. + min(lats) / 2.
        lon_0 = max_lons / 2. + min_lons / 2.
        if lon_0 > 180:
            lon_0 -= 360
        deg2m_lat = 2 * np.pi * 6371 * 1000 / 360
        deg2m_lon = deg2m_lat * np.cos(lat_0 / 180 * np.pi)
        if len(lats) > 1:
            height = (max(lats) - min(lats)) * deg2m_lat
            width = (max_lons - min_lons) * deg2m_lon
            margin = 0.2 * (width + height)
            height += margin
            width += margin
        else:
            height = 2.0 * deg2m_lat
            width = 5.0 * deg2m_lon
        # do intelligent aspect calculation for local projection
        # adjust to figure dimensions
        w, h = fig.get_size_inches()
        aspect = w / h
        if show_colorbar:
            aspect *= 1.2
        if width / height < aspect:
            width = height * aspect
        else:
            height = width / aspect

        bmap = Basemap(projection='aeqd', resolution=resolution,
                       area_thresh=1000.0, lat_0=lat_0, lon_0=lon_0,
                       width=width, height=height, ax=map_ax)
        # not most elegant way to calculate some round lats/lons

        def linspace2(val1, val2, N):
            """
            returns around N 'nice' values between val1 and val2
            """
            dval = val2 - val1
            round_pos = int(round(-np.log10(1. * dval / N)))
            # Fake negative rounding as not supported by future as of now.
            if round_pos < 0:
                factor = 10 ** (abs(round_pos))
                delta = round(2. * dval / N / factor) * factor / 2
            else:
                delta = round(2. * dval / N, round_pos) / 2
            new_val1 = np.ceil(val1 / delta) * delta
            new_val2 = np.floor(val2 / delta) * delta
            N = (new_val2 - new_val1) / delta + 1
            return np.linspace(new_val1, new_val2, N)

        N1 = int(np.ceil(height / max(width, height) * 8))
        N2 = int(np.ceil(width / max(width, height) * 8))
        parallels = linspace2(lat_0 - height / 2 / deg2m_lat,
                              lat_0 + height / 2 / deg2m_lat, N1)

        # Old basemap versions have problems with non-integer parallels.
        try:
            bmap.drawparallels(parallels, labels=[0, 1, 1, 0])
        except KeyError:
            parallels = sorted(list(set(map(int, parallels))))
            bmap.drawparallels(parallels, labels=[0, 1, 1, 0])

        if min(lons) < -150 and max(lons) > 150:
            lon_0 %= 360
        meridians = linspace2(lon_0 - width / 2 / deg2m_lon,
                              lon_0 + width / 2 / deg2m_lon, N2)
        meridians[meridians > 180] -= 360
        bmap.drawmeridians(meridians, labels=[1, 0, 0, 1])
    else:
        msg = "Projection '%s' not supported." % projection
        raise ValueError(msg)

    # draw coast lines, country boundaries, fill continents.
    map_ax.set_axis_bgcolor(water_fill_color)
    bmap.drawcoastlines(color="0.4")
    bmap.drawcountries(color="0.75")
    bmap.fillcontinents(color=continent_fill_color,
                        lake_color=water_fill_color)
    # draw the edge of the bmap projection region (the projection limb)
    bmap.drawmapboundary(fill_color=water_fill_color)
    # draw lat/lon grid lines every 30 degrees.
    bmap.drawmeridians(np.arange(-180, 180, 30))
    bmap.drawparallels(np.arange(-90, 90, 30))

    # compute the native bmap projection coordinates for events.
    x, y = bmap(lons, lats)
    # plot labels
    if labels:
        if 100 > len(lons) > 1:
            for name, xpt, ypt, _colorpt in zip(labels, x, y, color):
                # Check if the point can actually be seen with the current bmap
                # projection. The bmap object will set the coordinates to very
                # large values if it cannot project a point.
                if xpt > 1e25:
                    continue
                map_ax.text(xpt, ypt, name, weight="heavy",
                            color="k", zorder=100, **path_effect_kwargs)
        elif len(lons) == 1:
            map_ax.text(x[0], y[0], labels[0], weight="heavy", color="k",
                        **path_effect_kwargs)

    scatter = bmap.scatter(x, y, marker=marker, s=size, c=color,
                           zorder=10, cmap=colormap)

    if title:
        plt.suptitle(title)

    # Only show the colorbar for more than one event.
    if show_colorbar:
        if colorbar_ticklabel_format is not None:
            if isinstance(colorbar_ticklabel_format, (str, native_str)):
                formatter = FormatStrFormatter(colorbar_ticklabel_format)
            elif hasattr(colorbar_ticklabel_format, '__call__'):
                formatter = FuncFormatter(colorbar_ticklabel_format)
            elif isinstance(colorbar_ticklabel_format, Formatter):
                formatter = colorbar_ticklabel_format
            locator = MaxNLocator(5)
        else:
            if datetimeplot:
                locator = AutoDateLocator()
                formatter = AutoDateFormatter(locator)
                # Compat with old matplotlib versions.
                if hasattr(formatter, "scaled"):
                    formatter.scaled[1 / (24. * 60.)] = '%H:%M:%S'
            else:
                locator = None
                formatter = None
        cb = Colorbar(cm_ax, scatter, cmap=colormap,
                      orientation='horizontal',
                      ticks=locator,
                      format=formatter)
        # Compat with old matplotlib versions.
        if hasattr(cb, "update_ticks"):
            cb.update_ticks()

    if show:
        plt.show()

    return fig
