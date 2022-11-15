# -*- coding: utf-8 -*-
"""
Module for map related plotting in ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import datetime
import warnings

import numpy as np
import matplotlib
from matplotlib.colorbar import Colorbar
from matplotlib.dates import AutoDateFormatter, AutoDateLocator, date2num
from matplotlib import patheffects
from matplotlib.ticker import (FormatStrFormatter, Formatter, FuncFormatter,
                               MaxNLocator)

from obspy import UTCDateTime
from obspy.core.util import CARTOPY_VERSION
from obspy.core.util.decorator import deprecated_keywords
from obspy.geodetics.base import mean_longitude

if CARTOPY_VERSION and CARTOPY_VERSION >= [0, 12, 0]:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
else:
    HAS_CARTOPY = False

if not HAS_CARTOPY:
    msg = ("Cartopy not installed, map plots will not work.")
    warnings.warn(msg)


_CARTOPY_RESOLUTIONS = {
    'c': '110m',
    'l': '110m',
    'i': '50m',
    'h': '50m',
    'f': '10m',
    '110m': '110m',
    '50m': '50m',
    '10m': '10m',
}

if HAS_CARTOPY:
    _CARTOPY_FEATURES = {
        '110m': (cfeature.BORDERS, cfeature.LAND, cfeature.OCEAN),
    }


@deprecated_keywords({'bmap': None})
def _plot_cartopy_into_axes(
        ax, lons, lats, size, color, bmap=None, labels=None,
        projection='global', resolution='l', continent_fill_color='0.8',
        water_fill_color='1.0', colormap=None, marker="o", title=None,
        adjust_aspect_to_colorbar=False, **kwargs):  # @UnusedVariable

    """
    Creates a (or adds to existing) cartopy plot with a data point scatter
    plot in given axes.

    See :func:`plot_cartopy` for details on most args/kwargs.


    :type ax: :class:`matplotlib.axes.Axes` or
        :class:`cartopy.mpl.geoaxes.GeoAxes`
    :param ax: Existing matplotlib axes instance, optionally with previous
        cartopy plot. If a cartopy GeoAxes is provided, most setup steps will
        be skipped.
    :type bmap: :class:`matplotlib.axes.Axes`
    :param bmap: Deprecated and unused. Whether `ax` is a plain matplotlib Axes
        or a cartopy GeoAxes will determine if cartopy related setup on the
        axis is skipped (setting up projection etc.).
    :rtype: :class:`matplotlib.collections.PathCollection`
    :returns: Matplotlib path collection (e.g. to reuse for colorbars).
    """

    if not isinstance(ax, cartopy.mpl.geoaxes.GeoAxes):
        if projection in ['global', 'ortho']:
            pass
        elif projection == 'local':
            if min(lons) < -150 and max(lons) > 150:
                max_lons = max(np.array(lons) % 360)
                min_lons = min(np.array(lons) % 360)
            else:
                max_lons = max(lons)
                min_lons = min(lons)

            ax.set_extent([min_lons, max_lons, min(lats), max(lats)])

        else:
            msg = "Projection '%s' not supported." % projection
            raise ValueError(msg)
        # ax.gridlines()
        # ax.coastlines()
        # draw coast lines, country boundaries, fill continents.
        # ax.set_facecolor(water_fill_color)
        # newer matplotlib errors out if called with empty coastline data (no
        # coast on map)
        # if np.size(getattr(bmap, 'coastsegs', [])):
        #     bmap.drawcoastlines(color="0.4")
        # bmap.drawcountries(color="0.75")
        # bmap.fillcontinents(color=continent_fill_color,
        #                     lake_color=water_fill_color)
        # draw the edge of the bmap projection region (the projection limb)
        # bmap.drawmapboundary(fill_color=water_fill_color)
        # draw lat/lon grid lines every 30 degrees.
        # bmap.drawmeridians(np.arange(-180, 180, 30))
        # bmap.drawparallels(np.arange(-90, 90, 30))
        ax.stock_img()
        ax.gridlines()
        ax.coastlines()

    # compute the native bmap projection coordinates for events.
    # x, y = bmap(lons, lats)
    x, y = (lons, lats)
    # plot labels

    if labels:
        if 100 > len(lons) > 1:
            for name, xpt, ypt, _colorpt in zip(labels, x, y, color):
                # Check if the point can actually be seen with the current bmap
                # projection. The bmap object will set the coordinates to very
                # large values if it cannot project a point.
                if xpt > 1e25:
                    continue
                ax.text(xpt, ypt, name, weight="heavy",
                        color="k", zorder=100,
                        path_effects=[
                            patheffects.withStroke(linewidth=3,
                                                   foreground="white")],
                        transform=ccrs.Geodetic())
        elif len(lons) == 1:
            ax.text(x[0], y[0], labels[0], weight="heavy", color="k",
                    path_effects=[
                        patheffects.withStroke(linewidth=3,
                                               foreground="white")],
                    transform=ccrs.Geodetic())

    # scatter plot is removing valid x/y points with invalid color value,
    # so we plot those points separately.
    try:
        nan_points = np.isnan(np.array(color, dtype=float))
    except ValueError:
        # `color' was not a list of values, but a list of colors.
        pass
    else:
        if nan_points.any():
            x_ = np.array(x)[nan_points]
            y_ = np.array(y)[nan_points]
            size_ = np.array(size)[nan_points]
            ax.scatter(x_, y_, marker=marker, s=size_, c="0.3",
                       zorder=10, cmap=None, transform=ccrs.Geodetic())
    # Had to change transform to ccrs.PlateCarree, see:
    # https://stackoverflow.com/a/13657749/3645626
    scatter = ax.scatter(x, y, marker=marker, s=size, c=color, zorder=10,
                         cmap=colormap, transform=ccrs.PlateCarree(),)

    if title:
        ax.set_title(title)

    return scatter


def plot_cartopy(lons, lats, size, color, labels=None, projection='global',
                 resolution='110m', continent_fill_color='0.8',
                 water_fill_color='1.0', colormap=None, colorbar=None,
                 marker="o", title=None, colorbar_ticklabel_format=None,
                 show=True, proj_kwargs=None, ax=None,
                 **kwargs):  # @UnusedVariable
    """
    Creates a Cartopy plot with a data point scatter plot.

    :type lons: list[float] or tuple(float)
    :param lons: Longitudes of the data points.
    :type lats: list[float] or tuple(float)
    :param lats: Latitudes of the data points.
    :type size: float, list[float] or tuple(float)
    :param size: Size of the individual points in the scatter plot.
    :type color: list[float], tuple(float) or objects that can be
        converted to floats, like e.g.
        :class:`~obspy.core.utcdatetime.UTCDateTime`)
    :param color: Color information of the individual data points to be
        used in the specified color map (e.g. origin depths,
        origin times).
    :type labels: list[str] or tuple[float]
    :param labels: Annotations for the individual data points.
    :type projection: str, optional
    :param projection: The map projection.
        Currently supported are:

            * ``"global"`` (Will plot the whole world using
              :class:`~cartopy.crs.Mollweide`.)
            * ``"ortho"`` (Will center around the mean lat/long using
              :class:`~cartopy.crs.Orthographic`.)
            * ``"local"`` (Will plot around local events using
              :class:`~cartopy.crs.AlbersEqualArea`.)
            * Any other Cartopy :class:`~cartopy.crs.Projection`. An instance
              of this class will be created using the supplied ``proj_kwargs``.

        Defaults to "global"
    :type resolution: str, optional
    :param resolution: Resolution of the boundary database to use. Will be
        passed directly to the Cartopy module. Possible values are:

            * ``"110m"``
            * ``"50m"``
            * ``"10m"``

        Defaults to ``"110m"``. For compatibility, you may also specify any of
        the cartopy resolutions defined in :func:`plot_cartopy`.
    :type continent_fill_color: valid matplotlib color, optional
    :param continent_fill_color:  Color of the continents. Defaults to
        ``"0.9"`` which is a light gray.
    :type water_fill_color: valid matplotlib color, optional
    :param water_fill_color: Color of all water bodies.
        Defaults to ``"white"``.
    :type colormap: str, valid matplotlib colormap, optional
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
    :type proj_kwargs: dict
    :param proj_kwargs: Keyword arguments to pass to the Cartopy
        :class:`~cartopy.crs.Projection`. In this dictionary, you may specify
        ``central_longitude='auto'`` or ``central_latitude='auto'`` to have
        this function calculate the latitude or longitude as it would for other
        projections. Some arguments may be ignored if you choose one of the
        built-in ``projection`` choices.
    :type ax: :class:`matplotlib.axes.Axes` or
        :class:`cartopy.mpl.geoaxes.GeoAxes`
    :param ax: Existing matplotlib axes instance, optionally with previous
        cartopy plot. If a cartopy GeoAxes is provided, most setup steps will
        be skipped.
    """
    import matplotlib.pyplot as plt

    if isinstance(color[0], (datetime.datetime, UTCDateTime)):
        datetimeplot = True
        color = [date2num(getattr(t, 'datetime', t)) for t in color]
    else:
        datetimeplot = False

    # If ax wasn't specified, look for fig in kwargs
    if ax is None and kwargs.get("fig"):
        ax = kwargs['fig'].axes[0]

    if ax is None:
        fig, map_ax, cm_ax, show_colorbar = _basic_setup(
            lons=lons, lats=lats, size=size, color=color, labels=labels,
            projection=projection, resolution=resolution,
            continent_fill_color=continent_fill_color,
            water_fill_color=water_fill_color, colormap=colormap,
            colorbar=colorbar, marker=marker, title=title,
            colorbar_ticklabel_format=colorbar_ticklabel_format,
            proj_kwargs=proj_kwargs)
    else:
        if isinstance(ax, matplotlib.figure.Figure):
            fig = ax
            map_ax = fig.axes[0]
        else:
            fig = ax.figure
            map_ax = ax
        cm_ax = None
        show_colorbar = False

    # Plot labels
    if labels and len(lons) > 0:
        with map_ax.hold_limits():
            for name, xpt, ypt, _colorpt in zip(labels, lons, lats, color):
                map_ax.text(xpt, ypt, name, weight="heavy", color="k",
                            zorder=100, transform=ccrs.PlateCarree(),
                            path_effects=[
                                patheffects.withStroke(linewidth=3,
                                                       foreground="white")])

    scatter = map_ax.scatter(lons, lats, marker=marker, s=size, c=color,
                             zorder=10, cmap=colormap,
                             transform=ccrs.PlateCarree())

    if title:
        fig.suptitle(title)

    # Only show the colorbar for more than one event.
    if show_colorbar:
        if colorbar_ticklabel_format is not None:
            if isinstance(colorbar_ticklabel_format, str):
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
        cb = Colorbar(cm_ax, scatter,
                      orientation='horizontal',
                      ticks=locator,
                      format=formatter)
        # Compat with old matplotlib versions.
        if hasattr(cb, "update_ticks"):
            cb.update_ticks()

    if show:
        plt.show()

    return fig


def _basic_setup(
        lons, lats, size, color, labels, projection, resolution,
        continent_fill_color, water_fill_color, colormap, colorbar, marker,
        title, colorbar_ticklabel_format, proj_kwargs):
    import matplotlib.pyplot as plt

    fig = plt.figure()

    # The colorbar should only be plotted if more then one event is
    # present.
    if colorbar is not None:
        show_colorbar = colorbar
    else:
        if len(lons) > 1 and hasattr(color, "__len__") and \
                not isinstance(color, str):
            show_colorbar = True
        else:
            show_colorbar = False

    if projection == "local":
        ax_x0, ax_width = 0.10, 0.80
    elif projection == "global":
        ax_x0, ax_width = 0.01, 0.98
    else:
        ax_x0, ax_width = 0.05, 0.90

    proj_kwargs = proj_kwargs or {}
    if projection == 'global':
        proj_kwargs['central_longitude'] = np.mean(lons)
        proj = ccrs.Mollweide(**proj_kwargs)
    elif projection == 'ortho':
        proj_kwargs['central_latitude'] = np.mean(lats)
        proj_kwargs['central_longitude'] = mean_longitude(lons)
        proj = ccrs.Orthographic(**proj_kwargs)
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
        # Do intelligent aspect calculation for local projection
        # adjust to figure dimensions
        w, h = fig.get_size_inches()
        aspect = w / h
        if show_colorbar:
            aspect *= 1.2
        if width / height < aspect:
            width = height * aspect
        else:
            height = width / aspect

        proj_kwargs['central_latitude'] = lat_0
        proj_kwargs['central_longitude'] = lon_0
        proj_kwargs['standard_parallels'] = [lat_0, lat_0]
        proj = ccrs.AlbersEqualArea(**proj_kwargs)

    # User-supplied projection.
    elif isinstance(projection, type):
        if 'central_longitude' in proj_kwargs:
            if proj_kwargs['central_longitude'] == 'auto':
                proj_kwargs['central_longitude'] = mean_longitude(lons)
        if 'central_latitude' in proj_kwargs:
            if proj_kwargs['central_latitude'] == 'auto':
                proj_kwargs['central_latitude'] = np.mean(lats)
        if 'pole_longitude' in proj_kwargs:
            if proj_kwargs['pole_longitude'] == 'auto':
                proj_kwargs['pole_longitude'] = np.mean(lons)
        if 'pole_latitude' in proj_kwargs:
            if proj_kwargs['pole_latitude'] == 'auto':
                proj_kwargs['pole_latitude'] = np.mean(lats)

        proj = projection(**proj_kwargs)

    else:
        msg = "Projection '%s' not supported." % projection
        raise ValueError(msg)

    if show_colorbar:
        map_ax = fig.add_axes([ax_x0, 0.13, ax_width, 0.77], projection=proj)
        cm_ax = fig.add_axes([ax_x0, 0.05, ax_width, 0.05])
        plt.sca(map_ax)
    else:
        ax_y0, ax_height = 0.05, 0.85
        if projection == "local":
            ax_y0 += 0.05
            ax_height -= 0.05
        map_ax = fig.add_axes([ax_x0, ax_y0, ax_width, ax_height],
                              projection=proj)
        cm_ax = None

    if projection == 'local':
        x0, y0 = proj.transform_point(lon_0, lat_0, proj.as_geodetic())
        map_ax.set_xlim(x0 - width / 2, x0 + width / 2)
        map_ax.set_ylim(y0 - height / 2, y0 + height / 2)
    else:
        map_ax.set_global()

    # Pick features at specified resolution.
    resolution = _CARTOPY_RESOLUTIONS[resolution]
    try:
        borders, land, ocean = _CARTOPY_FEATURES[resolution]
    except KeyError:
        borders = cfeature.NaturalEarthFeature(
            cfeature.BORDERS.category, cfeature.BORDERS.name, resolution,
            edgecolor='none', facecolor='none')
        land = cfeature.NaturalEarthFeature(
            cfeature.LAND.category, cfeature.LAND.name, resolution,
            edgecolor='face', facecolor='none')
        ocean = cfeature.NaturalEarthFeature(
            cfeature.OCEAN.category, cfeature.OCEAN.name, resolution,
            edgecolor='face', facecolor='none')
        _CARTOPY_FEATURES[resolution] = (borders, land, ocean)

    # Draw coast lines, country boundaries, fill continents.
    map_ax.set_facecolor(water_fill_color)
    map_ax.add_feature(ocean, facecolor=water_fill_color)
    map_ax.add_feature(land, facecolor=continent_fill_color)
    map_ax.add_feature(borders, edgecolor='0.75')
    map_ax.coastlines(resolution=resolution, color='0.4')

    # Draw grid lines - TODO: draw_labels=True doesn't work yet.
    if projection == 'local':
        map_ax.gridlines()
    else:
        # Draw lat/lon grid lines every 30 degrees.
        map_ax.gridlines(xlocs=range(-180, 181, 30), ylocs=range(-90, 91, 30))

    return fig, map_ax, cm_ax, show_colorbar


def plot_map(method, *args, **kwargs):
    """
    Creates a map plot with a data point scatter plot.

    :type method: str
    :param method: Method to use for plotting. Possible values are:

        * ``'cartopy'`` to use the Cartopy library. For other arguments, see
          the :func:`plot_cartopy` function.
        * ``None`` will use the Cartopy library since it is the only supported
          method right now.
    """
    if method is None:
        if HAS_CARTOPY:
            return plot_cartopy(*args, **kwargs)
        else:
            raise ImportError('Cartopy could not be imported.')
    elif method == 'cartopy':
        if not HAS_CARTOPY:
            raise ImportError('Cartopy cannot be imported but was explicitly '
                              'requested.')
        return plot_cartopy(*args, **kwargs)
    else:
        raise ValueError("The method argument must be either 'None' or "
                         "'cartopy', not '%s'." % (method, ))
