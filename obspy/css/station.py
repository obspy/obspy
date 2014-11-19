# -*- coding: utf-8 -*-
"""
CSS bindings to ObsPy station module.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future.utils import native_str

from obspy import UTCDateTime


def _yearday(date):
    if date:
        return date.year * 1000 + date.julday
    else:
        return -1


def writeCSS(inventory, basename):
    """
    Writes an inventory object to a CSS database.

    The :class:`~obspy.station.inventory.Inventory`,
    :class:`~obspy.station.network.Network`,
    :class:`~obspy.station.station.Station`, and
    :class:`~obspy.station.channel.Channel` objects are included in the
    resulting database. Any :class:`~obspy.station.util.Comment` objects are
    only saved for the network. Any :class:`~obspy.station.response.Response`
    objects are not saved. For fields that are saved, most of the important
    information is used, but any extra metadata that cannot be represented in
    CSS will be lost.

    .. note::
        Because CSS stores data in multiple files, you cannot write to a
        file-like object. You should specify the basename of the CSS database
        only.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.station.inventory.Inventory.write` method of an
        ObsPy :class:`~obspy.station.inventory.Inventory` object, call this
        instead.

    :type inventory: :class:`~obspy.station.inventory.Inventory`
    :param inventory: The inventory instance to be written.
    :type basename: str
    :param basename: The base name of the files to be written. This export
        format currently writes to the following files ("relations" in CSS
        terms):

        ``basename.affiliation``
            Clusters seismic stations into networks.
        ``basename.network``
            Describes general information about seismic networks.
        ``basename.site``
            Contains site names and locations on the Earth where seismic
            measurements are made.
        ``basename.sitechan``
            Describes the orientation of recording channels at a site.
        ``basename.remark``
            Stores free-form comments that embellish records of other
            relations.
    """
    if not isinstance(basename, (str, native_str)):
        raise TypeError('Writing an Inventory to a file-like object in CSS '
                        'format is unsupported.')

    affiliation = []
    network = []
    site = []
    sitechan = []
    remark = []
    comment_id = 0
    auth = (inventory.source or inventory.sender or '-').replace('\n', ' ')
    lddate = (inventory.created or UTCDateTime()).strftime('%Y-%m-%dT%H%M%S')
    for net in inventory:
        if net.comments:
            comment_id += 1
            for i, comment in enumerate(net.comments):
                remark_line = '%8d %8d %-80.80s %-17.17s' % (
                    comment_id,
                    i + 1,
                    comment.value.replace('\n', ' '),
                    lddate)
                remark.append(remark_line)

        network_line = '%-8.8s %-80.80s %-4.4s %-15.15s %8d %-17.17s' % (
            net.code,
            net.description.replace('\n', ' ') if net.description else '-',
            '-',
            auth,
            comment_id if net.comments else -1,
            lddate)
        network.append(network_line)

        for sta in net:
            affiliation_line = '%-8.8s %-6.6s %-17.17s' % (
                net.code,
                sta.code,
                lddate)
            affiliation.append(affiliation_line)

            site_line = ('%-6.6s %8d %8d %9.4f %9.4f %9.4f %-50.50s %-4.4s '
                         '%-6.6s %9.4f %9.4f %-17.17s') % (
                sta.code,
                _yearday(sta.start_date),
                _yearday(sta.end_date),
                sta.latitude,
                sta.longitude,
                sta.elevation / 1000.0,
                (sta.site.name if sta.site
                    else sta.description or '-').replace('\n', ' '),
                '-',
                '-',
                0.0,
                0.0,
                lddate)
            site.append(site_line)

            for cha in sta:
                sitechan_line = ('%-6.6s %-8.8s %8d %8d %8d %-4.4s %9.4f '
                                 '%6.6s %6.6s %-50.50s %-17.17s') % (
                    sta.code,
                    cha.code,
                    _yearday(cha.start_date),
                    -1,
                    _yearday(cha.end_date),
                    'b' if 'BEAM' in cha.types else '-',
                    cha.depth,
                    ('%6.1f' % (cha.azimuth, )
                        if cha.azimuth is not None else 'NaN'),
                    ('%6.1f' % (cha.dip, )
                        if cha.dip is not None else 'NaN'),
                    (cha.description.replace('\n', ' ')
                        if cha.description else '-'),
                    lddate)
                sitechan.append(sitechan_line)

    if remark:
        with open(basename + '.remark', 'wt') as fh:
            fh.write('\n'.join(remark))
            fh.write('\n')
    if affiliation:
        with open(basename + '.affiliation', 'wt') as fh:
            fh.write('\n'.join(affiliation))
            fh.write('\n')
    if network:
        with open(basename + '.network', 'wt') as fh:
            fh.write('\n'.join(network))
            fh.write('\n')
    if site:
        with open(basename + '.site', 'wt') as fh:
            fh.write('\n'.join(site))
            fh.write('\n')
    if sitechan:
        with open(basename + '.sitechan', 'wt') as fh:
            fh.write('\n'.join(sitechan))
            fh.write('\n')
