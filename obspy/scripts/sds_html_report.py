#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a quality control HTML page.

Build QC html page with information on latency, data availability percentage
and gap/overlap count from a local SDS archive.

Example
=======

For a full scan, computing list of streams to check and data
availability percentage and gaps (note how to specify an empty location code):

.. code-block:: bash

    $ obspy-sds-report -r /bay200/mseed_online/archive -o /tmp/sds_report \
-l "" -l 00 -c EHZ -c HHZ -c ELZ -i BW.PFORL..HJZ -i BW.RLAS..HJZ

For a subsequent update of latency only:

.. code-block:: bash

    $ obspy-sds-report -r /bay200/mseed_online/archive -o /tmp/sds_report \
--update

Screenshot of resulting html page (cut off at the bottom):

.. image:: /_images/sds_report.png

"""
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
import numpy as np

from obspy import __version__, UTCDateTime
from obspy.core.util.base import ENTRY_POINTS
from obspy.core.util.misc import MatplotlibBackend
from obspy.core.util.obspy_types import ObsPyException
from obspy.clients.filesystem.sds import Client
from obspy.imaging.scripts.scan import scan


HTML_TEMPLATE = """\
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN"
"http://www.w3.org/TR/html4/strict.dtd">
<html>
<head><META HTTP-EQUIV=REFRESH CONTENT=60>
<title>SDS Report {sds_root}</title>
<style>
p.monospace
    {{font-family: "Lucida Console", Monaco, monospace;
     }}
p.monospace_wide_columns
    {{font-family: "Lucida Console", Monaco, monospace;
     -webkit-column-count: 4;
     -moz-column-count: 4;
     column-count: 4;
     -webkit-column-width: 480px;
     -moz-column-width: 480px;
     column-width: 480px;
     }}
p.monospace_narrow_columns
    {{font-family: "Lucida Console", Monaco, monospace;
     -webkit-column-count: 8;
     -moz-column-count: 8;
     column-count: 8;
     -webkit-column-width: 250px;
     -moz-column-width: 250px;
     column-width: 250px;
     }}
</style>
</head>
<body bgcolor="{background_color}">
<h1 style="font-size:medium">
    SDS Report for {sds_root}<br>
    Latency, data percentage, number of gaps/overlaps
    (last {check_quality_days} days)<br>{time} UTC</h1>
<p class="monospace_wide_columns">
{lines_normal}
</p>
<p class="monospace_narrow_columns">
{lines_outdated}
</p>
<p class="monospace">
  <span style="background-color: {ok_color}">
      {ok_info:*<40s}</span><br>
  <span style="background-color: {data_quality_warn_color}">
      {data_quality_warn_info:*<40s}</span><br>
  <span style="background-color: {latency_warn_color}">
      {latency_warn_info:*<40s}</span><br>
  <span style="background-color: {latency_error_color}">
      {latency_error_info:*<40s}</span><br>
  <span style="background-color: {outdated_color}">
      {outdated_info:*<40s}</span><br>
</p>
  <img src="{output_basename}.png" alt="obspy-scan image broken!">
</body>
</html>
"""
HTML_LINE = (
    '<span style="background-color: {color}">'
    '{network:*<2s} {station:*<6s} {location:*<2s} {channel:*<3s} '
    '{latency_string} {percentage_string} {gap_count_string}</span><br>\n')


def _latency_to_tuple(latency):
    """
    Convert latency in seconds to tuple of (days, hours, minutes, seconds).
    """
    latency = float(latency)
    latency /= 24 * 3600
    days = int(latency)
    latency -= days
    latency *= 24
    hours = int(latency)
    latency -= hours
    latency *= 60
    minutes = int(latency)
    latency -= minutes
    seconds = latency * 60
    return (days, hours, minutes, seconds)


def _latency_info_string(latency, only_days=False, pad=True):
    """
    Format latency as a plain ASCII string.
    """
    latency_string = ''
    days, hours, minutes, seconds = _latency_to_tuple(latency)
    if days:
        if pad:
            latency_string += '{:*>4d}d '.format(days)
        else:
            latency_string += '{:d}d '.format(days)
    elif pad:
        latency_string += '*' * 6
    if not only_days:
        if hours:
            if pad:
                latency_string += '{:*>2d}h '.format(hours)
            else:
                latency_string += '{:d}h '.format(hours)
        elif pad:
            latency_string += '*' * 4
        if minutes:
            if pad:
                latency_string += '{:*>2d}m'.format(minutes)
            else:
                latency_string += '{:d}m'.format(minutes)
        elif pad:
            latency_string += '*' * 3
    return latency_string


def _latency_line_html(latency_tuple, args, color=None, only_days=False,
                       gap_info=True):
    """
    Format a single latency information tuple (net, sta, loc, cha, latency,
    percentage, gap count) as a html line.
    """
    net, sta, loc, cha, latency, percentage, gap_count = latency_tuple
    if not color:
        if (latency is None or np.isinf(latency) or
                latency > args.check_back_days * 24 * 3600):
            color = args.outdated_color
        elif latency > args.latency_error:
            color = args.latency_error_color
        elif latency > args.latency_warn:
            color = args.latency_warn_color
        elif (percentage * 1e2 < args.percentage_warn or
              gap_count > args.gaps_warn):
            color = args.data_quality_warn_color
        else:
            color = args.ok_color
    if np.isinf(latency):
        latency_string = "{:*>3d}+d".format(int(args.check_back_days))
    else:
        latency_string = _latency_info_string(latency, only_days=only_days)
    if gap_info:
        if percentage * 1e2 < args.percentage_warn:
            percentage_string = "{:*>5.1f}%".format(percentage * 1e2)
        else:
            percentage_string = '*' * 6
        if gap_count > args.gaps_warn:
            gap_count_string = "{:*>4d}#".format(gap_count)
        else:
            gap_count_string = '*' * 5
    else:
        percentage_string = ''
        gap_count_string = ''
    line = HTML_LINE.format(
        network=net, station=sta, location=loc, channel=cha, color=color,
        latency_string=latency_string, percentage_string=percentage_string,
        gap_count_string=gap_count_string).replace(
            "*", "&nbsp;")
    return line


def _format_html(args, data_normal, data_outdated):
    lines_normal = "".join(
        [_latency_line_html(data, args) for data in data_normal])
    lines_outdated = "".join(
        [_latency_line_html(data, args, only_days=True, gap_info=False)
         for data in data_outdated])
    html_legend = {}
    html_legend['latency_error_info'] = (
        "Latency > " + _latency_info_string(args.latency_error, pad=False))
    html_legend['latency_warn_info'] = (
        "Latency > " + _latency_info_string(args.latency_warn, pad=False))
    html_legend['data_quality_warn_info'] = (
        "Last {check_quality_days:d} days: < {percentage_warn}% data or "
        "> {gaps_warn:d} gaps").format(**vars(args))
    html_legend['ok_info'] = "All checks pass"
    html_legend['outdated_info'] = (
        "No data within {check_back_days:d} days").format(**vars(args))
    html_legend['output_basename'] = Path(args.output).name
    html_legend.update(vars(args))
    html = HTML_TEMPLATE.format(
        time=UTCDateTime().strftime("%c"), lines_normal=lines_normal,
        lines_outdated=lines_outdated, **html_legend)
    html = html.replace("*", "&nbsp;")
    return html


def main(argv=None):
    MatplotlibBackend.switch_backend("AGG", sloppy=False)
    parser = ArgumentParser(
        prog='obspy-sds-report', description=__doc__,
        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument(
        '-r', '--sds-root', dest='sds_root', required=True,
        help='Root folder of SDS archive.')
    parser.add_argument(
        '-o', '--output', dest='output', required=True,
        help='Full path (absolute or relative) of output files, without '
             'suffix (e.g. ``/tmp/sds_report``).')
    parser.add_argument(
        '-u', '--update', dest='update', default=False, action="store_true",
        help='Only update latency information, reuse previously computed list '
             'of streams to check and data percentage and gap count. Many '
             'other options e.g. regarding stream selection (``--id``, '
             ' ``--location``, ..) and time span of data quality checks '
             '(``--check-quality-days``) will be without effect if this '
             'option is specified. Only updating latency is significantly '
             'faster than a full analysis run, a normal use case is to do a '
             'full run once or twice per day and update latency every 5 or '
             'ten minutes. An exception is raised if an update is specified '
             'but the necessary file is not yet present.')
    parser.add_argument(
        '-l', '--location', dest='locations', action="append",
        help='Location codes to look for (e.g. ``""`` for empty location code '
             'or ``"00"``). This option can be provided multiple times and '
             'must be specified at least once for a full run (i.e. without '
             '``--update`` option). While network/station combinations are '
             'automatically discovered, only streams whose location codes are '
             'provided here will be discovered and taken into account and '
             'ultimately displayed.')
    parser.add_argument(
        '-c', '--channel', dest='channels', action="append",
        help='Channel codes to look for (e.g. specified three times with '
             '``HHZ``, ``EHZ`` and ``ELZ`` to cover all stations that serve '
             'a broad-band, short-period or low gain vertical channel). '
             'This option can be provided multiple times and must be '
             'specified at least once for a full run (i.e. without '
             '``--update`` option). Only one stream per '
             'network/station/location combination will be displayed, '
             'selected by the lowest latency.')
    parser.add_argument(
        '-i', '--id', dest='ids', action="append", default=[],
        help='SEED IDs of streams that should be included in addition to the '
             'autodiscovery of streams controlled by ``--location`` and '
             '``--channel`` options (e.g. ``IU.ANMO..LHZ``). '
             'This option can be provided multiple times.')
    parser.add_argument(
        '--skip', dest='skip', action="append", default=[],
        help='Networks or stations that should be skipped (e.g. ``IU`` or '
             '``IU.ANMO``). This option can be provided multiple times.')
    parser.add_argument(
        '-f', '--format', default="MSEED", choices=ENTRY_POINTS['waveform'],
        help='Waveform format of SDS archive. Should be "MSEED" in most '
             'cases. Use ``None`` or empty string for format autodection '
             '(slower and should not be necessary in most all cases). '
             'Warning: formats that do not support ``headonly`` '
             'option in ``read()`` operation will be significantly slower).')
    parser.add_argument(
        '--check-backwards-days', dest='check_back_days', default=30,
        type=int, help='Check for latency backwards for this many days.')
    parser.add_argument(
        '--check-quality-days', dest='check_quality_days', default=7,
        type=int, help='Calculate and plot data availability and number of '
                       'gaps for a period of this many days.')
    parser.add_argument(
        '--latency-warn', dest='latency_warn', default=3600,
        type=float, help='Latency warning threshold in seconds.')
    parser.add_argument(
        '--latency-warn-color', dest='latency_warn_color', default="#FFFF33",
        help='Latency warning threshold color (valid HTML color string).')
    parser.add_argument(
        '--latency-error', dest='latency_error', default=24 * 3600,
        type=float, help='Latency error threshold in seconds.')
    parser.add_argument(
        '--latency-error-color', dest='latency_error_color', default="#E41A1C",
        help='Latency error threshold color (valid HTML color string).')
    parser.add_argument(
        '--percentage-warn', dest='percentage_warn', default=99.5, type=float,
        help='Data availability percentage warning threshold (``0`` to '
             '``100``).')
    parser.add_argument(
        '--gaps-warn', dest='gaps_warn', default=20, type=int,
        help='Gap/overlap number warning threshold.')
    parser.add_argument(
        '--data-quality-warn-color', dest='data_quality_warn_color',
        default="#377EB8",
        help='Data quality (percentage/gap count) warning color '
             '(valid HTML color string).')
    parser.add_argument(
        '--outdated-color', dest='outdated_color', default="#808080",
        help='Color for streams that have no data in check range '
             '(valid HTML color string).')
    parser.add_argument(
        '--ok-color', dest='ok_color', default="#4DAF4A",
        help='Color for streams that pass all checks (valid HTML color '
             'string).')
    parser.add_argument(
        '--background-color', dest='background_color', default="#999999",
        help='Color for background of page (valid HTML color string).')
    parser.add_argument(
        '-V', '--version', action='version', version='%(prog)s ' + __version__)

    args = parser.parse_args(argv)

    now = UTCDateTime()
    stop_time = now - args.check_back_days * 24 * 3600
    client = Client(args.sds_root)
    dtype_streamfile = np.dtype("U10, U30, U10, U10, f8, f8, i8")
    availability_check_endtime = now - 3600
    availability_check_starttime = (
        availability_check_endtime - (args.check_quality_days * 24 * 3600))
    streams_file = args.output + ".txt"
    html_file = args.output + ".html"
    scan_file = args.output + ".png"
    if args.format.upper() == "NONE" or args.format == "":
        args.format = None

    # check whether to set up list of streams to check or use existing list
    # update list of streams once per day at nighttime
    if args.update:
        if not Path(streams_file).is_file():
            msg = ("Update flag specified, but no output of previous full run "
                   "was present in the expected location (as determined by "
                   "``--output`` flag: {})").format(streams_file)
            raise IOError(msg)
        # use existing list of streams and availability information, just
        # update latency
        nslc = np.loadtxt(streams_file, delimiter=",", dtype=dtype_streamfile)
    else:
        if not args.locations or not args.channels:
            msg = ("At least one location code ``--location`` and at least "
                   "one channel code ``--channel`` must be specified.")
            raise ObsPyException(msg)
        nsl = set()
        # get all network/station combinations in SDS archive
        for net, sta in client.get_all_stations():
            if net in args.skip or ".".join((net, sta)) in args.skip:
                continue
            # for all combinations of user specified location and channel codes
            # check if data is in SDS archive
            for loc in args.locations:
                for cha in args.channels:
                    if client.has_data(net, sta, loc, cha):
                        # for now omit channel information, we only include the
                        # channel with lowest latency later on
                        nsl.add((net, sta, loc))
                        break
        nsl = sorted(nsl)
        nslc = []
        # determine which channel to check for each network/station/location
        # combination
        for net, sta, loc in nsl:
            latency = []
            # check latency of all channels that should be checked
            for cha in args.channels:
                latency_ = client.get_latency(
                    net, sta, loc, cha, stop_time=stop_time,
                    check_has_no_data=False)
                latency.append(latency_ or np.inf)
            # only include the channel with lowest latency in our stream list
            cha = args.channels[np.argmin(latency)]
            latency = np.min(latency)
            nslc.append((net, sta, loc, cha, latency))
        for id in args.ids:
            net, sta, loc, cha = id.split(".")
            latency = client.get_latency(
                net, sta, loc, cha, stop_time=stop_time,
                check_has_no_data=False)
            latency = latency or np.inf
            nslc.append((net, sta, loc, cha, latency))
        nslc_ = []
        # request and assemble availability information.
        # this takes pretty long (on network/slow file systems),
        # so we only do it during a full run here, not during update
        for net, sta, loc, cha, latency in nslc:
            percentage, gap_count = client.get_availability_percentage(
                net, sta, loc, cha, availability_check_starttime,
                availability_check_endtime)
            nslc_.append((net, sta, loc, cha, latency, percentage, gap_count))
        nslc = nslc_
        # write stream list and availability information to file
        nslc = np.array(sorted(nslc), dtype=dtype_streamfile)
        np.savetxt(streams_file, nslc, delimiter=",",
                   fmt=["%s", "%s", "%s", "%s", "%f", "%f", "%d"])
        # generate obspy-scan image
        files = []
        seed_ids = set()
        for nslc_ in nslc:
            net, sta, loc, cha, latency, _, _ = nslc_
            if np.isinf(latency) or latency > args.check_back_days * 24 * 3600:
                continue
            seed_ids.add(".".join((net, sta, loc, cha)))
            files += client._get_filenames(
                net, sta, loc, cha, availability_check_starttime,
                availability_check_endtime)
        scan(files, format=args.format, starttime=availability_check_starttime,
             endtime=availability_check_endtime, plot=scan_file, verbose=False,
             recursive=True, ignore_links=False, seed_ids=seed_ids,
             print_gaps=False)

    # request and assemble current latency information
    data = []
    for net, sta, loc, cha, latency, percentage, gap_count in nslc:
        if args.update:
            latency = client.get_latency(
                net, sta, loc, cha, stop_time=stop_time,
                check_has_no_data=False)
            latency = latency or np.inf
        data.append((net, sta, loc, cha, latency, percentage, gap_count))

    # separate out the long dead streams
    data_normal = []
    data_outdated = []
    for data_ in data:
        latency = data_[4]
        if np.isinf(latency) or latency > args.check_back_days * 24 * 3600:
            data_outdated.append(data_)
        else:
            data_normal.append(data_)

    # write html output to file
    html = _format_html(args, data_normal, data_outdated)
    with open(html_file, "wt") as fh:
        fh.write(html)


if __name__ == "__main__":
    main()
