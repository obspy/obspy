# -*- coding: utf-8 -*-
"""
FOCMEC file format support for ObsPy

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import re
import warnings

import numpy as np

from obspy import UTCDateTime, Catalog, __version__
from obspy.core.event import (
    Event, FocalMechanism, NodalPlanes, NodalPlane, Comment, CreationInfo)


# XXX some current PR was doing similar, should be merged to
# XXX core/utcdatetime.py eventually..
MONTHS = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12}
POLARITIES_IMPULSIVE = 'CUD<>LRFB'
POLARITIES_EMERGENT = '+-lrfb'
POLARITIES = POLARITIES_IMPULSIVE + POLARITIES_EMERGENT


def _is_focmec(filename):
    """
    Checks that a file is actually a FOCMEC output data file
    """
    try:
        with open(filename, 'rb') as fh:
            line = fh.readline()
    except Exception:
        return False
    # first line should be ASCII only, something like:
    #   Fri Sep  8 14:54:58 2017 for program Focmec
    try:
        line = line.decode('ASCII')
    except Exception:
        return False
    line = line.split()
    # program name 'focmec' at the end is written slightly differently
    # depending on how focmec was compiled, sometimes all lower case sometimes
    # capitalized..
    line[-1] = line[-1].lower()
    if line[-3:] == ['for', 'program', 'focmec']:
        return True
    return False


def _read_focmec(filename, **kwargs):
    """
    Reads a FOCMEC '.lst' or '.out' file to a
    :class:`~obspy.core.event.Catalog` object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.event.catalog.read_events()` function, call
        this instead.

    :param filename: File or file-like object in text mode.
    :rtype: :class:`~obspy.core.event.Catalog`
    """
    if not hasattr(filename, "read"):
        # Check if it exists, otherwise assume its a string.
        try:
            with open(filename, "rb") as fh:
                data = fh.read()
            data = data.decode("UTF-8")
        except Exception:
            try:
                data = filename.decode("UTF-8")
            except Exception:
                data = str(filename)
            data = data.strip()
    else:
        data = filename.read()
        if hasattr(data, "decode"):
            data = data.decode("UTF-8")

    # split lines
    lines = [line for line in data.splitlines()]

    # line 6 in 'lst' format should look like this:
    # " Statn  Azimuth    TOA   Key  Log10 Ratio  NumPol  DenTOA  Comment"
    if lines[5].split() == [
            'Statn', 'Azimuth', 'TOA', 'Key', 'Log10', 'Ratio', 'NumPol',
            'DenTOA', 'Comment']:
        event = _read_focmec_lst(lines)
    # line 16 in 'out' format should look like this:
    # "    Dip   Strike   Rake    Pol: P     SV    SH  AccR/TotR  RMS RErr..."
    # But on older program version output, it's instead line number 14, so it
    # might depend on input data (polarities and/or amplitude ratios) and thus
    # what the program outputs as info (different settings available depending
    # on input data)
    else:
        for line in lines[4:30]:
            if line.split() == [
                    'Dip', 'Strike', 'Rake', 'Pol:', 'P', 'SV', 'SH',
                    'AccR/TotR', 'RMS', 'RErr', 'AbsMaxDiff']:
                event = _read_focmec_out(lines)
                break
        else:
            msg = ("Input was not recognized as either FOCMEC 'lst' or "
                   "'out' file format. Please contact developers if input "
                   "indeed is one of these two file types.")
            raise ValueError(msg)

    cat = Catalog(events=[event])
    cat.creation_info.creation_time = UTCDateTime()
    cat.creation_info.version = "ObsPy %s" % __version__
    return cat


def _is_lst_block_start(line):
    if line.strip().startswith('+' * 20):
        return True
    return False


def _go_to_next_lst_block(lines):
    while lines and not _is_lst_block_start(lines[0]):
        lines.pop(0)
    return lines


def _match_polarity_summary_line(line):
    # there seem to be (at least) two flavors of lst files..
    # one has lines like this for polarity count summary:
    # "   8 P Pol.  7 SV Pol.  8 SH Pol.  0.2 allowed (weighted) errors"
    # .. the other one has lines like this:
    # " Polarities/Errors:  P 190/21  SV 000/00  SH 000/00"
    patterns_polarity_summary = (
        r'^ *([0-9]+) P Pol\. +([0-9]+) SV Pol\. +([0-9]+) SH Pol\. ',
        r'^ Polarities/Errors:  P +([0-9]+)/ *[0-9\.]+ +'
        r'SV +([0-9]+)/ *[0-9\.]+ +SH +([0-9]+)/ *[0-9\.]')
    for pattern_polarity_summary in patterns_polarity_summary:
        match = re.match(pattern_polarity_summary, line)
        if match:
            break
    return match


def _get_polarity_count(lines):
    for line in lines:
        match = _match_polarity_summary_line(line)
        if match:
            polarity_count = sum(int(x) for x in match.groups())
            break
    else:
        polarity_count = None
    weighted = '(weighted)' in line
    return polarity_count, weighted


def _read_focmec_lst(lines):
    """
    Read given data into an :class:`~obspy.core.event.Event` object.

    Unfortunately, "lst" is not a well defined file format but what it outputs
    depends on input data, program parameters, program version and also
    resulting focal mechanisms. But it has way more information than the "out"
    format, so it's worth the additional effort to try and parse all flavors of
    it.

    :type lines: list
    :param lines: List of decoded unicode strings with data from a FOCMEC lst
        file.
    """
    event, _ = _read_common_header(lines)
    # don't regard separator lines at end of file
    separator_indices = [i for i, line in enumerate(lines) if
                         _is_lst_block_start(line) and i < len(lines) - 1]
    if not separator_indices:
        return event
    header = lines[:separator_indices[0]]
    # get how many polarities are used
    polarity_count, _ = _get_polarity_count(header)
    # compute azimuthal gap
    for i, line in enumerate(header):
        if line.split()[:3] == ['Statn', 'Azimuth', 'TOA']:
            break
    azimuths = []
    emergent_ignored = False
    try:
        for line in header[i + 1:]:
            # some lst files have some comments on not using emergent
            # polarities right in the middle of the polarity block..
            if line.strip().lower() == 'not including emergent polarity picks':
                emergent_ignored = True
                continue
            # at the end of the polarity info block is the polarity summary
            # line..
            if _match_polarity_summary_line(line):
                break
            sta, azimuth, takeoff_angle, key = line.split()[:4]
            # these are all keys that identify a station polarity in FOCMEC,
            # because here we do not take into account amplitude ratios for the
            # azimuthal gap
            if key in POLARITIES:
                azimuths.append((float(azimuth), key))
    except IndexError:
        pass
    # if specified in output, only regard impulsive polarities
    azimuths = sorted(azimuths)
    azimuths = [azimuth_ for azimuth_, key_ in azimuths if
                not emergent_ignored or key_ in POLARITIES_IMPULSIVE]
    if polarity_count is not None and len(azimuths) != polarity_count:
        msg = ('Unexpected mismatch in number of polarity lines found ({:d}) '
               'and used polarities indicated by header ({:d})').format(
                   len(azimuths), polarity_count)
        warnings.warn(msg)
    if len(azimuths) > 1:
        # numpy diff on the azimuth list is missing to compare first and last
        # entry (going through North), so add that manually
        azimuthal_gap = np.diff(azimuths).max()
        azimuthal_gap = max(azimuthal_gap, azimuths[0] + 360 - azimuths[-1])
    else:
        azimuthal_gap = None

    event.comments.append(Comment(text='\n'.join(header)))
    blocks = []
    for i in separator_indices[::-1]:
        blocks.append(lines[i + 1:])
        lines = lines[:i]
    blocks = blocks[::-1]
    for block in blocks:
        focmec, lines = _read_focmec_lst_one_block(
            block, polarity_count)
        if focmec is None:
            continue
        focmec.azimuthal_gap = azimuthal_gap
        focmec.creation_info = CreationInfo(
            version='FOCMEC', creation_time=event.creation_info.creation_time)
        event.focal_mechanisms.append(focmec)
    return event


def _read_focmec_lst_one_block(lines, polarity_count=None):
    comment = Comment(text='\n'.join(lines))
    while lines and not lines[0].lstrip().startswith('Dip,Strike,Rake'):
        lines.pop(0)
    # the last block does not contain a focmec but only a short comment how
    # many solutions there were overall, so we hit a block that will not have
    # the above line and we exhaust the lines list
    if not lines:
        return None, []
    dip, strike, rake = [float(x) for x in lines[0].split()[1:4]]
    plane1 = NodalPlane(strike=strike, dip=dip, rake=rake)
    lines.pop(0)
    dip, strike, rake = [float(x) for x in lines[0].split()[1:4]]
    plane2 = NodalPlane(strike=strike, dip=dip, rake=rake)
    planes = NodalPlanes(nodal_plane_1=plane1, nodal_plane_2=plane2,
                         preferred_plane=1)
    focmec = FocalMechanism(nodal_planes=planes)
    focmec.comments.append(comment)
    if polarity_count is not None:
        polarity_errors = _get_polarity_error_count_lst_block(lines)
        focmec.station_polarity_count = polarity_count
        focmec.misfit = float(polarity_errors) / polarity_count
    return focmec, lines


def _get_polarity_error_count_lst_block(lines):
    # counting polarity errors is a bit tedious, as sometimes it's multiple
    # lines with station codes intersparsed with error weights and the error
    # summary lines sometimes only have the weighted sum and sometimes also
    # have the integer polarity error count, but we can't rely on it and thus
    # have to count station codes..
    pattern = re.compile(r'^ *(P|S[HV]) Polarity error at *([a-zA-Z]+.*)')
    pattern_summary = re.compile(r'^ *Total (P|S[HV]) polarity weight is')
    polarity_errors = 0
    for i, line in enumerate(lines):
        match = re.match(pattern, line)
        if match:
            polarity_errors += len(match.group(2).strip().split())
            # there's always two lines together, station codes and then weights
            # below it. there can be more than one 2-line tuple if there's many
            # polarity errors and at the end comes a summary line
            for j, line in enumerate(lines[i + 2:]):
                if re.match(pattern_summary, line) or not line.strip():
                    break
                if j % 2:
                    continue
                polarity_errors += len(line.strip().split())
    return polarity_errors


def _read_focmec_out(lines):
    """
    Read given data into an :class:`~obspy.core.event.Event` object.

    :type lines: list
    :param lines: List of decoded unicode strings with data from a FOCMEC out
        file.
    """
    event, _ = _read_common_header(lines)
    # now move to first line with a focal mechanism
    for i, line in enumerate(lines):
        if line.split()[:3] == ['Dip', 'Strike', 'Rake']:
            break
    else:
        return event
    header = lines[:i]
    polarity_count, weighted = _get_polarity_count(header)
    focmec_list_header = lines[i]
    event.comments.append(Comment(text='\n'.join(header)))
    try:
        lines = lines[i + 1:]
    except IndexError:
        return event
    for line in lines:
        # allow for empty lines (maybe they can happen at the end sometimes..)
        if not line.strip():
            continue
        comment = Comment(text='\n'.join((focmec_list_header, line)))
        items = line.split()
        dip, strike, rake = [float(x) for x in items[:3]]
        plane = NodalPlane(strike=strike, dip=dip, rake=rake)
        planes = NodalPlanes(nodal_plane_1=plane, preferred_plane=1)
        # XXX ideally should compute the auxilliary plane..
        focmec = FocalMechanism(nodal_planes=planes)
        focmec.station_polarity_count = polarity_count
        focmec.creation_info = CreationInfo(
            version='FOCMEC', creation_time=event.creation_info.creation_time)
        if not weighted:
            errors = sum([int(x) for x in items[3:6]])
            focmec.misfit = float(errors) / polarity_count
        focmec.comments.append(comment)
        event.focal_mechanisms.append(focmec)
    return event


def _read_common_header(lines):
    """
    Read given data into an :class:`~obspy.core.event.Event` object.

    Parses the first few common header lines and sets creation time and some
    other basic info.

    :type lines: list
    :param lines: List of decoded unicode strings with data from a FOCMEC out
        file.
    """
    event = Event()
    # parse time.. too much bother to mess around with switching locales, so do
    # it manually.. example:
    # "  Fri Sep  8 14:54:58 2017 for program Focmec"
    month, day, time_of_day, year = lines[0].split()[1:5]
    year = int(year)
    day = int(day)
    month = int(MONTHS[month.lower()])
    hour, minute, second = [int(x) for x in time_of_day.split(':')]
    event.creation_info = CreationInfo()
    event.creation_info.creation_time = UTCDateTime(
        year, month, day, hour, minute, second)
    # get rid of those common lines already parsed
    lines = lines[4:]
    return event, lines
