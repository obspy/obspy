# -*- coding: utf-8 -*-
"""
Integration with ObsPy's core classes.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport

import collections
import io
import re

import obspy
import obspy.core.inventory

from .parser import Parser, is_xseed


def _is_seed(filename):
    """
    Determine if the file is (dataless) SEED file.

    No comprehensive check - it only checks the initial record sequence 
    number and the very first blockette.

    :type filename: str
    :param filename: Path/filename of a local file to be checked.
    :rtype: bool
    :returns: `True` if file seems to be a RESP file, `False` otherwise.
    """
    try:
        if hasattr(filename, "read") and hasattr(filename, "seek") and \
                hasattr(filename, "tell"):
            pos = filename.tell()
            try:
                buf = filename.read(128)
            finally:
                filename.seek(pos, 0)
        else:
            with io.open(filename, "rb") as fh:
                buf = fh.read(128)
    except IOError:
        return False

    # Minimum record size.
    if len(buf) < 128:
        return False

    if buf[:8] != b"000001V ":
        return False

    if buf[8: 8 + 3] not in [b"010", b"008", b"005"]:
        return False

    return True


def _is_xseed(filename):
    """
    Determine if the file is an XML-SEED file.

    Does not do any schema validation but only check the root tag.

    :type filename: str
    :param filename: Path/filename of a local file to be checked.
    :rtype: bool
    :returns: `True` if file seems to be a RESP file, `False` otherwise.
    """
    return is_xseed(filename)


def _is_resp(filename):
    """
    Check if a file at the specified location appears to be a RESP file.

    :type filename: str
    :param filename: Path/filename of a local file to be checked.
    :rtype: bool
    :returns: `True` if file seems to be a RESP file, `False` otherwise.
    """
    try:
        with open(filename, "rb") as fh:
            try:
                # lookup the first line that does not start with a hash sign
                while True:
                    # use splitlines to correctly detect e.g. mac formatted
                    # files on Linux
                    lines = fh.readline().splitlines()
                    # end of file without finding an appropriate line
                    if not lines:
                        return False
                    # check each line after splitting them
                    for line in lines:
                        if line.decode().startswith("#"):
                            continue
                        # do the regex check on the first non-comment line
                        if re.match(r'[bB]0[1-6][0-9]F[0-9]{2} ',
                                    line.decode()):
                            return True
                        return False
            except UnicodeDecodeError:
                return False
    except IOError:
        return False


def _read_seed(filename, *args, **kwargs):
    """
    Read dataless SEED files to an ObsPy inventory object

    :param filename: File with a SEED file.
    :type filename: str or file-like object.
    """
    p = Parser(filename)

    # Parse to an inventory object.
    return _parse_to_inventory_object(p)


def _read_xseed(filename, *args, **kwargs):
    """
    Read XML-SEED files to an ObsPy inventory object

    :param filename: File with a XML-SEED file.
    :type filename: str or file-like object.
    """
    return _read_seed(filename=filename, *args, **kwargs)


def _read_resp(filename, *args, **kwargs):
    """
    Read resp files to an ObsPy inventory object

    RESP does not save vital information like the station coordinates so 
    this information will be missing from the inventory objects.

    :param filename: File with a RESP file.
    :type filename: str or file-like object.
    """
    if hasattr(filename, "read"):
        data = filename.read()
    else:
        with io.open(filename, "rb") as fh:
            data = fh.read()
    if hasattr(data, "decode"):
        data = data.decode()
    p = Parser()
    p._parse_resp(data)

    # Parse to an inventory object.
    return _parse_to_inventory_object(p)


def _parse_to_inventory_object(p):
    # The volume time in blockette 10 will be mapped to the creation data of
    # all the ObsPy objects. If it is not given, the current time will be used.
    creation_date = None
    blkt10 = p.blockettes.get(10, None)
    if blkt10:
        creation_date = blkt10[0].volume_time
    if not creation_date:
        creation_date = obspy.UTCDateTime()

    # Dictionary to collect network descriptions. While looping through all
    # stations it will attempt to figure out the network description. If it
    # encounters multiple network descriptions it will just use the first one.
    network_descriptions = {}

    n = collections.defaultdict(list)

    for station in p.stations:
        if station[0].id != 50:
            raise ValueError("Each station must start with blockette 50")

        # Blockette 50.
        b = station[0]
        network_code = b.network_code

        # Get the network description if it exists.
        nic = getattr(b, "network_identifier_code", None)
        if nic is not None:
            desc = p.resolve_abbreviation(33, nic).abbreviation_description
            if desc and network_code not in network_descriptions:
                network_descriptions[network_code] = desc

        s = obspy.core.inventory.Station(
            code=b.station_call_letters,
            # Set to bogus values if not set.
            latitude=getattr(b, "latitude", 0.0),
            longitude=getattr(b, "longitude", 0.0),
            elevation=getattr(b, "elevation", 123456.0),
            channels=None,
            site=obspy.core.inventory.Site(
                name=getattr(b, "site_name", None)),
            vault=None,
            geology=None,
            equipments=None,
            operators=None,
            creation_date=creation_date,
            termination_date=None,
            total_number_of_channels=None,
            selected_number_of_channels=None,
            description=None,
            comments=None,
            start_date=getattr(b, "start_effective_date", None),
            end_date=getattr(b, "end_effective_date", None),
            restricted_status=None,
            alternate_code=None,
            historical_code=None,
            data_availability=None)
        if not s.start_date:
            s.start_date = None
        if not s.end_date:
            s.end_date = None

        _c = [_b for _b in station if _b.id == 51]
        if _c:
            for c in _c:
                # Parse times.
                _start = c.beginning_effective_time \
                    if hasattr(c, "beginning_effective_time") else None
                if not _start:
                    _start = None
                _end = c.end_effective_time \
                    if hasattr(c, "end_effective_time") else None
                if not _end:
                    _end = None

                comment = p.resolve_abbreviation(31, c.comment_code_key)

                s.comments.append(obspy.core.inventory.Comment(
                    value=comment.description_of_comment,
                    begin_effective_time=_start,
                    end_effective_time=_end))

        # Split the rest into channels
        channels = []
        for _b in station[1:]:
            if _b.id == 51:
                continue
            elif _b.id == 52:
                channels.append([_b])
                continue
            channels[-1].append(_b)

        for channel in channels:
            if channel[0].id != 52:
                raise ValueError("Each station must start with blockette 52")
            # Blockette 50.
            b = channel[0]

            # Get the instrument name if it exists.
            sensor = getattr(b, "instrument_identifier", None)
            if sensor is not None:
                instrument_name= p.resolve_abbreviation(
                    33, sensor).abbreviation_description
                sensor = obspy.core.inventory.Equipment(type=instrument_name)

            c = obspy.core.inventory.Channel(
                code=b.channel_identifier,
                location_code=b.location_identifier,
                # Set to bogus values if not set.
                latitude=getattr(b, "latitude", 0.0),
                longitude=getattr(b, "longitude", 0.0),
                elevation=getattr(b, "elevation", 123456.0),
                depth=getattr(b, "local_depth", 123456.0),
                azimuth=getattr(b, "azimuth", None),
                dip=getattr(b, "dip", None),
                types=None,
                external_references=None,
                sample_rate=b.sample_rate,
                sample_rate_ratio_number_samples=None,
                sample_rate_ratio_number_seconds=None,
                storage_format=None,
                clock_drift_in_seconds_per_sample=None,
                calibration_units=None,
                calibration_units_description=None,
                sensor=sensor,
                pre_amplifier=None,
                data_logger=None,
                equipment=None,
                response=None,
                description=None,
                comments=None,
                start_date=b.start_date if b.start_date else None,
                end_date=b.end_date if b.end_date else None,
                restricted_status=None,
                alternate_code=None,
                historical_code=None,
                data_availability=None)

            # Parse the comments if any.
            comments = [b for b in channel if b.id == 59]
            if comments:
                for _c in comments:
                    # Parse times.
                    _start = _c.beginning_effective_time \
                        if hasattr(_c, "beginning_effective_time") else None
                    if not _start:
                        _start = None
                    _end = _c.end_effective_time \
                        if hasattr(_c, "end_effective_time") else None
                    if not _end:
                        _end = None

                    comment = p.resolve_abbreviation(31, _c.comment_code_key)
                    c.comments.append(obspy.core.inventory.Comment(
                        value=comment.description_of_comment,
                        begin_effective_time=_start,
                        end_effective_time=_end))

            resp = p.get_response_for_channel(blockettes_for_channel=channel)
            c.response = resp

            s.channels.append(c)

        n[network_code].append(s)

    networks = []
    for code, stations in n.items():
        networks.append(obspy.core.inventory.Network(
            code=code, stations=stations,
            description=network_descriptions.get(code, None)))

    inv = obspy.core.inventory.Inventory(
        networks=networks,
        source="ObsPy's obspy.io.xseed version %s" % obspy.__version__)

    return inv


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)