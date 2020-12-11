# -*- coding: utf-8 -*-
"""
Integration with ObsPy's core classes.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import collections
import io
import re
import warnings

import obspy
import obspy.core.inventory

from . import InvalidResponseError
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
    if hasattr(filename, "readline"):
        return _internal_is_resp(filename)
    try:
        with open(filename, "rb") as fh:
            return _internal_is_resp(fh)
    except (IOError, TypeError):
        return False


def _internal_is_resp(fh):
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
                if hasattr(line, "decode"):
                    try:
                        line = line.decode()
                    except UnicodeError:
                        return False
                if line.startswith("#"):
                    continue
                # do the regex check on the first non-comment line
                if re.match(r'[bB]0[1-6][0-9]F[0-9]{2} ', line):
                    return True
                return False
    except UnicodeDecodeError:
        return False


def _read_seed(filename, skip_invalid_responses=True, *args, **kwargs):
    """
    Read dataless SEED files to an ObsPy inventory object

    :param filename: File with a SEED file.
    :type filename: str or file-like object.
    :param skip_invalid_responses: If True, invalid responses will be replaced
        by None but a warning will be raised. Otherwise an exception will be
        raised. Only responses which are clearly invalid will not be read.
    """
    p = Parser(filename)

    # Parse to an inventory object.
    return _parse_to_inventory_object(
        p, skip_invalid_responses=skip_invalid_responses)


def _read_xseed(filename, skip_invalid_responses=True, *args, **kwargs):
    """
    Read XML-SEED files to an ObsPy inventory object

    :param filename: File with a XML-SEED file.
    :type filename: str or file-like object.
    :param skip_invalid_responses: If True, invalid responses will be replaced
        by None but a warning will be raised. Otherwise an exception will be
        raised. Only responses which are clearly invalid will not be read.
    """
    return _read_seed(filename=filename,
                      skip_invalid_responses=skip_invalid_responses,
                      *args, **kwargs)


def _read_resp(filename, skip_invalid_responses=True, *args, **kwargs):
    """
    Read resp files to an ObsPy inventory object

    RESP does not save vital information like the station coordinates so
    this information will be missing from the inventory objects.

    :param filename: File with a RESP file.
    :type filename: str or file-like object.
    :param skip_invalid_responses: If True, invalid responses will be replaced
        by None but a warning will be raised. Otherwise an exception will be
        raised. Only responses which are clearly invalid will not be read.
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
    return _parse_to_inventory_object(
        p, skip_invalid_responses=skip_invalid_responses)


def _parse_to_inventory_object(p, skip_invalid_responses=True):
    """
    Parses a Parser object to an obspy.core.inventory.Inventory object.

    :param p: A Parser object.
    :param skip_invalid_responses: If True, invalid responses will be replaced
        by None but a warning will be raised. Otherwise an exception will be
        raised. Only responses which are clearly invalid will not be read.
    """
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

    n = collections.OrderedDict()

    for station in p.stations:
        if station[0].id != 50:
            raise ValueError("Each station must start with blockette 50")

        # There might be multiple blockettes 50 if some information changed.
        blkts50 = [b for b in station if b.id == 50]
        station_info = collections.defaultdict(list)
        keys = ["network_identifier_code", "station_call_letters", "latitude",
                "longitude", "elevation", "site_name", "start_effective_date",
                "end_effective_date", "network_code"]
        for b in blkts50:
            for k in keys:
                if hasattr(b, k):
                    station_info[k].append(getattr(b, k))

        # For most fields we just choose the last variant.
        # A bit ugly but there is only so much one can do.
        def last_or_none(x):
            return station_info[x][-1] if x in station_info else None

        network_code = last_or_none("network_code")
        station_call_letters = last_or_none("station_call_letters")
        latitude = last_or_none("latitude")
        longitude = last_or_none("longitude")
        elevation = last_or_none("elevation")
        site_name = last_or_none("site_name")

        # handle None in mandatory elevation field with obvious bogus value
        if elevation is None:
            elevation = 123456.0

        # Take the first start-date and the last end-date.
        start_effective_date = station_info["start_effective_date"][0] \
            if "start_effective_date" in station_info else None
        end_effective_date = station_info["end_effective_date"][-1] \
            if "end_effective_date" in station_info else None
        if start_effective_date == "":
            start_effective_date = None
        if end_effective_date == "":
            end_effective_date = None

        # Get the network description if it exists.
        nic = last_or_none("network_identifier_code")
        if nic is not None:
            try:
                desc = p.resolve_abbreviation(33, nic).abbreviation_description
            except ValueError:
                pass
            else:
                if desc and network_code not in network_descriptions:
                    network_descriptions[network_code] = desc

        s = obspy.core.inventory.Station(
            code=station_call_letters,
            # Set to bogus values if not set.
            # elevation bogus is handled above
            latitude=latitude or 0.0,
            longitude=longitude or 0.0,
            elevation=elevation,
            channels=None,

            site=obspy.core.inventory.Site(name=site_name),
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
            start_date=start_effective_date,
            end_date=end_effective_date,
            restricted_status=None,
            alternate_code=None,
            historical_code=None,
            data_availability=None)

        _c = [_b for _b in station if _b.id == 51]
        # If there are comments but no comment description blockettes -
        # raise a warning but do not fail. Comments are not important enough
        # to fail parsing the file.
        if _c and 31 not in p.blockettes:
            msg = ("The file has comments but no comment descriptions "
                   "blockettes. This is an error - please fix the file! ObsPy "
                   "will still read the file as comments are not vital but "
                   "please be aware that some information might be missing "
                   "in the final file.")
            warnings.warn(msg, UserWarning)
        # Otherwise parse the comments.
        elif _c:
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
        for _b in [_i for _i in station if _i.id != 50]:
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
                try:
                    instrument_name = p.resolve_abbreviation(
                        33, sensor).abbreviation_description
                except ValueError:
                    sensor = None
                else:
                    sensor = obspy.core.inventory.Equipment(
                        type=instrument_name)

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
                sample_rate=b.sample_rate
                if hasattr(b, "sample_rate") else None,
                sample_rate_ratio_number_samples=None,
                sample_rate_ratio_number_seconds=None,
                storage_format=None,
                clock_drift_in_seconds_per_sample=None,
                calibration_units=None,
                calibration_units_description=None,
                sensor=sensor,
                pre_amplifier=None,
                data_logger=None,
                equipments=None,
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
            comments = [b_ for b_ in channel if b_.id == 59]
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

            try:
                # Epoch string used to generate nice warning and error
                # messages.
                epoch_str = "%s.%s.%s.%s [%s - %s]" % (
                    network_code, s.code, c.location_code, c.code,
                    c.start_date, c.end_date)
                resp = p.get_response_for_channel(
                    blockettes_for_channel=channel, epoch_str=epoch_str)
            except InvalidResponseError as e:
                if not skip_invalid_responses:
                    raise
                trace_id = "%s.%s.%s.%s" % (network_code, s.code,
                                            c.location_code, c.code)
                msg = ("Failed to calculate response for %s with epoch "
                       "%s - %s because: %s" % (trace_id, c.start_date,
                                                c.end_date, str(e)))
                warnings.warn(msg)
                resp = None
            c.response = resp

            s.channels.append(c)

        if network_code not in n:
            n[network_code] = []
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
