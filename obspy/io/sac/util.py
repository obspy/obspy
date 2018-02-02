# -*- coding: utf-8 -*-
"""
SAC module helper functions and data.

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import sys
import warnings

import numpy as np

from obspy import UTCDateTime
from obspy.core import Stats

from . import header as HD  # noqa

# ------------- DATA ----------------------------------------------------------
TWO_DIGIT_YEAR_MSG = ("SAC file with 2-digit year header field encountered. "
                      "This is not supported by the SAC file format standard. "
                      "Prepending '19'.")


# ------------- SAC-SPECIFIC EXCEPTIONS ---------------------------------------
class SacError(Exception):
    """
    Raised if the SAC file is corrupt or if necessary information
    in the SAC file is missing.
    """
    pass


class SacIOError(SacError, IOError):
    """
    Raised if the given SAC file can't be read.
    """
    pass


class SacInvalidContentError(SacError):
    """
    Raised if headers and/or data are not valid.
    """
    pass


class SacHeaderError(SacError):
    """
    Raised if header has issues.
    """
    pass


class SacHeaderTimeError(SacHeaderError, ValueError):
    """
    Raised if header has invalid "nz" times.
    """
    pass


# ------------- VALIDITY CHECKS -----------------------------------------------
def is_valid_enum_str(hdr, name):
    # is this a valid string name for this hdr
    # assume that, if a value isn't in HD.ACCEPTED_VALS, it's not valid
    if hdr in HD.ACCEPTED_VALS:
        tf = name in HD.ACCEPTED_VALS[hdr]
    else:
        tf = False
    return tf


def is_valid_enum_int(hdr, val, allow_null=True):
    # is this a valid integer for this hdr.
    if hdr in HD.ACCEPTED_VALS:
        accep = [HD.ENUM_VALS[nm] for nm in HD.ACCEPTED_VALS[hdr]]
        if allow_null:
            accep += [HD.INULL]
        tf = val in accep
    else:
        tf = False
    return tf


# ------------- GENERAL -------------------------------------------------------
def _convert_enum(header, converter, accep):
    # header : dict, SAC header
    # converter : dict, {source value: target value}
    # accep : dict, {header name: acceptable value list}

    # TODO: use functools.partial/wraps?
    for hdr, val in header.items():
        if hdr in HD.ACCEPTED_VALS:
            if val in accep[hdr]:
                header[hdr] = converter[val]
            else:
                msg = 'Unrecognized enumerated value "{}" for header "{}"'
                raise ValueError(msg.format(val, hdr))

    return header


def enum_string_to_int(header):
    """Convert enumerated string values in header dictionary to int values."""
    out = _convert_enum(header, converter=HD.ENUM_VALS, accep=HD.ACCEPTED_VALS)
    return out


def enum_int_to_string(header):
    """Convert enumerated int values in header dictionary to string values."""
    out = _convert_enum(header, converter=HD.ENUM_NAMES, accep=HD.ACCEPTED_INT)
    return out


def byteswap(*arrays):
    """
    Swapping of bytes for provided arrays.

    Notes
    -----
    arr.newbyteorder('S') swaps dtype interpretation, but not bytes in memory
    arr.byteswap() swaps bytes in memory, but not dtype interpretation
    arr.byteswap(True).newbyteorder('S') completely swaps both

    References
    ----------
    https://docs.scipy.org/doc/numpy/user/basics.byteswapping.html

    """
    return [arr.newbyteorder('S') for arr in arrays]


def is_same_byteorder(bo1, bo2):
    """
    Deal with all the ways to compare byte order string representations.

    :param bo1: Byte order string. Can be one of {'l', 'little', 'L', '<',
        'b', 'big', 'B', '>', 'n', 'native','N', '='}
    :type bo1: str
    :param bo2: Byte order string. Can be one of {'l', 'little', 'L', '<',
        'b', 'big', 'B', '>', 'n', 'native','N', '='}
    :type bo1: str

    :rtype: bool
    :return: True of same byte order.

    """
    # TODO: extend this as is_same_byteorder(*byteorders) using itertools
    be = ('b', 'big', '>')
    le = ('l', 'little', '<')
    ne = ('n', 'native', '=')
    ok = be + le + ne

    if (bo1.lower() not in ok) or (bo2.lower() not in ok):
        raise ValueError("Unrecognized byte order string.")

    # make native decide what it is
    bo1 = sys.byteorder if bo1.lower() in ne else bo1
    bo2 = sys.byteorder if bo2.lower() in ne else bo2

    return (bo1.lower() in le) == (bo2.lower() in le)


def _clean_str(value, strip_whitespace=True):
    """
    Remove null values and whitespace, return a str

    This fn is used in two places: in SACTrace.read, to sanitize strings for
    SACTrace, and in sac_to_obspy_header, to sanitize strings for making a
    Trace that the user may have manually added.
    """
    null_term = value.find('\x00')
    if null_term >= 0:
        value = value[:null_term] + " " * len(value[null_term:])

    if strip_whitespace:
        value = value.strip()

    return value


# TODO: do this in SACTrace?
def sac_to_obspy_header(sacheader):
    """
    Make an ObsPy Stats header dictionary from a SAC header dictionary.

    :param sacheader: SAC header dictionary.
    :type sacheader: dict

    :rtype: :class:`~obspy.core.Stats`
    :return: Filled ObsPy Stats header.

    """

    # 1. get required sac header values
    try:
        npts = sacheader['npts']
        delta = sacheader['delta']
    except KeyError:
        msg = "Incomplete SAC header information to build an ObsPy header."
        raise KeyError(msg)

    assert npts != HD.INULL
    assert delta != HD.FNULL
    #
    # 2. get time
    try:
        reftime = get_sac_reftime(sacheader)
    except (SacError, ValueError, TypeError):
        # ObsPy doesn't require a valid reftime
        reftime = UTCDateTime(0.0)

    b = sacheader.get('b', HD.FNULL)
    #
    # 3. get optional sac header values
    calib = sacheader.get('scale', HD.FNULL)
    kcmpnm = sacheader.get('kcmpnm', HD.SNULL)
    kstnm = sacheader.get('kstnm', HD.SNULL)
    knetwk = sacheader.get('knetwk', HD.SNULL)
    khole = sacheader.get('khole', HD.SNULL)
    #
    # 4. deal with null values
    b = b if (b != HD.FNULL) else 0.0
    calib = calib if (calib != HD.FNULL) else 1.0
    kcmpnm = kcmpnm if (kcmpnm != HD.SNULL) else ''
    kstnm = kstnm if (kstnm != HD.SNULL) else ''
    knetwk = knetwk if (knetwk != HD.SNULL) else ''
    khole = khole if (khole != HD.SNULL) else ''
    #
    # 5. transform to obspy values
    # nothing is null
    stats = {}
    stats['npts'] = npts
    stats['sampling_rate'] = np.float32(1.) / np.float32(delta)
    stats['network'] = _clean_str(knetwk)
    stats['station'] = _clean_str(kstnm)
    stats['channel'] = _clean_str(kcmpnm)
    stats['location'] = _clean_str(khole)
    stats['calib'] = calib

    # store _all_ provided SAC header values
    stats['sac'] = sacheader.copy()

    # get first sample absolute time as UTCDateTime
    # always add the begin time (if it's defined) to get the given
    # SAC reference time, no matter which iztype is given
    # b may be non-zero, even for iztype 'ib', especially if it was used to
    #   store microseconds from obspy_to_sac_header
    stats['starttime'] = UTCDateTime(reftime) + b

    return Stats(stats)


def split_microseconds(microseconds):
    # Returns milliseconds and remainder microseconds
    milliseconds = microseconds // 1000
    microseconds = (microseconds - milliseconds * 1000)

    return milliseconds, microseconds


def utcdatetime_to_sac_nztimes(utcdt):
    # Returns a dict of integer nz-times and remainder microseconds
    nztimes = {}
    nztimes['nzyear'] = utcdt.year
    nztimes['nzjday'] = utcdt.julday
    nztimes['nzhour'] = utcdt.hour
    nztimes['nzmin'] = utcdt.minute
    nztimes['nzsec'] = utcdt.second
    # nz times don't have enough precision, so push microseconds into b,
    # using integer arithmetic
    millisecond, microsecond = split_microseconds(utcdt.microsecond)
    nztimes['nzmsec'] = millisecond

    return nztimes, microsecond


def obspy_to_sac_header(stats, keep_sac_header=True):
    """
    Merge a primary with a secondary header, reconciling some differences.

    :param stats: Filled ObsPy Stats header
    :type stats: dict or :class:`~obspy.core.Stats`
    :param keep_sac_header: If keep_sac_header is True, old stats.sac
        header values are kept, and a minimal set of values are updated from
        the stats dictionary according to these guidelines:
        * npts, delta always come from stats
        * If a valid old reftime is found, the new b and e will be made
          and properly referenced to it. All other old SAC headers are simply
          carried along.
        * If the old SAC reftime is invalid and relative time headers are set,
          a SacHeaderError exception will be raised.
        * If the old SAC reftime is invalid, no relative time headers are set,
          and "b" is set, "e" is updated from stats and other old SAC headers
          are carried along.
        * If the old SAC reftime is invalid, no relative time headers are set,
          and "b" is not set, the reftime will be set from stats.starttime
          (with micro/milliseconds precision adjustments) and "b" and "e" are
          set accordingly.
        * If 'kstnm', 'knetwk', 'kcmpnm', or 'khole' are not set or differ
          from Stats values 'station', 'network', 'channel', or 'location',
          they are taken from the Stats values.
        If keep_sac_header is False, a new SAC header is constructed from only
        information found in the Stats dictionary, with some other default
        values introduced.  It will be an iztype 9 ("ib") file, with small
        reference time adjustments for micro/milliseconds precision issues.
        SAC headers nvhdr, level, lovrok, and iftype are always produced.
    :type keep_sac_header: bool
    :rtype merged: dict
    :return: SAC header

    """
    header = {}
    oldsac = stats.get('sac', {})

    if keep_sac_header and oldsac:
        header.update(oldsac)

        try:
            reftime = get_sac_reftime(header)
        except SacHeaderTimeError:
            reftime = None

        relhdrs = [hdr for hdr in HD.RELHDRS
                   if header.get(hdr) not in (None, HD.FNULL)]

        if reftime:
            # Set current 'b' relative to the old reftime.
            b = stats['starttime'] - reftime
        else:
            # Invalid reference time. Relative times like 'b' cannot be
            # unambiguously referenced to stats.starttime.
            if 'b' in relhdrs:
                # Assume no trimming/expanding of the Trace occurred relative
                # to the old 'b', and just use the old 'b' value.
                b = header['b']
            else:
                # Assume it's an iztype=ib (9) type file. Also set iztype?
                b = 0

            # Set the stats.starttime as the reftime and set 'b' and 'e'.
            # ObsPy issue 1204
            reftime = stats['starttime'] - b
            nztimes, microsecond = utcdatetime_to_sac_nztimes(reftime)
            header.update(nztimes)
            b += (microsecond * 1e-6)

        header['b'] = b
        header['e'] = b + (stats['endtime'] - stats['starttime'])

        # Merge some values from stats if they're missing in the SAC header
        # ObsPy issues 1204, 1457
        # XXX: If Stats values are empty/"" and SAC header values are real,
        #   this will replace the real SAC values with SAC null values.
        for sachdr, statshdr in [('kstnm', 'station'), ('knetwk', 'network'),
                                 ('kcmpnm', 'channel'), ('khole', 'location')]:
            if (header.get(sachdr) in (None, HD.SNULL)) or \
               (header.get(sachdr).strip() != stats[statshdr]):
                header[sachdr] = stats[statshdr] or HD.SNULL
    else:
        # SAC header from Stats only.

        # Here, set headers from Stats that would otherwise depend on the old
        # SAC header
        header['iztype'] = 9
        starttime = stats['starttime']
        # nz times don't have enough precision, so push microseconds into b,
        # using integer arithmetic
        nztimes, microsecond = utcdatetime_to_sac_nztimes(starttime)
        header.update(nztimes)

        header['b'] = microsecond * 1e-6

        # we now have correct b, npts, delta, and nz times
        header['e'] = header['b'] + (stats['npts'] - 1) * stats['delta']

        header['scale'] = stats.get('calib', HD.FNULL)

        # NOTE: overwrites existing SAC headers
        # nulls for these are '', which stats.get(hdr, HD.SNULL) won't catch
        header['kcmpnm'] = stats['channel'] if stats['channel'] else HD.SNULL
        header['kstnm'] = stats['station'] if stats['station'] else HD.SNULL
        header['knetwk'] = stats['network'] if stats['network'] else HD.SNULL
        header['khole'] = stats['location'] if stats['location'] else HD.SNULL

        header['lpspol'] = True
        header['lcalda'] = False

    # ObsPy issue 1204
    header['nvhdr'] = 6
    header['leven'] = 1
    header['lovrok'] = 1
    header['iftype'] = 1

    # ObsPy issue #1317
    header['npts'] = stats['npts']
    header['delta'] = stats['delta']

    return header


def get_sac_reftime(header):
    """
    Get SAC header reference time as a UTCDateTime instance from a SAC header
    dictionary.

    Builds the reference time from SAC "nz" time fields. Raises
    :class:`SacHeaderTimeError` if any time fields are null.

    :param header: SAC header
    :type header: dict

    :rtype: :class:`~obspy.core.UTCDateTime`
    :returns: SAC reference time.

    """
    # NOTE: epoch seconds can be got by:
    # (reftime - datetime.datetime(1970,1,1)).total_seconds()
    try:
        yr = header['nzyear']
        nzjday = header['nzjday']
        nzhour = header['nzhour']
        nzmin = header['nzmin']
        nzsec = header['nzsec']
        nzmsec = header['nzmsec']
    except KeyError as e:
        # header doesn't have all the keys
        msg = "Not enough time information: {}".format(e)
        raise SacHeaderTimeError(msg)

    if 0 <= yr <= 99:
        warnings.warn(TWO_DIGIT_YEAR_MSG)
        yr += 1900

    try:
        reftime = UTCDateTime(year=yr, julday=nzjday, hour=nzhour,
                              minute=nzmin, second=nzsec,
                              microsecond=nzmsec * 1000)
    except (ValueError, TypeError):
        msg = "Invalid time headers. May contain null values."
        raise SacHeaderTimeError(msg)

    return reftime
