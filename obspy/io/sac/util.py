# -*- coding: utf-8 -*-
"""
SAC module helper functions and data.

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.utils import native_str
from future.builtins import *

import sys
import warnings

import numpy as np

from obspy import UTCDateTime
from obspy.core import Stats

from ..sac import header as HD

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
    pass

# ------------- VALIDITY CHECKS -----------------------------------------------
def is_valid_enum_str(hdr, name):
    # is this a valid string name for this hdr
    # assume that, if a value isn't in HD.ACCEP_VALS, it's not valid
    if hdr in HD.ACCEP_VALS:
        tf = name in HD.ACCEP_VALS[hdr]
    else:
        tf = False
    return tf

def is_valid_enum_int(hdr, val, allow_null=True):
    # is this a valid integer for this hdr.
    if hdr in HD.ACCEP_VALS:
        accep = [HD.ENUM_VALS[nm] for nm in HD.ACCEP_VALS[hdr]]
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
        if hdr in HD.ACCEP_VALS:
            if val in accep[hdr]:
                header[hdr] = converter[val]
            else:
                msg = 'Unrecognized enumerated value "{}" for header "{}"'
                raise ValueError(msg.format(val, hdr))

    return header


def enum_string_to_int(header):
    """Convert enumerated string values in header dictionary to int values."""
    return _convert_enum(header, converter=HD.ENUM_VALS, accep=HD.ACCEP_VALS)


def enum_int_to_string(header):
    """Convert enumerated int values in header dictionary to string values."""
    return _convert_enum(header, converter=HD.ENUM_NAMES, accep=HD.ACCEP_INT)


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
    http://docs.scipy.org/doc/numpy/user/basics.byteswapping.html

    """
    return [arr.newbyteorder('S') for arr in arrays]


def is_same_byteorder(bo1, bo2):
    """
    Deal with all the ways to compare byte order string representations.

    Parameters
    ----------
    bo1, bo2 : str
        Byte order strings can be any of:
        {'l', 'little', 'L', '<', 'b', 'big', 'B', '>', 'n', 'native', 'N', '='}

    Returns
    -------
    isSAME : bool

    """
    # TODO: extend this as is_same_byteorder(*byteorders)
    BE = ('b', 'big', '>')
    LE = ('l', 'little', '<')
    NE = ('n', 'native', '=')
    OK = BE + LE + NE

    if (bo1.lower() not in OK) or (bo2.lower() not in OK):
        raise ValueError("Unrecognized byte order string.")

    # make native decide what it is
    bo1 = sys.byteorder if bo1.lower() in NE else bo1
    bo2 = sys.byteorder if bo2.lower() in NE else bo2

    return (bo1.lower() in LE) == (bo2.lower() in LE)


def _clean_str(value):
    null_term = value.find('\x00')
    if null_term >= 0:
        value = value[:null_term]
    value = value.strip()

    return value

# TODO: do this in SACTrace?
def sac_to_obspy_header(sacheader):
    """
    Make an ObsPy Stats header dictionary from a SAC header dictionary.

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
    except (SacError, ValueError):
        #reftime = datetime.datetime(1970, 1, 1, 0, 0)
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

# TODO: do this in SACTrace? it has some reftime handling overlap with set_reftime.
def obspy_to_sac_header(stats, ignore_old_header=True):
    """
    Make a SAC header dictionary from an obspy.stats instance.

    If ignore_old_header is True, a SAC header dictionary is built from scratch.
    Otherwise, any existing stats.sac dictionary is the base.  iztype and
    the nz- reference times are kept, and the stats.starttime is turned into a
    new 'b' and 'e' relative to the existing reference time.  Other SAC header
    values are also updated, like 'npts', 'depmin/men/max', and others.

    """
    # The Plan:
    # 0. Start with the old SAC header, if present
    # 1A. Find the new b (starttime-reftime+b) and e (new_b + (new_npts-1)*new_delta)
    #     relative to existing reftime, if keep_sac_reftime is rue
    # 1B. If no old SAC header is present, make the header iztype ib and stuff
    #     it with a reftime made from the stats.starttime modulo microseconds,
    #     and push microseconds from stats.starttime into b.
    # 2. Stuff the appropriate Stats headers into the sac header

    # 0.
    header = stats.get('sac', {}).copy()
    is_old_header = bool(header)

    # overprint with current trace information
    header['npts'] = stats.npts
    header['delta'] = stats.delta

    # 1.
    # get a reference time [UTCDateTime] and a b
    starttime = stats.starttime
    if is_old_header and (not ignore_old_header):
        # A. modify old header
        #
        # just the difference btwn the old reftime and the trace starttime is b
        # b naturally includes any microseconds
        reftime = get_sac_reftime(header)
        header['b'] = starttime - reftime
    else:
        # B. make the SAC header from scratch
        #
        # make reference time and b
        # an obspy trace is naturally iztype ib (9)
        header['iztype'] = 9
        header['nzyear'] = starttime.year
        header['nzjday'] = starttime.julday
        header['nzhour'] = starttime.hour
        header['nzmin'] = starttime.minute
        header['nzsec'] = starttime.second

        # nz times don't have enough precision, so push microseconds into b
        # using integer arithmetic
        millisecond, microsecond = split_microseconds(starttime.microsecond)

        header['nzmsec'] = millisecond
        header['b'] = (microsecond * 1e-6) if microsecond else 0.0
        reftime = starttime - header['b']

        # make other SAC values from scratch
        # TODO: are all of these necessary? Especially if header is being
        #       fed to SACTrace(), which has a lot of these in it already.
        header['internal0'] = 2
        header['cmpaz'] = 0
        header['cmpinc'] = 0
        header['nvhdr'] =  6
        header['leven'] = 1
        header['lpspol'] = 1
        header['lcalda'] = 0
        header['lovrok'] = 1
        header['evla'] = 0
        header['evlo'] = 0
        header['iftype'] = 1

    # we now have correct b, npts, delta, and nz times
    header['e'] = header['b'] + (header['npts'] - 1) * header['delta']

    # 2.
    header['scale'] = stats.calib
    header['kcmpnm'] = stats.channel if stats.channel else HD.SNULL
    header['kstnm'] = stats.station if stats.station else HD.SNULL
    header['knetwk'] = stats.network if stats.network else HD.SNULL
    header['khole'] = stats.location if stats.location else HD.SNULL

    return header


def get_sac_reftime(header):
    """
    Get SAC header reference time as a UTCDateTime instance from a SAC header
    dictionary.
    """
    # NOTE: epoch seconds can be got by:
    # (reftime - datetime.datetime(1970,1,1)).total_seconds()
    # TODO: let null nz values be 0?
    try:
        yr = header['nzyear']
        if 0 <= yr <= 99:
            warnings.warn(TWO_DIGIT_YEAR_MSG)
            yr += 1900
        nzjday = header['nzjday']
        nzhour = header['nzhour']
        nzmin = header['nzmin']
        nzsec = header['nzsec']
        nzmsec = header['nzmsec']
    except KeyError as e:
        msg = "Not enough time information: {}".format(e.message)
        raise SacError(msg)

    try:
        reftime = UTCDateTime(year=yr, julday=nzjday, hour=nzhour, minute=nzmin,
                              second=nzsec, microsecond=nzmsec * 1000)
        #reftime = datetime.datetime(yr, 1, 1, nzhour, nzmin, nzsec, nzmsec * 1000) + \
        #                            datetime.timedelta(int(nzjday-1))
    except ValueError:
        # may contain -12345 null values?
        msg = "Invalid time headers."
        raise SacHeaderTimeError(msg)

    return reftime


