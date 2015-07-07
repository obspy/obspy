# -*- coding: utf-8 -*-
"""
Low-level array interface to the SAC file format.

Functions in this module work directly with numpy arrays that mirror the SAC
format.  The 'primitives' in this module are the float, int, and string header
arrays, the float data array, and a header dictionary. Convenience functions
are provided to convert between header arrays and more user-friendly
dictionaries.

These read/write routines are very literal; there is almost no value or type
checking, except for byteorder and header/data array length.  File- and array-
based checking routines are provided for additional checks where desired.

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.utils import native_str
from future.builtins import *

import sys
import warnings

import numpy as np

from obspy.core.compatibility import frombuffer

from .. import header as HD
from .util import SacIOError, SacInvalidContentError
from .util import is_valid_enum_int, is_same_byteorder


def init_header_arrays(*args):
    """
    Initialize arbitrary header arrays.

    Parameters
    ----------
    which : str, {'float', 'int', 'str'}
        Specify which arrays to initialize, and desired order.
        If omitted, returned arrays are ('float', 'int', 'str').

    Returns
    -------
    arrays : list of numpy.ndarrays
        The desired SAC header arrays.

    """
    if len(args) == 0:
        args = ('float', 'int', 'str')
    #http://stackoverflow.com/a/13052254/745557
    out = []
    for itype in args:
        if itype == 'float':
            # null float header array
            # XXX: why not init native byte order?
            hf = np.ndarray(70, dtype=native_str('<f4'))
            hf.fill(HD.FNULL)
            out.append(hf)
        elif itype == 'int':
            # null integer header array
            hi = np.ndarray(40, dtype=native_str('<i4'))
            hi.fill(HD.INULL)
            # set logicals to 0, not -1234whatever
            for hdr in HD.INTHDRS:
                if hdr.startswith('l'):
                    hi[HD.INTHDRS.index(hdr)] = 0
            # TODO: initialize enumerated values to something?
            # calculate distances be default
            hi[HD.INTHDRS.index('lcalda')] = 1
            out.append(hi)
        elif itype == 'str':
            # null string header array
            hs = np.ndarray(24, dtype=native_str('|S8'))
            hs.fill(HD.SNULL)
            out.append(hs)
        else:
            raise ValueError("Unrecognized header array type {}".format(itype))

    return out


def read_sac(source, headonly=False, byteorder=None, checksize=False):
    """
    Read a SAC binary file.

    Parameters
    ----------
    source : str or file-like object
        Full path string for File-like object from a SAC binary file on disk.
        If the latter, open 'rb'.
    headonly : bool
        If headonly is True, only read the header arrays not the data array.
    byteorder : str {'little', 'big'}, optional
        If omitted or None, automatic byte-order checking is done, starting with
        native order. If byteorder is specified and incorrect, a SacIOError is
        raised.
    checksize : bool, default False
        If true, check that theoretical file size from header matches disk size.

    Returns
    -------
    hf, hi, hs : numpy.ndarray
        The float, integer, and string header arrays.
    data : numpy.ndarray or None
        float32 data array. If headonly is True, data will be None.

    Raises
    ------
    ValueError
        Unrecognized byte order.
    SacIOError
        File not found, incorrect specified byteorder, theoretical file size
        doesn't match header, or header arrays are incorrect length.

    """
    # TODO: rewrite using "with" statement instead of open/close management.
    # checks: byte order, header array length, file size, npts matches data length
    try:
        f = open(source, 'rb')
        is_file_name = True
    except IOError:
        raise SacIOError("No such file: " + source)
    except TypeError:
        # source is already a file-like object
        f = source
        is_file_name = False

    isSPECIFIED = byteorder is not None
    if not isSPECIFIED:
        byteorder = sys.byteorder

    if byteorder == 'little':
        endian_str = '<'
    elif byteorder == 'big':
        endian_str = '>'
    else:
        raise ValueError("Unrecognized byteorder. Use {'little', 'big'}")

    # --------------------------------------------------------------
    # READ HEADER
    # The sac header has 70 floats, 40 integers, then 192 bytes
    #    in strings. Store them in array (and convert the char to a
    #    list). That's a total of 632 bytes.
    # --------------------------------------------------------------
    hf = frombuffer(f.read(4 * 70), dtype=native_str(endian_str + 'f4'))
    hi = frombuffer(f.read(4 * 40), dtype=native_str(endian_str + 'i4'))
    hs = frombuffer(f.read(24 * 8), dtype=native_str('|S8'))

    isVALID = is_valid_byteorder(hi)
    if not isVALID:
        if isSPECIFIED:
            # specified but not valid. you dun messed up.
            raise SacIOError("Incorrect byteorder {}".format(byteorder))
        else:
            # not valid, but not specified.
            # swap the dtype interpretation (dtype.byteorder), but keep the
            # bytes, so the arrays in memory reflect the bytes on disk
            hf = hf.newbyteorder('S')
            hi = hi.newbyteorder('S')

    # we now have correct headers, let's use their correct byte order.
    endian_str = hi.dtype.byteorder

    # check header lengths
    if len(hf) != 70 or len(hi) != 40 or len(hs) != 24:
        hf = hi = hs = None
        f.close()
        raise SacIOError("Cannot read all header values")

    npts = hi[HD.INTHDRS.index('npts')]

    # check file size
    if checksize:
        cur_pos = f.tell()
        length = f.seek(0, 2)
        f.seek(cur_pos, 0)
        if length != (632 + 4 * int(npts)):
            msg = "File-size and theoretical size are inconsistent.\n" \
                  "Check that headers are consistent with time series."
            raise SacIOError(msg)


    # --------------------------------------------------------------
    # READ DATA
    # --------------------------------------------------------------
    if headonly:
        data = None
    else:
        data = frombuffer(f.read(npts * 4), dtype=native_str(endian_str + 'f4'))

        if len(data) != npts:
            f.close()
            raise SacIOError("Cannot read all data points")

    if is_file_name:
        f.close()

    return hf, hi, hs, data


def read_sac_ascii(source, headonly=False):
    """
    Read a SAC ASCII file.

    Parameters
    ----------
    source : str for File-like object
        Full path or File-like object from a SAC ASCII file on disk.
    headonly : bool
        If headonly is True, only read the header arrays not the data array.

    Returns
    -------
    hf, hi, hs : numpy.ndarray
        The float, integer, and string header arrays.
    data : numpy.ndarray or None
        float32 data array. If headonly is True, data will be None.

    """
    # checks: ASCII-ness, header array length, npts matches data length
    try:
        fh = open(source, 'rb')
        contents = fh.read()
        is_file_name = True
    except IOError:
        raise SacIOError("No such file: " + source)
    except TypeError:
        fh = source
        is_file_name = False

    contents = [_i.rstrip(b"\n\r") for _i in contents.splitlines(True)]
    if len(contents) < 14 + 8 + 8:
        raise SacIOError("%s is not a valid SAC file:" % fh.name)

    # --------------------------------------------------------------
    # parse the header
    #
    # The sac header has 70 floats, 40 integers, then 192 bytes
    #    in strings. Store them in array (and convert the char to a
    #    list). That's a total of 632 bytes.
    # --------------------------------------------------------------
    # read in the float values
    # TODO: use native '=' dtype byteorder instead of forcing little endian?
    hf = np.array([i.split() for i in contents[:14]],
                   dtype=native_str('<f4')).ravel()
    # read in the int values
    hi = np.array([i.split() for i in contents[14: 14 + 8]],
                   dtype=native_str('<i4')).ravel()
    # reading in the string part is a bit more complicated
    # because every string field has to be 8 characters long
    # apart from the second field which is 16 characters long
    # resulting in a total length of 192 characters
    hs, _ = init_header_arrays('str')
    for i, j in enumerate(range(0, 24, 3)):
        line = contents[14 + 8 + i]
        hs[j:j + 3] = np.fromstring(line, dtype=native_str('|S8'), count=3)
    # --------------------------------------------------------------
    # read in the seismogram points
    # --------------------------------------------------------------
    if headonly:
        data = None
    else:
        data = np.array([i.split() for i in contents[30:]],
                         dtype=native_str('<f4')).ravel()

        npts = hi[HD.INTHDRS.index('npts')]
        if len(data) != npts:
            if is_file_name:
                fh.close()
            raise SacIOError("Cannot read all data points")

    if is_file_name:
        fh.close()

    return hf, hi, hs, data


def write_sac(dest, hf, hi, hs, data=None, byteorder=None):
    """
    Write the header and (optionally) data arrays to a SAC binary file.

    Parameters
    ----------
    dest : str or File-like object
        Full path or File-like object from SAC binary file on disk.
        If data is None, file mode should be 'wb+'.
    hf, hi, hs : numpy.ndarray
        The float, integer, and string header arrays.
    data : numpy.ndarray, optional
        float32 data array.  If omitted or None, it is assumed that the user
        intends to overwrite/modify only the header arrays of an existing file.
        Equivalent to "writehdr".
    byteorder : str {'little', 'big'}, optional
        Desired output byte order.  If omitted, arrays are written as they are.
        If data=None, better make sure the file you're writing to has the same
        byte order as headers you're writing.

    Notes
    -----
    A user can/should not _create_ a header-only binary file.  Use mode 'wb+'
    for data=None (headonly) writing.

    """

    # deal with file name versus File-like object, and file mode
    if data is None:
        # file exists, just modify it (don't start from scratch)
        fmode = 'rb+'
    else:
        # start from scratch
        fmode = 'wb+'

    # TODO: use "with" statements (will always closes the file object?)
    try:
        f = open(dest, fmode)
        is_file_name = True
    except IOError:
        raise SacIOError("Cannot open file: " + dest)
    except TypeError:
        f = dest 
        is_file_name = False

    if data is None and (f.mode != 'rb+'):
        msg = "File mode must be 'wb+' for data=None."
        raise ValueError(msg)

    # deal with desired byte order
    if data is None:
        assert hf.dtype.byteorder == hi.dtype.byteorder
    else:
        assert hf.dtype.byteorder == hi.dtype.byteorder == data.dtype.byteorder

    if byteorder is None:
        pass
    else:
        if not is_same_byteorder(byteorder, hf.dtype.byteorder):
            hf = hf.byteswap(True).newbyteorder(byteorder)
            hi = hf.byteswap(True).newbyteorder(byteorder)
            if data is not None:
                data = data.byteswap(True).newbyteorder(byteorder)

    # actually write everything
    try:
        f.write(hf.data)
        f.write(hi.data)
        f.write(hs.data)
        if data is not None:
            f.write(data.data)
    except Exception as e:
        if is_file_name:
            f.close()
        msg = "Cannot write SAC-buffer to file: "
        raise SacIOError(msg, f.name, e)

    if is_file_name:
        f.close()



def write_sac_ascii(dest, hf, hi, hs, data=None):
    """
    Write the header and (optionally) data arrays to a SAC ASCII file.

    Parameters
    ----------
    dest : str or File-like object
        Full path or File-like object from SAC ASCII file on disk.
    hf, hi, hs : numpy.ndarray
        The float, integer, and string header arrays.
    data : numpy.ndarray, optional
        float32 data array.  If omitted or None, it is assumed that the user
        intends to overwrite/modify only the header arrays of an existing file.
        Equivalent to "writehdr". If data is None, better make sure the header
        you're writing matches any data already in the file.

    """
    # TODO: fix prodigious use of file open/close for "with" statements.

    if data is None:
        # file exists, just modify it (don't start from scratch)
        fmode = 'r+'
    else:
        # start from scratch
        fmode = 'w+'

    try:
        f = open(dest, fmode)
        is_file_name = True
    except IOError:
        raise SacIOError("Cannot open file: " + dest)
    except TypeError:
        f = dest
        is_file_name = False

    if data is None and (f.mode != 'r+'):
        msg = "File mode must be 'r+' for data=None."
        raise ValueError(msg)

    try:
        np.savetxt(f, np.reshape(hf, (14, 5)),
                   fmt=native_str("%#15.7g%#15.7g%#15.7g%#15.7g%#15.7g"))
        np.savetxt(f, np.reshape(hi, (8, 5)),
                   fmt=native_str("%10d%10d%10d%10d%10d"))
        for i in range(0, 24, 3):
            f.write(hs[i:i + 3].data)
            f.write(b'\n')
    except:
        if is_file_name:
            f.close()
        raise SacIOError("Cannot write header values: " + f.name)

    if data is not None:
        npts = hi[9]
        if npts in (HD.INULL, 0):
            if is_file_name:
                f.close()
            return
        try:
            rows = npts // 5
            np.savetxt(f, np.reshape(data[0:5 * rows], (rows, 5)),
                       fmt=native_str("%#15.7g%#15.7g%#15.7g%#15.7g%#15.7g"))
            np.savetxt(f, data[5 * rows:], delimiter=b'\t')
        except:
            f.close()
            raise SacIOError("Cannot write trace values: " + f.name)

    if is_file_name:
        f.close()


# ---------------------- HEADER ARRAY / DICTIONARY CONVERTERS -----------------
# TODO: this functionality is basically the same as the getters and setters in
#    sac.sactrace. find a way to avoid duplication?
# TODO: put these in sac.util?
def header_arrays_to_dict(hf, hi, hs, nulls=False):
    """
    Returns
    -------
    dict
        The correctly-ordered SAC header values, as a dictionary.
    nulls : bool
        If True, return all values including nulls.

    """
    if nulls:
        items = [(key, val) for (key, val) in zip(HD.FLOATHDRS, hf)] + \
                [(key, val) for (key, val) in zip(HD.INTHDRS, hi)] + \
                [(key, val) for (key, val) in zip(HD.STRHDRS, hs)]
    else:
        items = [(key, val) for (key, val) in zip(HD.FLOATHDRS, hf) if val != HD.FNULL] + \
                [(key, val) for (key, val) in zip(HD.INTHDRS, hi) if val != HD.INULL] + \
                [(key, val) for (key, val) in zip(HD.STRHDRS, hs) if val != HD.SNULL]

    header = dict(items)

    # here, we have to append the 2nd kevnm field into the first and remove
    #   it from the dictionary.
    # XXX: kevnm may be null when kevnm2 isn't
    if 'kevnm2' in header:
        if 'kevnm' in header:
            header['kevnm'] += header.pop('kevnm2').decode()
        else:
            header['kevnm'] = header.pop('kevnm2').decode()

    return header


def dict_to_header_arrays(header=None):
    """
    Returns null hf, hi, hs arrays, optionally filled with values from a
    dictionary.  No header checking.

    """
    hf, hi, hs = init_header_arrays()

    # have to split kevnm into two fields
    # TODO: add .lower() to hdr lookups, for safety
    if header is not None:
        for hdr, value in header.iteritems():
            if hdr in HD.FLOATHDRS:
                hf[HD.FLOATHDRS.index(hdr)] = value
            elif hdr in HD.INTHDRS:
                if not isinstance(value, (np.integer, int)):
                    warnings.warn("Non-integers may be truncated.")
                    print(" {}: {}".format(hdr, value))
                hi[HD.INTHDRS.index(hdr)] = value
            elif hdr in HD.STRHDRS:
                if hdr == 'kevnm':
                    # assumes users will not include a 'kevnm2' key
                    #XXX check for empty or null value?
                    kevnm = '{:<8s}'.format(value[0:8])
                    kevnm2 = '{:<8s}'.format(value[8:16])
                    hs[1] = kevnm.encode('ascii', 'strict')
                    hs[2] = kevnm2.encode('ascii', 'strict')
                else:
                    hs[HD.STRHDRS.index(hdr)] = value.encode('ascii', 'strict')
            else:
                raise ValueError("Unrecognized header name: {}.".format(hdr))

    return hf, hi, hs


def validate_sac_content(hf, hi, hs, data, *tests):
    """
    Check validity of loaded SAC file content, such as header/data consistency.

    Parameters
    ----------
    hf, hi, hs: numpy.ndarray
        Float, int, string SAC header arrays, respectively.
    data : numpy.ndarray or None
        SAC data array.
    tests : str {'delta', 'logicals', 'data_hdrs', 'enums', 'reftime', 'reltime'}
        Perform one or more of the following validity tests:

        'delta' : Time step "delta" is positive.
        'logicals' : Logical values are 0, 1, or null
        'data_hdrs' : Length, min, mean, max of data array match header values.
        'enums' : Check validity of enumerated values.
        'reftime' : Reference time values in header are all set.
        'reltime' : Relative time values in header can be absolutely referenced.
        'all' : Do all tests.

    Raises
    ------
    SacInvalidContentError
        Any of the specified tests fail.
    ValueError
        'data_hdrs' is specified and data is None, empty array
        No tests specified.

    """
    # TODO: move this to util.py and write and use individual test functions,
    # so that all validity checks are in one place?
    ALL = ('delta', 'logicals', 'data_hdrs', 'enums', 'reftime', 'reltime')

    if 'all' in tests:
        tests = ALL

    if not tests:
        raise ValueError("No validation tests specified.")
    elif any([(itest not in ALL) for itest in tests]):
        msg = "Unrecognized validataion test specified"
        raise ValueError(msg)

    if 'delta' in tests:
        val = hf[HD.FLOATHDRS.index('delta')]
        if not (val >= 0.0):
            msg = "Header 'delta' must be >= 0."
            raise SacInvalidContentError(msg)

    if 'logicals' in tests:
        for hdr in ('leven', 'lpspol', 'lovrok', 'lcalda'):
            val = hi[HD.INTHDRS.index(hdr)]
            if val not in (0, 1, HD.INULL):
                msg = "Header '{}' must be {{{}, {}, {}}}."
                raise SacInvalidContentError(msg.format(hdr, 0, 1, HD.INULL))

    if 'data_hdrs' in tests:
        try:
            isMIN = hf[HD.FLOATHDRS.index('depmin')] == data.min()
            isMAX = hf[HD.FLOATHDRS.index('depmax')] == data.max()
            isMEN = hf[HD.FLOATHDRS.index('depmen')] == data.mean()
            if not all([isMIN, isMAX, isMEN]):
                msg = "Data headers don't match data array."
                raise SacInvalidContentError(msg)
        except (AttributeError, ValueError) as e:
            msg = "Data array is None, empty array, or non-array. " + \
                  "Cannot check data headers."
            raise ValueError(msg)

    if 'enums' in tests:
        for hdr in HD.ACCEP_VALS:
            val = hi[HD.INTHDRS.index(hdr)]
            if not is_valid_enum_int(hdr, val, allow_null=True):
                msg = "Invalid enumerated value, '{}': {}".format(hdr, val)
                raise SacInvalidContentError(msg)

    if 'reftime' in tests:
        nzyear = hi[HD.INTHDRS.index('nzyear')]
        nzjday = hi[HD.INTHDRS.index('nzjday')]
        nzhour = hi[HD.INTHDRS.index('nzhour')]
        nzmin = hi[HD.INTHDRS.index('nzmin')]
        nzsec = hi[HD.INTHDRS.index('nzsec')]
        nzmsec = hi[HD.INTHDRS.index('nzmsec')]

        # all header reference time fields are set
        if not all([val != HD.INULL for val in [nzyear, nzjday, nzhour, nzmin, nzsec, nzmsec]]):
            msg = "Null reference time values detected."
            raise SacInvalidContentError(msg)

        # reference time fields are reasonable values
        nzjday_ok = 0 <= nzjday <= 366
        nzhour_ok = 0 <= nzhour <= 24
        nzmin_ok = 0 <= nzmin <= 59
        nzsec_ok = 0 <= nzsec <= 59
        nzmsec_ok = 0 <= nzsec <= 999
        if not all([nzjday_ok, nzhour_ok, nzmin_ok, nzsec_ok, nzmsec_ok]):
            msg = "Invalid reference time values detected."
            raise SacInvalidContentError(msg)


    if 'reltime' in tests:
        # iztype is set and points to a non-null header value
        iztype_val = hi[HD.INTHDRS.index('iztype')]
        if is_valid_enum_int('iztype', iztype_val, allow_null=False):
            if iztype_val == 9:
                hdr = 'b'
            elif iztype_val == 11:
                hdr = 'o'
            elif val == 12:
                hdr = 'a'
            elif val in range(13, 23):
                hdr = 'it'+str(val-13)

            if hi[HD.FLOATHDRS.index(hdr)] == HD.INULL:
                msg = "Reference header '{}' for iztype '{}' not set."
                raise SacInvalidContentError(msg.format(hdr, iztype_val))

        else:
            msg = "Invalid iztype: {}".format(iztype_val)
            raise SacInvalidContentError(msg)

    return


def is_valid_byteorder(hi):
    nvhdr = hi[HD.INTHDRS.index('nvhdr')]
    return (0 < nvhdr < 20)
