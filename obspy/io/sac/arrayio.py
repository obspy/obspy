# -*- coding: utf-8 -*-
"""
Low-level array interface to the SAC file format.

Functions in this module work directly with NumPy arrays that mirror the SAC
format.  The 'primitives' in this module are the float, int, and string header
arrays, the float data array, and a header dictionary. Convenience functions
are provided to convert between header arrays and more user-friendly
dictionaries.

These read/write routines are very literal; there is almost no value or type
checking, except for byteorder and header/data array length.  File- and array-
based checking routines are provided for additional checks where desired.

"""
import os
import sys
import warnings

import numpy as np

from obspy.core.compatibility import from_buffer
from obspy import UTCDateTime

from . import header as HD  # noqa
from .util import SacIOError, SacInvalidContentError
from .util import is_valid_enum_int


def init_header_arrays(arrays=('float', 'int', 'str'), byteorder='='):
    """
    Initialize arbitrary header arrays.

    :param arrays: Specify which arrays to initialize and the desired order.
        If omitted, returned arrays are ('float', 'int', 'str'), in that order.
    :type arrays: tuple(float, int, str)
    :param byteorder: Desired byte order of initialized arrays
        (little, native, big) as {'<', '=', '>'}.
    :type byteorder: str

    :rtype: list(:class:`~numpy.ndarray`)
    :returns: The desired SAC header arrays.

    """
    out = []
    for itype in arrays:
        if itype == 'float':
            # null float header array
            hf = np.empty(70, dtype=byteorder + 'f4')
            hf.fill(HD.FNULL)
            out.append(hf)
        elif itype == 'int':
            # null integer header array
            hi = np.empty(40, dtype=byteorder + 'i4')
            hi.fill(HD.INULL)
            # set logicals to 0, not -1234whatever
            for i, hdr in enumerate(HD.INTHDRS):
                if hdr.startswith('l'):
                    hi[i] = 0
            # TODO: make an init_header_array_values function that sets sane
            #   initial values, including lcalda, nvhdr, leven, etc..
            # calculate distances by default
            hi[HD.INTHDRS.index('lcalda')] = 1
            out.append(hi)
        elif itype == 'str':
            # null string header array
            hs = np.empty(24, dtype='|S8')
            hs.fill(HD.SNULL)
            out.append(hs)
        else:
            raise ValueError("Unrecognized header array type {}".format(itype))

    return out


def read_sac(source, headonly=False, byteorder=None, checksize=False):
    """
    Read a SAC binary file.

    :param source: Full path string for File-like object from a SAC binary file
        on disk.  If it is an open File object, open 'rb'.
    :type source: str or file
    :param headonly: If headonly is True, only read the header arrays not the
        data array.
    :type headonly: bool
    :param byteorder: If omitted or None, automatic byte-order checking is
        done, starting with native order. If byteorder is specified,
        {'little', 'big'} and incorrect, a SacIOError is raised.
    :type byteorder: str, optional
    :param checksize: If True, check that the theoretical file size from the
        header matches the size on disk.
    :type checksize: bool

    :return: The float, integer, and string header arrays, and data array,
        in that order. Data array will be None if headonly is True.
    :rtype: tuple(:class:`numpy.ndarray`)

    :raises: :class:`ValueError` if unrecognized byte order.  :class:`IOError`
        if file not found, incorrect specified byteorder, theoretical file size
        doesn't match header, or header arrays are incorrect length.

    """
    # TODO: rewrite using "with" statement instead of open/close management.
    # check byte order, header array length, file size, npts == data length
    try:
        f = open(source, 'rb')
        is_file_name = True
    except TypeError:
        # source is already a file-like object
        f = source
        is_file_name = False

    is_byteorder_specified = byteorder is not None
    if not is_byteorder_specified:
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
    hf = from_buffer(f.read(4 * 70), dtype=endian_str + 'f4')
    hi = from_buffer(f.read(4 * 40), dtype=endian_str + 'i4')
    hs = from_buffer(f.read(24 * 8), dtype='|S8')

    if not is_valid_byteorder(hi):
        if is_byteorder_specified:
            if is_file_name:
                f.close()
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
        if is_file_name:
            f.close()
        raise SacIOError("Cannot read all header values")

    npts = hi[HD.INTHDRS.index('npts')]

    # check file size
    if checksize:
        cur_pos = f.tell()
        f.seek(0, os.SEEK_END)
        length = f.tell()
        f.seek(cur_pos, os.SEEK_SET)
        th_length = (632 + 4 * int(npts))
        if length != th_length:
            if is_file_name:
                f.close()
            msg = "Actual and theoretical file size are inconsistent.\n" \
                  "Actual/Theoretical: {}/{}\n" \
                  "Check that headers are consistent with time series."
            raise SacIOError(msg.format(length, th_length))

    # --------------------------------------------------------------
    # READ DATA
    # --------------------------------------------------------------
    if headonly:
        data = None
    else:
        data = from_buffer(f.read(int(npts) * 4),
                           dtype=endian_str + 'f4')

        if len(data) != npts:
            if is_file_name:
                f.close()
            raise SacIOError("Cannot read all data points")

    if is_file_name:
        f.close()

    return hf, hi, hs, data


def read_sac_ascii(source, headonly=False):
    """
    Read a SAC ASCII/Alphanumeric file.

    :param source: Full path or File-like object from a SAC ASCII file on disk.
    :type source: str or file
    :param headonly: If headonly is True, return the header arrays not the
        data array.  Note, the entire file is still read in if headonly=True.
    :type headonly: bool

    :return: The float, integer, and string header arrays, and data array,
        in that order. Data array will be None if headonly is True.
    :rtype: :class:`numpy.ndarray`

    """
    # TODO: make headonly=True only read part of the file, not all of it.
    # checks: ASCII-ness, header array length, npts matches data length
    try:
        fh = open(source, 'rb')
        is_file_name = True
    except TypeError:
        fh = source
        is_file_name = False
    except IOError:
        raise SacIOError("No such file: " + source)
    finally:
        contents = fh.read()
        if is_file_name:
            fh.close()

    contents = [_i.rstrip(b"\n\r") for _i in contents.splitlines()]
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
                  dtype='<f4').ravel()
    # read in the int values
    hi = np.array([i.split() for i in contents[14: 14 + 8]],
                  dtype='<i4').ravel()
    # reading in the string part is a bit more complicated
    # because every string field has to be 8 characters long
    # apart from the second field which is 16 characters long
    # resulting in a total length of 192 characters
    hs, = init_header_arrays(arrays=('str',))
    for i, j in enumerate(range(0, 24, 3)):
        line = contents[14 + 8 + i]
        hs[j:j + 3] = from_buffer(line[:24], dtype='|S8')
    # --------------------------------------------------------------
    # read in the seismogram points
    # --------------------------------------------------------------
    if headonly:
        data = None
    else:
        data = np.array([i.split() for i in contents[30:]],
                        dtype='<f4').ravel()

        npts = hi[HD.INTHDRS.index('npts')]
        if len(data) != npts:
            raise SacIOError("Cannot read all data points")

    return hf, hi, hs, data


def write_sac(dest, hf, hi, hs, data=None, byteorder=None):
    """
    Write the header and (optionally) data arrays to a SAC binary file.

    :param dest: Full path or File-like object to SAC binary file on disk.
    :type dest: str or file
    :param hf: SAC float header array
    :type hf: :class:`numpy.ndarray` of floats
    :param hi: SAC int header array
    :type hi: :class:`numpy.ndarray` of ints
    :param hs: SAC string header array
    :type hs: :class:`numpy.ndarray` of str
    :param data: SAC data array, optional.  If omitted or None, it is assumed
        that the user intends to overwrite/modify only the header arrays of an
        existing file.  Equivalent to SAC's "writehdr".
    :type data: :class:`numpy.ndarray` of float32
    :param byteorder: Desired output byte order {'little', 'big'}.  If omitted,
        arrays are written as they are.  If data=None, better make sure the
        file you're writing to has the same byte order as headers you're
        writing.
    :type byteorder: str, optional

    :return: None

    .. rubric:: Notes

    A user can/should not _create_ a header-only binary file.  Use mode 'wb+'
    for data=None (headonly) writing to just write over the header part of an
    existing binary file with data in it.

    """
    # deal with file name versus File-like object, and file mode
    # file open modes in Python: http://stackoverflow.com/a/23566951/745557
    if data is None:
        # file exists, just modify it (don't start from scratch)
        fmode = 'rb+'
    else:
        # start from scratch
        fmode = 'wb+'

    if byteorder:
        if byteorder == 'little':
            endian_str = '<'
        elif byteorder == 'big':
            endian_str = '>'
        else:
            raise ValueError("Unrecognized byteorder. Use {'little', 'big'}")

        hf = hf.astype(endian_str + 'f4')
        hi = hi.astype(endian_str + 'i4')
        if data is not None:
            data = data.astype(endian_str + 'f4')

    # TODO: make sure all arrays have same byte order

    # TODO: use "with" statements (will always closes the file object?)
    try:
        f = open(dest, fmode)
        is_file_name = True
    except TypeError:
        f = dest
        is_file_name = False
    except IOError:
        raise SacIOError("Cannot open file: " + dest)

    # TODO: make sure all data have the same/desired byte order

    # actually write everything
    try:
        f.write(memoryview(hf))
        f.write(memoryview(hi))
        f.write(memoryview(hs))
        if data is not None:
            # TODO: this long way of writing it is to make sure that
            # 'f8' data, for example, is correctly cast as 'f4'
            f.write(memoryview(data.astype(data.dtype.byteorder + 'f4')))
    except Exception as e:
        if is_file_name:
            f.close()
        msg = "Cannot write SAC-buffer to file: "
        if hasattr(f, "name"):
            name = f.name
        else:
            name = "Unknown file name (file-like object?)"
        raise SacIOError(msg, name, e)

    if is_file_name:
        f.close()


def write_sac_ascii(dest, hf, hi, hs, data=None):
    """
    Write the header and (optionally) data arrays to a SAC ASCII file.

    :param dest: Full path or File-like object from SAC ASCII file on disk.
    :type dest: str or file
    :param hf: SAC float header array.
    :type hf: :class:`numpy.ndarray` of floats
    :param hi: SAC int header array.
    :type hi: :class:`numpy.ndarray` of ints
    :param hs: SAC string header array.
    :type hs: :class:`numpy.ndarray` of strings
    :param data: SAC data array.  If omitted or None, it is assumed that the
        user intends to overwrite/modify only the header arrays of an existing
        file.  Equivalent to "writehdr". If data is None, better make sure the
        header you're writing matches any data already in the file.
    :type data: :class:`numpy.ndarray` of float32 or None

    """
    # TODO: fix prodigious use of file open/close for "with" statements.

    # file open modes in Python: http://stackoverflow.com/a/23566951/745557
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

    try:
        np.savetxt(f, np.reshape(hf, (14, 5)), fmt="%#15.7g",
                   delimiter='')
        np.savetxt(f, np.reshape(hi, (8, 5)), fmt="%10d",
                   delimiter='')
        np.savetxt(f, np.reshape(hs, (8, 3)).astype('|U8'),
                   fmt='%-8s', delimiter='')
    except Exception:
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
                       fmt="%#15.7g", delimiter='')
            np.savetxt(f, data[5 * rows:], delimiter=b'\t')
        except Exception:
            if is_file_name:
                f.close()
            raise SacIOError("Cannot write trace values: " + f.name)

    if is_file_name:
        f.close()


# ---------------------- HEADER ARRAY / DICTIONARY CONVERTERS -----------------
# TODO: this functionality is basically the same as the getters and setters in
#    sac.sactrace. find a way to avoid duplication?
# TODO: put these in sac.util?
def header_arrays_to_dict(hf, hi, hs, nulls=False, encoding='ASCII'):
    """
    Convert SAC header arrays to a more user-friendly dict.

    :param hf: SAC float header array.
    :type hf: :class:`numpy.ndarray` of floats
    :param hi: SAC int header array.
    :type hi: :class:`numpy.ndarray` of ints
    :param hs: SAC string header array.
    :type hs: :class:`numpy.ndarray` of strings
    :param nulls: If True, return all header values, including nulls, else
        omit them.
    :type nulls: bool
    :param encoding: Encoding string that passes the user specified
        encoding scheme.
    :type nulls: str

    :return: SAC header dictionary
    :rtype: dict

    """
    if nulls:
        items = list(zip(HD.FLOATHDRS, hf)) + \
            list(zip(HD.INTHDRS, hi)) + \
            [(key, val.decode()) for (key, val) in zip(HD.STRHDRS, hs)]
    else:
        # more readable
        items = [(key, val) for (key, val) in zip(HD.FLOATHDRS, hf)
                 if val != HD.FNULL] + \
                [(key, val) for (key, val) in zip(HD.INTHDRS, hi)
                 if val != HD.INULL] + \
                [(key, val.decode(encoding).strip()) for (key, val)
                 in zip(HD.STRHDRS, hs) if val.decode(encoding) != HD.SNULL]

    header = dict(items)

    # here, we have to append the 2nd kevnm field into the first and remove
    #   it from the dictionary.
    # XXX: kevnm may be null when kevnm2 isn't
    if 'kevnm2' in header:
        if 'kevnm' in header:
            header['kevnm'] = header['kevnm']
            header['kevnm'] += header.pop('kevnm2')
        else:
            header['kevnm'] = header.pop('kevnm2')

    return header


def dict_to_header_arrays(header=None, byteorder='='):
    """
    Returns null hf, hi, hs arrays, optionally filled with values from a
    dictionary.

    No header checking.

    :param header: SAC header dictionary.
    :type header: dict
    :param byteorder: Desired byte order of initialized arrays (little, native,
        big, as  {'<', '=', '>'}).
    :type byteorder: str

    :return: The float, integer, and string header arrays, in that order.
    :rtype: tuple(:class:`numpy.ndarray`)

    """
    hf, hi, hs = init_header_arrays(byteorder=byteorder)

    # have to split kevnm into two fields
    # TODO: add .lower() to hdr lookups, for safety
    if header is not None:
        for hdr, value in header.items():
            if hdr in HD.FLOATHDRS:
                hf[HD.FLOATHDRS.index(hdr)] = value
            elif hdr in HD.INTHDRS:
                if not isinstance(value, (np.integer, int)):
                    msg = "Non-integers may be truncated: {} = {}"
                    warnings.warn(msg.format(hdr, value))
                hi[HD.INTHDRS.index(hdr)] = value
            elif hdr in HD.STRHDRS:
                if hdr == 'kevnm':
                    # assumes users will not include a 'kevnm2' key
                    # XXX check for empty or null value?
                    kevnm = '{:<8s}'.format(value[0:8])
                    kevnm2 = '{:<8s}'.format(value[8:16])
                    hs[1] = kevnm.encode('ascii', 'strict')
                    hs[2] = kevnm2.encode('ascii', 'strict')
                else:
                    # TODO: why was encoding done?
                    # hs[HD.STRHDRS.index(hdr)] = value.encode('ascii',
                    #                                          'strict')
                    hs[HD.STRHDRS.index(hdr)] = value.ljust(8)
            else:
                msg = "Unrecognized header name: {}. Ignored.".format(hdr)
                warnings.warn(msg)

    return hf, hi, hs


def validate_sac_content(hf, hi, hs, data, *tests):
    """
    Check validity of loaded SAC file content, such as header/data consistency.

    :param hf: SAC float header array
    :type hf: :class:`numpy.ndarray` of floats
    :param hi: SAC int header array
    :type hi: :class:`numpy.ndarray` of ints
    :param hs: SAC string header array
    :type hs: :class:`numpy.ndarray` of str
    :param data: SAC data array
    :type data: :class:`numpy.ndarray` of float32
    :param tests: One or more of the following validity tests:
        'delta' : Time step "delta" is positive.
        'logicals' : Logical values are 0, 1, or null
        'data_hdrs' : Length, min, mean, max of data array match header values.
        'enums' : Check validity of enumerated values.
        'reftime' : Reference time values in header are all set.
        'reltime' : Relative time values in header are absolutely referenced.
        'all' : Do all tests.
    :type tests: str

    :raises: :class:`obspy.io.sac.util.SacInvalidContentError` if any of the
        specified tests fail. :class:`ValueError` if 'data_hdrs' is specified
        and data is None, empty array, or no tests specified.

    """
    # TODO: move this to util.py and write and use individual test functions,
    # so that all validity checks are in one place?
    _all = ('delta', 'logicals', 'data_hdrs', 'enums', 'reftime', 'reltime')

    if 'all' in tests:
        tests = _all

    if not tests:
        raise ValueError("No validation tests specified.")
    elif any([(itest not in _all) for itest in tests]):
        msg = "Unrecognized validataion test specified"
        raise ValueError(msg)

    if 'delta' in tests:
        dval = hf[HD.FLOATHDRS.index('delta')]
        if not (dval >= 0.0):
            msg = "Header 'delta' must be >= 0."
            raise SacInvalidContentError(msg)

    if 'logicals' in tests:
        for hdr in ('leven', 'lpspol', 'lovrok', 'lcalda'):
            lval = hi[HD.INTHDRS.index(hdr)]
            if lval not in (0, 1, HD.INULL):
                msg = "Header '{}' must be {{{}, {}, {}}}."
                raise SacInvalidContentError(msg.format(hdr, 0, 1, HD.INULL))

    if 'data_hdrs' in tests:
        try:
            is_min = np.allclose(hf[HD.FLOATHDRS.index('depmin')], data.min())
            is_max = np.allclose(hf[HD.FLOATHDRS.index('depmax')], data.max())
            is_mean = np.allclose(hf[HD.FLOATHDRS.index('depmen')],
                                  data.mean())
            if not all([is_min, is_max, is_mean]):
                msg = "Data headers don't match data array."
                raise SacInvalidContentError(msg)
        except (AttributeError, ValueError):
            msg = "Data array is None, empty array, or non-array. " + \
                  "Cannot check data headers."
            raise SacInvalidContentError(msg)

    if 'enums' in tests:
        for hdr in HD.ACCEPTED_VALS:
            enval = hi[HD.INTHDRS.index(hdr)]
            if not is_valid_enum_int(hdr, enval, allow_null=True):
                msg = "Invalid enumerated value, '{}': {}".format(hdr, enval)
                raise SacInvalidContentError(msg)

    if 'reftime' in tests:
        nzyear = hi[HD.INTHDRS.index('nzyear')]
        nzjday = hi[HD.INTHDRS.index('nzjday')]
        nzhour = hi[HD.INTHDRS.index('nzhour')]
        nzmin = hi[HD.INTHDRS.index('nzmin')]
        nzsec = hi[HD.INTHDRS.index('nzsec')]
        nzmsec = hi[HD.INTHDRS.index('nzmsec')]

        # all header reference time fields are set
        if not all([val != HD.INULL for val in
                    [nzyear, nzjday, nzhour, nzmin, nzsec, nzmsec]]):
            msg = "Null reference time values detected."
            raise SacInvalidContentError(msg)

        # reference time fields are reasonable values
        try:
            UTCDateTime(year=nzyear, julday=nzjday, hour=nzhour, minute=nzmin,
                        second=nzsec, microsecond=nzmsec)
        except ValueError as e:
            raise SacInvalidContentError("Invalid reference time: %s" % str(e))

    if 'reltime' in tests:
        # iztype is set and points to a non-null header value
        iztype_val = hi[HD.INTHDRS.index('iztype')]
        if is_valid_enum_int('iztype', iztype_val, allow_null=False):
            if iztype_val == 9:
                hdr = 'b'
            elif iztype_val == 11:
                hdr = 'o'
            elif iztype_val == 12:
                hdr = 'a'
            elif iztype_val in range(13, 23):
                hdr = 'it' + str(iztype_val - 13)

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
