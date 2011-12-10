# -*- coding: utf-8 -*-
"""
MSEED bindings to ObsPy core module.
"""

import warnings
from obspy.core import Stream, Trace
from obspy.core.util import NATIVE_BYTEORDER
from obspy.mseed import LibMSEED
from obspy.mseed.headers import ENCODINGS
import numpy as np
import platform


def isMSEED(filename):
    """
    Checks whether a file is Mini-SEED or not.

    :type filename: string
    :param filename: Mini-SEED file to be checked.
    :rtype: bool
    :return: ``True`` if a Mini-SEED file.
    """
    __libmseed__ = LibMSEED()
    return __libmseed__.isMSEED(filename)


def readMSEED(filename, headonly=False, starttime=None, endtime=None,
              readMSInfo=True, reclen=-1, quality=False,
              **kwargs):  # @UnusedVariable
    """
    Reads a Mini-SEED file and returns a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :type filename: string
    :param filename: Mini-SEED file to be read.
    :type headonly: boolean, optional
    :param headonly: If True read only head of Mini-SEED file.
    :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
    :param starrttime: Specify the starttime to read. The remaining records are
        not extracted. Providing a starttime usually results into faster
        reading. Under Windows this only works if all the file's records have
        the same id and are chronologically ordered. Defaults to ``None``.
    :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
    :param endtime: See description of starttime.
    :type readMSInfo: bool, optional
    :param readMSInfo: If ``True`` the byteorder, record length and the
        encoding of the file will be read and stored in the Stream object. Only
        the very first record of the file will be read and all following
        records are assumed to be the same. Defaults to ``True``.
    :type reclen: int, optional
    :param reclen: Record length in bytes of Mini-SEED file to read. This
        option might be usefull if blockette 10 is missing and thus read cannot
        determine the reclen automatically.
    :type quality: bool, optional
    :param quality: Determines whether quality information is being read or
        not. Has a big impact on performance so only use when necessary
        (takes ~ 700 % longer). Two attributes will be written to each Trace's
        ``stats.mseed`` object: (1) ``data_quality_flags_count`` counts the
        bits in field 14 of the fixed header for each Mini-SEED record and (2)
        ``timing_quality`` is a :class:`~numpy.ndarray` which contains all
        timing qualities found in Blockette 1001 in the order of appearance. If
        no Blockette 1001 is found it will be an empty array.
    :rtype: :class:`~obspy.core.stream.Stream`
    :returns: Stream object containing header and data.

    .. rubric:: Example

    >>> from obspy.core import read
    >>> st = read("/path/to/test.mseed")
    """
    __libmseed__ = LibMSEED()
    # Read fileformat information if necessary.
    if readMSInfo:
        info = __libmseed__.getFileformatInformation(filename)
        # Better readability.
        if 'reclen' in info:
            info['record_length'] = info['reclen']
            del info['reclen']
        if 'encoding' in info:
            info['encoding'] = ENCODINGS[info['encoding']][0]
        if 'byteorder' in info:
            if info['byteorder'] == 1:
                info['byteorder'] = '>'
            else:
                info['byteorder'] = '<'
    # read MiniSEED file
    if headonly:
        trace_list = __libmseed__.readMSHeader(filename, reclen=reclen)
    else:
        if platform.system() == "Windows" or quality:
            # 4x slower on Mac
            trace_list = __libmseed__.readMSTracesViaRecords(filename,
                         starttime=starttime, endtime=endtime, reclen=reclen,
                         quality=quality)
        else:
            # problem on windows with big files (>=20 MB)
            trace_list = __libmseed__.readMSTraces(filename, reclen,
                starttime=starttime, endtime=endtime)
    # Create a list containing all the traces.
    traces = []
    # Loop over all traces found in the file.
    for _i in trace_list:
        # Convert header to be compatible with obspy.core.
        old_header = _i[0]
        header = {}
        # Create a dictionary to specify how to convert keys.
        convert_dict = {'station': 'station', 'sampling_rate': 'samprate',
                        'npts': 'numsamples', 'network': 'network',
                        'location': 'location', 'channel': 'channel',
                        'starttime': 'starttime', 'endtime': 'endtime'}
        # Convert header.
        for _j, _k in convert_dict.iteritems():
            header[_j] = old_header[_k]
        # Dataquality is Mini-SEED only and thus has an extra Stats attribute.
        header['mseed'] = {}
        header['mseed']['dataquality'] = old_header['dataquality']
        # Convert times to obspy.UTCDateTime objects.
        header['starttime'] = \
            __libmseed__._convertMSTimeToDatetime(header['starttime'])
        header['endtime'] = \
            __libmseed__._convertMSTimeToDatetime(header['endtime'])
        # Append quality informations if necessary.
        if quality:
                header['mseed']['timing_quality'] = \
                np.array(old_header['timing_quality'])
                header['mseed']['data_quality_flags_count'] = \
                                      old_header['data_quality_flags']
        # Append information if necessary.
        if readMSInfo:
            for key, value in info.iteritems():
                header['mseed'][key] = value
        # Append traces.
        if headonly:
            header['npts'] = int((header['endtime'] - header['starttime']) *
                                   header['sampling_rate'] + 1 + 0.5)
            traces.append(Trace(header=header))
        else:
            traces.append(Trace(header=header, data=_i[1]))
    return Stream(traces=traces)


def writeMSEED(stream, filename, encoding=None, **kwargs):
    """
    Write Mini-SEED file from a Stream object.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.stream.Stream.write` method of an

    :type stream: :class:`~obspy.core.stream.Stream`
    :param stream: The ObsPy Stream object to write.
    :type filename: string
    :param filename: Name of file to write.
    :type reclen: int, optional
    :param reclen: Should be set to the desired data record length in bytes
        which must be expressible as 2 raised to the power of X where X is
        between (and including) 8 to 20. Defaults to ``4096``.
    :type encoding: int or string, optional
    :param encoding: Should be set to one of the following supported
        Mini-SEED data encoding formats: ``ASCII`` (``0``)*, ``INT16`` (``1``),
        ``INT32`` (``3``), ``FLOAT32`` (``4``)*, ``FLOAT64`` (``5``)*,
        ``STEIM1`` (``10``) and ``STEIM2`` (``11``)*. Default data types a
        marked with an asterisk. Currently ``INT24`` (``2``) is not supported
        due to lacking NumPy support.
    :type byteorder: ``0``, ``'<'``, ``1``, ``'>'`` or ``-1``, optional
    :param byteorder: Must be either ``0`` or ``'<'`` for LSBF or
        little-endian, ``1`` or ``'>'`` for MBF or big-endian. The value ``-1``
        defaults to big-endian (``1``).
    :type flush: int, optional
    :param flush: If it is not zero all of the data will be packed into
        records, otherwise records will only be packed while there are
        enough data samples to completely fill a record.
    :type verbose: int, optional
    :param verbose: Controls verbosity, a value of zero will result in no
        diagnostic output. Defaults to ``0``.

    .. rubric:: Example

    >>> from obspy.core import read
    >>> st = read()
    >>> st.write('filename.mseed', format='MSEED') #doctest: +SKIP
    """
    # Write all fileformat descriptions that might occur in the
    # trace.stats.mseed dictionary to the kwargs if they do not exists yet.
    # This means that the kwargs will overwrite whatever is set in the
    # trace.stats.mseed attributes. Only check for the first trace!
    stats = stream[0].stats
    if hasattr(stats, 'mseed'):
        if not 'byteorder' in kwargs and hasattr(stats.mseed, 'byteorder'):
            kwargs['byteorder'] = stats.mseed.byteorder
        if not 'reclen' in kwargs and hasattr(stats.mseed, 'record_length'):
            kwargs['reclen'] = stats.mseed.record_length
    # Check if encoding kwarg is set and catch invalid encodings.
    # XXX: Currently INT24 is not working due to lacking numpy support.
    encoding_strings = dict([(v[0], k) for (k, v) in ENCODINGS.iteritems()])

    # If the encoding is enforced validate it and check if all data has the
    # correct dtype.
    if encoding is not None:
        if isinstance(encoding, int) and encoding in ENCODINGS:
            encoding = encoding
        elif isinstance(encoding, basestring) and encoding in encoding_strings:
            encoding = encoding_strings[encoding]
        else:
            msg = 'Invalid encoding %s. Valid encodings: %s'
            raise ValueError(msg % (encoding, encoding_strings))
        # Check if the dtype for all traces is compatible with the enforced
        # encoding.
        dtypes = [tr.data.dtype for tr in stream]
        # removing duplicates in list
        dtypes = list(set(dtypes))
        id, _, dtype = ENCODINGS[encoding]
        if len(dtypes) != 1 or dtypes[0].type != dtype:
            msg = """Wrong dtype of the data of one or more Traces for encoding
                %s expecting dtype %s. Please change the dtype of your data or
                use an appropriate encoding. See the obspy.mseed documentation
                for more information.
                """ % (id, dtype)
            raise Exception(msg)

    # translate byteorder
    if 'byteorder' in kwargs.keys() and kwargs['byteorder'] not in [0, 1, -1]:
        if kwargs['byteorder'] == '=':
            kwargs['byteorder'] = NATIVE_BYTEORDER
        if kwargs['byteorder'] == '<':
            kwargs['byteorder'] = 0
        elif kwargs['byteorder'] == '>':
            kwargs['byteorder'] = 1
        else:
            kwargs['byteorder'] = 1
    # Catch invalid record length.
    valid_record_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                            65536, 131072, 262144, 524288, 1048576]
    if 'reclen' in kwargs.keys() and \
                not kwargs['reclen'] in valid_record_lengths:
        msg = 'Invalid record length. The record length must be a value\n' + \
              'of 2 to the power of X where 8 <= X <= 20.'
        raise ValueError(msg)
    # libmseed instance.
    __libmseed__ = LibMSEED()
    traces = stream.traces
    trace_list = []
    convert_dict = {'station': 'station', 'samprate': 'sampling_rate',
                    'numsamples': 'npts', 'network': 'network',
                    'location': 'location', 'channel': 'channel',
                    'starttime': 'starttime', 'endtime': 'endtime'}
    for trace in traces:
        header = {}
        for _j, _k in convert_dict.iteritems():
            header[_j] = trace.stats[_k]
        # Set data quality to indeterminate (= D) if it is not already set.
        try:
            header['dataquality'] = trace.stats['mseed']['dataquality'].upper()
        except:
            header['dataquality'] = 'D'
        # Sanity check for the dataquality to get a nice Python exception
        # instead of a C error.
        if header['dataquality'] not in ['D', 'R', 'Q', 'M']:
            msg = 'The dataquality for Mini-SEED must be either D, R, Q ' + \
                   'or M. See the SEED manual for further information.'
            raise ValueError(msg)
        # Convert UTCDateTime times to Mini-SEED times.
        header['starttime'] = \
            __libmseed__._convertDatetimeToMSTime(header['starttime'])
        header['endtime'] = \
            __libmseed__._convertDatetimeToMSTime(header['endtime'])
        # Check that data are numpy.ndarrays
        if not isinstance(trace.data, np.ndarray):
            msg = "Unsupported data type %s" % type(trace.data)
            raise Exception(msg)

        enc = None
        if encoding is None:
            if hasattr(trace.stats, 'mseed') and \
                    hasattr(trace.stats.mseed, 'encoding'):
                mseed_encoding = stats.mseed.encoding
                # Check if the encoding is valid.
                if isinstance(mseed_encoding, int) and \
                        mseed_encoding in ENCODINGS:
                    enc = mseed_encoding
                elif isinstance(mseed_encoding, basestring) and \
                        mseed_encoding in encoding_strings:
                    enc = encoding_strings[mseed_encoding]
                else:
                    msg = 'Invalid encoding %s in ' + \
                          'stream[0].stats.mseed.encoding. Valid encodings: %s'
                    raise ValueError(msg % (mseed_encoding, encoding_strings))
                # Check if the encoding matches the data's dtype.
                if trace.data.dtype.type != ENCODINGS[enc][2]:
                    msg = 'The encoding specified in ' + \
                          'trace.stats.mseed.encoding does not match the ' + \
                          'dtype of the data.\nA suitable encoding will ' + \
                          'be chosen.'
                    warnings.warn(msg, UserWarning)
                    enc = None
            # automatically detect encoding if no encoding is given.
            if enc is None:
                if trace.data.dtype.type == np.dtype("int32"):
                    enc = 11
                elif trace.data.dtype.type == np.dtype("float32"):
                    enc = 4
                elif trace.data.dtype.type == np.dtype("float64"):
                    enc = 5
                elif trace.data.dtype.type == np.dtype("int16"):
                    enc = 1
                elif trace.data.dtype.type == np.dtype('|S1').type:
                    enc = 0
                else:
                    msg = "Unsupported data type %s" % trace.data.dtype
                    raise Exception(msg)
        else:
            enc = encoding

        id, sampletype, dtype = ENCODINGS[enc]
        data = trace.data
        # INT16 needs INT32 data type
        if enc == 1:
            # create a copy and don't modify the original data
            data = data.copy().astype(np.int32)
        header['sampletype'] = sampletype
        header['encoding'] = enc
        # Fill the samplecnt attribute.
        header['samplecnt'] = len(data)
        # Check if ndarray is contiguous (see #192, #193)
        if not data.flags.c_contiguous:
            msg = "Detected non contiguous data array during ctypes call. " \
                  "Trying to fix array."
            warnings.warn(msg)
        trace_list.append([header, np.require(data,
                                              requirements=('C_CONTIGUOUS',))])
    # Write resulting trace_list to Mini-SEED file.
    __libmseed__.writeMSTraces(trace_list, outfile=filename, **kwargs)
