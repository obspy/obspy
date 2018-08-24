# -*- coding: utf-8 -*-
"""
Data extraction and transfer from miniSEED files
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import re
import os
import ctypes
import bisect

from collections import namedtuple
from io import BytesIO
from obspy import read
from obspy.core import UTCDateTime
from obspy.core.stream import Stream
from obspy.clients.filesystem.msriterator import MSR_iterator


class NoDataError(Exception):
    """
    Error raised when no data is found
    """
    pass


class RequestLimitExceededError(Exception):
    """
    Error raised when the amount of data exceeds the configured limit
    """
    pass


class ExtractedDataSegment(object):
    """
    There are a few different forms that a chunk of extracted data can take,
    so we return a wrapped object that exposes a simple, consistent API
    for the handler to use.
    """
    def write(self, wfile):
        """
        Write the data to the given file-like object
        """
        raise NotImplementedError()

    def get_num_bytes(self):
        """
        Return the number of bytes in the segment
        """
        raise NotImplementedError()

    def get_src_name(self):
        """
        Return the name of the data source
        """
        raise NotImplementedError()


class MSRIDataSegment(ExtractedDataSegment):
    """
    Segment of data from a MSR_iterator
    """
    def __init__(self, msri, sample_rate, start_time, end_time, src_name,
                 debug=False):
        """
        :param msri: A `MSR_iterator`
        :param sample_rate: Sample rate of the data
        :param start_time: A `UTCDateTime` giving the start of the
                           requested data
        :param end_time: A `UTCDateTime` giving the end of the requested data
        :param src_name: Name of the data source for logging
        :type debug: bool
        :param debug: Debug flag.
        """
        self.msri = msri
        self.sample_rate = sample_rate
        self.start_time = start_time
        self.end_time = end_time
        self.src_name = src_name
        self.debug = debug

    def read_stream(self):
        msrstart = self.msri.get_startepoch()
        msrend = self.msri.get_endepoch()
        reclen = self.msri.msr.contents.reclen

        sepoch = self.start_time.timestamp
        eepoch = self.end_time.timestamp

        st = Stream()
        # Process records that intersect with request time window
        if msrstart < eepoch and msrend > sepoch:

            # Trim record if coverage and partial overlap with request
            if self.sample_rate > 0 and (msrstart < self.start_time or
                                         msrend > self.end_time):
                if self.debug is True:
                    print("Trimming record %s @ %s" %
                          (self.src_name, self.msri.get_starttime()))
                tr = \
                    read(BytesIO(ctypes.string_at(
                                    self.msri.msr.contents.record,
                                    reclen)),
                         format="MSEED")[0]
                tr.trim(self.start_time, self.end_time)
                st.traces.append(tr)
                return st
            # Otherwise, write un-trimmed record
            else:
                # Construct to avoid copying the data, supposedly
                if self.debug is True:
                    print("Writing full record %s @ %s" %
                          (self.src_name, self.msri.get_starttime()))
                out = (ctypes.c_char * reclen).from_address(
                    ctypes.addressof(self.msri.msr.contents.record.contents))
                data = BytesIO(out.raw)
                st = read(data, format="MSEED")
        return st

    def get_num_bytes(self):
        return self.msri.msr.contents.reclen

    def get_src_name(self):
        return self.src_name


class FileDataSegment(ExtractedDataSegment):
    """
    Segment of data that comes directly from a data file
    """
    def __init__(self, filename, start_byte, num_bytes, src_name):
        """
        :param filename: Name of data file
        :param start_byte: Return data starting from this offset
        :param num_bytes: Length of data to return
        :param src_name: Name of the data source for logging
        """
        self.filename = filename
        self.start_byte = start_byte
        self.num_bytes = num_bytes
        self.src_name = src_name

    def read_stream(self):
        st = Stream()
        with open(self.filename, "rb") as f:
            f.seek(self.start_byte)
            raw_data = BytesIO(f.read(self.num_bytes))
            st = read(raw_data, format="MSEED")
        return st

    def get_num_bytes(self):
        return self.num_bytes

    def get_src_name(self):
        return self.src_name


class MiniseedDataExtractor(object):
    """
    Component for extracting, trimming, and validating data.
    """
    def __init__(self, dp_replace=None, request_limit=0, debug=False):
        """
        :param dp_replace: optional tuple of (regex, replacement) indicating
          the location of data files. If regex is omitted, then the replacement
          string is appended to the beginning of the file name.
        :param request_limit: optional limit (in bytes) on how much data can
          be extracted at once
        :type debug: bool
        :param debug: Debug flag.
        """
        if dp_replace:
            self.dp_replace_re = re.compile(dp_replace[0]) if dp_replace[0] \
                                                            else None
            self.dp_replace_sub = dp_replace[1]
        else:
            self.dp_replace_re = None
            self.dp_replace_sub = None
        self.request_limit = request_limit
        self.debug = debug

    def handle_trimming(self, stime, etime, NRow):
        """
        Get the time & byte-offsets for the data in time range (stime, etime).

        This is done by finding the smallest section of the data in row that
        falls within the desired time range and is identified by the timeindex
        field of row.

        :returns: [(start time, start offset, trim_boolean),
                   (end time, end offset, trim_boolean)]
        """
        etime = UTCDateTime(NRow.requestend)
        row_stime = UTCDateTime(NRow.starttime)
        row_etime = UTCDateTime(NRow.endtime)

        # If we need a subset of the this block, trim it accordingly
        block_start = int(NRow.byteoffset)
        block_end = block_start + int(NRow.bytes)
        if stime > row_stime or etime < row_etime:
            tix = [x.split("=>") for x in NRow.timeindex.split(",")]
            if tix[-1][0] == 'latest':
                tix[-1] = [str(row_etime.timestamp), block_end]
            to_x = [float(x[0]) for x in tix]
            s_index = bisect.bisect_right(to_x, stime.timestamp) - 1
            if s_index < 0:
                s_index = 0
            e_index = bisect.bisect_right(to_x, etime.timestamp)
            off_start = int(tix[s_index][1])
            if e_index >= len(tix):
                e_index = -1
            off_end = int(tix[e_index][1])
            return ([to_x[s_index], off_start, stime > row_stime],
                    [to_x[e_index], off_end, etime < row_etime],)
        else:
            return ([row_stime.timestamp, block_start, False],
                    [row_etime.timestamp, block_end, False])

    def extract_data(self, index_rows):
        """
        Perform the data extraction.

        :param index_rows: requested data, as produced by
        `HTTPServer_RequestHandler.fetch_index_rows`
        :yields: sequence of `ExtractedDataSegment`s
        """

        # Pre-scan the index rows:
        # 1) Build processed list for extraction
        # 2) Check if the request is small enough to satisfy
        # Note: accumulated estimate of output bytes will be equal to or
        # higher than actual output
        total_bytes = 0
        request_rows = []
        Request = namedtuple('Request', ['srcname', 'filename', 'starttime',
                                         'endtime', 'triminfo', 'bytes',
                                         'samplerate'])
        try:
            for NRow in index_rows:
                srcname = "_".join(NRow[:4])
                filename = NRow.filename
                if self.debug:
                    print("EXTRACT: src=%s, file=%s, bytes=%s, rate:%s" %
                          (srcname, filename, NRow.bytes, NRow.samplerate))

                starttime = UTCDateTime(NRow.requeststart)
                endtime = UTCDateTime(NRow.requestend)
                triminfo = self.handle_trimming(starttime, endtime, NRow)
                total_bytes += triminfo[1][1] - triminfo[0][1]
                if self.request_limit > 0 and total_bytes > self.request_limit:
                    raise RequestLimitExceededError(
                            "Result exceeds limit of %d bytes" %
                            self.request_limit)
                if self.dp_replace_re and self.dp_replace_sub:
                    filename = self.dp_replace_re.sub(self.dp_replace_sub,
                                                      filename)
                elif self.dp_replace_sub:
                    filename = self.dp_replace_sub + filename
                if not os.path.exists(filename):
                    raise Exception("Data file does not exist: %s" % filename)
                request_rows.append(Request(srcname=srcname,
                                            filename=filename,
                                            starttime=starttime,
                                            endtime=endtime,
                                            triminfo=triminfo,
                                            bytes=NRow.bytes,
                                            samplerate=NRow.samplerate))
                if self.debug:
                    print("EXTRACT: src=%s, file=%s, bytes=%s, rate:%s" %
                          (srcname, filename, NRow.bytes, NRow.samplerate))
        except Exception as err:
            raise Exception("Error accessing data index: %s" % str(err))

        # Error if request matches no data
        if total_bytes == 0:
            raise NoDataError()

        # Get & return the actual data
        for NRow in request_rows:
            if self.debug:
                print("Extracting %s (%s - %s) from %s" % (NRow.srcname,
                                                           NRow.starttime,
                                                           NRow.endtime,
                                                           NRow.filename))

            # Iterate through records in section
            # if only part of the section is needed
            if NRow.triminfo[0][2] or NRow.triminfo[1][2]:

                for msri in MSR_iterator(filename=NRow.filename,
                                         startoffset=NRow.triminfo[0][1],
                                         dataflag=False):
                    offset = msri.get_offset()

                    # Done if we are beyond end offset
                    if offset >= NRow.triminfo[1][1]:
                        break

                    yield MSRIDataSegment(msri,
                                          NRow.samplerate,
                                          NRow.starttime,
                                          NRow.endtime,
                                          NRow.srcname,
                                          self.debug)

                    # Check for passing end offset
                    if (offset + msri.msr.contents.reclen) >= \
                            NRow.triminfo[1][1]:
                        break

            # Otherwise, return the entire section
            else:
                yield FileDataSegment(NRow.filename, NRow.triminfo[0][1],
                                      NRow.bytes, NRow.srcname)
