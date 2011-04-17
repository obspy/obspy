# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
#  Filename: libmseed.py
#  Purpose: Python wrapper for libmseed of Chad Trabant
#   Author: Lion Krischer, Robert Barsch, Moritz Beyreuther
#    Email: krischer@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2010 Lion Krischer, Robert Barsch, Moritz Beyreuther
#---------------------------------------------------------------------
"""
Low-level module internally used for handling MiniSEED files

Contains wrappers for libmseed - The MiniSEED library. The C library is
interfaced via Python ctypes. Currently only supports MiniSEED files with
integer data values.


GNU General Public License (GPL)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301, USA.

Selected examples of Methods in obspy.mseed.libmseed
----------------------------------------------------
All of the following methods can only be accessed with an instance of the
libmseed class.

>>> from obspy.mseed.libmseed import LibMSEED
>>> mseed = LibMSEED()
>>> mseed #doctest: +ELLIPSIS
<obspy.mseed.libmseed.LibMSEED object at 0x...>

isMSEED
^^^^^^^
Tests whether a file is a MiniSEED file or not. Returns True on success or
False otherwise.

This method only reads the first seven bytes of the file and checks whether it
is a MiniSEED or fullSEED file. It also is true for fullSEED files because
libmseed can read the data part of fullSEED files. If the method finds a
fullSEED file it also checks if it has a data part and returns False otherwise.
Thus it cannot be used to validate a MiniSEED or SEED file.

Parameters:
    * filename = MiniSEED file. 

(The first lines are just to get the absolute path of the test file)

>>> from obspy.core.util import getExampleFile
>>> filename = getExampleFile("test.mseed")
>>> mseed.isMSEED(filename)
True

getDataQualityFlagsCount
^^^^^^^^^^^^^^^^^^^^^^^^
Counts all data quality flags of the given MiniSEED file. This method will
count all set data quality flag bits in the fixed section of the data header in
a MiniSEED file and returns the total count for each flag type.

Data quality flags:

========  =================================================
Bit       Description
========  =================================================
[Bit 0]   Amplifier saturation detected (station dependent)
[Bit 1]   Digitizer clipping detected
[Bit 2]   Spikes detected
[Bit 3]   Glitches detected
[Bit 4]   Missing/padded data present
[Bit 5]   Telemetry synchronization error
[Bit 6]   A digital filter may be charging
[Bit 7]   Time tag is questionable
========  =================================================

This will only work correctly if each record in the file has the same record
length.

Parameters:
    * filename = MiniSEED file. 

(The first line are just to get the absolute path of the test file)

>>> filename = getExampleFile("qualityflags.mseed")
>>> mseed.getDataQualityFlagsCount(filename)
[9, 8, 7, 6, 5, 4, 3, 2]

getTimingQuality
^^^^^^^^^^^^^^^^
Reads timing quality and returns a dictionary containing statistics about it.
This method will read the timing quality in Blockette 1001 for each record in
the file if available and return the following statistics:
Minima, maxima, average, median and upper and lower quantile. It is probably
pretty safe to set the first_record parameter to True because the timing
quality is a vendor specific value and thus it will probably be set for each
record or for none.

Parameters:
* filename = MiniSEED file.
* first_record: Determines whether all records are assumed to either have a
  timing quality in Blockette 1001 or not depending on whether the first
  records has one. If True and the first records does not have a timing quality
  it will not parse the whole file. If False is will parse the whole file
  anyway and search for a timing quality in each record. Defaults to True.
* rl_autodetection: Determines the auto-detection of the record lengths in the
  file. If 0 only the length of the first record is detected automatically. All
  subsequent records are then assumed to have the same record length. If -1 the
  length of each record is automatically detected. Defaults to -1. 

(The first line are just to get the absolute path of the test file)

>>> filename = getExampleFile("timingquality.mseed")
>>> mseed.getTimingQuality(filename) #doctest: +NORMALIZE_WHITESPACE
{'min': 0.0, 'max': 100.0, 'average': 50.0, 'median': 50.0,
 'upper_quantile': 75.0, 'lower_quantile': 25.0}
"""

from obspy.core import UTCDateTime
from obspy.core.util import scoreatpercentile
from obspy.mseed.headers import MSFileParam, clibmseed, HPTMODULUS, MSRecord, \
    FRAME, DATATYPES, SAMPLESIZES, blkt_1001_s
from struct import unpack
import ctypes as C
import math
import numpy as np
import os
import warnings

# as defined in libmseed.h
MS_ENDOFFILE = 1
MS_NOERROR = 0

class LibMSEED(object):
    """
    Class for handling MiniSEED files.
    """

    def isMSEED(self, filename):
        """
        Tests whether a file is a MiniSEED file or not.
        
        Returns True on success or False otherwise.
        
        This method only reads the first seven bytes of the file and checks
        whether its a MiniSEED or fullSEED file.
        
        It also is true for fullSEED files because libmseed can read the data
        part of fullSEED files. If the method finds a fullSEED file it also
        checks if it has a data part and returns False otherwise.
        
        Thus it cannot be used to validate a MiniSEED or SEED file.
        
        :param filename: MiniSEED file.
        """
        fp = open(filename, 'rb')
        header = fp.read(7)
        # File has less than 7 characters
        if len(header) != 7:
            return False
        # Sequence number must contains a single number or be empty 
        seqnr = header[0:6].replace('\x00', ' ').strip()
        if not seqnr.isdigit() and seqnr != '':
            return False
        # Check for any valid control header types.
        if header[6] in ['D', 'R', 'Q', 'M']:
            return True
        # Check if Full-SEED
        if not header[6] == 'V':
            return False
        # Parse the whole file and check whether it has has a data record.
        fp.seek(1, 1)
        _i = 0
        # search for blockettes 010 or 008
        while True:
            if fp.read(3) in ['010', '008']:
                break
            # the next for bytes are the record length
            # as we are currently at position 7 (fp.read(3) fp.read(4))
            # we need to subtract this first before we seek
            # to the appropriate position
            try:
                fp.seek(int(fp.read(4)) - 7, 1)
            except:
                return False
            _i += 1
            # break after 3 cycles
            if _i == 3:
                return False
        # Try to get a record length.
        fp.seek(8, 1)
        try:
            record_length = pow(2, int(fp.read(2)))
        except:
            return False
        file_size = os.path.getsize(filename)
        # Jump to the second record.
        fp.seek(record_length + 6)
        # Loop over all records and return True if one record is a data
        # record
        while fp.tell() < file_size:
            flag = fp.read(1)
            if flag in ['D', 'R', 'Q', 'M']:
                return True
            fp.seek(record_length - 1, 1)
        return False

    def readMSTracesViaRecords(self, filename, reclen= -1, dataflag=1,
                               skipnotdata=1, verbose=0, starttime=None,
                               endtime=None, quality=False):
        """
        Read MiniSEED file. Returns a list with header informations and data
        for each trace in the file.
        
        The list returned contains a list for each trace in the file with the
        lists first element being a header dictionary and its second element
        containing the data values as a NumPy array.

        :param filename: Name of MiniSEED file.
        :param reclen, dataflag, skipnotdata, verbose: These are passed
            directly to the ms_readmsr.
        :param quality: Read quality information or not. Defaults to false.
        """
        # Open file handler necessary for reading quality informations.
        if quality:
            file = open(filename, 'rb')
        # Initialize list that will contain all traces, first dummy entry
        # will be removed at the end again
        trace_list = [[{'endtime':0}, np.array([])]]
        # Initialize quality informations.
        if quality:
            trace_list[-1][0]['timing_quality'] = []
            trace_list[-1][0]['data_quality_flags'] = [0] * 8
        ms = _MSStruct(filename)
        end_byte = 1e99
        if starttime or endtime:
            bytes = self._bytePosFromTime(filename, starttime=starttime,
                                          endtime=endtime)
            if bytes == '':
                del ms # for valgrind
                return ''
            ms.offset = bytes[0]
            end_byte = bytes[0] + bytes[1]
        else:
            end_byte = os.path.getsize(filename)
        # Loop over records and append to trace_list.
        last_msrid = None
        while True:
            if quality:
                filepos = ms.offset
            # Directly call ms_readmsr_r
            errcode = ms.read(reclen, skipnotdata, dataflag, verbose,
                              raise_flag=False)
            if errcode != MS_NOERROR:
                if errcode == MS_ENDOFFILE:
                    msg = "Broken last record in mseed file %s" % filename
                    warnings.warn(msg)
                    break
                raise Exception("Error %d in ms_readmsr_r" % errcode)
            chain = ms.msr.contents
            header = self._convertMSRToDict(chain)
            delta = HPTMODULUS / float(header['samprate'])
            header['endtime'] = long(header['starttime'] + delta * \
                                      (header['numsamples'] - 1))
            # Access data directly as NumPy array.
            data = self._ctypesArray2NumpyArray(chain.datasamples,
                                                chain.numsamples,
                                                chain.sampletype)
            msrid = self._MSRId(header)
            last_endtime = trace_list[-1][0]['endtime']
            record_delta = abs(last_endtime - header['starttime'])
            if record_delta <= 1.01 * delta and\
               record_delta >= 0.99 * delta and\
               last_msrid == msrid:
                # Append to trace
                trace_list[-1][0]['endtime'] = header['endtime']
                trace_list[-1][0]['numsamples'] += header['numsamples']
                trace_list[-1].append(data)
                # Read quality information.
                if quality:
                    self._readQuality(file, filepos, chain,
                         tq=trace_list[-1][0]['timing_quality'],
                         dq=trace_list[-1][0]['data_quality_flags'])
            else:
                # Concatenate last trace and start a new trace
                trace_list[-1] = [trace_list[-1][0],
                                  np.concatenate(trace_list[-1][1:])]
                trace_list.append([header, data])
                # Initialize quality informations.
                if quality:
                    trace_list[-1][0]['timing_quality'] = []
                    trace_list[-1][0]['data_quality_flags'] = [0] * 8
                    self._readQuality(file, filepos, chain,
                         tq=trace_list[-1][0]['timing_quality'],
                         dq=trace_list[-1][0]['data_quality_flags'])
            last_msrid = msrid
            if ms.offset >= end_byte:
                break
        # Finish up loop, concatenate last trace_list
        trace_list[-1] = [trace_list[-1][0],
                          np.concatenate(trace_list[-1][1:])]
        trace_list.pop(0) # remove first dummy entry of list
        del ms # for valgrind
        # Close file.
        if quality:
            file.close()
        return trace_list

    def _readQuality(self, file, filepos, chain, tq, dq):
        """
        Reads all quality informations from a file and writes it to tq and dq.
        """
        # Seek to correct position.
        file.seek(filepos, 0)
        # Skip non data records.
        data = file.read(39)
        if data[6] == 'D':
            # Read data quality byte.
            data_quality_flags = data[38]
            # Unpack the binary data.
            data_quality_flags = unpack('B', data_quality_flags)[0]
            # Add the value of each bit to the quality_count.
            for _j in xrange(8):
                if (data_quality_flags & (1 << _j)) != 0:
                    dq[_j] += 1
        try:
            # Get timing quality in blockette 1001.
            tq.append(float(chain.Blkt1001.contents.timing_qual))
        except:
            pass

    def readMSTraces(self, filename, reclen= -1, timetol= -1,
                     sampratetol= -1, dataflag=1, skipnotdata=1,
                     dataquality=1, verbose=0, starttime=None,
                     endtime=None):
        """
        Read MiniSEED file. Returns a list with header informations and data
        for each trace in the file.
        
        The list returned contains a list for each trace in the file with the
        lists first element being a header dictionary and its second element
        containing the data values as a NumPy array.

        :param filename: Name of MiniSEED file.
        :param reclen: Directly to the readFileToTraceGroup method.
        :param timetol: Directly to the readFileToTraceGroup method.
        :param sampratetol: Directly to the readFileToTraceGroup method.
        :param dataflag: Directly to the readFileToTraceGroup method.
        :param skipnotdata: Directly to the readFileToTraceGroup method.
        :param dataquality: Directly to the readFileToTraceGroup method.
        :param verbose: Directly to the readFileToTraceGroup method.
        """
        # Create empty list that will contain all traces.
        trace_list = []
        # Creates MSTraceGroup Structure and feed it with the MiniSEED data.
        mstg = self.readFileToTraceGroup(str(filename), reclen=reclen,
                                         timetol=timetol,
                                         sampratetol=sampratetol,
                                         dataflag=dataflag,
                                         skipnotdata=skipnotdata,
                                         dataquality=dataquality,
                                         verbose=verbose,
                                         starttime=starttime,
                                         endtime=endtime)
        chain = mstg.contents.traces.contents
        numtraces = mstg.contents.numtraces
        # Loop over traces and append to trace_list.
        for i in xrange(numtraces):
            header = self._convertMSTToDict(chain)
            # Access data directly as NumPy array.
            data = self._ctypesArray2NumpyArray(chain.datasamples,
                                                chain.numsamples,
                                                chain.sampletype)
            trace_list.append([header, data])
            # Set chain to next trace.
            if i != numtraces - 1:
                next = chain.next.contents
                clibmseed.mst_free(C.pointer(C.pointer(chain)))
                chain = next
        clibmseed.mst_free(C.pointer(C.pointer(chain)))
        mstg.contents.traces = None # avoid double free
        clibmseed.mst_freegroup(C.pointer(mstg))
        del mstg, chain
        return trace_list

    def writeMSTraces(self, trace_list, outfile, reclen= -1, encoding= -1,
                      byteorder= -1, flush= -1, verbose=0):
        """
        Write MiniSEED file from trace_list
        
        :param trace_list: List containing header informations and data.
        :param outfile: Name of the output file
        :param reclen: should be set to the desired data record length in bytes
            which must be expressible as 2 raised to the power of X where X is
            between (and including) 8 to 20. -1 defaults to 4096
        :type encoding: Integer
        :param encoding: should be set to one of the following supported
            MiniSEED data encoding formats: ASCII (0), INT16 (1),
            INT32 (3), FLOAT32 (4), FLOAT64 (5), STEIM1 (10)
            and STEIM2 (11). Defaults to STEIM2 (11)
        :param byteorder: must be either 0 (LSBF or little-endian) or 1 (MBF or 
            big-endian). -1 defaults to big-endian (1)
        :param flush: if it is not zero all of the data will be packed into 
            records, otherwise records will only be packed while there are
            enough data samples to completely fill a record.
        :param verbose: controls verbosity, a value of zero will result in no 
            diagnostic output.
        """
        try:
            f = open(outfile, 'wb')
        except TypeError:
            f = outfile
        # Figure out whether or not to use Blockette 1001. This check is done
        # once to ensure that Blockette 1001 is either written for every record
        # in the file or for none. It checks the starttime as well as the
        # sampling rate. If either one has a precision of more than 100
        # microseconds, Blockette 1001 will be written for every record.
        use_blkt_1001 = 0
        for trace in trace_list:
            if trace[0]['starttime'] % 100 != 0 or \
               (1.0 / trace[0]['samprate'] * HPTMODULUS) % 100 != 0:
                use_blkt_1001 += 1
        for trace in trace_list:
            # Populate MSTG Structure
            mstg = self._populateMSTG(trace)
            # Initialize packedsamples pointer for the mst_pack function
            self.packedsamples = C.c_int()
            # Callback function for mst_pack to actually write the file
            def record_handler(record, reclen, _stream):
                f.write(record[0:reclen])
            # Define Python callback function for use in C function
            recHandler = C.CFUNCTYPE(C.c_void_p, C.POINTER(C.c_char), C.c_int,
                                     C.c_void_p)(record_handler)

            # Fill up msr record structure, this is already contained in
            # mstg, however if blk1001 is set we need it anyway 
            msr = clibmseed.msr_init(None)
            msr.contents.network = trace[0]['network']
            msr.contents.station = trace[0]['station']
            msr.contents.location = trace[0]['location']
            msr.contents.channel = trace[0]['channel']
            msr.contents.dataquality = trace[0]['dataquality']
            # Only use Blockette 1001 if necessary.
            if use_blkt_1001:
                size = C.sizeof(blkt_1001_s)
                blkt1001 = C.c_char(' ')
                C.memset(C.pointer(blkt1001), 0, size)
                ret_val = clibmseed.msr_addblockette(msr, C.pointer(blkt1001),
                                           size, 1001, 0)
                # Usually returns a pointer to the added blockette in the
                # blockette link chain and a NULL pointer if it fails.
                # NULL pointers have a false boolean value according to the
                # ctypes manual.
                if bool(ret_val) is False:
                    raise Exception('Error msr_addblockette')
            try:
                enc = trace[0]['encoding']
            except:
                enc = encoding
            # Pack mstg into a MSEED file using record_handler as write method
            errcode = clibmseed.mst_packgroup(mstg, recHandler, None, reclen,
                                              enc, byteorder,
                                              C.byref(self.packedsamples),
                                              flush, verbose, msr)
            if errcode == -1:
                raise Exception('Error in mst_packgroup')
            # Cleaning up
            clibmseed.mst_freegroup(C.pointer(mstg))
            del mstg, msr
        if isinstance(f, file): # necessary for Python 2.5.2 BUG otherwise!
            f.close()

    def readFileToTraceGroup(self, filename, reclen= -1, timetol= -1,
                             sampratetol= -1, dataflag=1, skipnotdata=1,
                             dataquality=1, verbose=0, starttime=None,
                             endtime=None):
        """
        Reads MiniSEED data from file. Returns MSTraceGroup structure.
        
        :param filename: Name of MiniSEED file.
        :param reclen: If reclen is 0 the length of the first record is auto-
            detected. All subsequent records are then expected to have the
            same record length. If reclen is negative the length of every
            record is automatically detected. Defaults to -1.
        :param timetol: Time tolerance, default to -1 (1/2 sample period).
        :param sampratetol: Sample rate tolerance, defaults to -1 (rate
            dependent)
        :param dataflag: Controls whether data samples are unpacked, defaults
            to true (1).
        :param skipnotdata: If true (not zero) any data chunks read that to do
            not have valid data record indicators will be skipped. Defaults to
            true (1).
        :param dataquality: If the dataquality flag is true traces will be
            grouped by quality in addition to the source name identifiers.
            Defaults to true (1).
        :param verbose: Controls verbosity from 0 to 2. Defaults to None (0).
        """
        # Creates MSTraceGroup Structure
        mstg = clibmseed.mst_initgroup(None)
        if starttime and endtime:
            starttime = long(starttime.timestamp * HPTMODULUS + 0.5)
            endtime = long(endtime.timestamp * HPTMODULUS + 0.5)
            # Uses libmseed to read the file and populate the MSTraceGroup
            errcode = clibmseed.ms_readtraces_timewin(
                C.pointer(mstg), filename, reclen, timetol, sampratetol,
                starttime, endtime, dataquality, skipnotdata, dataflag, verbose)
            if errcode != MS_NOERROR:
                raise Exception("Error %d in ms_readtraces" % errcode)
        else:
            # Uses libmseed to read the file and populate the MSTraceGroup
            errcode = clibmseed.ms_readtraces(
                C.pointer(mstg), filename, reclen, timetol, sampratetol,
                dataquality, skipnotdata, dataflag, verbose)
            if errcode != MS_NOERROR:
                raise Exception("Error %d in ms_readtraces" % errcode)
        return mstg

    def getStartAndEndTime(self, filename):
        """
        Returns the start- and endtime of a MiniSEED file as a tuple
        containing two datetime objects.
        Method using ms_readmsr_r
        
        This method only reads the first and the last record. Thus it will only
        work correctly for files containing only one trace with all records
        in the correct order.
        
        The returned endtime is the time of the last datasample and not the
        time that the last sample covers.
        
        :param filename: MiniSEED file string.
        """
        # Get the starttime
        ms = _MSStruct(filename)
        starttime = ms.getStart()
        # Get the endtime
        ms.offset = ms.filePosFromRecNum(record_number= -1)
        endtime = ms.getEnd()
        del ms # for valgrind
        return starttime, endtime

    def readMSHeader(self, filename, reclen= -1, timetol= -1,
                     sampratetol= -1, dataflag=1, skipnotdata=1,
                     dataquality=1, verbose=0, starttime=None,
                     endtime=None):
        """
        Returns trace header information of a given file without reading
        the data part.
        
        :param filename: Name of MiniSEED file.
        :param reclen: Directly to the readFileToTraceGroup method.
        :param timetol: Directly to the readFileToTraceGroup method.
        :param sampratetol: Directly to the readFileToTraceGroup method.
        :param skipnotdata: Directly to the readFileToTraceGroup method.
        :param dataquality: Directly to the readFileToTraceGroup method.
        :param verbose: Directly to the readFileToTraceGroup method.
        :return: Dictionary containing header entries
        """
        # read file
        mstg = self.readFileToTraceGroup(str(filename), reclen=reclen,
                                         timetol=timetol,
                                         sampratetol=sampratetol,
                                         dataflag=0,
                                         skipnotdata=skipnotdata,
                                         dataquality=dataquality,
                                         verbose=verbose,
                                         starttime=starttime,
                                         endtime=endtime)
        # iterate through traces
        cur = mstg.contents.traces.contents
        header = [[self._convertMSTToDict(cur), None]]
        for _ in xrange(mstg.contents.numtraces - 1):
            next = cur.next.contents
            header.append([self._convertMSTToDict(next), None])
            cur = next
        clibmseed.mst_freegroup(C.pointer(mstg))
        del mstg
        return header

    def getFileformatInformation(self, filename):
        """
        Reads the first record and returns all information about the Mini-SEED
        file format is can find.
        """
        # Create _MSStruct instance to read the file.
        ms = _MSStruct(filename)
        chain = ms.msr.contents
        # Read all interesting attributes.
        attribs = ['byteorder', 'encoding', 'reclen']
        info = {}
        for attr in attribs:
            info[attr] = getattr(chain, attr)
        # Will delete C pointers and structures.
        del ms
        return info

    def getDataQualityFlagsCount(self, filename):
        """
        Counts all data quality flags of the given MiniSEED file.
        
        This method will count all set data quality flag bits in the fixed
        section of the data header in a MiniSEED file and returns the total
        count for each flag type.
        
        Data quality flags
         - [Bit 0] Amplifier saturation detected (station dependent)
         - [Bit 1] Digitizer clipping detected
         - [Bit 2] Spikes detected
         - [Bit 3] Glitches detected
         - [Bit 4] Missing/padded data present
         - [Bit 5] Telemetry synchronization error
         - [Bit 6] A digital filter may be charging
         - [Bit 7] Time tag is questionable
        
        This will only work correctly if each record in the file has the same
        record length.
        
        :param filename: MiniSEED file name.
        :return: List of all flag counts.
        """
        # Open the file.
        mseedfile = open(filename, 'rb')
        # Get record length of the file.
        info = self._getMSFileInfo(mseedfile, filename)
        # This will increase by one for each set quality bit.
        quality_count = [0, 0, 0, 0, 0, 0, 0, 0]
        record_length = info['record_length']
        # Loop over all records.
        for _i in xrange(info['number_of_records']):
            # Skip non data records.
            data = mseedfile.read(39)
            if data[6] != 'D':
                continue
            # Read data quality byte.
            data_quality_flags = data[38]
            # Jump to next byte.
            mseedfile.seek(record_length - 39, 1)
            # Unpack the binary data.
            data_quality_flags = unpack('B', data_quality_flags)[0]
            # Add the value of each bit to the quality_count.
            for _j in xrange(8):
                if (data_quality_flags & (1 << _j)) != 0:
                    quality_count[_j] += 1
        mseedfile.close()
        return quality_count

    def getTimingQuality(self, filename, first_record=True,
                         rl_autodetection= -1):
        """
        Reads timing quality and returns a dictionary containing statistics
        about it.
        
        This method will read the timing quality in Blockette 1001 for each
        record in the file if available and return the following statistics:
        Minima, maxima, average, median and upper and lower quantile.
        
        It is probably pretty safe to set the first_record parameter to True
        because the timing quality is a vendor specific value and thus it will
        probably be set for each record or for none.
        
        The method to calculate the quantiles uses a integer round outwards
        policy: lower quantiles are rounded down (probability < 0.5), and upper
        quantiles (probability > 0.5) are rounded up.
        This gives no more than the requested probability in the tails, and at
        least the requested probability in the central area.
        The median is calculating by either taking the middle value or, with an
        even numbers of values, the average between the two middle values.
        
        :param filename: Mini-SEED file to be parsed.
        :param first_record: Determines whether all records are assumed to 
            either have a timing quality in Blockette 1001 or not depending on
            whether the first records has one. If True and the first records
            does not have a timing quality it will not parse the whole file. If
            False is will parse the whole file anyway and search for a timing
            quality in each record. Defaults to True.
        :param rl_autodetection: Determines the auto-detection of the record
            lengths in the file. If 0 only the length of the first record is
            detected automatically. All subsequent records are then assumed
            to have the same record length. If -1 the length of each record
            is automatically detected. Defaults to -1.
        """
        # Get some information about the file.
        fp = open(filename, 'rb')
        fileinfo = self._getMSFileInfo(fp, filename)
        fp.close()
        ms = _MSStruct(filename, init_msrmsf=False)
        # Create Timing Quality list.
        data = []
        # Loop over each record
        for _i in xrange(fileinfo['number_of_records']):
            # Loop over every record.
            ms.read(rl_autodetection, 0, 0, 0)
            # Enclose in try-except block because not all records need to
            # have Blockette 1001.
            try:
                # Append timing quality to list.
                tq = ms.msr.contents.Blkt1001.contents.timing_qual
                data.append(float(tq))
            except:
                if first_record:
                    break
        # Deallocate for debugging with valgrind
        del ms
        # Length of the list.
        n = len(data)
        data = sorted(data)
        # Create new dictionary.
        result = {}
        # If no data was collected just return an empty list.
        if n == 0:
            return result
        # Calculate some statistical values.
        result['min'] = min(data)
        result['max'] = max(data)
        result['average'] = sum(data) / n
        data = sorted(data)
        result['median'] = scoreatpercentile(data, 50, issorted=False)
        result['lower_quantile'] = scoreatpercentile(data, 25, issorted=False)
        result['upper_quantile'] = scoreatpercentile(data, 75, issorted=False)
        return result

    def _bytePosFromTime(self, filename, starttime=None, endtime=None):
        """
        Return start and end byte position from mseed file.
        
        The method takes a MiniSEED file and tries to match it as good as
        possible to the supplied time range. It will simply return the byte
        position of records that are within the time range. The byte
        position of the record that covers the start time will be the first
        byte position. The byte length will be until the record that covers
        the end time.
        
        This method will only work correctly for files containing only traces
        from one single source. All traces have to be in chronological order.
        Also all records in the file need to have the same length in bytes.
        
        It will return an empty string if the file does not cover the desired
        range.
        
        :return: Byte position of beginning and total length of bytes
        
        :param filename: File string of the MiniSEED file to be cut.
        :param starttime: :class:`~obspy.core.utcdatetime.UTCDateTime` object.
        :param endtime: :class:`~obspy.core.utcdatetime.UTCDateTime` object.
        """
        # Read the start and end time of the file.
        ms = _MSStruct(filename)
        info = ms.fileinfo()
        start = ms.getStart()
        pos = (info['number_of_records'] - 1) * info['record_length']
        ms.offset = pos
        end = ms.getEnd()
        # Set the start time.
        if not starttime or starttime <= start:
            starttime = start
        # Set the end time.
        if not endtime or endtime >= end:
            endtime = end
        # Deallocate msr and msf memory for wrong input
        if starttime >= end or endtime <= start:
            del ms # for valgrind
            return None
        # Guess the most likely records that cover start- and end time.
        nr = info['number_of_records']
        start_record = int((starttime - start) / (end - start) * nr)
        end_record = int((endtime - start) / (end - start) * nr) + 1
        # Loop until the correct start_record is found
        delta = 0
        while True:
            # check boundaries
            if start_record < 0:
                start_record = 0
                break
            elif start_record > nr - 1:
                start_record = nr - 1
                break
            ms.offset = start_record * info['record_length']
            stime = ms.getStart()
            # Calculate last covered record.
            ms.offset = ms.offset + 30
            # Calculate sample rate.
            sample_rate = ms.msr.contents.samprate
            npts = ms.msr.contents.samplecnt
            # Calculate time of the first sample of new record
            etime = stime + ((npts - 1) / sample_rate)
            # Leave loop if correct record is found or change record number
            # otherwise. 
            if starttime >= stime and starttime <= etime:
                break
            elif delta == -1 and starttime > etime:
                break
            elif delta == 1 and starttime < stime:
                start_record += 1
                break
            elif starttime < stime:
                delta = -1
            else:
                delta = 1
            start_record += delta
        # Loop until the correct end_record is found
        delta = 0
        while True:
            # check boundaries
            if end_record < 0:
                end_record = 0
                break
            elif end_record > nr - 1:
                end_record = nr - 1
                break
            ms.offset = end_record * info['record_length']
            stime = ms.getStart()
            # Calculate last covered record.
            ms.offset = ms.offset + 30
            # Calculate sample rate.
            sample_rate = ms.msr.contents.samprate
            npts = ms.msr.contents.samplecnt
            # The time of the last covered sample is now:
            etime = stime + ((npts - 1) / sample_rate)
            # Leave loop if correct record is found or change record number
            # otherwise.
            if endtime >= stime and endtime <= etime:
                break
            elif delta == -1 and endtime > etime:
                end_record += 1
                break
            elif delta == 1 and endtime < stime:
                break
            elif endtime < stime:
                delta = -1
            else:
                delta = 1
            end_record += delta
        del ms # for valgrind
        # Calculate starting position
        record_length = info['record_length']
        start_byte = record_length * start_record
        # length in bytes to read
        length_byte = record_length * (end_record - start_record + 1)
        return start_byte, length_byte

    def unpack_steim2(self, data_string, npts, swapflag=0, verbose=0):
        """
        Unpack steim2 compressed data given as string.
        
        :param data_string: data as string
        :param npts: number of data points
        :param swapflag: Swap bytes, defaults to 0
        :return: Return data as numpy.ndarray of dtype int32
        """
        dbuf = data_string
        datasize = len(dbuf)
        samplecnt = npts
        datasamples = np.empty(npts , dtype='int32')
        diffbuff = np.empty(npts , dtype='int32')
        x0 = C.c_int32()
        xn = C.c_int32()
        nsamples = clibmseed.msr_unpack_steim2(\
                C.cast(dbuf, C.POINTER(FRAME)), datasize,
                samplecnt, samplecnt, datasamples, diffbuff,
                C.byref(x0), C.byref(xn), swapflag, verbose)
        if nsamples != npts:
            raise Exception("Error in unpack_steim2")
        return datasamples

    def unpack_steim1(self, data_string, npts, swapflag=0, verbose=0):
        """
        Unpack steim1 compressed data given as string.
        
        :param data_string: data as string
        :param npts: number of data points
        :param swapflag: Swap bytes, defaults to 0
        :return: Return data as numpy.ndarray of dtype int32
        """
        dbuf = data_string
        datasize = len(dbuf)
        samplecnt = npts
        datasamples = np.empty(npts , dtype='int32')
        diffbuff = np.empty(npts , dtype='int32')
        x0 = C.c_int32()
        xn = C.c_int32()
        nsamples = clibmseed.msr_unpack_steim1(\
                C.cast(dbuf, C.POINTER(FRAME)), datasize,
                samplecnt, samplecnt, datasamples, diffbuff,
                C.byref(x0), C.byref(xn), swapflag, verbose)
        if nsamples != npts:
            raise Exception("Error in unpack_steim1")
        return datasamples

    def _ctypesArray2NumpyArray(self, buffer, buffer_elements, sampletype):
        """
        Takes a Ctypes array and its length and type and returns it as a
        NumPy array.
        
        This works by reference and no data is copied.
        
        :param buffer: Ctypes c_void_p pointer to buffer.
        :param buffer_elements: length of the whole buffer
        :param sampletype: type of sample, on of "a", "i", "f", "d"
        """
        # Allocate NumPy array to move memory to
        numpy_array = np.empty(buffer_elements, dtype=sampletype)
        datptr = numpy_array.ctypes.get_data()
        # Manually copy the contents of the C allocated memory area to
        # the address of the previously created NumPy array
        C.memmove(datptr, buffer, buffer_elements * SAMPLESIZES[sampletype])
        return numpy_array

    def _convertDatetimeToMSTime(self, dt):
        """
        Takes obspy.util.UTCDateTime object and returns an epoch time in ms.
        
        :param dt: obspy.util.UTCDateTime object.
        """
        return int(dt.timestamp * HPTMODULUS)

    def _convertMSTimeToDatetime(self, timestring):
        """
        Takes Mini-SEED timestamp and returns a obspy.util.UTCDateTime object.
        
        :param timestamp: Mini-SEED timestring (Epoch time string in ms).
        """
        return UTCDateTime(timestring / HPTMODULUS)

    def _MSRId(self, header):
        ids = ['network', 'station', 'location', 'channel',
                'samprate', 'sampletype']
        return ".".join([str(header[_i]) for _i in ids])

    def _convertMSRToDict(self, m):
        h = {}
        attributes = ('network', 'station', 'location', 'channel',
                      'dataquality', 'starttime', 'samprate',
                      'samplecnt', 'numsamples', 'sampletype')
        # loop over attributes
        for _i in attributes:
            h[_i] = getattr(m, _i)
        return h

    def _convertMSTToDict(self, m):
        """
        Return dictionary from MSTrace Object m, leaving the attributes
        datasamples, ststate and next out
        
        :param m: MST structure to be read.
        """
        h = {}
        # header attributes to be converted
        attributes = ('network', 'station', 'location', 'channel',
                      'dataquality', 'starttime', 'endtime', 'samprate',
                      'samplecnt', 'numsamples', 'sampletype')
        # loop over attributes
        for _i in attributes:
            h[_i] = getattr(m, _i)
        return h

    def _convertDictToMST(self, m, h):
        """
        Takes dictionary containing MSTrace header data and writes them to the
        MSTrace Group
        
        :param m: MST structure to be modified.
        :param h: Dictionary containing all necessary information.
        """
        chain = m.contents
        h['type'] = '\x00'
        # header attributes to be converted
        attributes = ('network', 'station', 'location', 'channel',
                      'dataquality', 'type', 'starttime', 'endtime',
                      'samprate', 'samplecnt', 'numsamples', 'sampletype')
        # loop over attributes
        for attr in attributes:
            setattr(chain, attr, h[attr])

    def _getMSFileInfo(self, f, real_name):
        """
        Takes a Mini-SEED filename as an argument and returns a dictionary
        with some basic information about the file. Also suiteable for Full
        SEED.
        
        :param f: File pointer of opened file in binary format
        :param real_name: Realname of the file, needed for calculating size
        """
        # get size of file
        info = {'filesize': os.path.getsize(real_name)}
        pos = f.tell()
        f.seek(0)
        rec_buffer = f.read(512)
        info['record_length'] = clibmseed.ms_detect(rec_buffer, 512)
        # Calculate Number of Records
        info['number_of_records'] = long(info['filesize'] // \
                                         info['record_length'])
        info['excess_bytes'] = info['filesize'] % info['record_length']
        f.seek(pos)
        return info

    def _isRateTolerable(self, sr1, sr2):
        """
        Tests default sample rate tolerance: abs(1-sr1/sr2) < 0.0001
        """
        return math.fabs(1.0 - (sr1 / float(sr2))) < 0.0001

    def _populateMSTG(self, trace):
        """
        Populates MSTrace_Group structure from given header, data and
        numtraces and returns the MSTrace_Group
        
        :param trace: Trace.
        """
        # Initialize MSTraceGroup
        mstg = clibmseed.mst_initgroup(None)
        # Set numtraces.
        mstg.contents.numtraces = 1
        # Define starting point of the MSTG structure and the same as a string.
        # Initialize MSTrace object and connect with group
        mstg.contents.traces = clibmseed.mst_init(None)
        chain = mstg.contents.traces
        # fetch encoding options
        sampletype = trace[0]['sampletype']
        c_dtype = DATATYPES[sampletype]
        # Create variable with the number of samples in this trace for
        # faster future access.
        npts = trace[0]['numsamples']
        # Write header in MSTrace structure
        self._convertDictToMST(chain, trace[0])
        # Create a single datapoint and resize its memory to be able to
        # hold all datapoints.
        tempdatpoint = c_dtype()
        datasize = SAMPLESIZES[sampletype] * npts
        # XXX: Ugly workaround for bug writing ASCII.
        if sampletype == 'a' and datasize < 17:
            datasize = 17
        C.resize(tempdatpoint, datasize)
        # The datapoints in the MSTG structure are a pointer to the memory
        # area reserved for tempdatpoint.
        chain.contents.datasamples = C.cast(C.pointer(tempdatpoint),
                                            C.c_void_p)
        # Swap if wrong byte order
        if trace[1].dtype.byteorder != "=":
            trace[1] = trace[1].byteswap()
        # Pointer to the NumPy data buffer.
        datptr = trace[1].ctypes.get_data()
        # Manually move the contents of the NumPy data buffer to the
        # address of the previously created memory area.
        C.memmove(chain.contents.datasamples, datptr, datasize)
        return mstg


class _MSStruct(object):
    """
    Class for handling MSRecord and MSFileparam.

    It consists of a MSRecord and MSFileparam and an attached python file
    pointer.

    :ivar msr: MSRecord
    :ivar msf: MSFileparam
    :ivar file: filename
    :ivar offset: Current offset

    :param filename: file to attach to
    :param init_msrmsf: initialize msr and msf structure
        by a first pass of read. Setting this option to
        false will result in errors when setting e.g.
        the offset before a call to read
    """
    def __init__(self, filename, init_msrmsf=True):
        # Initialize MSRecord structure
        self.msr = clibmseed.msr_init(C.POINTER(MSRecord)())
        self.msf = C.POINTER(MSFileParam)() # null pointer
        self.file = filename
        # dummy read once, to avoid null pointer in ms.msf for e.g.
        # ms.offset
        if init_msrmsf:
            self.read(-1, 0, 1, 0)
            self.offset = 0

    def getEnd(self):
        """
        Return endtime
        """
        self.read(-1, 0, 1, 0)
        dtime = clibmseed.msr_endtime(self.msr)
        return UTCDateTime(dtime / HPTMODULUS)

    def getStart(self):
        """
        Return starttime
        """
        self.read(-1, 0, 1, 0)
        dtime = clibmseed.msr_starttime(self.msr)
        return UTCDateTime(dtime / HPTMODULUS)

    def fileinfo(self):
        """
        For details see libmseed._getMSFileInfo
        """
        fp = open(self.file, 'rb')
        self.info = LibMSEED()._getMSFileInfo(fp, self.file)
        fp.close()
        return self.info

    def filePosFromRecNum(self, record_number=0):
        """
        Return byte position of file given a certain record_number.

        The byte position can be used to seek to certain points in the file
        """
        if not hasattr(self, 'info'):
            self.info = self.fileinfo()
        # Calculate offset of the record to be read.
        if record_number < 0:
            record_number = self.info['number_of_records'] + record_number
        if record_number < 0 or \
           record_number >= self.info['number_of_records']:
            raise ValueError('Please enter a valid record_number')
        return record_number * self.info['record_length']

    def read(self, reclen= -1, dataflag=1, skipnotdata=1, verbose=0,
             raise_flag=True):
        """
        Read MSRecord using the ms_readmsr_r function. The following
        parameters are directly passed to ms_readmsr_r.
        
        :param ms: _MSStruct (actually consists of a LP_MSRecord,
            LP_MSFileParam and an attached file pointer). 
            Given an existing ms the function is much faster.
        :param reclen: If reclen is 0 the length of the first record is auto-
            detected. All subsequent records are then expected to have the 
            same record length. If reclen is negative the length of every 
            record is automatically detected. Defaults to -1.
        :param dataflag: Controls whether data samples are unpacked, defaults 
            to 1.
        :param skipnotdata: If true (not zero) any data chunks read that to do 
            not have valid data record indicators will be skipped. Defaults to 
            True (1).
        :param verbose: Controls verbosity from 0 to 2. Defaults to None (0).
        :param record_number: Number of the record to be read. The first record
            has the number 0. Negative numbers will start counting from the end
            of the file, e.g. -1 is the last complete record.
        """
        errcode = clibmseed.ms_readmsr_r(C.pointer(self.msf),
                                         C.pointer(self.msr),
                                         self.file, reclen, None, None,
                                         skipnotdata, dataflag, verbose)
        if raise_flag:
            if errcode != MS_NOERROR:
                raise Exception("Error %d in ms_readmsr_r" % errcode)
        return errcode

    def __del__(self):
        """
        Method for deallocating MSFileParam and MSRecord structure.
        """
        errcode = clibmseed.ms_readmsr_r(C.pointer(self.msf), C.pointer(self.msr),
                                         None, -1, None, None, 0, 0, 0)
        if errcode != MS_NOERROR:
            raise Exception("Error %d in ms_readmsr_r" % errcode)

    def setOffset(self, value):
        self.msf.contents.readoffset = C.c_int(value)

    def getOffset(self):
        return int(self.msf.contents.readoffset)

    offset = property(getOffset, setOffset)


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
