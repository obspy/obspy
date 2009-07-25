# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# Filename: libmseed.py
#  Purpose: Python wrapper for libmseed of Chad Trabant
#   Author: Lion Krischer, Robert Barsch, Moritz Beyreuther
#    Email: krischer@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2010 Lion Krischer, Robert Barsch, Moritz Beyreuther
#---------------------------------------------------------------------
"""
Class for handling Mini-SEED files.

Contains wrappers for libmseed - The Mini-SEED library. The C library is
interfaced via python-ctypes. Currently only supports Mini-SEED files with
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
"""

from StringIO import StringIO
from obspy.core import UTCDateTime
from obspy.core.util import scoreatpercentile, c_file_p
from obspy.mseed.headers import MSRecord, MSTraceGroup, MSTrace, HPTMODULUS, \
    MSFileParam
from struct import unpack
import ctypes as C
import math
import numpy as N
import os
import platform
import sys


# Use different shared libmseed library depending on the platform.
# 32 bit Windows.
if sys.platform == 'win32':
    lib_name = 'libmseed-2.2.win32.dll'
# 32 bit OSX, tested with 10.5.6
elif sys.platform == 'darwin':
    lib_name = 'libmseed.dylib'
# 32 and 64 bit UNIX
#XXX Check glibc version by platform.libc_ver()
else:
    if platform.architecture()[0] == '64bit':
        lib_name = 'libmseed.lin64.so'
    else:
        lib_name = 'libmseed-2.1.7.so'
clibmseed = C.CDLL(os.path.join(os.path.dirname(__file__), 'libmseed',
                                lib_name))


class libmseed(object):
    """
    Class for handling Mini-SEED files.
    """

    def printFileInformation(self, filename):
        """
        Prints some informations about the file.
        
        @param filename: Mini-SEED file.
        """
        try:
            #Read Trace Group
            mstg = self.readFileToTraceGroup(str(filename), dataflag=0)
            clibmseed.mst_printtracelist(mstg, 1, 1, 1)
        except:
            print 'The file could not be read.'

    def isMSEED(self, filename):
        """
        Tests whether a file is a Mini-SEED file or not.
        
        Returns True on success or False otherwise.
        This method will just read the first record and not the whole file.
        Thus it cannot be used to validate a Mini-SEED file.
        
        @param filename: Mini-SEED file.
        """
        try:
            msr = self.readSingleRecordToMSR(filename, dataflag=0)
            del msr
            return True
        except:
            return False

    def readMSTraces(self, filename, reclen= -1, timetol= -1,
                     sampratetol= -1, dataflag=1, skipnotdata=1,
                     dataquality=1, verbose=0):
        """
        Read Mini-SEED file. Returns a list with header informations and data
        for each trace in the file.
        
        The list returned contains a list for each trace in the file with the
        lists first element being a header dictionary and its second element
        containing the data values as a numpy array.

        @param filename: Name of Mini-SEED file.
        @param reclen, timetol, sampratetol, dataflag, skipnotdata,
            dataquality, verbose: These are passed directly to the 
            readFileToTraceGroup method.
        """
        # Create empty list that will contain all traces.
        trace_list = []
        # Creates MSTraceGroup Structure and feed it with the Mini-SEED data.
        mstg = self.readFileToTraceGroup(str(filename), reclen=reclen,
                                         timetol=timetol,
                                         sampratetol=sampratetol,
                                         dataflag=dataflag,
                                         skipnotdata=skipnotdata,
                                         dataquality=dataquality,
                                         verbose=verbose)
        chain = mstg.contents.traces.contents
        numtraces = mstg.contents.numtraces
        # Loop over traces and append to trace_list.
        for _i in xrange(numtraces):
            header = self._convertMSTToDict(chain)
            # Access data directly as numpy array.
            data = self._accessCtypesArrayAsNumpyArray(chain.datasamples,
                                                       chain.numsamples)
            trace_list.append([header, data])
            # Set chain to next trace.
            if _i != numtraces - 1:
                chain = chain.next.contents
        return trace_list

    def writeMSTraces(self, trace_list, outfile, reclen= -1, encoding= -1,
                      byteorder= -1, flush= -1, verbose=0):
        """
        Write Miniseed file from trace_list
        
        @param trace_list: List containing header informations and data.
        @param outfile: Name of the output file
        @param reclen: should be set to the desired data record length in bytes
            which must be expressible as 2 raised to the power of X where X is
            between (and including) 8 to 20. -1 defaults to 4096
        @param encoding: should be set to one of the following supported
            Mini-SEED data encoding formats: DE_ASCII (0), DE_INT16 (1),
            DE_INT32 (3), DE_FLOAT32 (4), DE_FLOAT64 (5), DE_STEIM1 (10)
            and DE_STEIM2 (11). -1 defaults to STEIM-2 (11)
        @param byteorder: must be either 0 (LSBF or little-endian) or 1 (MBF or 
            big-endian). -1 defaults to big-endian (1)
        @param flush: if it is not zero all of the data will be packed into 
            records, otherwise records will only be packed while there are
            enough data samples to completely fill a record.
        @param verbose: controls verbosity, a value of zero will result in no 
            diagnostic output.
        """
        # Populate MSTG Structure
        mstg = self._populateMSTG(trace_list)
        # Write File and loop over every trace in the MSTraceGroup Structure.
        numtraces = mstg.contents.numtraces
        openfile = open(outfile, 'wb')
        chain = mstg.contents.traces
        for _i in xrange(numtraces):
            self._packMSTToFile(chain, openfile, reclen, encoding, byteorder,
                                flush, verbose)
            if _i != numtraces - 1:
                chain = chain.contents.next
        openfile.close()

    def readFileToTraceGroup(self, filename, reclen= -1, timetol= -1,
                             sampratetol= -1, dataflag=1, skipnotdata=1,
                             dataquality=1, verbose=0):
        """
        Reads Mini-SEED data from file. Returns MSTraceGroup structure.
        
        @param filename: Name of Mini-SEED file.
        @param reclen: If reclen is 0 the length of the first record is auto-
            detected. All subsequent records are then expected to have the
            same record length. If reclen is negative the length of every
            record is automatically detected. Defaults to -1.
        @param timetol: Time tolerance, default to -1 (1/2 sample period).
        @param sampratetol: Sample rate tolerance, defaults to -1 (rate
            dependent)
        @param dataflag: Controls whether data samples are unpacked, defaults
            to true (0).
        @param skipnotdata: If true (not zero) any data chunks read that to do
            not have valid data record indicators will be skipped. Defaults to
            true (1).
        @param dataquality: If the dataquality flag is true traces will be
            grouped by quality in addition to the source name identifiers.
            Defaults to true (1).
        @param verbose: Controls verbosity from 0 to 2. Defaults to None (0).
        """
        # Creates MSTraceGroup Structure
        mstg = C.pointer(MSTraceGroup())
        # Uses libmseed to read the file and populate the MSTraceGroup
        errcode = clibmseed.ms_readtraces(
            C.pointer(mstg), str(filename), C.c_int(reclen),
            C.c_double(timetol), C.c_double(sampratetol),
            C.c_short(dataquality), C.c_short(skipnotdata),
            C.c_short(dataflag), C.c_short(verbose))
        if errcode != 0:
            assert 0, "\n\nError while reading Mini-SEED file: " + filename
        return mstg

    def readSingleRecordToMSR(self, filename, reclen= -1, dataflag=1,
                              skipnotdata=1, verbose=0, record_number=0):
        """
        Reads Mini-SEED record from file and populates MS Record data structure.
        
        @param filename: Mini-SEED file to be read.
        @param reclen: If reclen is 0 the length of the first record is auto-
            detected. All subsequent records are then expected to have the 
            same record length. If reclen is negative the length of every 
            record is automatically detected. Defaults to -1.
        @param dataflag: Controls whether data samples are unpacked, defaults 
            to 1.
        @param skipnotdata: If true (not zero) any data chunks read that to do 
            not have valid data record indicators will be skipped. Defaults to 
            True (1).
        @param verbose: Controls verbosity from 0 to 2. Defaults to None (0).
        @param record_number: Number of the record to be read. The first record
            has the number 0. Negative numbers will start counting from the end
            of the file, e.g. -1 is the last complete record.
        """
        # Get some information about the file.
        fileinfo = self._getMSFileInfo(filename)
        # Calculate offset of the record to be read.
        if record_number < 0:
            record_number = fileinfo['number_of_records'] + record_number
        if record_number < 0 or record_number >= fileinfo['number_of_records']:
            raise ValueError('Please enter a valid record_number')
        filepos = record_number * fileinfo['record_length']
        # init MSRecord structure
        clibmseed.msr_init.restype = C.POINTER(MSRecord)
        msr = clibmseed.msr_init(None)
        # defines return type
        clibmseed.ms_readmsr_r.restype = C.c_int
        # Init MSFileParam struct.
        FileParam = C.pointer(MSFileParam())
        # Open file and jump to the start of the record to be read.
        ff = open(filename, 'rb')
        ff.seek(filepos)
        fpp = self._convertToCFilePointer(ff)
        # Populate FileParam structure.
        FP_chain = FileParam.contents
        FP_chain.fp = fpp
        FP_chain.filepos = filepos
        FP_chain.filename = filename
        FP_chain.rawrec = None
        FP_chain.readlen = 256
        FP_chain.autodet = 1
        FP_chain.packtype = 0
        FP_chain.packhdroffset = 0
        FP_chain.recordcount = 0
        # Populate the MSRecord structure with the help of libmseed.
        clibmseed.ms_readmsr_r(C.pointer(FileParam), C.pointer(msr),
                               str(filename), C.c_int(reclen), None, None,
                               C.c_short(skipnotdata), C.c_short(dataflag),
                               C.c_short(verbose))
        # Clean up memory and close all open files.
        del fpp
        del FileParam
        ff.close()
        return msr

    def getFirstRecordHeaderInfo(self, filename):
        """
        Takes a Mini-SEED file and returns header of the first record.
        
        Returns a dictionary containing some header information from the first
        record of the Mini-SEED file only. It returns the location, network,
        station and channel information.
        
        @param filename: Mini-SEED file string.
        """
        # read first header only
        msr = self.readSingleRecordToMSR(filename, dataflag=0)
        header = {}
        chain = msr.contents
        # header attributes to be read
        attributes = ('location', 'network', 'station', 'channel')
        # loop over attributes
        for _i in attributes:
            header[_i] = getattr(chain, _i)
        return header

    def getStartAndEndTime(self, filename):
        """
        Returns the start- and endtime of a Mini-SEED file as a tuple
        containing two datetime objects.
        
        This method only reads the first and the last record. Thus it will only
        work correctly for files containing only one trace with all records
        in the correct order.
        
        The returned endtime is the time of the last datasample and not the
        time that the last sample covers.
        
        @param filename: Mini-SEED file string.
        """
        first_record = self.readSingleRecordToMSR(filename, dataflag=0)
        # Get the starttime using the libmseed method msr_starttime
        clibmseed.msr_starttime.restype = C.c_int64
        starttime = clibmseed.msr_starttime(first_record)
        starttime = self._convertMSTimeToDatetime(starttime)
        #Read last record.
        last_record = self.readSingleRecordToMSR(filename, dataflag=0,
                                      record_number= -1)
        # Get the endtime using the libmseed method msr_endtime
        clibmseed.msr_endtime.restype = C.c_int64
        endtime = clibmseed.msr_endtime(last_record)
        endtime = self._convertMSTimeToDatetime(endtime)
        return(starttime, endtime)

    def getGapList(self, filename, time_tolerance= -1,
                   samprate_tolerance= -1, min_gap=None, max_gap=None):
        """
        Returns gaps, overlaps and trace header information of a given file.
        
        Each item has a starttime and a duration value to characterize the gap.
        The starttime is the last correct data sample. If no gaps are found it
        will return an empty list.
        
        @param time_tolerance: Time tolerance while reading the traces, default 
            to -1 (1/2 sample period).
        @param samprate_tolerance: Sample rate tolerance while reading the 
            traces, defaults to -1 (rate dependent).
        @param min_gap: Omit gaps with less than this value if not None. 
        @param max_gap: Omit gaps with greater than this value if not None.
        @return: List of tuples in form of (network, station, location, 
            channel, starttime, endtime, gap, samples) 
        """
        # read file
        mstg = self.readFileToTraceGroup(filename, dataflag=0,
                                         skipnotdata=0,
                                         timetol=time_tolerance,
                                         sampratetol=samprate_tolerance)
        # iterate through traces
        cur = mstg.contents.traces.contents
        gap_list = [[self._convertMSTToDict(cur),None]]
        for _ in xrange(mstg.contents.numtraces - 1):
            next = cur.next.contents
            # Skip MSTraces with 0 sample rate, usually from SOH records
            if cur.samprate == 0:
                cur = next
                continue
            # Check that sample rates match using default tolerance
            if not self._isRateTolerable(cur.samprate, next.samprate):
                msg = "%s: Sample rate changed! %.10g -> %.10g\n"
                print msg % (filename, cur.samprate, next.samprate)
            gap = (next.starttime - cur.endtime) / HPTMODULUS
            # Check that any overlap is not larger than the trace coverage
            if gap < 0:
                if next.samprate:
                    delta = 1 / float(next.samprate)
                else:
                    delta = 0
                temp = (next.endtime - next.starttime) / HPTMODULUS + delta
                if (gap * -1) > temp:
                    gap = -1 * temp
            # Check gap/overlap criteria
            if min_gap and gap < min_gap:
                cur = next
                continue
            if max_gap and gap > max_gap:
                cur = next
                continue
            # Number of missing samples
            nsamples = math.fabs(gap) * cur.samprate
            if gap > 0:
                nsamples -= 1
            else:
                nsamples += 1
            # Convert to python datetime objects
            time1 = UTCDateTime.utcfromtimestamp(cur.endtime / HPTMODULUS)
            time2 = UTCDateTime.utcfromtimestamp(next.starttime / HPTMODULUS)
            #gap_list.append((cur.network, cur.station, cur.location,
            #                 cur.channel, time1, time2, gap, nsamples))
            _head = self._convertMSTToDict(cur)
            _head.update({"lastsamp":time1,"nextsamp":time2,
                          "gap":gap,"totsamp":nsamples})
            gap_list.append([_head,None]) # stay conform with readMSTraces
            cur = next
        return gap_list

    def printGapList(self, filename, time_tolerance= -1,
                     samprate_tolerance= -1, min_gap=None, max_gap=None):
        """
        Print gap/overlap list summary information for the given filename.
        """
        result = self.getGapList(filename, time_tolerance, samprate_tolerance,
                                 min_gap, max_gap)
        print "%-17s %-26s %-26s %-5s %-8s" % ('Source', 'Last Sample',
                                               'Next Sample', 'Gap', 'Samples')
        for _r in result:
            #print "%-17s %-26s %-26s %-5s %-.8g" % ('_'.join(r[0:4]),
            #                                        r[4].isoformat(),
            #                                        r[5].isoformat(),
            #                                        r[6], r[7])
            r = _r[0] # stay conform with readMSTraces
            print "%-5s_%-5s_%-5s_%-5s %-26s %-26s %-5s %-.8g" % (r['network'],
                r['station'], r['location'], r['channel'],
                r['lastsamp'].isoformat(), r['nextsamp'].isoformat(),
                r['totsamp'])
        print "Total: %d gap(s)" % len(result)

    def getDataQualityFlagsCount(self, filename):
        """
        Counts all data quality flags of the given Mini-SEED file.
        
        This method will count all set data quality flag bits in the fixed
        section of the data header in a Mini-SEED file and returns the total
        count for each flag type.
        
        Data quality flags:
          [Bit 0] - Amplifier saturation detected (station dependent)
          [Bit 1] - Digitizer clipping detected
          [Bit 2] - Spikes detected
          [Bit 3] - Glitches detected
          [Bit 4] - Missing/padded data present
          [Bit 5] - Telemetry synchronization error
          [Bit 6] - A digital filter may be charging
          [Bit 7] - Time tag is questionable
        
        This will only work correctly if each record in the file has the same
        record length.
        
        @param filename: Mini-SEED file name.
        @return: List of all flag counts.
        """
        # Get record length of the file.
        info = self._getMSFileInfo(filename)
        # Open the file.
        mseedfile = open(filename, 'rb')
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
        
        @param filename: Mini-SEED file to be parsed.
        @param first_record: Determines whether all records are assumed to 
            either have a timing quality in Blockette 1001 or not depending on
            whether the first records has one. If True and the first records
            does not have a timing quality it will not parse the whole file. If
            False is will parse the whole file anyway and search for a timing
            quality in each record. Defaults to True.
        @param rl_autodetection: Determines the auto-detection of the record
            lengths in the file. If 0 only the length of the first record is
            detected automatically. All subsequent records are then assumed
            to have the same record length. If -1 the length of each record
            is automatically detected. Defaults to -1.
        """
        # Get some information about the file.
        fileinfo = self._getMSFileInfo(filename)
        # init MSRecord structure
        clibmseed.msr_init.restype = C.POINTER(MSRecord)
        msr = clibmseed.msr_init(None)
        # defines return type
        clibmseed.ms_readmsr_r.restype = C.c_int
        # Init MSFileParam struct.
        FileParam = C.pointer(MSFileParam())
        # Open file.
        ff = open(filename, 'rb')
        # Convert to filepointer
        fpp = self._convertToCFilePointer(ff)
        # Populate FileParam structure.
        FP_chain = FileParam.contents
        FP_chain.fp = fpp
        FP_chain.filepos = 0
        FP_chain.filename = filename
        FP_chain.rawrec = None
        FP_chain.readlen = 256
        FP_chain.autodet = 1
        FP_chain.packtype = 0
        FP_chain.packhdroffset = 0
        FP_chain.recordcount = 0
        # Create Timing Quality list.
        data = []
        # Loop over each record
        for _i in xrange(fileinfo['number_of_records']):
            # Loop over every record.
            clibmseed.ms_readmsr_r(C.pointer(FileParam), C.pointer(msr),
                                   str(filename), C.c_int(rl_autodetection),
                                   None, None, C.c_short(1), C.c_short(0),
                                   C.c_short(0))
            # Enclose in try-except block because not all records need to
            # have Blockette 1001.
            try:
                # Append timing quality to list.
                data.append(float(msr.contents.Blkt1001.contents.timing_qual))
            except:
                if first_record:
                    break
        # Clean up and close file.
        del msr
        del fpp
        del FileParam
        ff.close()
        # Create new dictionary.
        result = {}
        # Length of the list.
        n = len(data)
        data = sorted(data)
        # If no data was collected just return an empty list.
        if n == 0:
            return result
        # Calculate some statistical values.
        result['min'] = min(data)
        result['max'] = max(data)
        result['average'] = sum(data) / n
        data = sorted(data)
        result['median'] = scoreatpercentile(data, 50, sort=False)
        result['lower_quantile'] = scoreatpercentile(data, 25, sort=False)
        result['upper_quantile'] = scoreatpercentile(data, 75, sort=False)
        return result

    def cutMSFileByRecords(self, filename, starttime=None, endtime=None):
        """
        Cuts a Mini-SEED file by cutting at records.
        
        The method takes a Mini-SEED file and tries to match it as good as
        possible to the supplied time range. It will simply cut off records
        that are not within the time range. The record that covers the
        start time will be the first record and the one that covers the 
        end time will be the last record.
        
        This method will only work correctly for files containing only traces
        from one single source. All traces have to be in chronological order.
        Also all records in the file need to have the same length.
        
        It will return an empty string if the file does not cover the desired
        range.
        
        @return: Byte string containing the cut file.
        
        @param filename: File string of the Mini-SEED file to be cut.
        @param starttime: obspy.util.UTCDateTime object.
        @param endtime: obspy.util.UTCDateTime object.
        """
        # Read the start and end time of the file.
        (start, end) = self.getStartAndEndTime(filename)
        # Set the start time.
        if not starttime or starttime <= start:
            starttime = start
        elif starttime >= end:
            return ''
        # Set the end time.
        if not endtime or endtime >= end:
            endtime = end
        elif endtime <= start:
            return ''
        # Guess the most likely records that cover start- and end time.
        info = self._getMSFileInfo(filename)
        nr = info['number_of_records']
        start_record = int((starttime.timestamp - start.timestamp) /
                           (end.timestamp - start.timestamp) * nr)
        end_record = int((endtime.timestamp - start.timestamp) /
                         (end.timestamp - start.timestamp) * nr) + 1
        # Loop until the correct start_record is found
        while True:
            # check boundaries
            if start_record < 0:
                start_record = 0
                break
            elif start_record > nr - 1:
                start_record = nr - 1
                break
            msr = self.readSingleRecordToMSR(filename, dataflag=0,
                                             record_number=start_record)
            chain = msr.contents
            stime = chain.starttime
            # Calculate last covered record.
            etime = self._convertMSTimeToDatetime(stime + ((chain.samplecnt - \
                            1) / chain.samprate) * HPTMODULUS)
            stime = self._convertMSTimeToDatetime(stime)
            # Leave loop if correct record is found or change record number
            # otherwise.
            if starttime >= stime and starttime <= etime:
                break
            elif starttime <= stime:
                start_record -= 1
            else:
                start_record += 1
        # Loop until the correct end_record is found
        while True:
            # check boundaries
            if end_record < 0:
                end_record = 0
                break
            elif end_record > nr - 1:
                end_record = nr - 1
                break
            msr = self.readSingleRecordToMSR(filename, dataflag=0,
                                             record_number=end_record)
            chain = msr.contents
            stime = chain.starttime
            # Calculate last covered record.
            etime = self._convertMSTimeToDatetime(stime + ((chain.samplecnt - \
                            1) / chain.samprate) * HPTMODULUS)
            stime = self._convertMSTimeToDatetime(stime)
            # Leave loop if correct record is found or change record number
            # otherwise.
            if endtime >= stime and endtime <= etime:
                break
            elif endtime <= stime:
                end_record -= 1
            else:
                end_record += 1
        # Open the file and read the cut file.
        fh = open(filename, 'rb')
        record_length = info['record_length']
        # Jump to starting location.
        fh.seek(record_length * start_record, 0)
        # Read until end_location.
        data = fh.read(record_length * (end_record - start_record + 1))
        fh.close()
        # Return the cut file string.
        return data

    def mergeAndCutMSFiles(self, file_list, starttime=None, endtime=None):
        """
        This method takes several Mini-SEED files and returns one merged file.
        
        It is also possible to specify a start- and a endtime and all records
        that are out of bounds will be cut.
        If two not identical files cover a common time frame they will still
        be merged and no data is lost.
        
        The input files can be given in any order but they have to be files
        that contain only traces from one source and one component and the
        traces inside the files have to be in chronological order. Otherwise
        the produced output will not be correct. All files also have to be from
        the same source.
        
        @param file_list: A list containing Mini-SEED filename strings.
        @param outfile: String of the file to be created.
        @param starttime: obspy.util.UTCDateTime object.
        @param endtime: obspy.util.UTCDateTime object.
        @return: Byte string containing the merged and cut file.
        """
        # Copy file_list to not alter the provided list.
        file_list = file_list[:]
        # Remove duplicates in list
        file_list = list(set(file_list))
        # Check if all files in the list are from one source. Raise ValueError
        # otherwise.
        check_list = [self.getFirstRecordHeaderInfo(filename) for filename in\
                      file_list]
        for _i in range(len(check_list) - 1):
            if check_list[_i] != check_list[_i + 1]:
                raise ValueError
        # Get the start- and the endtime for each file in filelist.
        file_list = [[filename, self.getStartAndEndTime(filename)] for \
                     filename in file_list]
        # Sort the list first by endtime and then by starttime. This results
        # in a list which is sorted by starttime first and then by endtime.
        file_list.sort(cmp=lambda x, y: int(self._convertDatetimeToMSTime(\
                       x[1][1]) - self._convertDatetimeToMSTime(y[1][1])))
        file_list.sort(cmp=lambda x, y: int(self._convertDatetimeToMSTime(\
                       x[1][0]) - self._convertDatetimeToMSTime(y[1][0])))
        # Set start- and endtime if they have not been set.
        if not starttime:
            starttime = file_list[0][1][0]
        if not endtime:
            endtime = max([file[1][1] for file in file_list])
        open_file = StringIO()
        try:
            # Loop over all files in file_list and append to final output file.
            for file in file_list:
                file_starttime = file[1][0]
                file_endtime = file[1][1]
                # If the starttime of the file is in range of the desired file.
                if file_starttime >= starttime and file_starttime <= endtime:
                    # If the whole file is inside the range, just append it.
                    if file_endtime <= endtime:
                        new_file = open(file[0], 'rb')
                        open_file.write(new_file.read())
                        new_file.close()
                    # Otherwise cut it.
                    else:
                        open_file.write(self.cutMSFileByRecords(filename=\
                                        file[0], starttime=starttime,
                                        endtime=endtime))
                # If some parts of the file are in range cut it. Neglect all
                # other cases as they are not necessary.
                elif file_starttime < starttime and file_endtime > starttime:
                    open_file.write(self.cutMSFileByRecords(filename=file[0],
                                    starttime=starttime, endtime=endtime))
            # Close the open file
            open_file.seek(0)
            return open_file.read()
        # Handle non existing files and files of the wrong type.
        except IOError, error:
            # Close file and remove the already written file.
            open_file.close()
            # Write to standard error.
            sys.stderr.write(str(error) + '\n')
            sys.stderr.write('No file has been written.\n')
            sys.stderr.write('Please check your files and try again.\n')

    def _accessCtypesArrayAsNumpyArray(self, buffer, buffer_elements):
        """
        Takes a Ctypes c_int32 array and its length and returns it as a numpy
        array.
        
        This works by reference and no data is copied.
        
        @param buffer: Ctypes c_int32 buffer.
        @param buffer_elements: length of the buffer
        """
        buffer_type = C.c_int32 * buffer_elements
        # Get address of array_in_c, which contains reference to the C array.
        array_address = C.addressof(buffer.contents)
        # Make ctypes style array from C array.
        ctypes_array = buffer_type.from_address(array_address)
        # Make a NumPy array from that.
        return N.ctypeslib.as_array(ctypes_array)

    def _convertDatetimeToMSTime(self, dt):
        """
        Takes obspy.util.UTCDateTime object and returns an epoch time in ms.
        
        @param dt: obspy.util.UTCDateTime object.
        """
        return int(dt.timestamp * HPTMODULUS)

    def _convertMSTimeToDatetime(self, timestring):
        """
        Takes Mini-SEED timestring and returns a obspy.util.UTCDateTime object.
        
        @param timestring: Mini-SEED timestring (Epoch time string in ms).
        """
        return UTCDateTime.utcfromtimestamp(timestring / HPTMODULUS)

    def _convertMSTToDict(self, m):
        """
        Return dictionary from MSTrace Object m, leaving the attributes
        datasamples, ststate and next out
        
        @param m: MST structure to be read.
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
        
        @param m: MST structure to be modified.
        @param h: Dictionary containing all necessary information.
        """
        chain = m.contents
        h['type'] = '\x00'
        # header attributes to be converted
        attributes = ('network', 'station', 'location', 'channel',
                      'dataquality', 'type', 'starttime', 'endtime',
                      'samprate', 'samplecnt', 'numsamples', 'sampletype')
        # loop over attributes
        for _i in attributes:
            setattr(chain, _i, h[_i])

    def _convertToCFilePointer(self, open_file):
        """
        Takes an open file and returns a C file pointer for use in ctypes.
        
        @param file: Open file.
        """
        if not isinstance(open_file, file):
            raise TypeError('Needs an open file.')
        # If not defined, define ctypes arg- and restypes.
        C.pythonapi.PyFile_AsFile.argtypes = [C.py_object]
        C.pythonapi.PyFile_AsFile.restype = c_file_p
        # Convert open python file to C file pointer.
        return C.pythonapi.PyFile_AsFile(open_file)

    def _getMSFileInfo(self, filename):
        """
        Takes a Mini-SEED filename as an argument and returns a dictionary
        with some basic information about the file.
        
        @param filename: Mini-SEED file string.
        """
        info = {}
        info['filesize'] = os.path.getsize(filename)
        #Open file and get record length using libmseed.
        msfile = open(filename, 'rb')
        rec_buffer = msfile.read(512)
        info['record_length'] = \
           clibmseed.ms_find_reclen(C.c_char_p(rec_buffer), C.c_int(512), None)
        #Calculate Number of Records
        info['number_of_records'] = long(info['filesize'] / \
                                         info['record_length'])
        msfile.close()
        return info

    def _isRateTolerable(self, sr1, sr2):
        """
        Tests default sample rate tolerance: abs(1-sr1/sr2) < 0.0001
        """
        return math.fabs(1.0 - (sr1 / float(sr2))) < 0.0001

    def _packMSTToFile(self, mst, outfile, reclen, encoding, byteorder, flush,
                       verbose):
        """
        Takes MS Trace object and writes it to a file
        """
        #Allow direclty passing of file pointers, usefull for appending
        #mseed records on existing mseed files
        if type(outfile) == file:
            mseedfile = outfile
        else:
            mseedfile = open(outfile, 'wb')
        #Initialize packedsamples pointer for the mst_pack function
        self.packedsamples = C.pointer(C.c_int(0))
        #Callback function for mst_pack to actually write the file
        def record_handler(record, reclen, _stream):
            mseedfile.write(record[0:reclen])
        #Define Python callback function for use in C function
        RECHANDLER = C.CFUNCTYPE(None, C.POINTER(C.c_char), C.c_int,
                                 C.c_void_p)
        rec_handler = RECHANDLER(record_handler)
        #Pack the file into a MiniSEED file
        clibmseed.mst_pack(mst, rec_handler, None, reclen, encoding, byteorder,
                           self.packedsamples, flush, verbose, None)
        if not type(outfile) == file:
            mseedfile.close()

    def _populateMSTG(self, trace_list):
        """
        Populates MSTrace_Group structure from given header, data and
        numtraces and returns the MSTrace_Group
        
        Currently only works with one continuous trace.
        
        @param header: Dictionary with the header values to be written to the
            structure.
        @param data: List containing the data values.
        @param numtraces: Number of traces in the structure. No function so
            far.
        """
        # Define return types for C functions.
        clibmseed.mst_initgroup.restype = C.POINTER(MSTraceGroup)
        clibmseed.mst_init.restype = C.POINTER(MSTrace)
        # Init MSTraceGroup
        mstg = clibmseed.mst_initgroup(None)
        # Set numtraces.
        numtraces = len(trace_list)
        mstg.contents.numtraces = numtraces
        # Define starting point of the MSTG structure and the same as a string.
        # Init MSTrace object and connect with group
        mstg.contents.traces = clibmseed.mst_init(None)
        chain = mstg.contents.traces
        # Loop over all traces in trace_list.
        for _i in xrange(numtraces):
            # Create variable with the number of sampels in this trace for
            # faster future access.
            npts = trace_list[_i][0]['numsamples']
            # Write header in MSTrace structure
            self._convertDictToMST(chain, trace_list[_i][0])
            # Create a single datapoint and resize its memory to be able to
            # hold all datapoints.
            tempdatpoint = C.c_int32()
            C.resize(tempdatpoint,
                     clibmseed.ms_samplesize(C.c_char(trace_list[_i][0]\
                                                      ['sampletype'])) * npts)
            # The datapoints in the MSTG structure are a pointer to the memory
            # area reserved for tempdatpoint.
            chain.contents.datasamples = C.pointer(tempdatpoint)
            # Pointer to the Numpy data buffer.
            datptr = trace_list[_i][1].ctypes.get_data()
            # Manually move the contents of the numpy data buffer to the
            # address of the previously created memory area.
            C.memmove(chain.contents.datasamples, datptr, npts * 4)
            if _i != numtraces - 1:
                chain.contents.next = clibmseed.mst_init(None)
                chain = chain.contents.next
        return mstg
