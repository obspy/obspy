# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# Filename: libmseed.py
#  Purpose: Python wrapper for libmseed of Chad Trabant
#   Author: Lion Krischer, Robert Barsch, Moritz Beyreuther
#    Email: krischer@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2010 Lion Krischer, Robert Barsch, Moritz Beyreuther
#---------------------------------------------------------------------
from StringIO import StringIO
from obspy.core import UTCDateTime
from obspy.core.util import scoreatpercentile
from obspy.mseed.headers import MSFileParam, _PyFile_callback, clibmseed, \
    PyFile_FromFile, HPTMODULUS, MSRecord, FRAME
from struct import unpack
import ctypes as C
import math
import numpy as np
import operator
import os
import sys
"""
Class for handling MiniSEED files.

Contains wrappers for libmseed - The MiniSEED library. The C library is
interfaced via python-ctypes. Currently only supports MiniSEED files with
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



class libmseed(object):
    """
    Class for handling MiniSEED files.
    """
    def printFileInformation(self, filename):
        """
        Prints some informations about the file.
        
        @param filename: MiniSEED file.
        """
        try:
            #Read Trace Group
            mstg = self.readFileToTraceGroup(str(filename), dataflag=0)
            clibmseed.mst_printtracelist(mstg, 1, 1, 1)
            del mstg
        except:
            raise

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
        
        @param filename: MiniSEED file.
        """
        f = open(filename, 'rb')
        header = f.read(7)
        if not header[0:6].isdigit:
            f.close()
            return False
        # Check for any valid control header types.
        if header[6] in ['D', 'R', 'Q', 'M']:
            f.close()
            return True
        # If it is a fullSEED record parse the whole file and check whether
        # it has has a data record.
        if header[6] == 'V':
            f.seek(1, 1)
            _i = 0
            # Check if one of the first three blockettes is blockette ten.
            while True:
                if f.read(3) == '010':
                    break
                f.seek(int(file.read(4)) - 7, 1)
                _i += 1
                if _i == 3:
                    f.close()
                    return False
            # Get record length.
            f.seek(8, 1)
            record_length = pow(2, int(f.read(2)))
            file_size = os.path.getsize(filename)
            # Jump to the second record.
            f.seek(record_length + 6)
            # Loop over all records and return True if one record is a data
            # record
            while f.tell() < file_size:
                xx = f.read(1)
                if xx in ['D', 'R', 'Q', 'M']:
                    f.close()
                    return True
                f.seek(record_length - 1, 1)
            f.close()
            return False
        f.close()
        return False

    def readMSTracesViaRecords(self, filename, reclen= -1, dataflag=1,
                               skipnotdata=1, verbose=0, starttime=None,
                               endtime=None):
        """
        Read MiniSEED file. Returns a list with header informations and data
        for each trace in the file.
        
        The list returned contains a list for each trace in the file with the
        lists first element being a header dictionary and its second element
        containing the data values as a numpy array.

        @param filename: Name of MiniSEED file.
        @param reclen, dataflag, skipnotdata, verbose: These are passed
            directly to the ms_readmsr.
        """
        # Initialise list that will contain all traces, first dummy entry
        # will be removed at the end again
        trace_list = [[{'endtime':0}, np.array([])]]
        #XXX: Put next lines till seek in a function Function, return msf, msr and f
        # Initialize MSRecord structure
        msr = clibmseed.msr_init(C.POINTER(MSRecord)())
        msf = C.POINTER(MSFileParam)() # null pointer
        # allocate file pointer, we need this to cut with start and endtime
        clibmseed.ms_readmsr_r(C.pointer(msf), C.pointer(msr),
                               str(filename), -1, None, None,
                               1, 0, 0)
        mf = C.pointer(MSFileParam.from_address(C.addressof(msf)))
        f = PyFile_FromFile(mf.contents.fp.contents.value,
                            str(filename), 'rb', _PyFile_callback)
        f.seek(0)
        end_byte = 1e99
        if starttime or endtime:
            bytes = self._bytePosFromTime(filename, starttime=starttime, endtime=endtime)
            if bytes == '':
                self.clear(msf, msr)
                return ''
            f.seek(bytes[0])
            end_byte = bytes[0] + bytes[1]
        # Loop over records and append to trace_list.
        # Directly call ms_readmsr
        last_msrid = None
        while True:
            errcode = clibmseed.ms_readmsr_r(C.pointer(msf),
                C.pointer(msr), filename, reclen,
                None, None, skipnotdata, dataflag, verbose)
            if errcode != 0:
                break
            chain = msr.contents
            header = self._convertMSRToDict(chain)
            delta = HPTMODULUS / float(header['samprate'])
            header['endtime'] = long(header['starttime'] + delta * \
                                      (header['numsamples'] - 1))
            # Access data directly as numpy array.
            data = self._accessCtypesArrayAsNumpyArray(chain.datasamples,
                                                       chain.numsamples)
            msrid = self._MSRId(header)
            last_endtime = trace_list[-1][0]['endtime']
            if abs(last_endtime - header['starttime']) <= 1.01 * delta and \
                    last_msrid == msrid:
                # Append to trace
                trace_list[-1][0]['endtime'] = header['endtime']
                trace_list[-1][0]['numsamples'] += header['numsamples']
                trace_list[-1].append(data)
            else:
                # Concatenate last trace and start a new trace
                trace_list[-1] = [trace_list[-1][0],
                                  np.concatenate(trace_list[-1][1:])]
                trace_list.append([header, data])
            last_msrid = msrid
            if f.tell() >= end_byte:
                break
        # Finish up loop, concatenate last trace_list
        trace_list[-1] = [trace_list[-1][0],
                          np.concatenate(trace_list[-1][1:])]
        trace_list.pop(0) # remove first dummy entry of list
        # Free MSRecord structure
        f.close()
        self.clear(msf, msr)
        del msf, msr, chain
        return trace_list

    def clear(self, msf, msr):
        """
        Method for deallocating MSFileParam and MSRecord structure.
        
        @param msf: MSFileParam structure.
        @param msr: MSRecord structure.
        """
        clibmseed.ms_readmsr_r(C.pointer(msf), C.pointer(msr),
                               None, -1, None, None, 0, 0, 0)

    def readMSTraces(self, filename, reclen= -1, timetol= -1,
                     sampratetol= -1, dataflag=1, skipnotdata=1,
                     dataquality=1, verbose=0):
        """
        Read MiniSEED file. Returns a list with header informations and data
        for each trace in the file.
        
        The list returned contains a list for each trace in the file with the
        lists first element being a header dictionary and its second element
        containing the data values as a numpy array.

        @param filename: Name of MiniSEED file.
        @param reclen: Directly to the readFileToTraceGroup method.
        @param timetol: Directly to the readFileToTraceGroup method.
        @param sampratetol: Directly to the readFileToTraceGroup method.
        @param dataflag: Directly to the readFileToTraceGroup method.
        @param skipnotdata: Directly to the readFileToTraceGroup method.
        @param dataquality: Directly to the readFileToTraceGroup method.
        @param verbose: Directly to the readFileToTraceGroup method.
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
                                         verbose=verbose)
        chain = mstg.contents.traces.contents
        numtraces = mstg.contents.numtraces
        # Loop over traces and append to trace_list.
        for i in xrange(numtraces):
            header = self._convertMSTToDict(chain)
            # Access data directly as numpy array.
            data = self._accessCtypesArrayAsNumpyArray(chain.datasamples,
                                                       chain.numsamples)
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
        Write Miniseed file from trace_list
        
        @param trace_list: List containing header informations and data.
        @param outfile: Name of the output file
        @param reclen: should be set to the desired data record length in bytes
            which must be expressible as 2 raised to the power of X where X is
            between (and including) 8 to 20. -1 defaults to 4096
        @param encoding: should be set to one of the following supported
            MiniSEED data encoding formats: DE_ASCII (0), DE_INT16 (1),
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
        try:
            f = open(outfile, 'wb')
        except TypeError:
            f = outfile
        # Populate MSTG Structure
        mstg = self._populateMSTG(trace_list)
        # Initialize packedsamples pointer for the mst_pack function
        self.packedsamples = C.c_int()
        # Callback function for mst_pack to actually write the file
        def record_handler(record, reclen, _stream):
            f.write(record[0:reclen])
        # Define Python callback function for use in C function
        recHandler = C.CFUNCTYPE(C.c_void_p, C.POINTER(C.c_char), C.c_int,
                                 C.c_void_p)(record_handler)
        # Pack mstg into a MSEED file using record_handler as write method
        msr = C.POINTER(MSRecord)()
        errcode = clibmseed.mst_packgroup(mstg, recHandler, None, reclen,
                                          encoding, byteorder,
                                          C.byref(self.packedsamples),
                                          flush, verbose, msr)
        if errcode == -1:
            raise Exception('Error in mst_packgroup')
        # Cleaning up
        clibmseed.mst_freegroup(C.pointer(mstg))
        if isinstance(f, file): # necessary for Python 2.5.2 BUG otherwise!
            f.close()
        del mstg, msr

    def readFileToTraceGroup(self, filename, reclen= -1, timetol= -1,
                             sampratetol= -1, dataflag=1, skipnotdata=1,
                             dataquality=1, verbose=0):
        """
        Reads MiniSEED data from file. Returns MSTraceGroup structure.
        
        @param filename: Name of MiniSEED file.
        @param reclen: If reclen is 0 the length of the first record is auto-
            detected. All subsequent records are then expected to have the
            same record length. If reclen is negative the length of every
            record is automatically detected. Defaults to -1.
        @param timetol: Time tolerance, default to -1 (1/2 sample period).
        @param sampratetol: Sample rate tolerance, defaults to -1 (rate
            dependent)
        @param dataflag: Controls whether data samples are unpacked, defaults
            to true (1).
        @param skipnotdata: If true (not zero) any data chunks read that to do
            not have valid data record indicators will be skipped. Defaults to
            true (1).
        @param dataquality: If the dataquality flag is true traces will be
            grouped by quality in addition to the source name identifiers.
            Defaults to true (1).
        @param verbose: Controls verbosity from 0 to 2. Defaults to None (0).
        """
        # Creates MSTraceGroup Structure
        mstg = clibmseed.mst_initgroup(None)
        # Uses libmseed to read the file and populate the MSTraceGroup
        errcode = clibmseed.ms_readtraces(
            C.pointer(mstg), filename, reclen, timetol, sampratetol,
            dataquality, skipnotdata, dataflag, verbose)
        if errcode != 0:
            raise Exception("Error in ms_readtraces")
        return mstg


    def readSingleRecordToMSR(self, filename, ms_p=(None, None),
                              reclen= -1, dataflag=1, skipnotdata=1,
                              verbose=0, record_number=0):
        """
        Reads MiniSEED record from file and populates MS Record data structure.
        
        @param filename: MiniSEED file to be read.
        @param ms_p: Use existing LP_MSRecord (msr) and LP_MSFileParam (msf)
            structures given by ms_p=(msr,msf), e.g. returned
            by this function. Given an existing msr and msf the function is
            much faster.
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
        @rtype: LP_MSRecord, LP_MSFileParam
        @return: msr, msf MSRecord structure and MSFileParam structure
        @requires: LP_MSRecord (msr), LP_MSFileParam (msf) need to be deallocated
            with the function call
            C{clibmseed.ms_readmsr_r(C.pointer(msf), C.pointer(msr), None, -1, None,
            None, 0, 0, 0)}
            or the wrapper method around it C{self.clear(msf, msr)}
        """
        # Get some information about the file.
        f = open(filename, 'rb')
        fileinfo = self._getMSFileInfo(f, filename)
        f.close()
        # Calculate offset of the record to be read.
        if record_number < 0:
            record_number = fileinfo['number_of_records'] + record_number
        if record_number < 0 or record_number >= fileinfo['number_of_records']:
            raise ValueError('Please enter a valid record_number')
        filepos = record_number * fileinfo['record_length']
        if isinstance(ms_p[0], C.POINTER(MSRecord)) and \
                isinstance(ms_p[1], C.POINTER(MSFileParam)):
            msr, msf = ms_p
        elif ms_p == (None, None):
            # Init MSRecord structure
            msr = clibmseed.msr_init(None)
            # Init null pointer, this pointer is needed for deallocation
            msf = C.POINTER(MSFileParam)()
            # Dummy-read/read the first record
            clibmseed.ms_readmsr_r(C.pointer(msf), C.pointer(msr),
                                   str(filename), reclen, None, None,
                                   skipnotdata, dataflag, verbose)
            if record_number == 0:
                return msr, msf
        else:
            cmd = 'Given ms_p arguments are not of type (LP_MSRecord, \
                   LP_MSFileParam)'
            raise Exception(cmd)
        # Parse msf structure in order to seek file pointer to special position
        mf = C.pointer(MSFileParam.from_address(C.addressof(msf)))
        f = PyFile_FromFile(mf.contents.fp.contents.value,
                            str(filename), 'rb', _PyFile_callback)
        f.seek(filepos)
        clibmseed.ms_readmsr_r(C.pointer(msf), C.pointer(msr),
                               filename, reclen, None, None,
                               skipnotdata, dataflag, verbose)
        f.close()
        del mf
        return msr, msf # need both for deallocation

    def getFirstRecordHeaderInfo(self, filename):
        """
        Takes a MiniSEED file and returns header of the first record.
        Method using ms_readmsr_r.
        
        Returns a dictionary containing some header information from the first
        record of the MiniSEED file only. It returns the location, network,
        station and channel information.
        
        @param filename: MiniSEED file string.
        """
        # read first header only
        msr, msf = self.readSingleRecordToMSR(filename, dataflag=0)
        header = {}
        # header attributes to be read
        attributes = ('location', 'network', 'station', 'channel')
        # loop over attributes
        for _i in attributes:
            header[_i] = getattr(msr.contents, _i)
        # Deallocate msr and msf memory
        self.clear(msf, msr)
        del msr, msf
        return header

    def getEndFromMSR(self, filename, msr, msf):
        """
        Return endtime of given msr and msf structure

        @param msr: LP_MSRecord of interest
        @param msf: associated LP_MSFileParam 
        @param filename: May be redundant but must be given
        """
        clibmseed.ms_readmsr_r(C.pointer(msf), C.pointer(msr),
                               str(filename), -1, None, None,
                               1, 0, 0)
        dtime = clibmseed.msr_endtime(msr)
        return UTCDateTime(dtime / HPTMODULUS)


    def getStartFromMSF(self, filename, msr, msf):
        """
        Return starttime of given msr and msf structure

        @param msr: LP_MSRecord of interest
        @param msf: associated LP_MSFileParam 
        @param filename: May be redundant but must be given
        """
        clibmseed.ms_readmsr_r(C.pointer(msf), C.pointer(msr),
                               str(filename), -1, None, None,
                               1, 0, 0)
        dtime = clibmseed.msr_starttime(msr)
        return UTCDateTime(dtime / HPTMODULUS)

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
        
        @param filename: MiniSEED file string.
        """
        # Get the starttime using the libmseed method msr_starttime
        msr, msf = self.readSingleRecordToMSR(filename, dataflag=0)
        starttime = clibmseed.msr_starttime(msr)
        starttime = self._convertMSTimeToDatetime(starttime)
        # Get the endtime using the libmseed method msr_endtime
        msr, msf = self.readSingleRecordToMSR(filename, ms_p=(msr, msf),
                                              dataflag=0, record_number= -1)
        endtime = clibmseed.msr_endtime(msr)
        endtime = self._convertMSTimeToDatetime(endtime)
        # Deallocate msr and msf memory
        self.clear(msf, msr)
        del msr, msf
        return starttime, endtime


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
        gap_list = []
        # iterate through traces
        cur = mstg.contents.traces.contents
        for _ in xrange(mstg.contents.numtraces - 1):
            next = cur.next.contents
            # Skip MSTraces with 0 sample rate, usually from SOH records
            if cur.samprate == 0:
                cur = next
                continue
            # Check that sample rates match using default tolerance
            if not self._isRateTolerable(cur.samprate, next.samprate):
                msg = "%s: Sample rate changed! %.10g -> %.10g\n"
                sys.stderr.write(msg % (filename, cur.samprate, next.samprate))
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
            gap_list.append((cur.network, cur.station, cur.location,
                             cur.channel, time1, time2, gap, nsamples))
            cur = next
        clibmseed.mst_freegroup(C.pointer(mstg))
        del mstg
        return gap_list

    def printGapList(self, filename, time_tolerance= -1,
                     samprate_tolerance= -1, min_gap=None, max_gap=None):
        """
        Print gap/overlap list summary information for the given filename.
        """
        result = self.getGapList(filename, time_tolerance, samprate_tolerance,
                                 min_gap, max_gap)
        msg = "%-17s %-26s %-26s %-5s %-8s\n" % ('Source', 'Last Sample',
                                                 'Next Sample', 'Gap',
                                                 'Samples')
        sys.stdout.write(msg)
        msg = "%-17s %-26s %-26s %-5s %-.8g"
        for r in result:
            sys.stdout.write(msg % ('_'.join(r[0:4]), r[4].isoformat(),
                                    r[5].isoformat(), r[6], r[7]))
        sys.stdout.write("Total: %d gap(s)\n" % len(result))

    def readMSHeader(self, filename, time_tolerance= -1,
                   samprate_tolerance= -1):
        """
        Returns trace header information of a given file.
        
        @param time_tolerance: Time tolerance while reading the traces, default 
            to -1 (1/2 sample period).
        @param samprate_tolerance: Sample rate tolerance while reading the 
            traces, defaults to -1 (rate dependent).
        @return: Dictionary containing header entries
        """
        # read file
        mstg = self.readFileToTraceGroup(filename, dataflag=0,
                                         skipnotdata=0,
                                         timetol=time_tolerance,
                                         sampratetol=samprate_tolerance)
        # iterate through traces
        cur = mstg.contents.traces.contents
        header = [[self._convertMSTToDict(cur), None]]
        for _ in xrange(mstg.contents.numtraces - 1):
            next = cur.next.contents
            header.append([self._convertMSTToDict(cur), None])
            cur = next
        clibmseed.mst_freegroup(C.pointer(mstg))
        del mstg
        return header

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
        
        @param filename: MiniSEED file name.
        @return: List of all flag counts.
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
        f = open(filename, 'rb')
        fileinfo = self._getMSFileInfo(f, filename)
        f.close()
        # Init MSRecord structure
        msr = clibmseed.msr_init(None)
        # Init null pointer, this pointer is needed for deallocation
        msf = C.POINTER(MSFileParam)()
        # Create Timing Quality list.
        data = []
        # Loop over each record
        for _i in xrange(fileinfo['number_of_records']):
            # Loop over every record.
            errcode = clibmseed.ms_readmsr_r(C.pointer(msf), C.pointer(msr),
                                             str(filename), C.c_int(rl_autodetection),
                                             None, None, C.c_short(1), C.c_short(0),
                                             C.c_short(0))
            if errcode != 0:
                raise Exception("Error in ms_readmsr_r")
            # Enclose in try-except block because not all records need to
            # have Blockette 1001.
            try:
                # Append timing quality to list.
                data.append(float(msr.contents.Blkt1001.contents.timing_qual))
            except:
                if first_record:
                    break
        # Deallocate msr and msf memory
        self.clear(msf, msr)
        del msr, msf
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
        result['median'] = scoreatpercentile(data, 50, sort=False)
        result['lower_quantile'] = scoreatpercentile(data, 25, sort=False)
        result['upper_quantile'] = scoreatpercentile(data, 75, sort=False)
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
        
        @return: Byte position of beginning and total length of bytes
        
        @param filename: File string of the MiniSEED file to be cut.
        @param starttime: L{obspy.core.UTCDateTime} object.
        @param endtime: L{obspy.core.UTCDateTime} object.
        """
        # Read the start and end time of the file.
        msr = clibmseed.msr_init(None)
        msf = C.POINTER(MSFileParam)()
        clibmseed.ms_readmsr_r(C.pointer(msf), C.pointer(msr),
                               str(filename), -1, None, None,
                               1, 0, 0)
        mf = C.pointer(MSFileParam.from_address(C.addressof(msf)))
        f = PyFile_FromFile(mf.contents.fp.contents.value,
                            str(filename), 'rb', _PyFile_callback)
        f.seek(0)
        info = self._getMSFileInfo(f, filename)
        start = self.getStartFromMSF(filename, msr, msf)
        pos = (info['number_of_records'] - 1) * info['record_length']
        f.seek(pos)
        end = self.getEndFromMSR(filename, msr, msf)
        # Set the start time.
        if not starttime or starttime <= start:
            starttime = start
        # Set the end time.
        if not endtime or endtime >= end:
            endtime = end
        # Deallocate msr and msf memory for wrong input
        if starttime >= end or endtime <= start:
            self.clear(msf, msr)
            del msr, msf
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
            f.seek(start_record * info['record_length'])
            stime = self.getStartFromMSF(filename, msr, msf)
            # Calculate last covered record.
            f.seek(30, 1)
            # Calculate sample rate.
            sample_rate = msr.contents.samprate
            npts = msr.contents.samplecnt
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
            f.seek(end_record * info['record_length'])
            stime = self.getStartFromMSF(filename, msr, msf)
            # Calculate last covered record.
            f.seek(30, 1)
            # Calculate sample rate.
            sample_rate = msr.contents.samprate
            npts = msr.contents.samplecnt
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
        # Deallocate msr and msf memory
        self.clear(msf, msr)
        del msr, msf
        # Calculate starting position
        record_length = info['record_length']
        start_byte = record_length * start_record
        # length in bytes to read
        length_byte = record_length * (end_record - start_record + 1)
        return start_byte, length_byte

    def cutMSFileByRecords(self, filename, starttime=None, endtime=None):
        """
        Cuts a MiniSEED file by cutting at records.
        
        For details see method _bytePosFromTime.
        
        @return: Byte string containing the cut file.
        
        @param filename: File string of the MiniSEED file to be cut.
        @param starttime: L{obspy.core.UTCDateTime} object.
        @param endtime: L{obspy.core.UTCDateTime} object.
        """
        bytes = self._bytePosFromTime(filename, starttime=starttime, endtime=endtime)
        if bytes == None:
            return ''
        # Open file a seek to location
        f = open(filename)
        f.seek(bytes[0], 0)
        # Read until end_location.
        data = f.read(bytes[1])
        f.close()
        # Return the cut file string.
        return data

    def mergeAndCutMSFiles(self, file_list, starttime=None, endtime=None):
        """
        This method takes several MiniSEED files and returns one merged file.
        
        It is also possible to specify a start- and a endtime and all records
        that are out of bounds will be cut.
        If two not identical files cover a common time frame they will still
        be merged and no data is lost.
        
        The input files can be given in any order but they have to be files
        that contain only traces from one source and one component and the
        traces inside the files have to be in chronological order. Otherwise
        the produced output will not be correct. All files also have to be from
        the same source.
        
        @param file_list: A list containing MiniSEED filename strings.
        @param outfile: String of the file to be created.
        @param starttime: L{obspy.core.UTCDateTime} object.
        @param endtime: L{obspy.core.UTCDateTime} object.
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
        file_list = sorted(file_list, key=operator.itemgetter(1))
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
        # 1. METHOD LION
        #buffer_type = C.c_int32 * buffer_elements
        # Get address of array_in_c, which contains reference to the C array.
        #array_address = C.addressof(buffer.contents)
        # Make ctypes style array from C array.
        #ctypes_array = buffer_type.from_address(array_address)
        # Allocate numpy array to move memory to
        # Make a NumPy array from that.
        #return np.ctypeslib.as_array(ctypes_array)
        # 2. METHOD MORITZ 
        numpy_array = np.ndarray(buffer_elements, dtype='int32')
        datptr = numpy_array.ctypes.get_data()
        # Manually copy the contents of the C malloced memory area to
        # the address of the previously created numpy array
        C.memmove(datptr, buffer, buffer_elements * 4)
        # free the memory of the buffer, do not do that when you used 
        # mst_freegroup before, else Segmentation fault
        #libc.free( C.cast(buffer, C.c_void_p) )
        return numpy_array
        # 3. METHOD MORITZ
        # reading C memory into buffer which can be converted to numpy array
        # this is read only too
        #C.pythonapi.PyBuffer_FromMemory.argtypes = [C.c_void_p, C.c_int]
        #C.pythonapi.PyBuffer_FromMemory.restype = C.py_object
        #return np.frombuffer(C.pythonapi.PyBuffer_FromMemory(buffer,
        #                                                    buffer_elements*4),
        #                    dtype='int32',count=buffer_elements)

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

    def _getMSFileInfo(self, f, real_name):
        """
        Takes a Mini-SEED filename as an argument and returns a dictionary
        with some basic information about the file.
        
        @param f: File pointer of opened file in binary format
        @param real_name: Realname of the file
        """
        info = {}
        #
        # get size of file
        info['filesize'] = os.path.getsize(real_name)
        pos = f.tell()
        f.seek(0)
        rec_buffer = f.read(512)
        info['record_length'] = \
           clibmseed.ms_find_reclen(rec_buffer, 512, None)
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
            # Check that data are numpy.ndarrays of dtype int32
            if not isinstance(trace_list[_i][1], np.ndarray) or \
                    trace_list[_i][1].dtype != 'int32':
                raise Exception("Data must me of type numpy.ndarray, dtype "
                                "int32")
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
            # old segmentationfault try
            # chain.contents.datasamples = \
            # trace_list[_i][1].ctypes.data_as(C.POINTER(C.c_long))
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

    def unpack_steim2(self, data_string, npts, swapflag=0, verbose=0):
        """
        Unpack steim2 compressed data given as string.
        
        @param data_string: data as string
        @param npts: number of data points
        @param swapflag: Swap bytes, defaults to 0
        @return: Return data as numpy.ndarray of dtype int32
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
        
        @param data_string: data as string
        @param npts: number of data points
        @param swapflag: Swap bytes, defaults to 0
        @return: Return data as numpy.ndarray of dtype int32
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
