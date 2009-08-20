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

from StringIO import StringIO
from obspy.core import UTCDateTime
from obspy.core.util import scoreatpercentile
from obspy.mseed.headers import MSTraceGroup, MSTrace, HPTMODULUS, MSRecord
from obspy.mseed.headers import MSFileParam, Py_ssize_t
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
    lib_name = 'libmseed-2.3.win32.dll'
# 32 bit OSX, tested with 10.5.6
elif sys.platform == 'darwin':
    lib_name = 'libmseed.dylib'
# 32 and 64 bit UNIX
#XXX Check glibc version by platform.libc_ver()
else:
    if platform.architecture()[0] == '64bit':
        lib_name = 'libmseed.lin64.so'
    else:
        lib_name = 'libmseed-2.3.so'
clibmseed = C.CDLL(os.path.join(os.path.dirname(__file__), 'libmseed',
                                lib_name))

#
# Declare function of libmseed library, argument parsing
#
clibmseed.mst_init.artypes = [C.POINTER(MSTrace)]
clibmseed.mst_init.restype = C.POINTER(MSTrace)

clibmseed.mst_free.argtypes = [C.POINTER(C.POINTER(MSTrace))]
clibmseed.mst_free.restype = C.c_void_p

clibmseed.mst_initgroup.artypes = [C.POINTER(MSTraceGroup)]
clibmseed.mst_initgroup.restype = C.POINTER(MSTraceGroup)

clibmseed.mst_freegroup.argtypes = [C.POINTER(C.POINTER(MSTraceGroup))]
clibmseed.mst_freegroup.restype = C.c_void_p

clibmseed.msr_init.argtypes = [C.POINTER(MSRecord)]
clibmseed.msr_init.restype = C.POINTER(MSRecord)

clibmseed.mst_printtracelist.argtypes = [C.POINTER(MSTraceGroup), C.c_int, C.c_int,
                                         C.c_int]
clibmseed.mst_printtracelist.restype = C.c_void_p

clibmseed.ms_readmsr_r.argtypes = [C.POINTER(C.POINTER(MSFileParam)), 
                                   C.POINTER(C.POINTER(MSRecord)), C.c_char_p, C.c_int,
                                   C.POINTER(Py_ssize_t), C.POINTER(C.c_int), 
                                   C.c_short, C.c_short, C.c_short]
clibmseed.ms_readmsr_r.restypes = C.c_int

clibmseed.ms_readtraces.argtypes = [C.POINTER(C.POINTER(MSTraceGroup)),
                                    C.c_char_p, C.c_int, C.c_double,
                                    C.c_double, C.c_short, C.c_short,
                                    C.c_short, C.c_short]
clibmseed.ms_readtraces.restype = C.c_int

clibmseed.msr_starttime.artypes = [C.POINTER(MSRecord)]
clibmseed.msr_starttime.restype = C.c_int64

clibmseed.msr_endtime.artypes = [C.POINTER(MSRecord)]
clibmseed.msr_endtime.restype = C.c_int64

clibmseed.mst_packgroup.artypes = [C.POINTER(C.POINTER(MSTraceGroup)),
                                   C.CFUNCTYPE(C.c_char_p,C.c_int,C.c_void_p),
                                   C.c_void_p, C.c_int, C.c_short,
                                   C.c_short, C.POINTER(C.c_int), 
                                   C.c_short, C.c_short,
                                   C.POINTER(MSRecord)]
clibmseed.mst_packgroup.restype = C.c_int

PyFile_FromFile = C.pythonapi.PyFile_FromFile
PyFile_FromFile.artypes = [Py_ssize_t, C.c_char_p, C.c_char_p,
                           C.CFUNCTYPE(C.c_int, Py_ssize_t)]
PyFile_FromFile.restype = C.py_object


#
# Python callback functions for C
#
def yes_(f):
    return 1
yes = C.CFUNCTYPE(C.c_int, Py_ssize_t)(yes_)



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
        This method will just read the first record and not the whole file.
        Thus it cannot be used to validate a MiniSEED file.
        
        @param filename: MiniSEED file.
        """
        f = open(filename, 'rb')
        # Read part of the MiniSEED header.
        f.seek(6)
        header = f.read(16)
        f.close()
        # Read big- and little endian word order!
        big_endian = unpack('>cxxxxxxxxxxxxxH', header)
        #little_endian = unpack('>cxxxxxxxxxxxxxH', header)
        if big_endian[0] not in ['D', 'R', 'Q', 'M', 'V']:
            return False
        #if ((big_endian[1] < 2100 and big_endian[1] > 1900) or
        #    (little_endian[0] < 2100 and little_endian[0] > 1900)):
        #    return True
        return True

    def readMSTracesViaRecords(self, filename, reclen= -1, dataflag=1, skipnotdata=1,
                               verbose=0):
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
        trace_list = [[{'endtime':0}, N.array([])]]
        # Initialize MSRecord structure
        msr = clibmseed.msr_init(C.POINTER(MSRecord)())
        msf = C.POINTER(MSFileParam)() # null pointer
        # Loop over records and append to trace_list.
        # Directly call ms_readmsr
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
                concat_flag = True
            else:
                # Concatenate last trace and start a new trace
                trace_list[-1] = [trace_list[-1][0],
                                  N.concatenate(trace_list[-1][1:])]
                trace_list.append([header, data])
                concat_flag = False
            last_msrid = msrid
        # Finish up loop, concatenate trace_list if not already done
        if concat_flag:
                trace_list[-1] = [trace_list[-1][0],
                                  N.concatenate(trace_list[-1][1:])]
        trace_list.pop(0) # remove first dummy entry of list
        # Free MSRecord structure
        clibmseed.ms_readmsr_r(C.pointer(msf), C.pointer(msr),
                               None, 0, None, None, 0, 0, 0)
        del msr, chain
        return trace_list

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
        @param reclen, timetol, sampratetol, dataflag, skipnotdata,
            dataquality, verbose: These are passed directly to the 
            readFileToTraceGroup method.
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
        recHandler = C.CFUNCTYPE(None, C.POINTER(C.c_char), C.c_int,
                         C.c_void_p)(record_handler)
        # Pack mstg into a MSEED file using record_handler as write method
        errcode = clibmseed.mst_packgroup(mstg, recHandler, None, reclen,
                                          encoding, byteorder,
                                          C.byref(self.packedsamples),
                                          flush, verbose, None)
        if errcode == -1:
            raise Exception('Error in mst_packgroup')
        clibmseed.mst_freegroup(C.pointer(mstg))
        del mstg

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
        mstg = clibmseed.mst_initgroup(None)
        # Uses libmseed to read the file and populate the MSTraceGroup
        errcode = clibmseed.ms_readtraces(
            C.pointer(mstg), filename, reclen, timetol, sampratetol,
            dataquality, skipnotdata, dataflag, verbose)
        if errcode != 0:
            raise Exception("Error in ms_readtraces")
        return mstg

    def getFirstRecordHeaderInfo(self, filename):
        """
        Takes a MiniSEED file and returns header of the first record.
        
        Returns a dictionary containing some header information from the first
        record of the MiniSEED file only. It returns the location, network,
        station and channel information.
        
        @param filename: MiniSEED file string.
        """
        # open file and jump to beginning of the data of interest.
        mseed_file = open(filename, 'rb')
        mseed_file.seek(8)
        # Unpack the information using big endian byte order.
        unpacked_tuple = unpack('>cccccccccccc', mseed_file.read(12))
        # Close the file.
        mseed_file.close()
        # Return a dictionary containing the necessary information.
        return \
            {'station' : ''.join([_i for _i in unpacked_tuple[0:5]]).strip(),
             'location' : ''.join([_i for _i in unpacked_tuple[5:7]]).strip(),
             'channel' :''.join([_i for _i in unpacked_tuple[7:10]]).strip(),
             'network' : ''.join([_i for _i in unpacked_tuple[10:12]]).strip()}


    def readSingleRecordToMSR(self, filename, ms_p=(None,None),
                              reclen= -1, dataflag=1, skipnotdata=1,
                              verbose=0, record_number=0):
        """
        Reads Mini-SEED record from file and populates MS Record data structure.
        
        @param filename: Mini-SEED file to be read.
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
        @required: LP_MSRecord (msr), LP_MSFileParam (msf) need to be deallocated
            with the function call:
            clibmseed.ms_readmsr_r(C.pointer(msf), C.pointer(msr),
                                   None, 0, None, None, 0, 0, 0)
        """
        # Get some information about the file.
        fileinfo = self._getMSFileInfo(filename)
        # Calculate offset of the record to be read.
        if record_number < 0:
            record_number = fileinfo['number_of_records'] + record_number
        if record_number < 0 or record_number >= fileinfo['number_of_records']:
            raise ValueError('Please enter a valid record_number')
        filepos = record_number * fileinfo['record_length']
        if isinstance(ms_p[0],C.POINTER(MSRecord)) and \
                isinstance(ms_p[1],C.POINTER(MSFileParam)):
            msr, msf = ms_p
        elif ms_p == (None,None):
            # Init MSRecord structure
            msr = clibmseed.msr_init(None)
            # Init null pointer, this pointer is needed for deallocation
            msf = C.POINTER(MSFileParam)()
            # Dummy-read/read the first record
            clibmseed.ms_readmsr_r(C.pointer(msf), C.pointer(msr),
                                   str(filename), reclen, None, None,
                                   skipnotdata, dataflag, verbose)
            #import pdb;pdb.set_trace()
            if record_number == 0:
                return msr, msf
        else:
            cmd = 'Given ms_p arguments are not of type (LP_MSRecord, \
                   LP_MSFileParam)'
            raise Exception(cmd)
        # Parse msf structure in order to seek file pointer to special position
        mf = C.pointer(MSFileParam.from_address(C.addressof(msf)))
        f = PyFile_FromFile(mf.contents.fp.contents.value,
                            str(filename), 'rb', yes)
        f.seek(filepos)
        clibmseed.ms_readmsr_r(C.pointer(msf), C.pointer(msr),
                               filename, reclen, None, None,
                               skipnotdata, dataflag, verbose)
        f.close()
        del mf
        return msr, msf # need both for deallocation

    def getFirstRecordHeaderInfo2(self, filename):
        """
        Takes a Mini-SEED file and returns header of the first record.
        Method using ms_readmsr_r.
        
        Returns a dictionary containing some header information from the first
        record of the Mini-SEED file only. It returns the location, network,
        station and channel information.
        
        @param filename: Mini-SEED file string.
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
        clibmseed.ms_readmsr_r(C.pointer(msf), C.pointer(msr),
                               None, 0, None, None, 0, 0, 0)
        del msr, msf
        return header


    def getStartAndEndTime2(self, filename):
        """
        Returns the start- and endtime of a Mini-SEED file as a tuple
        containing two datetime objects.
        Method using ms_readmsr_r
        
        This method only reads the first and the last record. Thus it will only
        work correctly for files containing only one trace with all records
        in the correct order.
        
        The returned endtime is the time of the last datasample and not the
        time that the last sample covers.
        
        @param filename: Mini-SEED file string.
        """
        # Get the starttime using the libmseed method msr_starttime
        msr, msf = self.readSingleRecordToMSR(filename, dataflag=0)
        starttime = clibmseed.msr_starttime(msr)
        starttime = self._convertMSTimeToDatetime(starttime)
        # Get the endtime using the libmseed method msr_endtime
        msr, msf = self.readSingleRecordToMSR(filename, ms_p = (msr,msf), 
                                              dataflag=0, record_number= -1)
        endtime = clibmseed.msr_endtime(msr)
        endtime = self._convertMSTimeToDatetime(endtime)
        # Deallocate msr and msf memory
        clibmseed.ms_readmsr_r(C.pointer(msf), C.pointer(msr),
                               None, 0, None, None, 0, 0, 0)
        del msr, msf
        return starttime, endtime


    def getStartAndEndTime(self, filename):
        """
        Returns the start- and end time of a MiniSEED file as a tuple
        containing two datetime objects.
        
        This method only reads the first and the last record. Thus it will only
        work correctly for files containing only one trace with all records
        in the correct order and all records necessarily need to have the same
        record length.
        
        The returned end time is the time of the last datasample and not the
        time that the last sample covers.
        
        It is written in pure Python to resolve some memory issues present
        with creating file pointers and passing them to the libmseed.
        
        @param filename: MiniSEED file string.
        """
        # Open the file of interest ad jump to the beginning of the timing
        # information in the file.
        mseed_file = open(filename, 'rb')
        # Get some general information of the file.
        info = self._getMSFileInfo(filename)
        # Get the start time.
        starttime = self._getMSStarttime(mseed_file)
        # Jump to the last record.
        mseed_file.seek(info['filesize'] - info['excess_bytes'] - \
                        info['record_length'])
        # Starttime of the last record.
        last_record_starttime = self._getMSStarttime(mseed_file)
        # Get the number of samples, the sample rate factor and the sample
        # rate multiplier.
        mseed_file.seek(30, 1)
        (npts, sample_rate_factor, sample_rate_multiplier) = \
            unpack('>Hhh', mseed_file.read(6))
        # Calculate sample rate.
        sample_rate = self._calculateSamplingRate(sample_rate_factor, \
                                                  sample_rate_multiplier)
        # The time of the last covered sample is now:
        endtime = last_record_starttime + ((npts - 1) / sample_rate)
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
        print "%-17s %-26s %-26s %-5s %-8s" % ('Source', 'Last Sample',
                                               'Next Sample', 'Gap', 'Samples')
        for r in result:
            print "%-17s %-26s %-26s %-5s %-.8g" % ('_'.join(r[0:4]),
                                                    r[4].isoformat(),
                                                    r[5].isoformat(),
                                                    r[6], r[7])
        print "Total: %d gap(s)" % len(result)

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
        
        @param filename: MiniSEED file name.
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

    def getTimingQuality(self, filename, first_record=True):
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
        
        @param filename: MiniSEED file to be parsed.
        @param first_record: Determines whether all records are assumed to 
            either have a timing quality in Blockette 1001 or not depending on
            whether the first records has one. If True and the first records
            does not have a timing quality it will not parse the whole file. If
            False is will parse the whole file anyway and search for a timing
            quality in each record. Defaults to True.
        """
        # Create Timing Quality list.
        data = []
        # Open file.
        mseed_file = open(filename, 'rb')
        filesize = os.path.getsize(filename)
        # Loop over all records. After each loop the file pointer is supposed
        # to be at the beginning of the next record.
        while True:
            starting_pointer = mseed_file.tell()
            # Unpack field 17 and 18 of the fixed section of the data header.
            mseed_file.seek(44, 1)
            (beginning_of_data, first_blockette) = unpack('>HH',
                                                          mseed_file.read(4))
            # Jump to the first blockette.
            mseed_file.seek(first_blockette - 48, 1)
            # Read all blockettes.
            blockettes = mseed_file.read(beginning_of_data - first_blockette)
            # Loop over all blockettes and find number 1000 and 1001.
            offset = 0
            record_length = None
            timing_quality = None
            blockettes_length = len(blockettes)
            while True:
                # Double check to avoid infinite loop.
                if offset >= blockettes_length:
                    break
                (blkt_number, next_blkt) = unpack('>HH',
                                            blockettes[offset : offset + 4])
                if blkt_number == 1000:
                    record_length = 2 ** unpack('>B',
                                                blockettes[offset + 6])[0]
                elif blkt_number == 1001:
                    timing_quality = unpack('>B',
                                            blockettes[offset + 4])[0]
                # Leave loop if no more blockettes follow.
                if next_blkt == 0:
                    break
                # New offset.
                offset = next_blkt - first_blockette
            # If no Blockette 1000 could be found raise warning.
            if not record_length:
                msg = 'No blockette 1000 found to determine record length'
                raise Exception(msg)
            end_pointer = starting_pointer + record_length
            # Set the file pointer to the beginning of the next record.
            mseed_file.seek(end_pointer)
            # Leave the loop if first record is set and no timing quality
            # could be found.
            if first_record and timing_quality == None:
                break
            if timing_quality != None:
                data.append(timing_quality)
            # Leave the loop when all records have been processed.
            if end_pointer >= filesize:
                break
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
        Cuts a MiniSEED file by cutting at records.
        
        The method takes a MiniSEED file and tries to match it as good as
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
        
        @param filename: File string of the MiniSEED file to be cut.
        @param starttime: L{obspy.core.UTCDateTime} object.
        @param endtime: L{obspy.core.UTCDateTime} object.
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
        start_record = int((starttime - start) / (end - start) * nr)
        end_record = int((endtime - start) / (end - start) * nr) + 1
        fh = open(filename, 'rb')
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
            fh.seek(start_record * info['record_length'])
            stime = self._getMSStarttime(fh)
            # Calculate last covered record.
            fh.seek(30, 1)
            (npts, sr_factor, sr_multiplier) = unpack('>Hhh', fh.read(6))
            # Calculate sample rate.
            sample_rate = self._calculateSamplingRate(sr_factor, sr_multiplier)
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
            fh.seek(end_record * info['record_length'])
            stime = self._getMSStarttime(fh)
            # Calculate last covered record.
            fh.seek(30, 1)
            (npts, sr_factor, sr_multiplier) = unpack('>Hhh', fh.read(6))
            # Calculate sample rate.
            sample_rate = self._calculateSamplingRate(sr_factor, sr_multiplier)
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
        # Open the file and read the cut file.
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
        # 1. METHOD LION
        #buffer_type = C.c_int32 * buffer_elements
        # Get address of array_in_c, which contains reference to the C array.
        #array_address = C.addressof(buffer.contents)
        # Make ctypes style array from C array.
        #ctypes_array = buffer_type.from_address(array_address)
        # Allocate numpy array to move memory to
        # Make a NumPy array from that.
        #return N.ctypeslib.as_array(ctypes_array)
        # 2. METHOD MORITZ 
        numpy_array = N.ndarray(buffer_elements, dtype='int32')
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
        #return N.frombuffer(C.pythonapi.PyBuffer_FromMemory(buffer,
        #                                                    buffer_elements*4),
        #                    dtype='int32',count=buffer_elements)

    def _calculateSamplingRate(self, samp_rate_factor, samp_rate_multiplier):
        """
        Calculates the actual sampling rate of the record.
        
        This is needed for manual readimg of MiniSEED headers. See the SEED
        Manual page 100 for details.
        
        @param samp_rate_factor: Field 10 of the fixed header in MiniSEED.
        @param samp_rate_multiplier: Field 11 of the fixed header in MiniSEED.
        """
        # Case 1
        if samp_rate_factor > 0 and samp_rate_multiplier > 0:
            return samp_rate_factor * float(samp_rate_multiplier)
        # Case 2
        elif samp_rate_factor > 0 and samp_rate_multiplier < 0:
            # Using float is needed to avoid integer division.
            return - 1 * samp_rate_factor / float(samp_rate_multiplier)
        # Case 3
        elif samp_rate_factor < 0 and samp_rate_multiplier > 0:
            return - 1 * samp_rate_multiplier / float(samp_rate_factor)
        # Case 4
        elif samp_rate_factor < 0 and samp_rate_multiplier < 0:
            return float(1) / (samp_rate_multiplier * samp_rate_factor)
        else:
            msg = 'The sampling rate of the record could not be determined.'
            raise Exception(msg)

    def _convertDatetimeToMSTime(self, dt):
        """
        Takes obspy.util.UTCDateTime object and returns an epoch time in ms.
        
        @param dt: obspy.util.UTCDateTime object.
        """
        return int(dt.timestamp * HPTMODULUS)

    def _convertMSTimeToDatetime(self, timestring):
        """
        Takes MiniSEED timestring and returns a obspy.util.UTCDateTime object.
        
        @param timestring: MiniSEED timestring (Epoch time string in ms).
        """
        return UTCDateTime.utcfromtimestamp(timestring / HPTMODULUS)

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

    def _getMSFileInfo(self, filename, real_name=None):
        """
        Takes a MiniSEED filename or an open file/StringIO as an argument and
        returns a dictionary with some basic information about the file.
        
        The information returned is: filesize, record_length,
        number_of_records and excess_bytes (bytes at the end not used by any
        record).
        
        If filename is an open file/StringIO the file pointer will not be
        changed by this method.
        
        @param filename: MiniSEED file string or an already open file.
        @param real_name: If filename is an open file you need to support the
            filesystem name of it so that the method is able to determine the
            file size. Use None if filename is a file string. Defaults to None.
        """
        info = {}
        # Filename is a true filename.
        if isinstance(filename, basestring) and not real_name:
            info['filesize'] = os.path.getsize(filename)
            #Open file and get record length using libmseed.
            mseed_file = open(filename, 'rb')
            starting_pointer = None
        # Filename is an open file and real_name is a string that refers to
        # a file.
        elif (isinstance(filename, file) or isinstance(filename, StringIO)) \
                and isinstance(real_name, basestring):
            # Save file pointer to restore it later on.
            starting_pointer = filename.tell()
            mseed_file = filename
            info['filesize'] = os.path.getsize(real_name)
        # Otherwise raise error.
        else:
            msg = 'filename either needs to be a string with a filename or ' + \
                  'an open file/StringIO object. If its a filename real_' + \
                  'name needs to be None, otherwise a string with a filename.'
            raise TypeError(msg)
        # Read all blockettes.
        mseed_file.seek(44)
        unpacked_tuple = unpack('>HH', mseed_file.read(4))
        blockettes_offset = unpacked_tuple[1] - 48
        mseed_file.seek(blockettes_offset, 1)
        blockettes = mseed_file.read(unpacked_tuple[0] - unpacked_tuple[1])
        # Loop over blockettes until Blockette 1000 is found.
        offset = 0
        while True:
            two_fields = unpack('>HH', blockettes[offset:offset + 4])
            if two_fields[0] == 1000:
                info['record_length'] = 2 ** unpack('>B', blockettes[6])[0]
                break
            else:
                # Only true when no blockette 1000 is present.
                if two_fields[1] <= 0:
                    msg = 'Record length could not be determined due to ' + \
                          'missing blockette 1000'
                    raise Exception(msg)
                offset = two_fields[1] - blockettes_offset
            # To avoid an infinite loop the total offset is checked.
            if offset >= len(blockettes):
                msg = 'Record length could not be determined due to ' + \
                          'missing blockette 1000'
                raise Exception(msg)
        # Number of total records.
        info['number_of_records'] = int(info['filesize'] / \
                                        info['record_length'])
        # Excess bytes that do not belong to a record.
        info['excess_bytes'] = info['filesize'] % info['record_length']
        if starting_pointer:
            mseed_file.seek(starting_pointer)
        else:
            mseed_file.close()
        return info

    def _getMSStarttime(self, open_file):
        """
        Returns the starttime of the given MiniSEED record and returns a 
        UTCDateTime object.
        
        Due to various possible time correction it is complicated to get the
        actual start time of a MiniSEED file. This method hopefully handles
        all possible cases.
        
        It evaluates fields 8, 12 and 16 of the fixed header section and
        additionally blockettes 500 and 1001 which both contain (the same?)
        microsecond correction. If both blockettes are available only blockette
        1001 is used. Not sure if this is the correct way to handle it but
        I could not find anything in the SEED manual nor an example file.
        
        Please see the SEED manual for additional information.
        
        @param open_file: Open file or StringIO. The pointer has to be set
            at the beginning of the record of interst. When the method is
            done with the calulations it will reset the file pointer to the
            original state.
        """
        # Save the originial state of the file pointer.
        file_pointer_start = open_file.tell()
        # Jump to the beginning of field 8 and read the rest of the fixed
        # header section.
        open_file.seek(20, 1)
        # Unpack the starttime, field 12, 16, 17 and 18.
        unpacked_tuple = unpack('>HHBBBxHxxxxxxBxxxiHH', open_file.read(28))
        # Use field 17 to calculate how long all blockettes are and read them.
        blockettes = open_file.read(unpacked_tuple[-2] - 48)
        # Reset the file_pointer
        open_file.seek(file_pointer_start, 0)
        time_correction = 0
        # Check if bit 1 of field 12 has not been set.
        if unpacked_tuple[6] & 2 == 0:
            # If it has not been set the time correction of field 16 still
            # needs to be applied. The units are in 0.0001 seconds.
            time_correction += unpacked_tuple[7] * 100
        # Loop through the blockettes to find blockettes 500 and 1001.
        offset = 0
        blkt_500 = 0
        blkt_1001 = 0
        while True:
            # Get blockette number.
            cur_blockette = unpack('>H', blockettes[offset : offset + 2])
            if cur_blockette == 1001:
                blkt_1001 = unpack('>H', blockettes[5])
                if unpack('>H', blockettes[offset + 2 : offset + 4]) == 0:
                    break
            if cur_blockette == 500:
                blkt_500 = unpack('>H', blockettes[19])
                if unpack('>H', blockettes[offset + 2 : offset + 4]) == 0:
                    break
            next_blockette = unpack('>H',
                                    blockettes[offset + 2 : offset + 4])[0]
            # Leave the loop if no further blockettes follow.
            if next_blockette == 0:
                break
            # New offset.
            offset = next_blockette - 48
        # Adjust the starrtime. Blockette 1001 will override blkt_500.
        additional_correction = 0
        if blkt_500:
            additional_correction = blkt_500
        if blkt_1001:
            additional_correction = blkt_1001
        # Return a UTCDateTime object with the applied corrections.
        starttime = UTCDateTime(year=unpacked_tuple[0],
                        julday=unpacked_tuple[1], hour=unpacked_tuple[2],
                        minute=unpacked_tuple[3], second=unpacked_tuple[4],
                        microsecond=unpacked_tuple[5] * 100)
        # Due to weird bug a difference between positive and negative offsets
        # is needed.
        total_correction = time_correction + additional_correction
        if total_correction < 0:
            starttime = starttime - abs(total_correction) / 1e6
        else:
            starttime = starttime + total_correction / 1e6
        return starttime

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
            #chain.contents.datasamples = trace_list[_i][1].ctypes.data_as(C.POINTER(C.c_long))
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
