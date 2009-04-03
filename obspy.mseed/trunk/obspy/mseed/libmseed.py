# -*- coding: utf-8 -*-
"""
Wrapper class for libmseed - The Mini-SEED library.

Currently only supports MiniSEED files with integer data values.

This library is free software; you can redistribute it and/or modify
it under the terms of the GNU Library General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.

This library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Library General Public License (GNU-LGPL) for more details.  The
GNU-LGPL and further information can be found here:
http://www.gnu.org/
"""

from datetime import datetime
from obspy.mseed.headers import MSRecord, MSTraceGroup, MSTrace, HPTMODULUS
from time import mktime
import StringIO
import ctypes as C
import math
import os
import platform
import sys


#Import libmseed library.
if sys.platform=='win32':
    lib_name = 'libmseed.win32.dll'
else:
    if platform.architecture()[0] == '64bit':
        lib_name = 'libmseed.lin64.so'
    else:
        lib_name = 'libmseed.so'
clibmseed = C.CDLL(os.path.join(os.path.dirname(__file__), 'libmseed', 
                                lib_name))


class libmseed(object):
    """
    Wrapper class for libmseed.
    """
    def msr2dict(self, m):
        """
        """
        h = {}
        h['reclen'] = m.contents.reclen
        h['sequence_number'] = m.contents.sequence_number
        h['network'] = m.contents.network
        h['station'] = m.contents.station
        h['location'] = m.contents.location
        h['channel'] = m.contents.channel
        h['dataquality'] = m.contents.dataquality
        h['starttime'] = m.contents.starttime
        h['samprate'] = m.contents.samprate
        h['samplecnt'] = m.contents.samplecnt
        h['encoding'] = m.contents.encoding
        h['byteorder'] = m.contents.byteorder
        h['encoding'] = m.contents.encoding
        h['sampletype'] = m.contents.sampletype
        return h
    
    def mst2dict(self, m):
        """
        Return dictionary from MSTrace Object m, leaving the attributes
        datasamples, ststate and next out
        """
        h = {}
        h["network"] = m.contents.network
        h["station"] = m.contents.station
        h["location"] = m.contents.location
        h["channel"] = m.contents.channel
        h["dataquality"] = m.contents.dataquality
        h["type"] = m.contents.type
        h["starttime"] = m.contents.starttime
        h["endtime"] = m.contents.endtime
        h["samprate"] = m.contents.samprate
        h["samplecnt"] = m.contents.samplecnt
        h["numsamples"] = m.contents.numsamples
        h["sampletype"] = m.contents.sampletype
        return h

    def dict2mst(self, m, h):
        """
        Takes dictionary containing MSTrace header data and writes them to the
        MSTrace Group
        """
        m.contents.network =h["network"]
        m.contents.station = h["station"] 
        m.contents.location = h["location"]
        m.contents.channel = h["channel"]
        m.contents.dataquality = h["dataquality"]
        m.contents.type = h["type"]
        m.contents.starttime = h["starttime"]
        m.contents.endtime = h["endtime"]
        m.contents.samprate = h["samprate"]
        m.contents.samplecnt = h["samplecnt"]
        m.contents.numsamples = h["numsamples"]
        m.contents.sampletype = h["sampletype"]
        
    def compareHeaders(self, header1, msrecord):
        if len(msrecord) == 0:
            return True
        #msrecord[len(msrecord)-1][0]
        else:
            return False
    
    def read_ms_using_traces(self, filename, dataflag = 1):
        """
        Read Mini-SEED file. Header, Data and numtraces are returned

        filename        - Name of file to read Mini-SEED data from
        timetol           - Time tolerance, default is 1/2 sample period (-1)
        sampratetol   - Sample rate tolerance, default is rate dependent (-1)
        verbosity       - Level of diagnostic messages, default 0
        """
        #Creates MSTraceGroup Structure
        mstg = self.readTraces(filename)
        data=[]
        header=[]
        mst = mstg.contents.traces
        numtraces = mstg.contents.numtraces
        for _i in range(numtraces):
            data.extend(mst.contents.datasamples[0:mst.contents.numsamples])
            header.append(self.mst2dict(mst))
            mst = mst.contents.next
        return header[0],data, numtraces
    
    def readMSusingRecords(self, filename):
        """
        Reads a given Mini-SEED file and parses all information.
        
        Structure of the returned list:
        [[header for trace 1, data] , [header for trace 2, data], ...]
        """
        msrecord=[]
        retcode=0
        while retcode == 0:
            msr, retcode = self.read_MSRec(filename)
            if retcode == 0:
                header=self.msr2dict(msr)
                #Sanity check
                if header['samplecnt'] != msr.contents.numsamples:
                    print "Warning: The number of samples unpacked does not"
                    print "correspond with the number of samples specified in the header."
#                if len(msrecord) == 0:
#                data=msr.contents.datasamples[0:msr.contents.numsamples]
                if self.compareHeaders(header, msrecord):
                    print "Same Trace"
                else:
                    msrecord.append([header, 0])
        return msrecord
    
    def read_MSRec(self, filename, reclen = -1, dataflag = 1, skipnotdata = 1, 
                   verbose = 0):
        """
        Reads Mini-SEED file and populates MS Record data structure.
        
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
        """
        # init MSRecord structure
        clibmseed.msr_init.restype = C.POINTER(MSRecord)
        msr = clibmseed.msr_init(None)
        # defines return type
        clibmseed.ms_readmsr.restype = C.c_int
        # read the file and write the relevant information to msr
        retcode = clibmseed.ms_readmsr(C.pointer(msr), filename, 
                                       C.c_int(reclen), None, None,
                                       C.c_short(skipnotdata), 
                                       C.c_short(dataflag), C.c_short(verbose))
        return msr, retcode
    
    def populate_MSTG(self, header, data, numtraces=1):
        """
        Populates MSTrace_Group structure from given header, data and
        numtraces and returns the MSTrace_Group
        """
        #Init MSTraceGroupint
        clibmseed.mst_initgroup.restype = C.POINTER(MSTraceGroup)
        mstg = clibmseed.mst_initgroup(None)
        #Init MSTrace object
        clibmseed.mst_init.restype = C.POINTER(MSTrace)
        #Connect Group with Traces
        mstg.contents.traces=clibmseed.mst_init(None)
        #Write header in MSTrace structure
        self.dict2mst(mstg.contents.traces, header)
        #Needs to be redone, dynamic??
        mstg.contents.numtraces=numtraces
        #Create void pointer and allocates more memory to it
        #tempdatpoint=C.c_void_p()
        tempdatpoint=C.c_int()
        C.resize(tempdatpoint,
                 clibmseed.ms_samplesize(C.c_char(header['sampletype']))*
                 header['numsamples'])
        #Set pointer to tempdatpoint
        mstg.contents.traces.contents.datasamples=C.pointer(tempdatpoint)
        #Write data in MSTrace structure
        for i in range(header['numsamples']):
            #mstg.contents.traces.contents.datasamples[i]=C.c_void_p(data[i])
            mstg.contents.traces.contents.datasamples[i]=C.c_int(data[i])
        return mstg

    def mst2file(self, mst, outfile, reclen, encoding, byteorder, flush, verbose):
        """
        Takes MS Trace object and writes it to a file
        """
        mseedfile=open(outfile, 'wb')
        #Initialize packedsamples pointer for the mst_pack function
        self.packedsamples = C.pointer(C.c_int(0))
        #Callback function for mst_pack to actually write the file
        def record_handler(record, reclen, _stream):
            mseedfile.write(record[0:reclen])
        #Define Python callback function for use in C function
        RECHANDLER = C.CFUNCTYPE(None, C.POINTER(C.c_char), C.c_int, C.c_void_p)
        rec_handler = RECHANDLER(record_handler)
        #Pack the file into a MiniSEED file
        clibmseed.mst_pack(mst, rec_handler, None, reclen, encoding, byteorder,
                           self.packedsamples, flush, verbose, None)
        mseedfile.close()
    
    def write_ms(self,header,data, outfile, numtraces=1, reclen= -1,
                 encoding=-1, byteorder=-1, flush=-1, verbose=0):
        """
        Write Miniseed file from header, data and numtraces
        
        header    - Dictionary containing the header files
        data      - List of the datasamples
        outfile   - Name of the output file
        numtraces - Number of traces in trace chain (Use??)
        reclen    - should be set to the desired data record length in bytes
                    which must be expressible as 2 raised to the power of X 
                    where X is between (and including) 8 to 20. -1 defaults to
                    4096
        encoding  - should be set to one of the following supported Mini-SEED
                    data encoding formats: DE_ASCII (0), DE_INT16 (1), 
                    DE_INT32 (3), DE_FLOAT32 (4), DE_FLOAT64 (5), DE_STEIM1 (10)
                    and DE_STEIM2 (11). -1 defaults to STEIM-2 (11)
        byteorder - must be either 0 (LSBF or little-endian) or 1 (MBF or 
                    big-endian). -1 defaults to big-endian (1)
        flush     - if it is not zero all of the data will be packed into 
                    records, otherwise records will only be packed while there
                    are enough data samples to completely fill a record.
        verbose   - controls verbosity, a value of zero will result in no 
                    diagnostic output.
        """
        #Populate MSTG Structure
        mstg=self.populate_MSTG(header, data, numtraces)
        #Write File from MS-Trace structure
        self.mst2file(mstg.contents.traces, outfile, reclen, encoding, byteorder,
                      flush, verbose)
    
    def cut_ms(self, data, header, stime, cutsamplecount, outfile='cut.mseed'):
        """
        Takes a data file list, a header dictionary, a starttime, the number of 
        samples to cut and writes it in outfile.
        stime             - The time in microseconds with the origin set to the
                                      beginning of the file
        cutsamplecount  - The number of samples to cut
        outfile                  - filename of the Record to write
        """
        samprate_in_microsecs = header['samprate']/1e6
        #Modifiy the header
        header['starttime'] = header['starttime'] + stime
        header['endtime'] = int(header['starttime'] + cutsamplecount/\
                                samprate_in_microsecs)
        header['numsamples'] = cutsamplecount
        header['samplecnt'] = cutsamplecount
        #Make new data list, some rounding issues need to be solved
        cutdata=data[int(stime/samprate_in_microsecs):cutsamplecount+1]
        #Write cutted file
        self.write_ms(header, cutdata, outfile)
    
    def readTraces(self, filename, reclen = -1, timetol = -1, sampratetol = -1,
                   dataflag = 1, skipnotdata = 1, verbose = 0):
        """
        Reads MiniSEED data from file. Returns MSTraceGroup structure.
        
        @param filename: Mini-SEED file to be read.
        @param reclen: If reclen is 0 the length of the first record is auto- 
            detected. All subsequent records are then expected to have the 
            same record length. If reclen is negative the length of every 
            record is automatically detected. Defaults to -1.
        @param timetol: Time tolerance, default to -1 (1/2 sample period).
        @param sampratetol: Sample rate tolerance, defaults to -1 (rate 
            dependent)
        @param dataflag: Controls whether data samples are unpacked, defaults 
            to 1
        @param skipnotdata: If true (not zero) any data chunks read that to do 
            not have valid data record indicators will be skipped. Defaults to 
            true (1).
        @param verbose: Controls verbosity from 0 to 2. Defaults to None (0).
        """
        # Creates MSTraceGroup Structure
        mstg = C.pointer(MSTraceGroup())
        # Uses libmseed to read the file and populate the MSTraceGroup structure
        errcode = clibmseed.ms_readtraces(
            C.pointer(mstg), filename, C.c_int(reclen), 
            C.c_double(timetol), C.c_double(sampratetol),
            C.c_short(dataflag), C.c_short(skipnotdata), 
            C.c_short(dataflag), C.c_short(verbose))
        if errcode != 0:
            assert 0, "\n\nError while reading Mini-SEED file: "+filename
        #Reset memory
        self.resetMs_readmsr()
        return mstg
    
    def resetMs_readmsr(self):
        """
        Cleans memory and resets the clibmseed.ms_readmsr() function.
        
        This function needs to be called after reading a Mini-SEED file with
        any of the provided methods if you want to read another Mini-SEED file
        without restarting Python. If this is not done it will probably still
        work but raise a warning.
        """
        clibmseed.msr_init.restype = C.POINTER(MSRecord)
        msr=clibmseed.msr_init(None)
        clibmseed.ms_readmsr(msr, None, 0, None, None, 0, 0, 0)
    
    def getFirstRecordHeaderInfo(self, file):
        """
        Takes a Mini-SEED file and returns header of the first record.
        
        Returns a dictionary containing some header information from the first
        record of the Mini-SEED file only. It returns the location, network,
        station and channel information.
        
        @param file: Mini-SEED file string.
        """
        # read first header only
        msr = self.read_MSRec(file, dataflag = 0)
        header = {}
        chain = msr[0].contents
        # header attributes to be read
        attributes = ('location', 'network', 'station', 'channel')
        # loop over attributes
        for _i in attributes:
            header[_i] = getattr(chain, _i)
        return header
    
    def getStartAndEndTime(self, file):
        """
        Returns the start- and endtime of a Mini-SEED file.
        
        This method only reads the first and the last record. Thus it will work
        only work correctly for files containing only one trace.
        
        @param file: Mini-SEED file string.
        """
        # read starttime and record length from first header
        msr = self.read_MSRec(file, dataflag = 0)
        chain = msr[0].contents
        starttime = chain.starttime
        record_length = chain.reclen
        # cleanup memory and close old msr file
        self.resetMs_readmsr()
        # open Mini-SEED file and write last record into temporary file
        mseed_file = open(file, 'rb')
        mseed_file.seek(-record_length, 2)
        temp_record = open('temp', 'wb')
        temp_record.write(mseed_file.read(record_length))
        mseed_file.close()
        temp_record.close()
        # read last record
        msr = self.read_MSRec('temp', dataflag = 0)
        chain = msr[0].contents
        # calculate endtime
        endtime = chain.starttime + (chain.samplecnt -1) / chain.samprate *\
                  HPTMODULUS
        # remove temporary file
        #os.remove('temp')
        return (self.MSTime2Datetime(starttime), self.MSTime2Datetime(endtime))
        
    def Datetime2MSTime(self, dt):
        """
        Takes datetime object and returns an epoch time in ms.
        
        @param dt: Datetime object.
        """
        return long((mktime(dt.timetuple()) * HPTMODULUS) + dt.microsecond)
    
    def MSTime2Datetime(self, timestring):
        """
        Takes Mini-SEED timestring and returns a Python datetime object.
        
        @param timestring: Mini-SEED timestring (Epoch time string in ms).
        """
        return datetime.fromtimestamp(timestring / HPTMODULUS)
    
    def printGapList(self, filename, time_tolerance = -1, 
                     samprate_tolerance = -1, min_gap = None, max_gap = None):
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
    
    def getGapList(self, filename, time_tolerance = -1, 
                   samprate_tolerance = -1, min_gap = None, max_gap = None):
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
        mstg = self.readTraces(str(filename), dataflag = 0, skipnotdata = 0,
                               timetol = time_tolerance,
                               sampratetol = samprate_tolerance)
        gap_list = []
        # iterate through traces
        cur = mstg.contents.traces.contents
        for _ in xrange(mstg.contents.numtraces-1):
            next = cur.next.contents
            # Skip MSTraces with 0 sample rate, usually from SOH records
            if cur.samprate == 0:
                cur = next
                continue
            # Check that sample rates match using default tolerance
            if not self._isRateTolerable(cur.samprate, next.samprate):
                msg = "%s Sample rate changed! %.10g -> %.10g\n"
                print msg % (cur.samprate, next.samprate)
            gap = (next.starttime - cur.endtime) / HPTMODULUS
            # Check that any overlap is not larger than the trace coverage
            if gap < 0:
                if next.samprate:
                    delta =  1 / float(next.samprate)
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
                nsamples-=1
            else:
                nsamples+=1
            # Convert to python datetime objects
            time1 = datetime.utcfromtimestamp(cur.endtime / HPTMODULUS)
            time2 = datetime.utcfromtimestamp(next.starttime / HPTMODULUS)
            gap_list.append((cur.network, cur.station, cur.location, 
                             cur.channel, time1, time2, gap, nsamples))
            cur = next
        return gap_list
    
    def _isRateTolerable(self, sr1, sr2):
        """
        Tests default sample rate tolerance: abs(1-sr1/sr2) < 0.0001
        """
        return math.fabs(1.0 - (sr1 / float(sr2))) < 0.0001
    
    def graphCreateMinMaxTimestampList(self, file, width, starttime = None, 
                                       endtime = None):
        """
        Creates a list with tuples containing a minimum value, a maximum value
        and a timestamp in microseconds.
        
        Only values between the start- and the endtime will be calculated. The
        first two items of the returned list are the actual start- and endtimes
        of the returned list. This is needed to cope with many different
        Mini-SEED files.
        The returned timestamps are the mean times of the minmax value pair.
        
        @requires: The Mini-SEED file has to contain only one trace. It may
            contain gaps and overlaps and it may be arranged in any order but
            the first and last records must be in chronological order as they
            are used to determine the start- and endtime.
        
        @param file: Mini-SEED file string.
        @param width: Number of tuples in the list. Corresponds to the width
            in pixel of the graph.
        @param starttime: Starttime of the List/Graph as a Datetime object. If
            none is supplied the starttime of the file will be used.
            Defaults to None.
        @param endtime: Endtime of the List/Graph as a Datetime object. If none
            is supplied the endtime of the file will be used.
            Defaults to None.
        """
        #Read traces using the readTraces method.
        mstg = self.readTraces(file, skipnotdata = 0)
        #Create list with start-, endtime and number in chain.
        timeslist = []
        cur = mstg.contents.traces.contents
        for _i in range(mstg.contents.numtraces):
            timeslist.append((cur.starttime, cur.endtime, _i + 1))
            if _i + 1 < mstg.contents.numtraces:
                cur = cur.next.contents
        #Sort list according to starttime.
        timeslist.sort()
        #Get start- and endtime and convert them too microsecond timestamp.
        start_and_end_time = self.getStartAndEndTime(file)
        if not starttime:
            starttime = self.Datetime2MSTime(start_and_end_time[0])
        else:
            starttime = self.Datetime2MSTime(starttime)
        if not endtime:
            endtime = self.Datetime2MSTime(start_and_end_time[1])
        else:
            endtime = self.Datetime2MSTime(endtime)
        #Calculate time for one pixel.
        stepsize = (endtime - starttime) / width
        #First two items are start- and endtime.
        minmaxlist=[starttime, endtime]
        #While loop over the plotting duration.
        while starttime < endtime:
            pixel_endtime = starttime + stepsize
            tempdatlist = []
            #Inner Loop over all times.
            for _i in timeslist:
                #Calculate current chain in the MSTraceGroup Structure.
                chain = mstg.contents.traces.contents
                for _ in xrange(_i[2] - 1):
                    chain = chain.next.contents
                #If the starttime is bigger than the endtime of the current
                #trace delete the item from the list.
                if starttime > _i[1]:
                    #Still need to figure out how to delete the item from the
                    #list.
                    pass
                elif starttime < _i[0]:
                    #If starttime and endtime of the current pixel are too
                    #small than leave the list.
                    if pixel_endtime < _i[0]:
                        #Leave the loop.
                        pass
                    #Otherwise append the border to tempdatlist.
                    else:
                        end = float((pixel_endtime - _i[0])) / \
                              (_i[1] - _i[0]) * chain.samplecnt
                        if end > _i[1]:
                            end = _i[1]
                        tempdatlist.extend(chain.datasamples[0 : int(end)])
                #Starttime is right in the current trace.
                else:
                    #Endtime also is in the trace. Append to tempdatlist.
                    if pixel_endtime < _i[1]:
                        start = float((starttime - _i[0])) / (_i[1] - _i[0]) *\
                                chain.samplecnt
                        end = float((pixel_endtime - _i[0])) / \
                              (_i[1] - _i[0]) * chain.samplecnt
                        tempdatlist.extend(chain.datasamples[int(start) : \
                                                             int(end)])
                    #Endtime is not in the trace. Append to tempdatlist.
                    else:
                        start = float((starttime - _i[0])) / (_i[1] - _i[0]) *\
                                chain.samplecnt
                        tempdatlist.extend(chain.datasamples[int(start) : \
                                                             chain.samplecnt])
            #If empty list do nothing.
            if tempdatlist == []:
                pass
            #If not empty append min, max and timestamp values to list.
            else:
                minmaxlist.append((min(tempdatlist), max(tempdatlist), 
                                   starttime + 0.5 * stepsize))
            #New starttime for while loop.
            starttime = pixel_endtime
        #Reset memory
        self.resetMs_readmsr()
        return minmaxlist
    
    def graph_create_graph(self, file, outfile = None, format = None,
                           size = (800, 200), starttime = False,
                           endtime = False, dpi = 100, color = 'red',
                           bgcolor = 'white', transparent = False,
                           minmaxlist = False):
        """
        Creates a graph of any given Mini-SEED file. It either saves the image
        directly to the file system or returns an binary image string.
        
        Currently only supports files with one continuous trace. I still have
        to figure out how to remove the frame around the graph and create the
        option to set a start and end time of the graph.
        
        The option to set a start- and endtime to plot currently only works
        for starttime smaller and endtime greater than the file's times.
        
        For all color values you can use legit html names, html hex strings
        (e.g. '#eeefff') or you can pass an R , G , B tuple, where each of
        R , G , B are in the range [0,1]. You can also use single letters for
        basic builtin colors ('b' = blue, 'g' = green, 'r' = red, 'c' = cyan,
        'm' = magenta, 'y' = yellow, 'k' = black, 'w' = white) and gray shades
        can be given as a string encoding a float in the 0-1 range.
        
        @param file: Mini-SEED file string
        @param outfile: Output file string. Also used to automatically
            determine the output format. Currently supported is emf, eps, pdf,
            png, ps, raw, rgba, svg and svgz output.
            Defaults to None.
        @param format: Format of the graph picture. If no format is given the
            outfile parameter will be used to try to automatically determine
            the output format. If no format is found it defaults to png output.
            If no outfile is specified but a format is than a binary
            imagestring will be returned.
            Defaults to None.
        @param size: Size tupel in pixel for the output file. This corresponds
            to the resolution of the graph for vector formats.
            Defaults to 800x200 px.
        @param starttime: Starttime of the graph as a datetime object. If not
            set the graph will be plotted from the beginning.
            Defaults to False.
        @param endtime: Endtime of the graph as a datetime object. If not set
            the graph will be plotted until the end.
            Defaults to False.
        @param dpi: Dots per inch of the output file. This also affects the
            size of most elements in the graph (text, linewidth, ...).
            Defaults to 100.
        @param color: Color of the graph. Defaults to 'red'.
        @param bgcolor: Background color of the graph. Defaults to 'white'.
        @param transparent: Make all backgrounds transparent (True/False). This
            will overwrite the bgcolor param.
            Defaults to False.
        @param minmaxlist: A list containing minimum, maximum and timestamp
            values. If none is supplied it will be created automatically.
            Useful for caching.
            Defaults to False.
        """
        #Either outfile or format needs to be set.
        if not outfile and not format:
            raise ValueError('Either outfile or format needs to be set.')
        #Get a list with minimum and maximum values.
        if not minmaxlist:
            minmaxlist = self.graphCreateMinMaxTimestampList(file = file,
                                                    width = size[0],
                                                    starttime = starttime,
                                                    endtime = endtime)
        starttime = minmaxlist[0]
        endtime = minmaxlist[1]
        minmaxlist = minmaxlist[2:]
        length = len(minmaxlist)
        #Importing pyplot and numpy.
        import matplotlib.pyplot as plt
        #Setup figure and axes
        fig = plt.figure(num = None, figsize = (float(size[0])/dpi,
                         float(size[1])/dpi))
        ax = fig.add_subplot(111)
        # hide axes + ticks
        ax.axison = False
        #Make the graph fill the whole image.
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        #Determine range for the y axis. This may not be the smartest way to
        #do it.
        miny = 99999999999999999
        maxy = -9999999999999999
        for _i in range(length):
            try:
                if minmaxlist[_i][0] < miny:
                    miny = minmaxlist[_i][0]
            except:
                pass
            try:
                if minmaxlist[_i][1] > maxy:
                    maxy = minmaxlist[_i][1]
            except:
                pass
        #Set axes and disable ticks
        plt.ylim(miny, maxy)
        plt.xlim(starttime, endtime)
        plt.yticks([])
        plt.xticks([])
        #Draw horizontal lines.
        for _i in range(length):
            #Calculate relative values needed for drawing the lines.
            yy = (float(minmaxlist[_i][0])-miny)/(maxy-miny)
            xx = (float(minmaxlist[_i][1])-miny)/(maxy-miny)
            plt.axvline(x = minmaxlist[_i][2], ymin = yy, ymax = xx,
                        color = color)
        #Save file.
        if outfile:
            #If format is set use it.
            if format:
                plt.savefig(outfile, dpi = dpi, transparent = transparent,
                    facecolor = bgcolor, edgecolor = bgcolor, format = format)
            #Otherwise try to get the format from outfile or default to png.
            else:
                plt.savefig(outfile, dpi = dpi, transparent = transparent,
                    facecolor = bgcolor, edgecolor = bgcolor)
        #Return an binary imagestring if outfile is not set but format is.
        if not outfile:
            imgdata = StringIO.StringIO()
            plt.savefig(imgdata, dpi = dpi, transparent = transparent,
                    facecolor = bgcolor, edgecolor = bgcolor, format = format)
            imgdata.seek(0)
            return imgdata.read()
