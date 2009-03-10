# -*- coding: utf-8 -*-
"""
Wrapper class for libmseed - The Mini-SEED library.

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

from obspy.mseed.headers import MSRecord, MSTraceGroup, MSTrace, HPTMODULUS
import ctypes as C
from datetime import datetime
import math
import os
import sys

# import libmseed library
if sys.platform=='win32':
    lib_name = 'libmseed.win32.dll'
else:
    lib_name = 'libmseed.so'
clibmseed = C.CDLL(os.path.join(os.path.dirname(__file__), 'libmseed', 
                                lib_name))


class libmseed(object):
    """
    Wrapper class for libmseed.
    """
    
    def printtracelist(self,filename,timeformat= 0,details = 0, gaps=0):
        """
        Prints information about the traces in a Mini-SEED file using the libmseed
        method printtracelist.
        
        Prints all informations to stdout.
        
        filename          - Name of file to read Mini-SEED data from
        timeformat      - Controls the format of the resulting time string, default = 0
                                    0 : SEED time format (2005,146,00:00:00.000000)
                                    1 : ISO time format (2005-05-26T00:00:00.000000)
                                    2 : Epoch time, seconds since the epoch (1117065600.00000000)
        details             - If the details flag is greater than 0 the sample rate and 
                                  sample count are printed for each trace, default = 0
        gaps                - If the gaps flag is greater than zero the time gap from the 
                                  previous MSTrace (if the source name matches) is printed, 
                                  default = 0
        """
        mstg = self.readtraces(filename, dataflag = 0,skipnotdata = 0)
        clibmseed.mst_printtracelist(mstg,timeformat,details,gaps)
    
    def printgaplist(self,filename,timeformat= 0,mingap = 0, maxgap=0):
        """
        Prints a formatted list of the gaps between MSTrace segments in the
        given MSTraceGroup to stdout. If mingap or maxgap is not NULL their 
        values will be enforced and only gaps/overlaps matching their implied
        criteria will be printed.
        
        Uses the libmseed function printgapslist.
        
        filename          - Name of file to read Mini-SEED data from
        timeformat      - Controls the format of the resulting time string, defaults to 0
                                    0 : SEED time format (2005,146,00:00:00.000000)
                                    1 : ISO time format (2005-05-26T00:00:00.000000)
                                    2 : Epoch time, seconds since the epoch (1117065600.00000000)
        mingap            - defaults to 0
        maxgap           - defaults to 0
        """
        mstg = self.readtraces(filename, dataflag = 0, skipnotdata = 0)
        clibmseed.mst_printgaplist(mstg,timeformat,mingap,maxgap)
    
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

    def read_MSRec(self, filename, reclen = -1, dataflag = 1, 
                   skipnotdata = 1, verbose = 0):
        """
        Reads Mini-SEED file and populates MS Record data structure with subsequent
        calls.
        
        filename        - Mini-SEED file to be read
        reclen            - If reclen is 0 the length of the first record is auto-
                                detected. All subsequent records are then expected to
                                have the same record length.
                                If reclen is negative the length of every record is
                                automatically detected.
                                Defaults to -1.
        dataflag        - Controls whether data samples are unpacked, defaults to 1
        skipnotdata   - If true (not zero) any data chunks read that to do not
                                have valid data record indicators will be skipped.
                                Defaults to true (1).
        verbose         - Controls verbosity from 0 to 2. Defaults to None (0).
        """
        #Init MSRecord structure
        clibmseed.msr_init.restype = C.POINTER(MSRecord)
        msr=clibmseed.msr_init(None)
        #Defines return type
        clibmseed.ms_readmsr.restype = C.c_int
        #Read the file and write the relevant information to msr
        retcode=clibmseed.ms_readmsr(C.pointer(msr), filename, C.c_int(reclen),
                             None, None,
                             C.c_short(skipnotdata), C.c_short(dataflag),
                             C.c_short(verbose))
        return msr,retcode

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
        tempdatpoint=C.c_void_p()
        C.resize(tempdatpoint,
                 clibmseed.ms_samplesize(C.c_char(header['sampletype']))*
                 header['numsamples'])
        #Set pointer to tempdatpoint
        mstg.contents.traces.contents.datasamples=C.pointer(tempdatpoint)
        #Write data in MSTrace structure
        for i in range(header['numsamples']):
            mstg.contents.traces.contents.datasamples[i]=C.c_void_p(data[i])
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
        header['endtime'] = int(header['starttime'] + cutsamplecount/samprate_in_microsecs)
        header['numsamples'] = cutsamplecount
        header['samplecnt'] = cutsamplecount
        #Make new data list, some rounding issues need to be solved
        cutdata=data[int(stime/samprate_in_microsecs):cutsamplecount+1]
        #Write cutted file
        self.write_ms(header, cutdata, outfile)
    
    def printrecordinfo(self, file):
        """
        Reads Mini-SEED file using subsequent calls to read_MSRec and prints
        general information about all records in the file and any gaps/overlaps
        present in the file
        """
        print "Records in",file,":"
        print "---------------------------------------------------"
        retcode=0
        oldstarttime=0
        while retcode == 0:
            msr, retcode=self.read_MSRec(file, dataflag=0, skipnotdata=0)
            if retcode == 0:
                    if oldstarttime!=0:
                        if msr.contents.starttime-oldstarttime==0:
                            print "NO GAPS/OVERLAPS"
                        elif msr.contents.starttime-oldstarttime<0:
                            print "OVERLAP"
                        else:
                            print "GAP"
                    oldstarttime=long(msr.contents.starttime+msr.contents.samplecnt*(1/msr.contents.samprate)*1e6)
                    print "Sequence number:",msr.contents.sequence_number,"--",
                    print "starttime:",msr.contents.starttime,", # of samples:",
                    print msr.contents.samplecnt,"=> endtime :",
                    print oldstarttime
    
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
        return mstg
    
    def isRateTolerable(self, sr1, sr2):
        """
        Tests default sample rate tolerance: abs(1-sr1/sr2) < 0.0001
        """
        return math.fabs(1.0 - (sr1 / float(sr2))) < 0.0001
    
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
            if not self.isRateTolerable(cur.samprate, next.samprate):
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
    def graph_createMinMaxList(self, file, width):
        """
        Returns a list that consists of pairs of minimum and maxiumum data
        values.
        
        file  -    Mini-SEED file
        width -    desired width in pixel of the data graph/Number of pairs of
                   mimima and maxima.
        """
        minmaxlist=[]
        #Read traces using readTraces
        mstg = self.readTraces(file, skipnotdata = 0)
        chain = mstg.contents.traces.contents
        #Number of datasamples in one pixel
        stepsize = chain.numsamples/width
        #Loop over datasamples and append minmaxlist
        for _i in range(width):
            tempdatlist = chain.datasamples[_i*stepsize: (_i+1)*stepsize]
            minmaxlist.append([min(tempdatlist),max(tempdatlist)])
        return minmaxlist
    
    def getMinMaxList(self, file, width):
        """
        Returns a list that consists of minimum and maximum data values.
        
        @param file: Mini-SEED file string.
        @param width: Desired width in pixel of the data graph/number of 
            values of returned data list.
        """
        # Read traces
        mstg = self.readTraces(file, skipnotdata = 0)
        chain = mstg.contents.traces.contents
        # Number of datasamples in one pixel
        if width >= chain.numsamples:
            width = chain.numsamples
        stepsize = int(chain.numsamples/width)
        # Loop over datasamples and append to minmaxlist
        data=[]
        for x in xrange(0, width):
            temp = chain.datasamples[x*stepsize:(x+1)*stepsize]
            if x%2:
                data.append(min(temp))
            else:
                data.append(max(temp))
        return data