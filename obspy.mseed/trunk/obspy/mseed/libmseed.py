# -*- coding: utf-8 -*-
"""
Wrapper class for libmseed.
"""

from obspy.mseed.headers import MSRecord, MSTraceGroup, MSTrace
import ctypes as C
import os
import sys

# import libmseed library
if sys.platform=='win32':
    lib_name = 'libmseed.win32.dll'
else:
    lib_name = 'libmseed.so'
clibmseed = C.CDLL(os.path.join(os.path.dirname(__file__),'libmseed', lib_name))


class libmseed(object):
    
    def __init__(self, file="test.mseed"):
        """
        Inits the class and writes all attributes of the MSRecord structure in
        self.attrs
        """
        self.attrs = []
        for (name,type) in MSRecord._fields_:
            self.attrs.append(name)
        self.defaultfile=file

    def __getattr__(self, name):
        """
        Fetches C-Attributes from the MSRecord Structure.
        
        Currently only works with the first record of the Mseedfile and the 
        test.mseed file.
        """
       
        if name in self.attrs:
            clibmseed.ms_readmsr.restype = C.c_int
            #needed to write the MSRecord
            msr = C.pointer(MSRecord())
            #Checks whether the returned msr is the last record. If it is the last record it will be 1.
            islast = C.pointer(C.c_int(0))
            for _i in range(1):
                clibmseed.ms_readmsr(C.pointer(msr), self.defaultfile, 0, None, islast,1, 1, 0)
                if islast.contents.value == 1:
                    break
            return getattr(msr.contents, name)
        else:
            return self.__dict__[name] 
    
    def __setattr__(self, name, val):
        """
        Set C-Attributes in the MSRecord Structure.
        
        Currently only works with the first record of the Mseedfile.
        NOT TESTED!
        """
        if self.__dict__.has_key("attrs") and name in self.attrs:
            clibmseed.ms_readmsr.restype = C.c_int
            #needed to write the MSRecord
            msr = C.pointer(MSRecord())
            #Checks whether the returned msr is the last record. If it is the last record it will be 1.
            islast = C.pointer(C.c_int(0))
            for _i in range(1):
                clibmseed.ms_readmsr(C.pointer(msr), file, 0, None, islast,1, 1, 0)
                if islast.contents.value == 1:
                    break
                    setattr(msr.contents, name, val)
        else:
            #does what??
            self.__dict__[name] = val
    
    def printtracelist(self,filename,timeformat= 0,details = 0, gaps=0):
        """
        Mst_printtracelist prints a formated list of the MSTrace segments 
        in the given MSTraceGroup. All output is printed using ms_log(3)
        at level 0.
        
        filename      - Name of file to read Mini-SEED data from
        timeformat    - Controls the format of the resulting time string, default = 0
                            0 : SEED time format (2005,146,00:00:00.000000)
                            1 : ISO time format (2005-05-26T00:00:00.000000)
                            2 : Epoch time, seconds since the epoch (1117065600.00000000)
        details       - If the details flag is greater than 0 the sample rate and 
                        sample count are printed for each trace, default = 0
        gaps          - If the gaps flag is greater than zero the time gap from the 
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
        
        filename      - Name of file to read Mini-SEED data from
        timeformat    - Controls the format of the resulting time string, defaults to 0
                            0 : SEED time format (2005,146,00:00:00.000000)
                            1 : ISO time format (2005-05-26T00:00:00.000000)
                            2 : Epoch time, seconds since the epoch (1117065600.00000000)
        mingap        - defaults to 0
        maxgap        - defaults to 0
        """
        mstg = self.readtraces(filename, dataflag = 0, skipnotdata = 0)
        clibmseed.mst_printgaplist(mstg,timeformat,mingap,maxgap)
        
    def findgaps(self, filename):
        """
        Finds gaps and returns a list for each found gap
        .
        Each item has a starttime and a duration value. The starttime is the last
        correct data sample plus one step. If no gaps are
        found it will return an empty list.
        All time and duration values are in microseconds.
        """
        gaplist=[]
        retcode=0
        oldstarttime=0
        while retcode == 0:
            msr, retcode=self.read_MSRec(filename, dataflag=0, skipnotdata=0)
            if retcode == 0:
                if oldstarttime!=0:
                    if msr.contents.starttime-oldstarttime != 0:
                        gaplist.append([oldstarttime , msr.contents.starttime-oldstarttime+(1/msr.contents.samprate)*1e6])
                oldstarttime=long(msr.contents.starttime+msr.contents.samplecnt*(1/msr.contents.samprate)*1e6-(1/msr.contents.samprate)*1e6)
        return gaplist
    
    def fastfindgaps(self, filename):
        """
        Find gaps using Traces
        Not done yet!
        """
        mstg = self.readtraces(filename, dataflag = 0,skipnotdata = 0)
        gapslist=[]
        for i in range(2,mstg.contents.numtraces+1)
            pass

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

    def readtraces(self, filename, reclen = -1, timetol = -1, sampratetol = -1,
                   dataflag = 1, skipnotdata = 1, verbose = 0):
        """
        Reads Mini-SEED data from file. Returns MSTraceGroup structure.
        
        filename        - Mini-SEED file to be read
        reclen          - If reclen is 0 the length of the first record is auto-
                          detected. All subsequent records are then expected to
                          have the same record length.
                          If reclen is negative the length of every record is
                          automatically detected.
                          Defaults to -1.
        timetol         - Time tolerance, default to -1 (1/2 sample period)
        sampratetol     - Sample rate tolerance, defaults to -1 (rate dependent)
        dataflag        - Controls whether data samples are unpacked, defaults to 1
        skipnotdata     - If true (not zero) any data chunks read that to do not
                          have valid data record indicators will be skipped.
                          Defaults to true (1).
        verbose         - Controls verbosity from 0 to 2. Defaults to None (0).
        """
        #Creates MSTraceGroup Structure
        mstg = C.pointer(MSTraceGroup())
        #Uses libmseed to read the file and populate the MSTraceGroup structure
        errcode=clibmseed.ms_readtraces(C.pointer(mstg), filename, C.c_int(reclen),
                            C.c_double(timetol), C.c_double(sampratetol),
                            C.c_short(dataflag), C.c_short(skipnotdata), C.c_short(dataflag),
                            C.c_short(verbose))
        if errcode != 0:
            assert 0, "\n\nError while reading Mini-SEED file: "+filename
        return mstg

    def read_ms_using_traces(self, filename):
        """
        Read Mini-SEED file. Header, Data and numtraces are returned

        filename    - Name of file to read Mini-SEED data from
        timetol     - Time tolerance, default is 1/2 sample period (-1)
        sampratetol - Sample rate tolerance, default is rate dependent (-1)
        verbosity   - Level of diagnostic messages, default 0
        """
        #Creates MSTraceGroup Structure
        mstg = self.readtraces(filename)
        data=[]
        header=[]
        mst = mstg.contents.traces
        numtraces = mstg.contents.numtraces
        for _i in range(numtraces):
            data.extend(mst.contents.datasamples[0:mst.contents.numsamples])
            header.append(self.mst2dict(mst))
            mst = mst.contents.next
        return header[0],data, numtraces

    def read_MSRec(self, filename, reclen = -1, dataflag = 1, 
                   skipnotdata = 1, verbose = 0):
        """
        Reads Mini-SEED file and populates MS Record data structure with subsequent
        calls.
        
        For subsequent calls, call the function with an MSRecord structure.
        ilename        - Mini-SEED file to be read
        reclen          - If reclen is 0 the length of the first record is auto-
                          detected. All subsequent records are then expected to
                          have the same record length.
                          If reclen is negative the length of every record is
                          automatically detected.
                          Defaults to -1.
        dataflag        - Controls whether data samples are unpacked, defaults to 1
        skipnotdata     - If true (not zero) any data chunks read that to do not
                          have valid data record indicators will be skipped.
                          Defaults to true (1).
        verbose         - Controls verbosity from 0 to 2. Defaults to None (0).
        """
        #Init MSRecord structure
        clibmseed.msr_init.restype = C.POINTER(MSRecord)
        msr=clibmseed.msr_init(None)

        #Creates MSTraceGroup Structure
#        msr = C.pointer(MSRecord(None))

        islast=C.c_int(1)
        fpos=C.c_longlong()
        clibmseed.ms_readmsr.restype = C.c_int
        retcode=clibmseed.ms_readmsr(C.pointer(msr), filename, C.c_int(reclen),
                             None, None,
                             C.c_short(1), C.c_short(1),
                             C.c_short(verbose))
#        int ms_readmsr (MSRecord **ppmsr, char *msfile, int reclen, off_t *fpos,
#            int *last, flag skipnotdata, flag dataflag, flag verbose)

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