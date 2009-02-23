#Wrapper class for libmseed

from obspy.mseed.libmseed_header import MSRecord, MSTraceGroup, MSTrace
import ctypes as C
import os
import sys

# import libmseed library
if sys.platform=='win32':
    lib_name = 'libmseed.win32.dll'
else:
    lib_name = 'libmseed.so'
clibmseed = C.CDLL(os.path.join(os.path.dirname(__file__),'libmseed', lib_name))


#Some methods are not needed anymore and will be removed for the final version
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
    
    def ms_printtracelist(self,filename,timeformat= 0,details = 0, gaps=0):
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
        mstg = C.pointer(MSTraceGroup())
        #needed to write the MSTraceGroup
        clibmseed.ms_readtraces(C.pointer(mstg), filename, C.c_int(-1),
                                C.c_double(1), C.c_double(1),
                                C.c_short(1), C.c_short(1), C.c_short(1),
                                C.c_short(0))
        clibmseed.mst_printtracelist(mstg,timeformat,details,gaps)
    
    def ms_printgaplist(self,filename,timeformat= 0,mingap = 0, maxgap=0):
        """
        mst_printgaplist prints a formatted list of the gaps between MSTrace
        segments in the given MSTraceGroup to stdout. If mingap or maxgap is
        not NULL their values will be enforced and only gaps/overlaps matching
        their implied criteria will be printed.
        
        filename      - Name of file to read Mini-SEED data from
        timeformat    - Controls the format of the resulting time string, default = 0
                            0 : SEED time format (2005,146,00:00:00.000000)
                            1 : ISO time format (2005-05-26T00:00:00.000000)
                            2 : Epoch time, seconds since the epoch (1117065600.00000000)
        mingap        - default = 0
        maxgap        - default = 0
        """
        mstg = C.pointer(MSTraceGroup())
        #needed to write the MSTraceGroup
        clibmseed.ms_readtraces(C.pointer(mstg), filename, C.c_int(-1),
                                C.c_double(1), C.c_double(1),
                                C.c_short(1), C.c_short(1), C.c_short(1),
                                C.c_short(0))
        clibmseed.mst_printgaplist(mstg,timeformat,mingap,maxgap)
    
    def msr_print(self,filename,details = 0,recnum = 0):
        """
        msr_print prints formatted details from the given MSRecord struct (parsed
        record header), i.e. fixed section and blockettes. All output is printed using
        ms_log at level 0.
        
        filename      - Name of file to read Mini-SEED data from
        details       - Controls how much information is printed, default = 0
                            0  : a single line summarizing the record
                            1  : most commonly desired header details
                            2+ : all header details
        recnum        - Number of records parsed. If it is bigger than the total number
                        details of all records will be printed.
                        default = 0 - all records will be parsed
        
        If no fixed section header information is available at MSRecord then a single line 
        is printed from the other information in the MSRecord structure.
        """
        
        #needed to write the MSRecord
        self.msr = C.pointer(MSRecord())
        #Checks whether the returned msr is the last record. If it is the last record it will be 1.
        self.islast = C.pointer(C.c_int(0))
        #loop over the MS records
        if recnum == 0:
            while clibmseed.ms_readmsr(C.pointer(self.msr), filename, 0, None, self.islast,0, 1, 0)==0:
                clibmseed.msr_print(self.msr,details)
                #print "Samplerate: ",self.msr.contents.samprate
                #print "Number of Datasamples: ", self.msr.contents.numsamples
                #print "Sampletyp: ", self.msr.contents.sampletype
                #self.msr.contents.datasamples
                if self.islast.contents.value == 1:
                    break
        else:
            for _i in range(recnum):
                clibmseed.ms_readmsr(C.pointer(self.msr), filename, 0, None, self.islast,0, 1, 0)
                clibmseed.msr_print(self.msr,details)
                if self.islast.contents.value == 1:
                    break

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

    def read_ms(self, filename, timetol=-1,sampratetol=-1,verbose=0):
        """
        Read Mini-SEED file. Header, Data and numtraces are returned

        filename    - Name of file to read Mini-SEED data from
        timetol     - Time tolerance, default is 1/2 sample period (-1)
        sampratetol - Sample rate tolerance, default is rate dependent (-1)
        verbosity   - Level of diagnostic messages, default 0
        """
        #Creates MSTraceGroup Structure
        mstg = C.pointer(MSTraceGroup())
        #Uses libmseed to read the file and returns a MSTraceGroup structure
        netstat=clibmseed.ms_readtraces(C.pointer(mstg), filename, C.c_int(-1),
                            C.c_double(timetol), C.c_double(sampratetol),
                            C.c_short(1), C.c_short(1), C.c_short(1),
                            C.c_short(verbose))
        if netstat != 0:
            assert 0, "\n\nError while reading mseed file %s" % file
        data=[]
        header=[]
        mst = mstg.contents.traces
        numtraces = mstg.contents.numtraces
        for _i in range(numtraces):
            data.extend(mst.contents.datasamples[0:mst.contents.numsamples])
            header.append(self.mst2dict(mst))
            mst = mst.contents.next
        return header[0],data, numtraces

    def populate_MSTG(self, header, data, numtraces=1):
        """
        Populates MSTrace_Group structure from given header, data and
        numtraces and returns the MSTrace_Group
        """
        #Init MSTraceGroup
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