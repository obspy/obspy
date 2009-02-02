#Wrapper class for libmseed

from obspy.mseed.libmseed_header import MSRecord, MSTraceGroup, MSTrace
import ctypes as C
import os
#if sys.platform=='win32':
#    clibmseed = C.cdll.libmseed
#    clib = C.cdll.msvcrt
#    #Windows DLL not tested
#else:
#Directory where the test files are located
lib_path = os.path.dirname(__file__)
lib_path = os.path.join(lib_path, 'libmseed', 'libmseed.so')
clibmseed = C.CDLL(lib_path)
clib = C.CDLL('libc.so.6')

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
        
        
        #loops over the MS records
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
    
    def cut(self,filename):
        """
        WORK IN PROGRESS
        """
        #Defines return type of ms_readmsr
        clibmseed.ms_readmsr.restype = C.c_int
        #needed to write the MSRecord
        #MSRec=MSRecord()
        msr = C.pointer(MSRecord())
        #opens output file
        outdat = clib.fopen("out.mseed","wb")
        #outdat = open('out.mseed', 'wb')
        def record_handler(record, recleng, stream):
            clib.fwrite(msr.contents.record, msr.contents.reclen, 1, outdat)
        #outdat.write(msr.contents.record)
        #Defines Python callback function for use in C function
        RECHANDLER = C.CFUNCTYPE(None, C.POINTER(C.c_char_p), C.c_int, C.c_void_p)
        rec_handler = RECHANDLER(record_handler)
        
        for _i in range(1):
            clibmseed.ms_readmsr(C.pointer(msr), filename, 0, None, None,1, 1, 0)
            #packedrecords = clibmseed.msr_pack(msr, rec_handler, None, None, 1, 3)
            #print "Packed ", packedrecords, " records"
        
        #closes output file   
        clib.close(outdat)
        #outdat.close()
        clibmseed.msr_free(C.pointer(msr))

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

    def populate_MSTG(self,header, data, numtraces=1):
        """
        Populates MSTrace_Group structure from given header, data and
        numtraces and returns the MSTrace_Group
        """
        #Init MSTraceGroup
        clibmseed.mst_initgroup.restype = C.POINTER(MSTraceGroup)
        mstg = clibmseed.mst_initgroup(None)
        #Init MSTrace object
        clibmseed.mst_init.restype = C.POINTER(MSTrace)
        mst = clibmseed.mst_init(None)
        #Connect Group with Traces
        mstg.contents.traces=mst
        mst=mstg.contents.traces
        #Write header in MSTrace structure
        self.dict2mst(mst,header)
        #Needs to be redone, dynamic??
        mstg.contents.numtraces=numtraces
        #Create void pointer and allocates more memory to it
        tempdatpoint=C.c_void_p()
        C.resize(tempdatpoint,
                 clibmseed.ms_samplesize(C.c_char(header['sampletype']))*
                 header['numsamples'])
        #Set pointer to tempdatpoint
        mst.contents.datasamples=C.pointer(tempdatpoint)
        #Write data in MSTrace structure
        for i in range(header['numsamples']):
            mst.contents.datasamples[i]=C.c_void_p(data[i])
        return mstg

    def write_ms(self,header,data, outfile='out.mseed', numtraces=1):
        """
        Write Miniseed file from header, data and numtraces
        
        header    - Dictionary containing the header files
        data      - List of the datasamples
        outfile   - Name of the output file
        
        Does not work completely. Has some memory issues in the record_handler
        callback function.
        """
        #Populate MSTG Structure
        mstg=self.populate_MSTG(header, data, numtraces)
        mst=mstg.contents.traces
        
        #WRITE MINISEED FILE FROM MS TRACE STRUCTURE
        def record_handler(record, reclen, stream):
#            import pdb;pdb.set_trace()
            out=open(outfile,'ab')
            out.write(record)
            out.close()

        #Define Python callback function for use in C function
        charp=C.c_char_p
        C.resize(charp(), 4096)
        RECHANDLER = C.CFUNCTYPE(None, charp, C.c_int, C.c_void_p)
        rec_handler = RECHANDLER(record_handler)
        
        packedrecords = clibmseed.mst_pack(mst,rec_handler,None,-1,-1,-1,None,1,0,None)
        print "Packed ", packedrecords, " records"