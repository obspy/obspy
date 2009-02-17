#!/usr/bin/env python

from obspy import parser
from obspy.mseed.libmseed import libmseed
import os

class MseedParser(parser.Parser):
    """
    %s
    """

    def __init__(self,file=False):
        mseed=libmseed()
        if file:
            self.read(file)
    
    def read(self,mseedfile):
        """Read seismogram from MiniSEED file
        
        Header attributes are assigned as attributes
        """
        try:
            os.path.exists(mseedfile)
            #numsamples is the number of traces in the group
            header,data,self.numsamples = self.mseed.read_ms(mseedfile)
        except IOError:
            assert 1, "File not found: " + mseedfile
        ##### define the header entries as attributes
        ### Common header information
        self.station = header['station']     # station name (string)
        # start time of seismogram in seconds since 1970 (float)
        self.julsec = float(header['starttime']/1000000)
        # the actual seismogram data (list of floats)
        self.trace=[]
        for _i in data:
            self.trace.append(float(data[_i]))             
        self.df = header['samprate']         # sampling rate in Hz (float)
        self.npts = header['samplecnt']      # number of samples/data points (int)
        ### MiniSEED specific header files
        self.network = header['network']     #network name (string)
        self.location = header['location']   #location
        self.channel = header['channel']     #channel
        self.dataquality = header['dataquality']    #data quality indicator
        self.sampletype = 'f'                #sample type
        # start time of seismogram in seconds since 1970 (float)
        self.endtime = float(header['endtime']/1000000)
        self.type = header['type']           #type, not actually used by libmseed
    
    def write(self,mseedfile):
        """
        Not written yet
        """
        pass