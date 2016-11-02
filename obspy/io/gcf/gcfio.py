# -*- coding: utf-8 -*-

"""
GCF Block

Tasks:
add channel name conversion (HHZ, etc)


"""

import numpy as np
import struct
from obspy import Trace, Stream, UTCDateTime
from numpy.f2py.auxfuncs import throw_error

#temporary:
import difflib

_base_36_dict=map(str,range(10)) + [chr(i) for i in range(ord('A'),ord('Z')+1)]
_rev_base_36_dict=dict(zip(_base_36_dict,range(len(_base_36_dict))))
_base_26_dict=[chr(i) for i in range(ord('A'),ord('Z')+1)]
_rev_base_26_dict=dict(zip(_base_26_dict,range(len(_base_26_dict))))

_gcf_block_size=1024 #bytes - without the transport layer 
_gcf_base_day=UTCDateTime("1989-11-17")
_gcf_header_size=16 #bytes
_special_sampling_rates={157:0.1,
                        161:0.125,
                        162:0.2,
                        164:0.25,
                        167:0.5,
                        171:400,
                        174:500,
                        176:1000,
                        179:2000,
                        181:4000}

#Time Fractional Offset Denominator:
_special_tfod={400:8,
               500:2,
               1000:4,
               2000:8,
               4000:16}

#def test_fractional_offset_read:
#    pass


def hodo():
    for i in range(len(Glist)):
        bl.ParseBlockMemory(buf[Glist[i]+4:Glist[i]+4+1024])
        print "Glist[i],i: ",Glist[i],i 
        if bl.header['compression_info'] in [0,1,2,4]:
            print bl.header

            if raw_input("so?")=="y":
                print "hododhoy"
                break


def PrintHexNice(s1):
    return ":".join("{:02x}".format(ord(c)) for c in s1)

def test_stream_analyze(s1,s2):
    s1=":".join("{:02x}".format(ord(c)) for c in s1)
    s2=":".join("{:02x}".format(ord(c)) for c in s2)
    
    #print('{} => {}'.format(s1,s2))
    for i,s in enumerate(difflib.ndiff(s1,s2)):
        if s[0]==' ': 
            continue
        elif s[0]=='-':
            print(u'Delete "{}" from position {}'.format(s[-1],i)),
        elif s[0]=='+':
            print(u'Add "{}" to position {}'.format(s[-1],i)),
        print

def test_read_gcf_file(filename):
    f=open(filename,'rb')
    buf=f.read(1024) #read one block
    gcf_block=GCFBlock()
    gcf_block.ParseBlockMemory(buf)
    tr=MakeGcfTrace(gcf_block)
    f.close()
    return tr

def EncodeBase36String(strn): #to utils
    sum=0
    for letter,i in zip(strn[::-1],range(len(strn))):
        sum+=_rev_base_36_dict[letter]*(36**(i))
    return sum

def DecodeBase36String(num): #to utils
    dec_string=''
    num36=0
    while True:
        num36=num % 36
        dec_string+=_base_36_dict[num36]
        num=num/36
        if num==0: break
    return dec_string[::-1].lstrip('0') #strip leading 0's per gcf specs

def GuralpToUtcTime(g_day,g_sec): #to utils
    return _gcf_base_day+86400*g_day+g_sec 
    #Leap second concerns: the timing is off for only the
    #duration of the leap second with this method.
    #since UTCDateTime does not have a "add_day" method,
    #this is the only way at least for now.

def MakeGcfTrace(gcf_block): 
    tr=Trace()
    #tr.data=np.array(gcf_block.data,dtype='float64')
    #to make compatible with STEIM1 compression:
    tr.data=np.array(gcf_block.data,dtype='int32')
    tr.stats.sampling_rate=gcf_block.header['sample_rate']
    tr.stats.starttime=GuralpToUtcTime(gcf_block.header['g_day'],
                                        gcf_block.header['g_sec'])
    tr.stats.channel=gcf_block.header['streamId'][4]
    tr.stats.location=gcf_block.header['streamId'][5]
    
    #NOT TESTED:
    if gcf_block.header['sample_rate'] in _special_tfod:
        offset=gcf_block.header['tfon']/ \
            _special_tfod[gcf_block.header['sample_rate']]
        tr.stats.start_time+=offset
        
    tr.stats.station=gcf_block.header['systemId']
    
    return tr

def ReadGcfFile(gcf_file,return_gcf_block=False):
    #read blocks of 1024 into different traces then join them
    gcf_block=GCFBlock()
    st=Stream()
    max_size=1024
    with open(gcf_file,'rb') as f:
        while True:
            buf=f.read(_gcf_block_size)
            if not buf: break
            gcf_block.ParseBlockMemory(buf)
            st.append(MakeGcfTrace(gcf_block))
            if(gcf_block.header['is_status'])==True:
                print "status"
            
    #st.merge() #causes problems when writing MSEED if resulting data is a masked array
    st._cleanup()
    if return_gcf_block:
        return [st,gcf_block]
    else:
        return st

def ReadGcf1Hdr(gcf_file):
    #reads header of the first packet in the file and prints
    gcf_block=GCFBlock()
    with open(gcf_file,'rb') as f:
        buf=f.read(_gcf_header_size)
        gcf_block.ParseHeaderMemory(buf)
        gcf_block.PrintHeader()
    
    return gcf_block
    
    

class GCFBlock(object):
    __doc__="""
    GCF Block class. An instance is created after reading a file
    or reading from a stream. The block includes the 16 byte header,
    4 byte FIC, differential data with variable length and 4 byte RIC.
    """
    
    def __getattr__(self, attr):
        return self.get(attr)
    
    def __getattr__(self, attr):
        return self.header.get(attr)

    def __init__(self,systemId="000000",streamId="000000",
                 compression_code=1,is_status=False,n_data_blocks=0,
                 npts=0,data_length=4,sample_rate=1,FIC=0,RIC=0,
                 g_day=0,g_sec=0,is_legit=True,**kwargs):
        
        self.header={'systemId':systemId,'streamId':streamId,
                'compression_code':compression_code,'is_status':is_status,
                'n_data_blocks':n_data_blocks,'npts':npts,
                'data_length':data_length,'sample_rate':data_length,
                'sample_rate':sample_rate,'FIC':FIC,'RIC':RIC,
                'g_day':g_day,'g_sec':g_sec,'is_legit':is_legit}
        
        self.status=''
        self.data=np.array([])
        self.endianness='>' #Big Endiann
        #User can put arbitrary variables into the header
        self.header.update(kwargs)
        
    
    
    def ParseHeaderMemory(self,bytes_in_memory):
        """
        Parses a header block into the memory using
        a 1024 byte gcf block
        """
        
        #if len(bytes_in_memory)!=1024: raise ValueError('block not 1024 bytes')
        
        systemIdRawBytes=struct.unpack(self.endianness+'I',bytes_in_memory[0:4])[0]
        if systemIdRawBytes>>31==0: #top bit is unset
            #decode bottom 31 bits
            self.header['systemId']=DecodeBase36String(systemIdRawBytes & 0x7fffffff)
        if systemIdRawBytes>>31==1: #top bit is set
            #decode bottom 26 bits
            self.header['systemId']=DecodeBase36String(systemIdRawBytes & 0x3ffffff)
        
        streamIdRawBytes=struct.unpack(self.endianness+'I',bytes_in_memory[4:8])[0]
        self.header['streamId']=DecodeBase36String(streamIdRawBytes)
        
        gTimeRawBytes=struct.unpack(self.endianness+'I',bytes_in_memory[8:12])[0]
        self.header['g_day']=gTimeRawBytes>>17
        self.header['g_sec']=gTimeRawBytes & 0b11111111111111111
        
        self.header['sample_rate']=struct.unpack(self.endianness+'B',bytes_in_memory[13])[0]
        
        self.header['compression_info']=struct.unpack(self.endianness+'B',bytes_in_memory[14])[0]
        
        #Special sampling rates:
        if self.header['sample_rate'] in _special_sampling_rates:
            self.header['sample_rate']=_special_sampling_rates[self.header['sample_rate']]
        
        elif self.header['sample_rate'] == 0 and \
            self.header['compression_info']==4 and \
            self.header['streamId'][-2:]=='00':
            
            self.header['is_status']=True
        
        self.header['n_data_blocks']=struct.unpack(self.endianness+'B',bytes_in_memory[15])[0]
        self.header['npts']=self.header['n_data_blocks']*self.header['compression_info']
        if self.header['npts']>1000: self.header['npts']=0
        
        
    
    
    def ParseDataMemory(self,bytes_in_memory):
        if len(bytes_in_memory)!=1024: raise ValueError('gcf block not 1024 bytes')
        
        if self.header['is_status'] is True:
            #just return the raw block for now:
            self.status=''.join([struct.unpack('c',bytes_in_memory[i])[0] for i in range(16,1024) if ord(struct.unpack('c',bytes_in_memory[i])[0]) < 127])
            return 0
        
        #bottom 3 bits
        if self.header['compression_info'] & 111==1:
            dattype=np.dtype(np.int32).newbyteorder(self.endianness)
        elif self.header['compression_info'] & 111==2:
            dattype=np.dtype(np.int16).newbyteorder(self.endianness)
        elif self.header['compression_info'] & 111==4:
            dattype=np.dtype(np.int8).newbyteorder(self.endianness)
        else:
            print "compression info error:",self.header['compression_info']
            return 1
        
        #top 4 bits for fractional time offset nominator:
        self.header['tfon']=self.header['compression_info']>>28
        
        ric_address=_gcf_header_size+ \
            np.dtype(np.int32).itemsize + \
            self.header['npts']*dattype.itemsize    
        
        self.header['FIC']=struct.unpack(self.endianness+'i',
                                         bytes_in_memory[_gcf_header_size:_gcf_header_size+np.dtype(np.int32).itemsize])[0]
        self.header['RIC']=struct.unpack(self.endianness+'i',
                                         bytes_in_memory[ric_address:ric_address+np.dtype(np.int32).itemsize])[0]
        #print self.header['FIC']

        self.diffdata=np.frombuffer(bytes_in_memory,
                                dtype=dattype,
                                count=self.header['npts'],
                                offset=_gcf_header_size+np.dtype(np.int32).itemsize)
        
        self.data=self.diffdata.cumsum()+self.header['FIC']
        
        #if self.data[-1]==self.header['RIC']: print "MATCH"
        
    def ParseBlockMemory(self,bytes_in_memory):
        if len(bytes_in_memory)!=1024: raise ValueError('gcf block not 1024 bytes')
        
        self.ParseHeaderMemory(bytes_in_memory)
        self.ParseDataMemory(bytes_in_memory)
        #self.CheckIsLegit()
        
    def PrintHeader(self,type='verbose'):
        if type is 'verbose':
            print "system id:",self.header['systemId']
            print "stream id:",self.header['streamId']
            print "npts:",self.header['npts']
            print "sampling rate:",self.header['sample_rate']
        elif type is 'oneline':
            print self.header['systemId'],self.header['streamId'],self.header['npts'],self.header['sample_rate']
        elif type is 'string':
            retstr=self.header['systemId']+' '+self.header['streamId']
            return retstr
    
    def CheckIsLegit(self):
        #checks if it's a legit block
        if self.header['compression_code'] not in [1,2,4]:
            print 'comp code error'
            self.header['is_legit']=False
            return
        elif self.data[-1] != self.header['RIC']:
            print 'RIC mismatch'
            self.header['is_legit']=False
            return
        else:
            self.header['is_legit']=True
                                                   
        #check if compression code makes sense
        #check the checksum of
    
    def Ack(self):
        
        #print 'hedere'
        ackstr='\x01'
        rawStreamId=struct.pack('<I',EncodeBase36String(self.header['streamId']))
        
        ackstr+=rawStreamId[0]
        ackstr+='\x13'
        ackstr+=rawStreamId[1]
        ackstr+=rawStreamId[2]
        ackstr+=rawStreamId[3]
        
        return ackstr
    
    def Nack(self,block_no):
        
        #print 'hedere'
        nackstr='\x02'
        rawStreamId=struct.pack('<I',EncodeBase36String(self.header['streamId']))
        
        nackstr+=rawStreamId[0]
        nackstr+=struct.pack('>B',block_no)
        nackstr+=rawStreamId[1]
        nackstr+=rawStreamId[2]
        nackstr+=rawStreamId[3]
        
        return nackstr
        
        