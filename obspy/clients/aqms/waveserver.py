"""
AQMS waveserver for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & ISTI (Instrumental Software Technologies, Inc)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)

"""

import struct
import socket
import sys
import io
from obspy import read
import logging

TN_TCP_GETDATA_REQ   = 3000
TN_TCP_GETTIMES_REQ  = 3001
TN_TCP_GETRATE_REQ   = 3002
TN_TCP_GETCHAN_REQ   = 3003
TN_TCP_ERROR_RESP    = 3004
TN_TCP_GETDATA_RESP  = 3005
TN_TCP_GETTIMES_RESP = 3006
TN_TCP_GETRATE_RESP  = 3007
TN_TCP_GETCHAN_RESP  = 3008

DF_NONE     = 0
DF_INTEGER  = 1
DF_FLOAT    = 2
DF_STRING   = 3  
DF_BINARY   = 4

df_fmt = [ '', 'i', 'd', 's', 'b'  ]

TCP_CONN_VERSION = 1
TCP_CONN_MAX_DATA_SIZE = 16366

DATAFIELD_MAX_WIRE = 4224
DATAFIELD_MAX_SIZE = 4200

'''
// Generic function return codes 
const int TN_SUCCESS    = 0;
const int TN_FAILURE    = -1;
const int TN_EOF        = -2;
const int TN_SIGNAL     = -3;
const int TN_NODATA     = -4;
const int TN_NOTVALID   = -5;
const int TN_TIMEOUT    = -6;
// These two codes have been replaced by TN_BEGIN, TN_END
const int TN_STARTGROUP = -7;
const int TN_ENDGROUP   = -8;
const int TN_BEGIN      = -7;
const int TN_END        = -8;
//
const int TN_PARENT     = -9;
const int TN_CHILD      = -10;
// Operational failure return codes
const int TN_FAIL_WRITE = -20;
const int TN_FAIL_READ  = -21;
// Seismic-specific return codes
const int TN_BAD_SAMPRATE = -100;
'''

# typedef enum {SENSOR_BROADBAND, SENSOR_STRONG_MOTION, SENSOR_SHORT_PERIOD, 
#                       SENSOR_UNKNOWN} inst_type;
SENSOR_UNKNOWN = 3

class Channel():
    def __init__(self, network = "", station = "", channel = "", 
                        location = "") :
        self.network = network  # nscl, each char[10]
        self.station = station
        self.channel = channel
        self.location = location

        self.gain_units = ""   # char[20]
        self.instrument_type= SENSOR_UNKNOWN
        self.samprate = 0.0
        self.latitude = 0.0
        self.longitude = 0.0
        self.elevation = 0.0
        self.gain = 0.0
        self.mlcor = 0.0
        self.mecor = 0.0

    def serialize(self):
        # fmt = '!10s10s10s10s' + 'dddddddi20s'  # old fmt, from C client code
        fmt = '!3s6s4s3s' + 'dddddddi20s'        # new fmt, from java client code
        wire = struct.pack( fmt ,   self.network.encode(), 
                                    self.station.encode() ,
                                    self.channel.encode(), 
                                    self.location.encode(), 
                                    self.samprate, 
                                    self.latitude, 
                                    self.longitude, 
                                    self.elevation, 
                                    self.gain, 
                                    self.mlcor, 
                                    self.mecor, 
                                    self.instrument_type, 
                                    self.gain_units.encode() )
        return(wire)

class TimeWindow():
    def __init__(self, start = 0, end = 0) -> None:
        self.start = start
        self.end = end

    def serialize(self):
        fmt = '!dd'
        len = struct.calcsize(fmt)
        wire = struct.pack( fmt , self.start, self.end )
        return(wire)

class Packet():
    def __init__(self, ver=TCP_CONN_VERSION, mtype = None) -> None:
        self.ver = ver
        self.mtype = mtype
        self.packnum = 1
        self.packtot = 0
        self.datalen = 0
        self.data = bytearray(0)

    def serialize_hdr(self):
        sh = struct.pack('!iiiii',  self.ver,
                                    self.mtype,
                                    self.packnum,
                                    self.packtot,
                                    self.datalen)
        return sh                                 

class DataField():
    def __init__(self, dtype=0, value=None) -> None:
        self.dtype = dtype
        self.value = value

    def serialize(self):
        if self.dtype == DF_INTEGER or self.dtype == DF_FLOAT:
            fmt = df_fmt[self.dtype]
            size = struct.calcsize("!"+ fmt )
            wire = struct.pack("!ii"+fmt, self.dtype, size, self.value)
        else:
            size = len(self.value)
            wire = struct.pack("!ii", self.dtype, size ) + self.value
        return wire

class TCPMessage():
    def __init__(self, mtype=None) -> None:
        self.valid = True
        self.mtype = mtype
        self.dlist = []

    def append(self, data_field):
        self.dlist.append(data_field)

class TCPConn():

    seq_nr = 1

    def __init__(self,socket) -> None:
        # self.server = server
        # self.port = port
        # self.timeout = timeout
        self.socket = socket
        # self.seq_nr = 1

        # self.connect()

    # def connect(self):
    #     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     s.settimeout(self.timeout)
    #     s.connect((self.server, self.port))
    #     self.socket = s
    #     return self.socket

    def get_seq_nr(self):
        ret = self.seq_nr
        self.seq_nr += 1
        # print(f'seq nr: {ret}')
        return ret

    def send_msg(self, msg):
        # print('send msg, mtype:', msg.mtype)        
        sendlist = [] # list of packets
        packnum = 1

        # each message DataFields is serialized into 'packets'
        # the list of 'packets' is then sent.

        # Initialize the first packet
        p = Packet()
        p.ver = TCP_CONN_VERSION
        p.mtype = msg.mtype
        p.packnum = packnum
        p.datalen = 0
        # print('packet',p.ver, p.mtype, p.packnum, len(msg.dlist))

        # Convert the data fields into their wire format representations
        for df in msg.dlist:

            dfs = df.serialize()
            fieldlen = len(dfs)
            
            if fieldlen > ( TCP_CONN_MAX_DATA_SIZE - p.datalen) :
                # This packet would overflow; insert it and start a new one
                sendlist.append(p)
                p = Packet( TCP_CONN_VERSION, msg.mtype )
                packnum = packnum + 1
                p.packnum = packnum
                p.datalen = 0

            w_buf = dfs
            w_len = struct.pack('!i',fieldlen)
            w = w_len + w_buf

            # p.data[p.datalen: p.datalen + len(w) ] = w
            p.data = p.data + w
            p.datalen = p.datalen + len(w)
        
        # insert last packet
        if p.datalen > 0:
            sendlist.append(p)

        # Fill in the total packets field and send the packets
        for p in sendlist:
            # print(f'sending p {p.mtype} {p.packnum} {p.datalen}')
            p.packtot = len(sendlist)
            data = p.serialize_hdr() + p.data[0:p.datalen]
            totalsent = 0
            while totalsent < len(data):
                sent = self.socket.send(data[totalsent:])
                if sent == 0:
                    raise Runtimewarning("socket connection broken")
                totalsent = totalsent + sent

    def receive_msg(self, dur=None):

        # print(f'rcv_msg {dur}')

        p_mtype, packets = self.receivePackets( dur )
        # print('rcv pkts', p_mtype, len(packets) )

        msg = TCPMessage(p_mtype)

        # packet.data is a seq of wirePacketField:
        #         struct wirePacketField {
        #           int len;
        #           char buf[DATAFIELD_MAX_WIRE];
        #         };
        # 'buf' contains a serialized DataField [dtype, len, value].


        # parse packets' wire DF into Message's DataField list
        for pkt in packets:
            recv = 0 
            while recv < len(pkt.data):
                # wire data DF is: len(int) + buf[len] 
                wlen = struct.unpack('!i',pkt.data[recv:recv+4])[0]
                wbuf = pkt.data[recv+4:recv+4+wlen]

                # parse wirePacketField buf into a DataField and add it to Message
                # DF is type(int) + len(int) + value(type)
                df_type, df_len = struct.unpack('!ii',wbuf[0:8])
                if df_type in (DF_INTEGER,DF_FLOAT):
                    df_value = struct.unpack('!'+df_fmt[df_type], wbuf[8:])[0]
                elif df_type == DF_BINARY:
                    df_value = wbuf[8:]
                data_field = DataField( df_type , df_value )
                msg.append( data_field )

                recv = recv + 4 + wlen

        # print(f'got msg: {msg.mtype} {len(msg.dlist)}')
        return msg

    def receivePackets(self, dur=None ):
        # print(f'rcv pkts {dur}')
        hdr_len = struct.calcsize('!iiiii')
        packets = []
        mtype = None
        lastpacket = 0

        while True:
            hdr = self.receiveBuffer( hdr_len ) 
            p =  struct.unpack('!iiiii', hdr) 
            p_ver, p_mtype, p_packnum, p_packtot, p_datalen = p
            if p_ver != TCP_CONN_VERSION:
                logging.warning(f'Version mismatch in packet')
            if mtype == None:
                mtype = p_mtype
            elif p_mtype != mtype:
                logging.warning('Message type mismatch in packets')
            lastpacket += 1
            if p_packnum != lastpacket:
                logging.warning('One or more packets missing')
            data = self.receiveBuffer( p_datalen, dur)
            pkt = Packet( p_ver, p_mtype )
            pkt.packnum = p_packnum
            pkt.packtot = p_packtot
            pkt.datalen = p_datalen
            pkt.data = data
            packets.append(pkt)
            if p_packnum >= p_packtot:
                break

        return mtype, packets

    def receiveBuffer(self, nbytes, dur=None):
        """
        Listens for nbytes from open socket.
        Returns byte array as python string or None if timeout
        """
        # print(f'rcv buf {nbytes} {dur}')

        sock = self.socket
        # sock.settimeout(dur)
        chunks = []
        btoread = nbytes
        try:
            while btoread:
                indat = sock.recv(min(btoread, 8192))
                # print(f'rb: got {len(indat)}')
                if not indat:
                    break
                btoread -= len(indat)
                chunks.append(indat)
        except socket.timeout:
            print('socket timeout in get_sock_bytes()', file=sys.stderr)
            return None
        if chunks:
            response = b''.join(chunks)
            return response
        else:
            return None

def get_ws_data( socket, network = "", station = "", channel = "", 
                                location = "", start=None, end=None , timeout=20 ):

    # print(f'get_data {socket} {network} {station} {channel} \
    #             {location} {start} {end}')

    s = TCPConn( socket )

    msg = TCPMessage(TN_TCP_GETDATA_REQ)


    data_type = DF_INTEGER   # seq nr
    data_value = s.get_seq_nr()
    data_field = DataField( data_type , data_value )
    msg.append( data_field )
    # print(data_type , data_value)

    chans = Channel(network, station, channel, location)
    chans_wire =  chans.serialize()

    data_type = DF_BINARY
    data_value = chans_wire
    data_field = DataField( data_type , data_value )
    msg.append( data_field )

    time_window = TimeWindow( start , end)
    tw_wire = time_window.serialize()

    data_type = DF_BINARY
    data_value = tw_wire
    data_field = DataField( data_type , data_value )
    msg.append( data_field )
    s.send_msg( msg )

    res = s.receive_msg( timeout )


    if res.mtype != TN_TCP_GETDATA_RESP: # error
        #print('error')
        #for dl in res.dlist:
            #print(f'dl t:{dl.dtype} v:{dl.value}')
        return None

    seq_nr = res.dlist[0].value
    samp_rate = res.dlist[1].value
    num_seg = res.dlist[2].value

    # print('data msg', seq_nr, samp_rate, num_seg)

    data = bytearray(0)
    for seg in res.dlist[3:]:
        # ds_hdr = struct.unpack('!dii',seg.value[:16])
        # print(ds_hdr)
        data = data + seg.value[16:]

    f = io.BytesIO(data)
    st = read(f)

    return st

def get_ws_samplerate( socket, network = "", station = "", channel = "", 
                                location = "" ):
    # not implemented on pws

    # print(f'get_samplerate {socket} {network} {station} {channel} \
    #             {location} ')

    s = TCPConn( socket )

    msg = TCPMessage(TN_TCP_GETRATE_REQ)

    data_type = DF_INTEGER   # seq nr
    data_value = s.get_seq_nr()

    data_field = DataField( data_type , data_value )
    msg.append( data_field )
    # print(data_type , data_value)

    chans = Channel(network, station, channel, location)
    chans_wire =  chans.serialize()

    data_type = DF_BINARY
    data_value = chans_wire
    data_field = DataField( data_type , data_value )
    msg.append( data_field )


    s.send_msg( msg )

    res = s.receive_msg( )


    if res.mtype != TN_TCP_GETRATE_RESP: # error
        for dl in res.dlist:
            print('error dl:',dl.dtype,  dl.value)
        return None

    # parse result
    seq_nr = res.dlist[0].value
    samp_rate = res.dlist[1].value

    # print('data msg', seq_nr, samp_rate)

    return samp_rate

def get_ws_times( socket, network = "", station = "", channel = "", 
                                location = "" ):
    # not implemented on pws

    # print(f'get_times {socket} {network} {station} {channel} \
    #             {location} ')

    s = TCPConn(socket )

    msg = TCPMessage(TN_TCP_GETTIMES_REQ)

    data_type = DF_INTEGER   # seq nr
    data_value = s.get_seq_nr()

    data_field = DataField( data_type , data_value )
    msg.append( data_field )
    # print(data_type , data_value)

    chans = Channel(network, station, channel, location)
    chans_wire =  chans.serialize()

    data_type = DF_BINARY
    data_value = chans_wire
    data_field = DataField( data_type , data_value )
    msg.append( data_field )


    s.send_msg( msg )

    res = s.receive_msg( )


    if res.mtype != TN_TCP_GETTIMES_RESP: # error
        for dl in res.dlist:
            print('error dl:',dl.dtype,  dl.value)
        return None

    # parse result
    seq_nr = res.dlist[0].value
    num_win = res.dlist[1].value

    ret = []
    for dl in res.dlist[2:]:
        times = struct.unpack('!dd', dl.value)
        ret.append(times)

    # print('times', seq_nr, num_win, ret)

    return ret

def get_ws_channels(socket):
    # not implemented on pws
    # print(f'get_ws_channels {server} {port} {timeout}')

    s = TCPConn(socket)

    msg_type = TN_TCP_GETCHAN_REQ
    msg = TCPMessage(msg_type)

    reqseqnum = s.get_seq_nr()
    data_type = DF_INTEGER
    data_value = reqseqnum
    # print(data_type , data_value)

    data_field = DataField( data_type , data_value )    

    msg.append( data_field )

    s.send_msg( msg )

    res = s.receive_msg( timeout )


    if res.mtype != TN_TCP_GETCHAN_RESP: # error
        for dl in res.dlist:
            print('error dl:',dl.dtype,  dl.value)
        return None

    # parse result
    seq_nr = res.dlist[0].value
    samp_rate = res.dlist[1].value

    # print('data msg', seq_nr, samp_rate)

    return samp_rate
