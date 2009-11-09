from omniORB import CORBA
import CosNaming
from idl import Fissures

orb = CORBA.ORB_init([
    #"-ORBtraceLevel", "40",
    "-ORBgiopMaxMsgSize", "2097152",
    "-ORBInitRef",
        "NameService=corbaloc:iiop:dmc.iris.washington.edu:6371/NameService",
    ], CORBA.ORB_ID)

obj = orb.resolve_initial_references("NameService")
name = [CosNaming.NameComponent(id='Fissures', kind='dns'),
         CosNaming.NameComponent(id='edu', kind='dns'),
         CosNaming.NameComponent(id='iris', kind='dns'),
         CosNaming.NameComponent(id='dmc', kind='dns'),
         CosNaming.NameComponent(id='NetworkDC', kind='interface'),
         CosNaming.NameComponent(id='IRIS_NetworkDC', kind='object_FVer1.0')]

rootContext = obj._narrow(CosNaming.NamingContext)
netDC = rootContext.resolve(name)
netDC = netDC._narrow(Fissures.IfNetwork.NetworkDC)
netFind = netDC._get_a_finder()
netFind = netFind._narrow(Fissures.IfNetwork.NetworkFinder)
networkCode = "GE"
network = netFind.retrieve_by_code(networkCode)[0]
network = network._narrow(Fissures.IfNetwork.ConcreteNetworkAccess)
stations = network.retrieve_stations()
stations2 = [s for s in stations if s.id.station_code == 'APE']

channels = network.retrieve_for_station(stations2[0].id)

#http://www.iris.edu/dhi/servers.htm
name = [CosNaming.NameComponent(id='Fissures', kind='dns'),
         CosNaming.NameComponent(id='edu', kind='dns'),
         CosNaming.NameComponent(id='iris', kind='dns'),
         CosNaming.NameComponent(id='dmc', kind='dns'),
         CosNaming.NameComponent(id='DataCenter', kind='interface'),
         CosNaming.NameComponent(id='IRIS_DataCenter', kind='object_FVer1.0')]
seisDC = rootContext.resolve(name)
seisDC = seisDC._narrow(Fissures.IfSeismogramDC.DataCenter)

t1 = Fissures.Time("2003-06-20T05:59:00.0000Z", -1)
t2 = Fissures.Time("2003-06-20T06:32:00.0000Z", -1)

# request all channels
req = [Fissures.IfSeismogramDC.RequestFilter(c.id, t1, t2) for c in channels]

seis = seisDC.retrieve_seismograms(req)



# http://www.seis.sc.edu/software/SeedCodec/apidocs/edu/iris/dmc/seedcodec/package-summary.html
# http://www.seis.sc.edu/software/fissuresUtil/xref/edu/sc/seis/fissuresUtil/sac/FissuresToSac.html

# build up obspy Trace object
from obspy.core import Trace, UTCDateTime, Stream
from obspy.mseed import libmseed
import numpy as np
import sys

if sys.byteorder == 'little':
    byteorder = True
else:
    byteorder = False

mseed = libmseed()

st = Stream()
i = 0
for sei in seis:
    if sei.num_points == 0:
        #XXX these are mostly blocketes R with compression 0
        #XXX e.g. 000000R APE    HHNGE
        #print sei.data.encoded_values[0].compression
        #print sei.data.encoded_values[0].values
        continue
    tr = Trace()
    tr.stats.starttime = UTCDateTime(sei.begin_time.date_time)
    tr.stats.npts = sei.num_points
    # calculate sampling rate
    if str(sei.sampling_info.interval.the_units.the_unit_base) != 'SECOND':
        raise Exception("Wrong unit!")
    value = sei.sampling_info.interval.value
    power = sei.sampling_info.interval.the_units.power
    multi_factor = sei.sampling_info.interval.the_units.multi_factor
    exponent = sei.sampling_info.interval.the_units.exponent
    # sampling rate is given in Hertz within ObsPy!
    delta = pow(value * pow(10, power) * multi_factor, exponent)
    sr = sei.num_points / float(delta)
    tr.stats.sampling_rate = sr
    # calculate end time 
    temp = 1 / sr * (sei.num_points - 1)
    # set all kind of stats
    tr.stats.endtime = tr.stats.starttime + temp
    tr.stats.station = sei.channel_id.station_code
    tr.stats.network = sei.channel_id.network_id.network_code
    tr.stats.channel = sei.channel_id.channel_code
    tr.stats.location = sei.channel_id.site_code.strip()
    # loop over data chunks
    data = []
    for chunk in sei.data.encoded_values:
        i+=1
        # for now we only support STEIM2
        # src/IfTimeSeries.idl:43:    //  const EncodingFormat STEIM1=10;
        # src/IfTimeSeries.idl:44:    //  const EncodingFormat STEIM2=11;
        compression = chunk.compression
        if compression != 11:
            raise NotImplementedError("Compression %d not implemented" % compression)
        # swap byte order in decompression routine if byte orders differ 
        # src/IfTimeSeries.idl:52:       *  FALSE = big endian format -
        # src/IfTimeSeries.idl:54:       *  TRUE  = little endian format -
        swapflag = (byteorder != chunk.byte_order)
        data.append(mseed.unpack_steim2(chunk.values, chunk.num_points,
                                        swapflag=swapflag))
    # merge data chunks
    tr.data = np.concatenate(data)
    tr._verify()
    st.append(tr)

print st
import pdb; pdb.set_trace()
