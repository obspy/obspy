from omniORB import CORBA
import CosNaming
from idl import Fissures

orb = CORBA.ORB_init( [
    #"-ORBtraceLevel", "40",
    "-ORBgiopMaxMsgSize","2097152",
    "-ORBInitRef", 
        "NameService=corbaloc:iiop:dmc.iris.washington.edu:6371/NameService",
    ], CORBA.ORB_ID)

obj = orb.resolve_initial_references("NameService")
name =  [CosNaming.NameComponent(id='Fissures', kind='dns'),
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
networkCode = "IU"
network =  netFind.retrieve_by_code(networkCode)[0]
network = network._narrow(Fissures.IfNetwork.ConcreteNetworkAccess)
stations = network.retrieve_stations()
channels = network.retrieve_for_station(stations[0].id)

name =  [CosNaming.NameComponent(id='Fissures', kind='dns'),
         CosNaming.NameComponent(id='edu', kind='dns'),
         CosNaming.NameComponent(id='iris', kind='dns'),
         CosNaming.NameComponent(id='dmc', kind='dns'),
         CosNaming.NameComponent(id='DataCenter', kind='interface'),
         CosNaming.NameComponent(id='IRIS_PondDataCenter', kind='object_FVer1.0')]
seisDC = rootContext.resolve(name)
seisDC = seisDC._narrow(Fissures.IfSeismogramDC.DataCenter)

t1 = Fissures.Time("2003-06-20T06:23:25.0000Z", -1)
t2 = Fissures.Time("2003-06-20T06:43:25.0000Z", -1)
req = [Fissures.IfSeismogramDC.RequestFilter(channels[0].id, t1, t2)]

print "querying for %s.%s.%s.%s" % ( \
        req[0].channel_id.network_id.network_code,
        req[0].channel_id.station_code,
        req[0].channel_id.site_code.strip(),
        req[0].channel_id.channel_code )

seis = seisDC.retrieve_seismograms(req)

print "seis[0] has %d points and starts at %s" % (seis[0].num_points, \
        seis[0].begin_time.date_time)

seis_data = seis[0].data
_i = seis_data.encoded_values[0] # 8 values in total

# src/IfTimeSeries.idl:43:    //  const EncodingFormat STEIM1=10;
# src/IfTimeSeries.idl:44:    //  const EncodingFormat STEIM2=11;
compression = _i.compression
byte_order = _i.byte_order
npts = _i.num_points
data = _i.values

fmt = "Extracted data. First of %s records has %d points, type %d \
compressed, byte order %d, type of data %s, byte length of data is %s"
print fmt % ( len(seis_data.encoded_values), npts, compression, byte_order, 
              type(data), len(data) )

# http://www.seis.sc.edu/software/SeedCodec/apidocs/edu/iris/dmc/seedcodec/package-summary.html
# http://www.seis.sc.edu/software/fissuresUtil/xref/edu/sc/seis/fissuresUtil/sac/FissuresToSac.html
import pdb; pdb.set_trace()
