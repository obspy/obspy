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

