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
networkCode = "II"
network =  netFind.retrieve_by_code(networkCode)[0]
network = network._narrow(Fissures.IfNetwork.ConcreteNetworkAccess)
stations = network.retrieve_stations()
_i = stations[0]
_i = _i._narrow(Fissures.IfNetwork.Station)

print "Network %s has %d stations" % (networkCode, len(stations))
print "%s, %s located at %6.2f %6.2f"  % (_i.id.station_code, _i.name, \
        _i.my_location.latitude, _i.my_location.longitude)

channels = network.retrieve_for_station(_i.id)

print "%s %s has %d channels" % ( _i.id.network_id.network_code, \
        _i.id.station_code, len(channels))


