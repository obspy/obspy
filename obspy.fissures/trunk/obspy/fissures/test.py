from omniORB import CORBA
import CosNaming
from idl import IfNetwork_idl
from idl import Fissures_idl
from corba_walk import walk_print

orb = CORBA.ORB_init( [
    #"-ORBInitialPort", "6371", 
    #"-ORBthreadPerConnectionPolicy","0",
    #"-ORBmaxGIOPConnectionPerServer","10",
    #"-ORBmaxGIOPVersion","1.2",
    #"-ORBuseTypeCodeIndirections", "0",
    "-ORBacceptMisalignedTcIndirections", "1",
    "-ORBtraceLevel", "40",
    "-ORBverifyObjectExistsAndType", "0",
    "-ORBInitRef", 
        "NameService=corbaloc:iiop:dmc.iris.washington.edu:6371/NameService",
    ], CORBA.ORB_ID)
obj = orb.resolve_initial_references("NameService") #returns just none
# Uncomment to activate verbose module scanning
#walk_print(obj)

print """\nTry to retrieve/register object, see
    http://omniorb.sourceforge.net/omnipy2/omniORBpy/omniORBpy002.html\n"""

name =  [CosNaming.NameComponent(id='Fissures', kind='dns'),
         CosNaming.NameComponent(id='edu', kind='dns'),
         CosNaming.NameComponent(id='iris', kind='dns'),
         CosNaming.NameComponent(id='dmc', kind='dns'),
         CosNaming.NameComponent(id='NetworkDC', kind='interface'),
         CosNaming.NameComponent(id='IRIS_NetworkDC', kind='object_FVer1.0')]

rootContext = obj._narrow(CosNaming.NamingContext)
childContext = rootContext.resolve(name)
netDC = childContext._narrow(Fissures_idl._0_Fissures.IfNetwork.NetworkDC)
#netDC = childContext._narrow(Fissures_idl._0_Fissures__POA.IfNetwork.NetworkDC)
netFind = netDC._get_a_finder()
net =  netFind.retrieve_by_code("II")[0]
print net
print dir(net)
#import pdb; pdb.set_trace()
print net.retrieve_stations()


# XXX:
# * create new python IDL, may be dependent on version
# * firewall problems?
# http://marc.info/?l=omniorb-list&m=112016990924842&w=2
# http://thread.gmane.org/gmane.comp.corba.omniorb.user/6473/focus=6478
# 

# old version of walk_print
#obj_list = [obj]
#while True:
#    childContext = obj_list[-1]._narrow(CosNaming.NamingContext)
#    buf = childContext.list(10000)
#    print buf
#    binding = buf[0][0]
#    childContext = childContext.resolve(binding.binding_name)
#    if binding.binding_type is CosNaming.ncontext:
#        obj_list.append(childContext)
#    else:
#        break


#
# Java reference URL
#
# http://www.seis.sc.edu/software/fissuresUtil/xref/edu/sc/seis/fissuresUtil/namingService/FissuresNamingService.html


#
# Network Java
#
# FissuresNamingService namingService = new FissuresNamingService(orb);
# namingService.setNameServiceCorbaLoc("corbaloc:iiop:dmc.iris.washington.edu:6371/NameService");
# NetworkDCOperations netDC = namingService.getNetworkDC("edu/iris/dmc", "IRIS_NetworkDC");
# NetworkFinder netFinder = netDC.a_finder();
#


#
# Waveform Java
#
#        // Pick a name server to get FISSURES servers.
#        FissuresNamingService namingService = new FissuresNamingService(orb);
#        namingService.setNameServiceCorbaLoc("corbaloc:iiop:dmc.iris.washington.edu:6371/NameService");



#import edu.iris.Fissures.IfNetwork.NetworkDCOperations;
#import edu.iris.Fissures.model.AllVTFactory;
#import edu.sc.seis.fissuresUtil.namingService.FissuresNamingService;
#
#public class SeismogramClient {
#
#        // used.
#        org.omg.CORBA_2_3.ORB orb = (org.omg.CORBA_2_3.ORB)org.omg.CORBA.ORB.init(new String[] {},
#                                                                                  new Properties());
#        // Registers the FISSURES classes with the ORB
#        new AllVTFactory().register(orb);
#        // Pick a name server to get FISSURES servers.
#        FissuresNamingService namingService = new FissuresNamingService(orb);
#        namingService.setNameServiceCorbaLoc("corbaloc:iiop:dmc.iris.washington.edu:6371/NameService");
#        NetworkDCOperations netDC = namingService.getNetworkDC("edu/iris/dmc",
#                                                               "IRIS_NetworkDC");
#        NetworkFinder netFinder = netDC.a_finder();
#        NetworkAccess net = netFinder.retrieve_by_code("IU")[0];
#        Station[] stations = net.retrieve_stations();
#        Channel[] channels = net.retrieve_for_station(stations[0].get_id());
#        DataCenter seisDC = namingService.getSeismogramDC("edu/iris/dmc",
#                                                          "IRIS_PondDataCenter");
#        RequestFilter[] seismogramRequest = new RequestFilter[1];
#        Time start = new Time("2003-06-20T06:23:25.0000Z", -1);
#        Time end = new Time("2003-06-20T06:43:25.0000Z", -1);
#        seismogramRequest[0] = new RequestFilter(channels[0].get_id(),
#                                                 start,
#                                                 end);
#        logger.info("querying for " + seismogramRequest[0].channel_id.network_id.network_code + "."
#                    + seismogramRequest[0].channel_id.station_code + "."
#                    + seismogramRequest[0].channel_id.site_code + "."
#                    + seismogramRequest[0].channel_id.channel_code);
#        LocalSeismogram[] seis = seisDC.retrieve_seismograms(seismogramRequest);
#        for(int i=0; i < seis.length; i + +) {
#            logger.info("seis[" + i + "] has " + seis[i].num_points
#                    + " points and starts at " + seis[i].begin_time.date_time);
#        }
