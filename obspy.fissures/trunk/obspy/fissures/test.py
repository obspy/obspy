from omniORB import CORBA
import CosNaming
from idl import IfSeismogramDC_idl
from idl import Fissures_idl
from idl import IfNetwork_idl
from idl import IfSeismogramDC_idl

#
# http://www.seis.sc.edu/software/fissuresUtil/xref/edu/sc/seis/fissuresUtil/namingService/FissuresNamingService.html
# FissuresNamingService namingService = new FissuresNamingService(orb);
# namingService.setNameServiceCorbaLoc("corbaloc:iiop:dmc.iris.washington.edu:6371/NameService");
# NetworkDCOperations netDC = namingService.getNetworkDC("edu/iris/dmc", "IRIS_NetworkDC");
# NetworkFinder netFinder = netDC.a_finder();
#
orb = CORBA.ORB_init( ["-ORBInitRef",
       "NameService=corbaloc:iiop:dmc.iris.washington.edu:6371/NameService"], CORBA.ORB_ID)
obj = orb.resolve_initial_references("NameService") #returns just none
rootContext = obj._narrow(CosNaming.NamingContext)
bl, bi = rootContext.list(10000)
binding = bl[0]
new_obj = rootContext.resolve(binding.binding_name)
print orb.list_initial_services()
print bl, bi
print binding.binding_name
print binding.binding_type is CosNaming.ncontext


#networkDC = IfNetwork_idl._0_Fissures.IfNetwork.NetworkDC#("edu/iris/dmc", "IRIS_NetworkDC")
#print "Repository ID", NetworkDC._NP_RepositoryId
#name = [CosNaming.NameComponent("edu","iris","dmc"),
#        CosNaming.NameComponent("IRIS_NetworkDC")]
#http://www.bioinformatics.org/pipermail/pipet-devel/2000-March/001317.html





























#
# Try very simple version
#
#   orb = CORBA.ORB_init()
#   o = orb.string_to_object("corbaloc:iiop:dmc.iris.washington.edu:6371/NameService")
#   #o = orb.resolve_initial_references("corbaloc:iiop:dmc.iris.washington.edu:6371/NameService")
#   network_id = IfNetwork_idl._0_Fissures.IfNetwork.NetworkId("edu/iris/dmc", "IRIS_NetworkDC")
#   #network_id = IfNetwork_idl._0_Fissures.IfNetwork.NetworkId
#   print orb.list_initial_services()
#   print o._narrow(network_id)

#y = IfNetwork_idl._0_Fissures.IfNetwork._objref_NetworkDC()


#orb = CORBA.ORB_init(sys.argv, CORBA.ORB_ID)
#poa = orb.resolve_initial_references("RootPOA")
#ref = Counter()._this() # implicit activation
#
#poa._get_the_POAManager().activate()
#
## print IOR to console
##
#sys.stdout.write (orb.object_to_string(ref)+"\n")
#sys.stdout.flush();
#orb.run()

#import pdb;pdb.set_trace()
#  orb.register_value_factory('blah',x)
#o = o._narrow(Fortune.CookieServer)
#print o.get_cookie()

#>>> import CORBA, Fortune
#>>> orb = CORBA.ORB_init()
#>>> o = orb.string_to_object("corbaloc::host.example.com/fortune")
#>>> o = o._narrow(Fortune.CookieServer)
#>>> print o.get_cookie()

#orb = CORBA.ORB_init()
#o = orb.string_to_object("corbaloc:iiop:dmc.iris.washington.edu:6371/NameService")
#network_id = IfNetwork_idl._0_Fissures.IfNetwork.NetworkId('BW','20091010')
#_0_Fissures.IfNetwork.NetworkFinder()
#poa = orb.resolve_initial_references("RootPOA")


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
