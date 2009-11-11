from omniORB import CORBA
import CosNaming
from obspy.fissures.idl import Fissures
from util import walk_print

orb = CORBA.ORB_init( [
    "-ORBInitRef", 
        "NameService=corbaloc:iiop:dmc.iris.washington.edu:6371/NameService",
    ], CORBA.ORB_ID)


obj = orb.resolve_initial_references("NameService")
walk_print(obj)
