# Small hack to make omniORB.CORBA appear as CORBA

import sys, omniORB.CORBA
sys.modules["CORBA"] = omniORB.CORBA
