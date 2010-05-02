# Small hack to make omniORB.PortableServer appear as PortableServer

import sys, omniORB.PortableServer
sys.modules["PortableServer"] = omniORB.PortableServer
