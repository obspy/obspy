# Small hack to make omniORB.PortableServer__POA appear as PortableServer__POA

import sys, omniORB.PortableServer__POA
sys.modules["PortableServer__POA"] = omniORB.PortableServer__POA
