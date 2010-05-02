# -*- Mode: Python; -*-
#                            Package   : omniORBpy
# PortableServer__POA.py     Created on: 2000/02/24
#                            Author    : Duncan Grisby (dpg1)
#
#    Copyright (C) 2000 AT&T Laboratories Cambridge
#
#    This file is part of the omniORBpy library
#
#    The omniORBpy library is free software; you can redistribute it
#    and/or modify it under the terms of the GNU Lesser General
#    Public License as published by the Free Software Foundation;
#    either version 2.1 of the License, or (at your option) any later
#    version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free
#    Software Foundation, Inc., 59 Temple Place - Suite 330, Boston,
#    MA 02111-1307, USA
#
#
# Description:
#    PortableServer skeletons

# $Id: PortableServer__POA.py,v 1.2.2.1 2003/03/23 21:51:43 dgrisby Exp $
# $Log: PortableServer__POA.py,v $
# Revision 1.2.2.1  2003/03/23 21:51:43  dgrisby
# New omnipy3_develop branch.
#
# Revision 1.1.4.1  2002/01/18 15:49:45  dpg1
# Context support. New system exception construction. Fix None call problem.
#
# Revision 1.1  2000/06/27 15:07:10  dpg1
# Change from POA_M to M__POA mapping
#
# Revision 1.1  2000/03/03 17:41:28  dpg1
# Major reorganisation to support omniORB 3.0 as well as 2.8.
#

import omniORB
from omniORB import CORBA, PortableServer
import _omnipy

# ServantManager skeleton
class ServantManager (PortableServer.Servant):
    _NP_RepositoryId = PortableServer.ServantManager._NP_RepositoryId

    _omni_op_d = {}
    _omni_special = 1

ServantManager._omni_skeleton = ServantManager


# ServantActivator skeleton
class ServantActivator (ServantManager):
    _NP_RepositoryId = PortableServer.ServantActivator._NP_RepositoryId

    _omni_op_d = {"incarnate":  PortableServer.ServantActivator._d_incarnate,
                  "etherealize":PortableServer.ServantActivator._d_etherealize}
    _omni_op_d.update(ServantManager._omni_op_d)
    _omni_special = 1

ServantActivator._omni_skeleton = ServantActivator


# ServantLocator skeleton
class ServantLocator (ServantManager):
    _NP_RepositoryId = PortableServer.ServantLocator._NP_RepositoryId

    _omni_op_d = {"preinvoke":  PortableServer.ServantLocator._d_preinvoke,
                  "postinvoke": PortableServer.ServantLocator._d_postinvoke}
    _omni_op_d.update(ServantManager._omni_op_d)
    _omni_special = 1

ServantLocator._omni_skeleton = ServantLocator



# AdapterActivator skeleton
class AdapterActivator (PortableServer.Servant):
    _NP_RepositoryId = PortableServer.AdapterActivator._NP_RepositoryId

    _omni_op_d = {"unknown_adapter":
                  PortableServer.AdapterActivator._d_unknown_adapter}
    _omni_special = 1

AdapterActivator._omni_skeleton = AdapterActivator
