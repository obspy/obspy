# -*- Mode: Python; -*-
#                            Package   : omniORBpy
# PortableServer.py          Created on: 1999/09/22
#                            Author    : Duncan Grisby (dpg1)
#
#    Copyright (C) 2005-2006 Apasphere Ltd
#    Copyright (C) 1999 AT&T Laboratories Cambridge
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
#    PortableServer module


# $Id: PortableServer.py,v 1.9.2.5 2009/05/06 16:50:24 dgrisby Exp $
# $Log: PortableServer.py,v $
# Revision 1.9.2.5  2009/05/06 16:50:24  dgrisby
# Updated copyright.
#
# Revision 1.9.2.4  2006/01/19 17:28:44  dgrisby
# Merge from omnipy2_develop.
#
# Revision 1.9.2.3  2005/11/09 12:33:31  dgrisby
# Support POA LocalObjects.
#
# Revision 1.9.2.2  2005/01/07 00:22:35  dgrisby
# Big merge from omnipy2_develop.
#
# Revision 1.9.2.1  2003/03/23 21:51:43  dgrisby
# New omnipy3_develop branch.
#
# Revision 1.7.4.9  2003/01/27 11:58:51  dgrisby
# Correct IfR scoping.
#
# Revision 1.7.4.8  2002/05/28 22:04:41  dgrisby
# Incorrect repoIds.
#
# Revision 1.7.4.7  2002/03/18 12:40:38  dpg1
# Support overriding _non_existent.
#
# Revision 1.7.4.6  2002/03/11 15:40:05  dpg1
# _get_interface support, exception minor codes.
#
# Revision 1.7.4.5  2002/01/18 15:49:45  dpg1
# Context support. New system exception construction. Fix None call problem.
#
# Revision 1.7.4.4  2001/09/20 14:51:26  dpg1
# Allow ORB reinitialisation after destroy(). Clean up use of omni namespace.
#
# Revision 1.7.4.3  2001/08/01 10:12:36  dpg1
# Main thread policy.
#
# Revision 1.7.4.2  2001/06/11 13:06:26  dpg1
# Support for PortableServer::Current.
#
# Revision 1.7.4.1  2000/11/28 14:51:11  dpg1
# Typo in method name.
#
# Revision 1.7  2000/05/25 16:07:44  dpg1
# Servant._default_POA now caches the root POA.
#
# Revision 1.6  2000/03/03 17:41:27  dpg1
# Major reorganisation to support omniORB 3.0 as well as 2.8.
#
# Revision 1.5  1999/11/25 11:21:36  dpg1
# Proper support for server-side _is_a().
#
# Revision 1.4  1999/09/28 16:19:41  dpg1
# Small memory management issues fixed.
#
# Revision 1.3  1999/09/24 13:26:00  dpg1
# _default_POA() operation added to Servant.
#
# Revision 1.2  1999/09/24 09:22:01  dpg1
# Added copyright notices.
#
# Revision 1.1  1999/09/22 15:46:11  dpg1
# Fake POA implemented.
#

import _omnipy
import omniORB
from omniORB import CORBA

# native Servant
class Servant:
    _NP_RepositoryId = ""

    def _this(self):
        return _omnipy.poa_func.servantThis(self)

    def _default_POA(self):
        if omniORB.rootPOA: return omniORB.rootPOA
        assert(omniORB.orb)
        omniORB.rootPOA = omniORB.orb.resolve_initial_references("RootPOA")
        return omniORB.rootPOA

    def _get_interface(self):
        omniORB.importIRStubs() # Make sure IR stubs are loaded
        ir = omniORB.orb.resolve_initial_references("InterfaceRepository")
        ir = ir._narrow(CORBA.Repository)
        if ir is None:
            raise CORBA.INTF_REPOS(omniORB.INTF_REPOS_NotAvailable,
                                   CORBA.COMPLETED_NO)
        interf = ir.lookup_id(self._NP_RepositoryId)
        return interf._narrow(CORBA.InterfaceDef)

    def _non_existent(self):
        return 0

_d_Servant = omniORB.tcInternal.tv_native


# interface POAManager
class POAManager (CORBA.Object) :
    _NP_RepositoryId = "IDL:omg.org/PortableServer/POAManager:1.0"

    def __init__(self):
        self.__release = _omnipy.poamanager_func.releaseRef

    def __del__(self):
        self.__release(self)

    def activate(self):
        _omnipy.poamanager_func.activate(self)

    def hold_requests(self, wait_for_completion):
        _omnipy.poamanager_func.hold_requests(self, wait_for_completion)
    
    def discard_requests(self, wait_for_completion):
        _omnipy.poamanager_func.discard_requests(self, wait_for_completion)
    
    def deactivate(self, etherialize_objects, wait_for_completion):
        _omnipy.poamanager_func.deactivate(self, etherialize_objects,
                                           wait_for_completion)

    def get_state(self):
        return self.State._item(_omnipy.poamanager_func.get_state(self))

    __methods__ = ["activate", "hold_requests", "discard_requests",
                   "deactivate", "get_state"] + CORBA.Object.__methods__

    # Generated declarations

    class AdapterInactive (CORBA.UserException):
        _NP_RepositoryId = "IDL:omg.org/PortableServer/POAManager/AdapterInactive:1.0"

    _d_AdapterInactive  = (omniORB.tcInternal.tv_except, AdapterInactive,
                           AdapterInactive._NP_RepositoryId, "AdapterInactive")
    _tc_AdapterInactive = omniORB.tcInternal.createTypeCode(_d_AdapterInactive)
    omniORB.registerType(AdapterInactive._NP_RepositoryId,
                         _d_AdapterInactive, _tc_AdapterInactive)
    
    HOLDING    = omniORB.EnumItem("HOLDING", 0)
    ACTIVE     = omniORB.EnumItem("ACTIVE", 1)
    DISCARDING = omniORB.EnumItem("DISCARDING", 2)
    INACTIVE   = omniORB.EnumItem("INACTIVE", 3)
    State = omniORB.Enum("IDL:omg.org/PortableServer/POAManager/State:1.0",
                         (HOLDING, ACTIVE, DISCARDING, INACTIVE))
    
    _d_State  = (omniORB.tcInternal.tv_enum, State._NP_RepositoryId,
                 "State", State._items)
    _tc_State = omniORB.tcInternal.createTypeCode(_d_State)
    omniORB.registerType(State._NP_RepositoryId, _d_State, _tc_State)



# interface POA
_d_POA = (omniORB.tcInternal.tv_objref,
          "IDL:omg.org/PortableServer/POA:1.0", "POA")

class POA (CORBA.Object) :
    """POA implementation."""
    
    _NP_RepositoryId = _d_POA[1]

    def __init__(self):
        self.__release = _omnipy.poa_func.releaseRef

    def __del__(self):
        self.__release(self)

    def create_POA(self, adapter_name, a_POAManager, policies):
        return _omnipy.poa_func.create_POA(self, adapter_name,
                                           a_POAManager, policies)

    def find_POA(self, adapter_name, activate_it):
        return _omnipy.poa_func.find_POA(self, adapter_name, activate_it)

    def destroy(self, etherialize_objects, wait_for_completion):
        _omnipy.poa_func.destroy(self, etherialize_objects,wait_for_completion)
        omniORB.poaCache.clear()

    def create_thread_policy(self, value):
        return ThreadPolicy(value)

    def create_lifespan_policy(self, value):
        return LifespanPolicy(value)

    def create_id_uniqueness_policy(self, value):
        return IdUniquenessPolicy(value)

    def create_id_assignment_policy(self, value):
        return IdAssignmentPolicy(value)

    def create_implicit_activation_policy(self, value):
        return ImplicitActivationPolicy(value)

    def create_servant_retention_policy(self, value):
        return ServantRetentionPolicy(value)

    def create_request_processing_policy(self, value):
        return RequestProcessingPolicy(value)

    def _get_the_name(self):
        return _omnipy.poa_func._get_the_name(self)

    def _get_the_parent(self):
        return _omnipy.poa_func._get_the_parent(self)

    def _get_the_children(self):
        return _omnipy.poa_func._get_the_children(self)

    def _get_the_POAManager(self):
        try:
            return self.__manager
        except AttributeError:
            self.__manager = _omnipy.poa_func._get_the_POAManager(self)
            return self.__manager

    def _get_the_activator(self):
        return _omnipy.poa_func._get_the_activator(self)

    def _set_the_activator(self, value):
        return _omnipy.poa_func._set_the_activator(self, value)

    def get_servant_manager(self):
        return _omnipy.poa_func.get_servant_manager(self)

    def set_servant_manager(self, imgr):
        return _omnipy.poa_func.set_servant_manager(self, imgr)

    def get_servant(self):
        return _omnipy.poa_func.get_servant(self)

    def set_servant(self, p_servant):
        return _omnipy.poa_func.set_servant(self, p_servant)

    def activate_object(self, p_servant):
        return _omnipy.poa_func.activate_object(self, p_servant)

    def activate_object_with_id(self, id, p_servant):
        return _omnipy.poa_func.activate_object_with_id(self, id, p_servant)

    def deactivate_object(self, oid):
        return _omnipy.poa_func.deactivate_object(self, oid)

    def create_reference(self, intf):
        return _omnipy.poa_func.create_reference(self, intf)

    def create_reference_with_id(self, oid, intf):
        return _omnipy.poa_func.create_reference_with_id(self, oid, intf)

    def servant_to_id(self, p_servant):
        return _omnipy.poa_func.servant_to_id(self, p_servant)

    def servant_to_reference(self, p_servant):
        return _omnipy.poa_func.servant_to_reference(self, p_servant)

    def reference_to_servant(self, reference):
        return _omnipy.poa_func.reference_to_servant(self, reference)

    def reference_to_id(self, reference):
        return _omnipy.poa_func.reference_to_id(self, reference)

    def id_to_servant(self, oid):
        return _omnipy.poa_func.id_to_servant(self, oid)

    def id_to_reference(self, oid):
        return _omnipy.poa_func.id_to_reference(self, oid)

    __methods__ = ["create_POA",
                   "find_POA",
                   "destroy",
                   "create_thread_policy",
                   "create_lifespan_policy",
                   "create_id_uniqueness_policy",
                   "create_id_assignment_policy",
                   "create_implicit_activation_policy",
                   "create_servant_retention_policy",
                   "create_request_processing_policy",
                   "_get_the_name",
                   "_get_the_parent",
                   "_get_the_children",
                   "_get_the_POAManager",
                   "_get_the_activator",
                   "_set_the_activator",
                   "get_servant_manager",
                   "set_servant_manager",
                   "get_servant",
                   "set_servant",
                   "activate_object",
                   "activate_object_with_id",
                   "deactivate_object",
                   "create_reference",
                   "create_reference_with_id",
                   "servant_to_id",
                   "servant_to_reference",
                   "reference_to_servant",
                   "reference_to_id",
                   "id_to_servant",
                   "id_to_reference"] + CORBA.Object.__methods__

    # Generated exception declarations
    # exception AdapterAlreadyExists
    class AdapterAlreadyExists (CORBA.UserException):
        _NP_RepositoryId = "IDL:omg.org/PortableServer/POA/AdapterAlreadyExists:1.0"

    _d_AdapterAlreadyExists  = (omniORB.tcInternal.tv_except,
                                AdapterAlreadyExists,
                                AdapterAlreadyExists._NP_RepositoryId,
                                "AdapterAlreadyExists")
    _tc_AdapterAlreadyExists = omniORB.tcInternal.createTypeCode(_d_AdapterAlreadyExists)
    omniORB.registerType(AdapterAlreadyExists._NP_RepositoryId,
                         _d_AdapterAlreadyExists, _tc_AdapterAlreadyExists)
    
    # exception AdapterInactive
    class AdapterInactive (CORBA.UserException):
        _NP_RepositoryId = "IDL:omg.org/PortableServer/POA/AdapterInactive:1.0"
    
    _d_AdapterInactive  = (omniORB.tcInternal.tv_except, AdapterInactive,
                           AdapterInactive._NP_RepositoryId, "AdapterInactive")
    _tc_AdapterInactive = omniORB.tcInternal.createTypeCode(_d_AdapterInactive)
    omniORB.registerType(AdapterInactive._NP_RepositoryId,
                         _d_AdapterInactive, _tc_AdapterInactive)
    
    # exception AdapterNonExistent
    class AdapterNonExistent (CORBA.UserException):
        _NP_RepositoryId = "IDL:omg.org/PortableServer/POA/AdapterNonExistent:1.0"
    
    _d_AdapterNonExistent  = (omniORB.tcInternal.tv_except,
                              AdapterNonExistent,
                              AdapterNonExistent._NP_RepositoryId,
                              "AdapterNonExistent")
    _tc_AdapterNonExistent = omniORB.tcInternal.createTypeCode(_d_AdapterNonExistent)
    omniORB.registerType(AdapterNonExistent._NP_RepositoryId,
                         _d_AdapterNonExistent, _tc_AdapterNonExistent)
    
    # exception InvalidPolicy
    class InvalidPolicy (CORBA.UserException):
        _NP_RepositoryId = "IDL:omg.org/PortableServer/POA/InvalidPolicy:1.0"
    
        def __init__(self, index):
            self.index = index
    
    _d_InvalidPolicy  = (omniORB.tcInternal.tv_except, InvalidPolicy,
                         InvalidPolicy._NP_RepositoryId, "InvalidPolicy",
                         "index", omniORB.tcInternal.tv_ushort)
    _tc_InvalidPolicy = omniORB.tcInternal.createTypeCode(_d_InvalidPolicy)
    omniORB.registerType(InvalidPolicy._NP_RepositoryId,
                         _d_InvalidPolicy, _tc_InvalidPolicy)
    
    # exception NoServant
    class NoServant (CORBA.UserException):
        _NP_RepositoryId = "IDL:omg.org/PortableServer/POA/NoServant:1.0"
    
    _d_NoServant  = (omniORB.tcInternal.tv_except, NoServant,
                     NoServant._NP_RepositoryId, "NoServant")
    _tc_NoServant = omniORB.tcInternal.createTypeCode(_d_NoServant)
    omniORB.registerType(NoServant._NP_RepositoryId,
                         _d_NoServant, _tc_NoServant)
    
    # exception ObjectAlreadyActive
    class ObjectAlreadyActive (CORBA.UserException):
        _NP_RepositoryId = "IDL:omg.org/PortableServer/POA/ObjectAlreadyActive:1.0"
    
    _d_ObjectAlreadyActive  = (omniORB.tcInternal.tv_except,
                               ObjectAlreadyActive,
                               ObjectAlreadyActive._NP_RepositoryId,
                               "ObjectAlreadyActive")
    _tc_ObjectAlreadyActive = omniORB.tcInternal.createTypeCode(_d_ObjectAlreadyActive)
    omniORB.registerType(ObjectAlreadyActive._NP_RepositoryId,
                         _d_ObjectAlreadyActive, _tc_ObjectAlreadyActive)
    
    # exception ObjectNotActive
    class ObjectNotActive (CORBA.UserException):
        _NP_RepositoryId = "IDL:omg.org/PortableServer/POA/ObjectNotActive:1.0"
    
    _d_ObjectNotActive  = (omniORB.tcInternal.tv_except, ObjectNotActive,
                           ObjectNotActive._NP_RepositoryId, "ObjectNotActive")
    _tc_ObjectNotActive = omniORB.tcInternal.createTypeCode(_d_ObjectNotActive)
    omniORB.registerType(ObjectNotActive._NP_RepositoryId,
                         _d_ObjectNotActive, _tc_ObjectNotActive)
    
    # exception ServantAlreadyActive
    class ServantAlreadyActive (CORBA.UserException):
        _NP_RepositoryId = "IDL:omg.org/PortableServer/POA/ServantAlreadyActive:1.0"
    
    _d_ServantAlreadyActive  = (omniORB.tcInternal.tv_except,
                                ServantAlreadyActive,
                                ServantAlreadyActive._NP_RepositoryId,
                                "ServantAlreadyActive")
    _tc_ServantAlreadyActive = omniORB.tcInternal.createTypeCode(_d_ServantAlreadyActive)
    omniORB.registerType(ServantAlreadyActive._NP_RepositoryId,
                         _d_ServantAlreadyActive, _tc_ServantAlreadyActive)
    
    # exception ServantNotActive
    class ServantNotActive (CORBA.UserException):
        _NP_RepositoryId = "IDL:omg.org/PortableServer/POA/ServantNotActive:1.0"
    
    _d_ServantNotActive  = (omniORB.tcInternal.tv_except, ServantNotActive,
                            ServantNotActive._NP_RepositoryId,
                            "ServantNotActive")
    _tc_ServantNotActive = omniORB.tcInternal.createTypeCode(_d_ServantNotActive)
    omniORB.registerType(ServantNotActive._NP_RepositoryId,
                         _d_ServantNotActive, _tc_ServantNotActive)
    
    # exception WrongAdapter
    class WrongAdapter (CORBA.UserException):
        _NP_RepositoryId = "IDL:omg.org/PortableServer/POA/WrongAdapter:1.0"
    
    _d_WrongAdapter  = (omniORB.tcInternal.tv_except, WrongAdapter,
                        WrongAdapter._NP_RepositoryId, "WrongAdapter")
    _tc_WrongAdapter = omniORB.tcInternal.createTypeCode(_d_WrongAdapter)
    omniORB.registerType(WrongAdapter._NP_RepositoryId,
                         _d_WrongAdapter, _tc_WrongAdapter)
    
    # exception WrongPolicy
    class WrongPolicy (CORBA.UserException):
        _NP_RepositoryId = "IDL:omg.org/PortableServer/POA/WrongPolicy:1.0"
    
    _d_WrongPolicy  = (omniORB.tcInternal.tv_except, WrongPolicy,
                       WrongPolicy._NP_RepositoryId, "WrongPolicy")
    _tc_WrongPolicy = omniORB.tcInternal.createTypeCode(_d_WrongPolicy)
    omniORB.registerType(WrongPolicy._NP_RepositoryId,
                         _d_WrongPolicy, _tc_WrongPolicy)


# interface Current
class Current (CORBA.Object) :
    _NP_RepositoryId = "IDL:omg.org/PortableServer/Current:1.0"

    def __init__(self):
        self.__release = _omnipy.poacurrent_func.releaseRef

    def __del__(self):
        self.__release(self)

    def get_POA(self):
        return _omnipy.poacurrent_func.get_POA(self)

    def get_object_id(self):
        return _omnipy.poacurrent_func.get_object_id(self)

    def get_reference(self):
        return _omnipy.poacurrent_func.get_reference(self)

    def get_servant(self):
        return _omnipy.poacurrent_func.get_servant(self)

    __methods__ = ["get_POA", "get_object_id",
                   "get_reference", "get_servant"] + CORBA.Object.__methods__

    # Generated declarations

    class NoContext (CORBA.UserException):
        _NP_RepositoryId = "IDL:omg.org/PortableServer/Current/NoContext:1.0"

    _d_NoContext  = (omniORB.tcInternal.tv_except, NoContext,
                     NoContext._NP_RepositoryId, "NoContext")
    _tc_NoContext = omniORB.tcInternal.createTypeCode(_d_NoContext)
    omniORB.registerType(NoContext._NP_RepositoryId,
                         _d_NoContext, _tc_NoContext)


# Generated declarations

# ObjectId
class ObjectId:
    _NP_RepositoryId = "IDL:omg.org/PortableServer/ObjectId:1.0"
    def __init__(self):
        raise RuntimeError("Cannot construct objects of this type.")

_d_ObjectId  = (omniORB.tcInternal.tv_sequence, omniORB.tcInternal.tv_octet, 0)
_ad_ObjectId = (omniORB.tcInternal.tv_alias,
                "IDL:omg.org/PortableServer/ObjectId:1.0", "ObjectId",
                (omniORB.tcInternal.tv_sequence,
                 omniORB.tcInternal.tv_octet, 0))
_tc_ObjectId = omniORB.tcInternal.createTypeCode(_ad_ObjectId)


# exception ForwardRequest
class ForwardRequest (CORBA.UserException):
    _NP_RepositoryId = "IDL:omg.org/PortableServer/ForwardRequest:1.0"

    def __init__(self, forward_reference):
        self.forward_reference = forward_reference

_d_ForwardRequest  = (omniORB.tcInternal.tv_except, ForwardRequest,
                      ForwardRequest._NP_RepositoryId, "ForwardRequest",
                      "forward_reference", CORBA._d_Object)
_tc_ForwardRequest = omniORB.tcInternal.createTypeCode(_d_ForwardRequest)
omniORB.registerType(ForwardRequest._NP_RepositoryId,
                     _d_ForwardRequest, _tc_ForwardRequest)


# Policies

def _create_policy(ptype, val):
    if ptype == 16:
        return ThreadPolicy(val)
    elif ptype == 17:
        return LifespanPolicy(val)
    elif ptype == 18:
        return IdUniquenessPolicy(val)
    elif ptype == 19:
        return IdAssignmentPolicy(val)
    elif ptype == 20:
        return ImplicitActivationPolicy(val)
    elif ptype == 21:
        return ServantRetentionPolicy(val)
    elif ptype == 22:
        return RequestProcessingPolicy(val)
    return None


class ThreadPolicy (CORBA.Policy):
    _NP_RepositoryId = "IDL:omg.org/PortableServer/ThreadPolicy:1.0"

    def __init__(self, value):
        if value not in ThreadPolicyValue._items:
            raise CORBA.PolicyError(CORBA.BAD_POLICY_TYPE)
        self._value       = value
        self._policy_type = 16

    def _get_value(self):
        return self._value

    __methods__ = ["_get_value"] + CORBA.Policy.__methods__

class LifespanPolicy (CORBA.Policy):
    _NP_RepositoryId = "IDL:omg.org/PortableServer/LifespanPolicy:1.0"

    def __init__(self, value):
        if value not in LifespanPolicyValue._items:
            raise CORBA.PolicyError(CORBA.BAD_POLICY_TYPE)
        self._value       = value
        self._policy_type = 17

    def _get_value(self):
        return self._value

    __methods__ = ["_get_value"] + CORBA.Policy.__methods__

class IdUniquenessPolicy (CORBA.Policy):
    _NP_RepositoryId = "IDL:omg.org/PortableServer/IdUniquenessPolicy:1.0"

    def __init__(self, value):
        if value not in IdUniquenessPolicyValue._items:
            raise CORBA.PolicyError(CORBA.BAD_POLICY_TYPE)
        self._value       = value
        self._policy_type = 18

    def _get_value(self):
        return self._value

    __methods__ = ["_get_value"] + CORBA.Policy.__methods__

class IdAssignmentPolicy (CORBA.Policy):
    _NP_RepositoryId = "IDL:omg.org/PortableServer/IdAssignmentPolicy:1.0"

    def __init__(self, value):
        if value not in IdAssignmentPolicyValue._items:
            raise CORBA.PolicyError(CORBA.BAD_POLICY_TYPE)
        self._value       = value
        self._policy_type = 19

    def _get_value(self):
        return self._value

    __methods__ = ["_get_value"] + CORBA.Policy.__methods__

class ImplicitActivationPolicy (CORBA.Policy):
    _NP_RepositoryId = "IDL:omg.org/PortableServer/ImplicitActivationPolicy:1.0"

    def __init__(self, value):
        if value not in ImplicitActivationPolicyValue._items:
            raise CORBA.PolicyError(CORBA.BAD_POLICY_TYPE)
        self._value       = value
        self._policy_type = 20

    def _get_value(self):
        return self._value

    __methods__ = ["_get_value"] + CORBA.Policy.__methods__

class ServantRetentionPolicy (CORBA.Policy):
    _NP_RepositoryId = "IDL:omg.org/PortableServer/ServantRetentionPolicy:1.0"

    def __init__(self, value):
        if value not in ServantRetentionPolicyValue._items:
            raise CORBA.PolicyError(CORBA.BAD_POLICY_TYPE)
        self._value       = value
        self._policy_type = 21

    def _get_value(self):
        return self._value

    __methods__ = ["_get_value"] + CORBA.Policy.__methods__

class RequestProcessingPolicy (CORBA.Policy):
    _NP_RepositoryId = "IDL:omg.org/PortableServer/RequestProcessingPolicy:1.0"

    def __init__(self, value):
        if value not in RequestProcessingPolicyValue._items:
            raise CORBA.PolicyError(CORBA.BAD_POLICY_TYPE)
        self._value       = value
        self._policy_type = 22

    def _get_value(self):
        return self._value

    __methods__ = ["_get_value"] + CORBA.Policy.__methods__


# enum ThreadPolicyValue
ORB_CTRL_MODEL      = omniORB.EnumItem("ORB_CTRL_MODEL", 0)
SINGLE_THREAD_MODEL = omniORB.EnumItem("SINGLE_THREAD_MODEL", 1)
MAIN_THREAD_MODEL   = omniORB.EnumItem("MAIN_THREAD_MODEL", 2)
ThreadPolicyValue   = omniORB.Enum(\
    "IDL:omg.org/PortableServer/ThreadPolicyValue:1.0",
    (ORB_CTRL_MODEL, SINGLE_THREAD_MODEL, MAIN_THREAD_MODEL))

_d_ThreadPolicyValue  = (omniORB.tcInternal.tv_enum,
                         ThreadPolicyValue._NP_RepositoryId,
                         "ThreadPolicyValue", ThreadPolicyValue._items)
_tc_ThreadPolicyValue = omniORB.tcInternal.createTypeCode(_d_ThreadPolicyValue)
omniORB.registerType(ThreadPolicyValue._NP_RepositoryId,
                     _d_ThreadPolicyValue, _tc_ThreadPolicyValue)

# enum LifespanPolicyValue
TRANSIENT  = omniORB.EnumItem("TRANSIENT", 0)
PERSISTENT = omniORB.EnumItem("PERSISTENT", 1)
LifespanPolicyValue = omniORB.Enum(\
    "IDL:omg.org/PortableServer/LifespanPolicyValue:1.0",
    (TRANSIENT, PERSISTENT))

_d_LifespanPolicyValue  = (omniORB.tcInternal.tv_enum,
                           LifespanPolicyValue._NP_RepositoryId,
                           "LifespanPolicyValue", LifespanPolicyValue._items)
_tc_LifespanPolicyValue = omniORB.tcInternal.createTypeCode(_d_LifespanPolicyValue)
omniORB.registerType(LifespanPolicyValue._NP_RepositoryId,
                     _d_LifespanPolicyValue, _tc_LifespanPolicyValue)

# enum IdUniquenessPolicyValue
UNIQUE_ID   = omniORB.EnumItem("UNIQUE_ID", 0)
MULTIPLE_ID = omniORB.EnumItem("MULTIPLE_ID", 1)
IdUniquenessPolicyValue = omniORB.Enum(\
    "IDL:omg.org/PortableServer/IdUniquenessPolicyValue:1.0",
    (UNIQUE_ID, MULTIPLE_ID))

_d_IdUniquenessPolicyValue  = (omniORB.tcInternal.tv_enum,
                               IdUniquenessPolicyValue._NP_RepositoryId,
                               "IdUniquenessPolicyValue",
                               IdUniquenessPolicyValue._items)
_tc_IdUniquenessPolicyValue = omniORB.tcInternal.createTypeCode(_d_IdUniquenessPolicyValue)
omniORB.registerType(IdUniquenessPolicyValue._NP_RepositoryId,
                     _d_IdUniquenessPolicyValue, _tc_IdUniquenessPolicyValue)

# enum IdAssignmentPolicyValue
USER_ID   = omniORB.EnumItem("USER_ID", 0)
SYSTEM_ID = omniORB.EnumItem("SYSTEM_ID", 1)
IdAssignmentPolicyValue = omniORB.Enum(\
    "IDL:omg.org/PortableServer/IdAssignmentPolicyValue:1.0",
    (USER_ID, SYSTEM_ID))

_d_IdAssignmentPolicyValue  = (omniORB.tcInternal.tv_enum,
                               IdAssignmentPolicyValue._NP_RepositoryId,
                               "IdAssignmentPolicyValue",
                               IdAssignmentPolicyValue._items)
_tc_IdAssignmentPolicyValue = omniORB.tcInternal.createTypeCode(_d_IdAssignmentPolicyValue)
omniORB.registerType(IdAssignmentPolicyValue._NP_RepositoryId,
                     _d_IdAssignmentPolicyValue, _tc_IdAssignmentPolicyValue)


# enum ImplicitActivationPolicyValue
IMPLICIT_ACTIVATION    = omniORB.EnumItem("IMPLICIT_ACTIVATION", 0)
NO_IMPLICIT_ACTIVATION = omniORB.EnumItem("NO_IMPLICIT_ACTIVATION", 1)
ImplicitActivationPolicyValue = omniORB.Enum(\
    "IDL:omg.org/PortableServer/ImplicitActivationPolicyValue:1.0",
    (IMPLICIT_ACTIVATION, NO_IMPLICIT_ACTIVATION))

_d_ImplicitActivationPolicyValue  = (omniORB.tcInternal.tv_enum,
                                     ImplicitActivationPolicyValue._NP_RepositoryId,
                                     "ImplicitActivationPolicyValue",
                                     ImplicitActivationPolicyValue._items)
_tc_ImplicitActivationPolicyValue = omniORB.tcInternal.createTypeCode(_d_ImplicitActivationPolicyValue)
omniORB.registerType(ImplicitActivationPolicyValue._NP_RepositoryId,
                     _d_ImplicitActivationPolicyValue,
                     _tc_ImplicitActivationPolicyValue)

# enum ServantRetentionPolicyValue
RETAIN     = omniORB.EnumItem("RETAIN", 0)
NON_RETAIN = omniORB.EnumItem("NON_RETAIN", 1)
ServantRetentionPolicyValue = omniORB.Enum(\
    "IDL:omg.org/PortableServer/ServantRetentionPolicyValue:1.0",
    (RETAIN, NON_RETAIN))

_d_ServantRetentionPolicyValue  = (omniORB.tcInternal.tv_enum,
                                   ServantRetentionPolicyValue._NP_RepositoryId,
                                   "ServantRetentionPolicyValue",
                                   ServantRetentionPolicyValue._items)
_tc_ServantRetentionPolicyValue = omniORB.tcInternal.createTypeCode(_d_ServantRetentionPolicyValue)
omniORB.registerType(ServantRetentionPolicyValue._NP_RepositoryId,
                     _d_ServantRetentionPolicyValue,
                     _tc_ServantRetentionPolicyValue)

# enum RequestProcessingPolicyValue
USE_ACTIVE_OBJECT_MAP_ONLY = omniORB.EnumItem("USE_ACTIVE_OBJECT_MAP_ONLY", 0)
USE_DEFAULT_SERVANT        = omniORB.EnumItem("USE_DEFAULT_SERVANT", 1)
USE_SERVANT_MANAGER        = omniORB.EnumItem("USE_SERVANT_MANAGER", 2)
RequestProcessingPolicyValue = omniORB.Enum(\
    "IDL:omg.org/PortableServer/RequestProcessingPolicyValue:1.0",
    (USE_ACTIVE_OBJECT_MAP_ONLY, USE_DEFAULT_SERVANT, USE_SERVANT_MANAGER))

_d_RequestProcessingPolicyValue  = (omniORB.tcInternal.tv_enum,
                                    RequestProcessingPolicyValue._NP_RepositoryId,
                                    "RequestProcessingPolicyValue",
                                    RequestProcessingPolicyValue._items)
_tc_RequestProcessingPolicyValue = omniORB.tcInternal.createTypeCode(_d_RequestProcessingPolicyValue)
omniORB.registerType(RequestProcessingPolicyValue._NP_RepositoryId,
                     _d_RequestProcessingPolicyValue,
                     _tc_RequestProcessingPolicyValue)


# ServantManagers

# interface ServantManager
_d_ServantManager = (omniORB.tcInternal.tv_local_interface,
                     "IDL:omg.org/PortableServer/ServantManager:1.0",
                     "ServantManager")

class ServantManager (CORBA.LocalObject):
    _NP_RepositoryId = _d_ServantManager[1]

    _nil = CORBA.Object._nil

_tc_ServantManager = omniORB.tcInternal.createTypeCode(_d_ServantManager)
omniORB.registerType(ServantManager._NP_RepositoryId,
                     _d_ServantManager, _tc_ServantManager)

# ServantManager object reference
class _objref_ServantManager (CORBA.Object):
    _NP_RepositoryId = ServantManager._NP_RepositoryId

    def __init__(self):
        CORBA.Object.__init__(self)

    __methods__ = [] + CORBA.Object.__methods__

omniORB.registerObjref(ServantManager._NP_RepositoryId, _objref_ServantManager)


# interface ServantActivator
_d_ServantActivator = (omniORB.tcInternal.tv_local_interface,
                       "IDL:omg.org/PortableServer/ServantActivator:1.0",
                       "ServantActivator")

class ServantActivator (ServantManager):
    _NP_RepositoryId = _d_ServantActivator[1]

    _nil = CORBA.Object._nil

    def incarnate(self, oid, adapter):
        raise CORBA.NO_IMPLEMENT(omniORB.NO_IMPLEMENT_NoPythonMethod,
                                 CORBA.COMPLETED_NO)

    def etherialize(self, oid, adapter, serv,
                    cleanup_in_progress, remaining_activations):
        raise CORBA.NO_IMPLEMENT(omniORB.NO_IMPLEMENT_NoPythonMethod,
                                 CORBA.COMPLETED_NO)


_tc_ServantActivator = omniORB.tcInternal.createTypeCode(_d_ServantActivator)
omniORB.registerType(ServantActivator._NP_RepositoryId,
                     _d_ServantActivator, _tc_ServantActivator)

ServantActivator._d_incarnate = ((_d_ObjectId, _d_POA), (_d_Servant, ),
                                 {ForwardRequest._NP_RepositoryId:
                                  _d_ForwardRequest})

ServantActivator._d_etherealize = ((_d_ObjectId, _d_POA, _d_Servant,
                                    omniORB.tcInternal.tv_boolean,
                                    omniORB.tcInternal.tv_boolean), (), None)


# ServantActivator object reference
class _objref_ServantActivator (_objref_ServantManager):
    _NP_RepositoryId = ServantActivator._NP_RepositoryId

    def __init__(self):
        _objref_ServantManager.__init__(self)

    def incarnate(self, oid, adapter):
        return _omnipy.invokeOp(self, "incarnate",
                                ServantActivator._d_incarnate,
                                (oid, adapter))

    def etherealize(self, oid, adapter, serv, cleanup_in_progress,
                    remaining_activations):
        return _omnipy.invokeOp(self, "etherialize",
                                ServantActivator._d_etherialize,
                                (oid, adapter, serv, cleanup_in_progress,
                                 remaining_activations))

    __methods__ = ["incarnate", "etherealize"] + \
                  _objref_ServantManager.__methods__

omniORB.registerObjref(ServantActivator._NP_RepositoryId,
                       _objref_ServantActivator)


# interface ServantLocator
_d_ServantLocator = (omniORB.tcInternal.tv_local_interface,
                     "IDL:omg.org/PortableServer/ServantLocator:1.0",
                     "ServantLocator")

class ServantLocator (ServantManager):
    _NP_RepositoryId = _d_ServantLocator[1]

    _nil = CORBA.Object._nil

    _d_Cookie = omniORB.tcInternal.tv_native

    def preinvoke(self, oid, adapter, operation):
        raise CORBA.NO_IMPLEMENT(omniORB.NO_IMPLEMENT_NoPythonMethod,
                                 CORBA.COMPLETED_NO)

    def postinvoke(self, oid, adapter, operations, the_cookie, the_servant):
        raise CORBA.NO_IMPLEMENT(omniORB.NO_IMPLEMENT_NoPythonMethod,
                                 CORBA.COMPLETED_NO)

_tc_ServantLocator = omniORB.tcInternal.createTypeCode(_d_ServantLocator)
omniORB.registerType(ServantLocator._NP_RepositoryId,
                     _d_ServantLocator, _tc_ServantLocator)

# ServantLocator operations and attributes
ServantLocator._d_preinvoke = ((_d_ObjectId, _d_POA, CORBA._d_Identifier),
                               (_d_Servant, ServantLocator._d_Cookie),
                               {ForwardRequest._NP_RepositoryId:
                                _d_ForwardRequest})

ServantLocator._d_postinvoke = ((_d_ObjectId, _d_POA, CORBA._d_Identifier,
                                 ServantLocator._d_Cookie, _d_Servant), (),
                                None)

# ServantLocator object reference
class _objref_ServantLocator (_objref_ServantManager):
    _NP_RepositoryId = ServantLocator._NP_RepositoryId

    def __init__(self):
        _objref_ServantManager.__init__(self)

    def preinvoke(self, oid, adapter, operation):
        return _omnipy.invokeOp(self, "preinvoke",
                                ServantLocator._d_preinvoke,
                                (oid, adapter, operation))

    def postinvoke(self, oid, adapter, operation, the_cookie, the_servant):
        return _omnipy.invokeOp(self, "postinvoke",
                                ServantLocator._d_postinvoke,
                                (oid, adapter, operation,
                                 the_cookie, the_servant))

    __methods__ = ["preinvoke", "postinvoke"] + \
                  _objref_ServantManager.__methods__

omniORB.registerObjref(ServantLocator._NP_RepositoryId, _objref_ServantLocator)



# AdapterActivator

# interface AdapterActivator
_d_AdapterActivator = (omniORB.tcInternal.tv_local_interface,
                       "IDL:omg.org/PortableServer/AdapterActivator:1.0",
                       "AdapterActivator")

class AdapterActivator (CORBA.LocalObject):
    _NP_RepositoryId = _d_AdapterActivator[1]

    _nil = CORBA.Object._nil

    def unknown_adapter(self, parent, name):
        raise CORBA.NO_IMPLEMENT(omniORB.NO_IMPLEMENT_NoPythonMethod,
                                 CORBA.COMPLETED_NO)


_tc_AdapterActivator = omniORB.tcInternal.createTypeCode(_d_AdapterActivator)
omniORB.registerType(AdapterActivator._NP_RepositoryId,
                     _d_AdapterActivator, _tc_AdapterActivator)

# AdapterActivator operations and attributes
AdapterActivator._d_unknown_adapter = ((_d_POA,
                                        (omniORB.tcInternal.tv_string,0)),
                                       (omniORB.tcInternal.tv_boolean, ), None)

# AdapterActivator object reference
class _objref_AdapterActivator (CORBA.Object):
    _NP_RepositoryId = AdapterActivator._NP_RepositoryId

    def __init__(self):
        CORBA.Object.__init__(self)

    def unknown_adapter(self, parent, name):
        return _omnipy.invokeOp(self, "unknown_adapter",
                                AdapterActivator._d_unknown_adapter,
                                (parent, name))

    __methods__ = ["unknown_adapter"] + CORBA.Object.__methods__

omniORB.registerObjref(AdapterActivator._NP_RepositoryId,
                       _objref_AdapterActivator)
