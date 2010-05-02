# -*- Mode: Python; -*-
#                            Package   : omniORBpy
# CORBA.py                   Created on: 1999/06/08
#                            Author    : Duncan Grisby (dpg1)
#
#    Copyright (C) 2002-2006 Apasphere Ltd
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
#    Definitions for CORBA module


# $Id: CORBA.py,v 1.31.2.15 2009/05/06 16:50:25 dgrisby Exp $
# $Log: CORBA.py,v $
# Revision 1.31.2.15  2009/05/06 16:50:25  dgrisby
# Updated copyright.
#
# Revision 1.31.2.14  2009/03/13 13:57:56  dgrisby
# Bind orb.register_initial_reference. Thanks Wei Jiang.
#
# Revision 1.31.2.13  2006/09/07 15:29:57  dgrisby
# Use boxes.idl to build standard value boxes.
#
# Revision 1.31.2.12  2006/07/26 17:49:59  dgrisby
# Support unchecked_narrow.
#
# Revision 1.31.2.11  2006/07/11 13:53:09  dgrisby
# Implement missing TypeCode creation functions.
#
# Revision 1.31.2.10  2006/03/06 18:30:31  dgrisby
# Missing module name in use of minor code.
#
# Revision 1.31.2.9  2006/02/28 12:41:59  dgrisby
# New _NP_postUnmarshal hook on valuetypes.
#
# Revision 1.31.2.8  2006/02/22 13:05:15  dgrisby
# __repr__ and _narrow methods for valuetypes.
#
# Revision 1.31.2.7  2006/01/19 17:28:44  dgrisby
# Merge from omnipy2_develop.
#
# Revision 1.31.2.6  2005/11/09 12:33:31  dgrisby
# Support POA LocalObjects.
#
# Revision 1.31.2.5  2005/04/25 18:28:16  dgrisby
# Implement narrow as a no-op if the Python classes have the right inheritance.
#
# Revision 1.31.2.4  2005/01/07 00:22:34  dgrisby
# Big merge from omnipy2_develop.
#
# Revision 1.31.2.3  2003/09/04 14:08:41  dgrisby
# Correct register_value_factory semantics.
#
# Revision 1.31.2.2  2003/05/20 17:10:25  dgrisby
# Preliminary valuetype support.
#
# Revision 1.31.2.1  2003/03/23 21:51:43  dgrisby
# New omnipy3_develop branch.
#
# Revision 1.28.2.17  2003/03/07 11:56:04  dgrisby
# Missing TypeCode creation functions.
#
# Revision 1.28.2.16  2002/09/21 23:27:11  dgrisby
# New omniORB.any helper module.
#
# Revision 1.28.2.15  2002/06/11 20:21:31  dgrisby
# Missed out wchar, wstring TypeCodes.
#
# Revision 1.28.2.14  2002/05/27 01:02:37  dgrisby
# Fix bug with scope lookup in generated code. Fix TypeCode clean-up bug.
#
# Revision 1.28.2.13  2002/05/26 00:56:57  dgrisby
# Minor bug in ORB __del__.
#
# Revision 1.28.2.12  2002/03/11 15:40:05  dpg1
# _get_interface support, exception minor codes.
#
# Revision 1.28.2.11  2002/01/18 15:49:45  dpg1
# Context support. New system exception construction. Fix None call problem.
#
# Revision 1.28.2.10  2001/09/20 14:51:25  dpg1
# Allow ORB reinitialisation after destroy(). Clean up use of omni namespace.
#
# Revision 1.28.2.9  2001/08/21 12:48:27  dpg1
# Meaningful exception minor code strings.
#
# Revision 1.28.2.8  2001/08/01 10:12:36  dpg1
# Main thread policy.
#
# Revision 1.28.2.7  2001/06/15 10:59:27  dpg1
# Apply fixes from omnipy1_develop.
#
# Revision 1.28.2.6  2001/05/14 12:48:27  dpg1
# Report exception minor code in hex.
#
# Revision 1.28.2.5  2001/04/10 11:11:14  dpg1
# TypeCode support and tests for Fixed point.
#
# Revision 1.28.2.4  2001/04/09 15:22:17  dpg1
# Fixed point support.
#
# Revision 1.28.2.3  2000/11/22 14:43:58  dpg1
# Support code set conversion and wchar/wstring.
#
# Revision 1.28.2.2  2000/11/01 15:29:01  dpg1
# Support for forward-declared structs and unions
# RepoIds in indirections are now resolved at the time of use
#
# Revision 1.28.2.1  2000/10/13 13:55:30  dpg1
# Initial support for omniORB 4.
#
# Revision 1.28  2000/08/21 10:20:19  dpg1
# Merge from omnipy1_develop for 1.1 release
#
# Revision 1.27.2.2  2000/08/17 08:46:06  dpg1
# Support for omniORB.LOCATION_FORWARD exception
#
# Revision 1.27.2.1  2000/08/07 09:19:24  dpg1
# Long long support
#
# Revision 1.27  2000/07/10 18:44:26  dpg1
# Remove partial IR stubs. Add TypeCodes for System Exceptions.
#
# Revision 1.26  2000/06/27 15:09:01  dpg1
# Fix to _is_a(). CORBA.Object registered in the objref map.
#
# Revision 1.25  2000/06/02 14:25:51  dpg1
# orb.run() now properly exits when the ORB is shut down
#
# Revision 1.24  2000/05/16 10:43:24  dpg1
# Add TC_foo names for TypeCode constants.
#
# Revision 1.23  2000/04/27 11:05:26  dpg1
# Add perform_work(), work_pending(), shutdown(), and destroy() operations.
#
# Revision 1.22  2000/03/29 10:15:47  dpg1
# Exceptions now more closely follow the interface of
# exceptions.Exception.
#
# Revision 1.21  2000/03/28 14:46:36  dpg1
# Undo the last change.
#
# Revision 1.20  2000/03/28 09:33:21  dpg1
# CORBA.Exception no longer derives from exceptions.Exception.
#
# Revision 1.19  2000/03/03 17:41:28  dpg1
# Major reorganisation to support omniORB 3.0 as well as 2.8.
#
# Revision 1.17  2000/01/31 10:51:42  dpg1
# Fix to exception throwing.
#
# Revision 1.16  2000/01/18 17:14:13  dpg1
# Support for pickle
#
# Revision 1.15  1999/12/07 12:35:33  dpg1
# id() function added.
#
# Revision 1.14  1999/11/25 14:12:34  dpg1
# sleep()ing for maxint seconds wasn't a good idea, since some platforms
# use milliseconds for their sleep system call.
#
# Revision 1.13  1999/11/25 14:01:45  dpg1
# orb.run() now uses time.sleep() to sleep, rather than blocking in
# impl_is_ready(). This means Python can interrupt the sleep.
#
# Revision 1.12  1999/11/10 16:08:21  dpg1
# Some types weren't registered properly.
#
# Revision 1.11  1999/10/18 08:25:57  dpg1
# _is_a() now works properly for local objects.
#
# Revision 1.10  1999/09/29 11:25:55  dpg1
# Nil objects now map to None. They work too, which is more than can be
# said for the old mapping...
#
# Revision 1.9  1999/09/27 09:06:37  dpg1
# Friendly error message if there is no thread support.
#
# Revision 1.8  1999/09/24 13:28:37  dpg1
# RootPOA added to list_initial_services() list.
#
# Revision 1.7  1999/09/24 09:22:01  dpg1
# Added copyright notices.
#
# Revision 1.6  1999/09/22 15:46:11  dpg1
# Fake POA implemented.
#
# Revision 1.5  1999/09/13 14:51:46  dpg1
# Any coercion implemented. TypeCode() constructor now complies to
# latest spec.
#
# Revision 1.4  1999/09/13 09:55:02  dpg1
# Initial references support. __methods__ added.
#
# Revision 1.3  1999/07/29 14:16:56  dpg1
# Server side, TypeCode creation interface.
#
# Revision 1.2  1999/07/19 15:47:05  dpg1
# TypeCode and Any support.
#
# Revision 1.1  1999/06/08 16:21:52  dpg1
# Initial revision
#

"""
Main omniORB CORBA module
"""


import _omnipy
import omniORB

import threading, types, time, exceptions, string


#############################################################################
#                                                                           #
# MISCELLANEOUS STUFF                                                       #
#                                                                           #
#############################################################################

TRUE  = 1
FALSE = 0



#############################################################################
#                                                                           #
# EXCEPTIONS                                                                #
#                                                                           #
#############################################################################

class Exception (exceptions.Exception):
    pass

OMGVMCID = 0x4f4d0000

# Completion status:

COMPLETED_YES     = omniORB.EnumItem("COMPLETED_YES",   0)
COMPLETED_NO      = omniORB.EnumItem("COMPLETED_NO",    1)
COMPLETED_MAYBE   = omniORB.EnumItem("COMPLETED_MAYBE", 2)
completion_status = omniORB.Enum("IDL:omg.org/CORBA/completion_status:1.0",
                                 (COMPLETED_YES,
                                  COMPLETED_NO,
                                  COMPLETED_MAYBE))

class SystemException (Exception):
    def __init__(self, minor=0, completed=COMPLETED_NO):
        self.minor = minor
        if type(completed) == types.IntType:
            self.completed = completion_status._item(completed)
        else:
            self.completed = completed
        Exception.__init__(self, minor, self.completed)

    def __repr__(self):
        minorName = omniORB.minorCodeToString(self)
        if minorName is None:
            minorName = hex(self.minor)
        else:
            minorName = "omniORB." + minorName

        return "CORBA.%s(%s, CORBA.%s)" % (self.__class__.__name__,
                                           minorName,
                                           self.completed)
    def __str__(self):
        return self.__repr__()



class UserException (Exception):
    _NP_RepositoryId = None
    _NP_ClassName = None

    def __init__(self, *args):
        self.__args = args

    def __repr__(self):
        cname = self._NP_ClassName
        if cname is None:
            cname = "%s.%s" % (self.__module__, self.__class__.__name__)

        desc = omniORB.findType(self._NP_RepositoryId)
        if desc is None:
            # Type is not properly registered
            return "<%s instance at 0x%x>" % (cname, builtin_id(self))
        vals = []
        for i in range(4, len(desc), 2):
            attr = desc[i]
            try:
                val = getattr(self, attr)
                vals.append("%s=%s" % (attr,repr(val)))
            except AttributeError:
                vals.append("%s=<not set>" % attr)

        return "%s(%s)" % (cname, string.join(vals, ", "))

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, i):
        return self.__args[i]

    def _tuple(self):
        return tuple(self)


# All the standard system exceptions...

g = globals()

for exc in _omnipy.system_exceptions:
    class _omni_sys_exc (SystemException):
        _NP_RepositoryId = "IDL:omg.org/CORBA/" + exc + ":1.0"

    _omni_sys_exc.__name__ = exc
    g[exc] = _omni_sys_exc

del g, exc


#############################################################################
#                                                                           #
# TypeCode                                                                  #
#                                                                           #
#############################################################################

tk_null               = omniORB.EnumItem("CORBA.tk_null",               0)
tk_void               = omniORB.EnumItem("CORBA.tk_void",               1)
tk_short              = omniORB.EnumItem("CORBA.tk_short",              2)
tk_long               = omniORB.EnumItem("CORBA.tk_long",               3)
tk_ushort             = omniORB.EnumItem("CORBA.tk_ushort",             4)
tk_ulong              = omniORB.EnumItem("CORBA.tk_ulong",              5)
tk_float              = omniORB.EnumItem("CORBA.tk_float",              6)
tk_double             = omniORB.EnumItem("CORBA.tk_double",             7)
tk_boolean            = omniORB.EnumItem("CORBA.tk_boolean",            8)
tk_char	              = omniORB.EnumItem("CORBA.tk_char",               9)
tk_octet              = omniORB.EnumItem("CORBA.tk_octet",              10)
tk_any	              = omniORB.EnumItem("CORBA.tk_any",        	11)
tk_TypeCode           = omniORB.EnumItem("CORBA.tk_TypeCode",           12)
tk_Principal          = omniORB.EnumItem("CORBA.tk_Principal",          13)
tk_objref             = omniORB.EnumItem("CORBA.tk_objref",             14)
tk_struct             = omniORB.EnumItem("CORBA.tk_struct",             15)
tk_union              = omniORB.EnumItem("CORBA.tk_union",              16)
tk_enum	              = omniORB.EnumItem("CORBA.tk_enum",               17)
tk_string             = omniORB.EnumItem("CORBA.tk_string",             18)
tk_sequence           = omniORB.EnumItem("CORBA.tk_sequence",           19)
tk_array              = omniORB.EnumItem("CORBA.tk_array",              20)
tk_alias              = omniORB.EnumItem("CORBA.tk_alias",              21)
tk_except             = omniORB.EnumItem("CORBA.tk_except",             22)
tk_longlong           = omniORB.EnumItem("CORBA.tk_longlong",           23)
tk_ulonglong          = omniORB.EnumItem("CORBA.tk_ulonglong",          24)
tk_longdouble         = omniORB.EnumItem("CORBA.tk_longdouble",         25)
tk_wchar              = omniORB.EnumItem("CORBA.tk_wchar",              26)
tk_wstring            = omniORB.EnumItem("CORBA.tk_wstring",            27)
tk_fixed              = omniORB.EnumItem("CORBA.tk_fixed",              28)
tk_value              = omniORB.EnumItem("CORBA.tk_value",              29)
tk_value_box          = omniORB.EnumItem("CORBA.tk_value_box",          30)
tk_native             = omniORB.EnumItem("CORBA.tk_native",             31)
tk_abstract_interface = omniORB.EnumItem("CORBA.tk_abstract_interface", 32)
tk_local_interface    = omniORB.EnumItem("CORBA.tk_local_interface",    33)

TCKind = omniORB.Enum("IDL:omg.org/CORBA/TCKind:1.0",
                      (tk_null, tk_void, tk_short, tk_long, tk_ushort,
                       tk_ulong, tk_float, tk_double, tk_boolean,
                       tk_char, tk_octet, tk_any, tk_TypeCode,
                       tk_Principal, tk_objref, tk_struct, tk_union,
                       tk_enum, tk_string, tk_sequence, tk_array,
                       tk_alias, tk_except, tk_longlong, tk_ulonglong,
                       tk_longdouble, tk_wchar, tk_wstring, tk_fixed,
                       tk_value, tk_value_box, tk_native,
                       tk_abstract_interface, tk_local_interface))

# ValueModifiers
VM_NONE        = 0
VM_CUSTOM      = 1
VM_ABSTRACT    = 2
VM_TRUNCATABLE = 3

# Value Visibility
PRIVATE_MEMBER = 0
PUBLIC_MEMBER  = 1


class TypeCode:
    class Bounds (UserException):
        pass
    class BadKind (UserException):
        pass

    def __init__(self, cr):
        self._t = tcInternal.typeCodeFromClassOrRepoId(cr)
        self._d = self._t._d
        self._k = self._t._k

    def equal(self, tc):                return self._t.equal(tc)
    def equivalent(self, tc):           return self._t.equivalent(tc)
    def get_compact_typecode(self):     return self._t.get_compact_typecode()
    def kind(self):                     return self._t.kind()
    def __repr__(self):                 return self._t.__repr__()
    
    # Operations which are only available for some kinds:
    def id(self):                       return self._t.id()
    def name(self):                     return self._t.name()
    def member_count(self):             return self._t.member_count()
    def member_name(self, index):       return self._t.member_name(index)
    def member_type(self, index):       return self._t.member_type(index)
    def member_label(self, index):      return self._t.member_label(index)

    def discriminator_type(self):       return self._t.discriminator_type()
    def default_index(self):            return self._t.default_index()
    def length(self):                   return self._t.length()
    def content_type(self):             return self._t.content_type()

    def fixed_digits(self):             return self._t.fixed_digits()
    def fixed_scale(self):              return self._t.fixed_scale()

    def member_visibility(self, index): return self._t.member_visibility(index)
    def type_modifier(self):            return self._t.type_modifier()
    def concrete_base_type(self):       return self._t.concrete_base_type()

    __methods__ = ["equal", "equivalent", "get_compact_typecode",
                   "kind", "id", "name", "member_count", "member_name",
                   "member_type", "member_label", "discriminator_type",
                   "default_index", "length", "content_type",
                   "fixed_digits", "fixed_scale", "member_visibility",
                   "type_modifier", "concrete_base_type"]

import tcInternal
_d_TypeCode = tcInternal.tv_TypeCode


# TypeCodes of basic types. The CORBA mapping says the TypeCode
# constants should start TC, but omniORBpy previously used _tc, so we
# support both:

TC_null     = _tc_null     = tcInternal.createTypeCode(tcInternal.tv_null)
TC_void     = _tc_void     = tcInternal.createTypeCode(tcInternal.tv_void)
TC_short    = _tc_short    = tcInternal.createTypeCode(tcInternal.tv_short)
TC_long     = _tc_long     = tcInternal.createTypeCode(tcInternal.tv_long)
TC_ushort   = _tc_ushort   = tcInternal.createTypeCode(tcInternal.tv_ushort)
TC_ulong    = _tc_ulong    = tcInternal.createTypeCode(tcInternal.tv_ulong)
TC_float    = _tc_float    = tcInternal.createTypeCode(tcInternal.tv_float)
TC_double   = _tc_double   = tcInternal.createTypeCode(tcInternal.tv_double)
TC_boolean  = _tc_boolean  = tcInternal.createTypeCode(tcInternal.tv_boolean)
TC_char     = _tc_char     = tcInternal.createTypeCode(tcInternal.tv_char)
TC_octet    = _tc_octet    = tcInternal.createTypeCode(tcInternal.tv_octet)
TC_any      = _tc_any      = tcInternal.createTypeCode(tcInternal.tv_any)
TC_TypeCode = _tc_TypeCode = tcInternal.createTypeCode(tcInternal.tv_TypeCode)
TC_Principal= _tc_Principal= tcInternal.createTypeCode(tcInternal.tv_Principal)
TC_string   = _tc_string   =tcInternal.createTypeCode((tcInternal.tv_string,0))
TC_longlong = _tc_longlong = tcInternal.createTypeCode(tcInternal.tv_longlong)
TC_ulonglong= _tc_ulonglong= tcInternal.createTypeCode(tcInternal.tv_ulonglong)
TC_longdouble = _tc_longdouble \
            = tcInternal.createTypeCode(tcInternal.tv_longdouble)
TC_wchar    = _tc_wchar    = tcInternal.createTypeCode(tcInternal.tv_wchar)
TC_wstring  = _tc_wstring \
            = tcInternal.createTypeCode((tcInternal.tv_wstring,0))

# id() function returns the repository ID of an object
builtin_id = id

def id(obj):
    try:
        return obj._NP_RepositoryId
    except AttributeError:
        raise BAD_PARAM(omniORB.BAD_PARAM_WrongPythonType, COMPLETED_NO)


#############################################################################
#                                                                           #
# Any                                                                       #
#                                                                           #
#############################################################################

class Any:
    def __init__(self, t, v):
        if not isinstance(t, TypeCode):
            raise TypeError("Argument 1 must be a TypeCode.")
        self._t = t
        self._v = v

    def typecode(self):
        return self._t

    def value(self, coerce=None):
        if coerce is None:
            return self._v

        if not isinstance(coerce, TypeCode):
            raise TypeError("Argument 1 must be a TypeCode if present.")

        return omniORB.coerceAny(self._v, self._t._d, coerce._d)

    def __repr__(self):
        return "CORBA.Any(%s, %s)" % (repr(self._t), repr(self._v))

    __methods__ = ["typecode", "value"]

_d_any = tcInternal.tv_any


#############################################################################
#                                                                           #
# ORB                                                                       #
#                                                                           #
#############################################################################

if _omnipy.coreVersion() == "2.8.0":
    ORB_ID = "omniORB2"
elif _omnipy.coreVersion()[0] == "3":
    ORB_ID = "omniORB3"
elif _omnipy.coreVersion()[0] == "4":
    ORB_ID = "omniORB4"
else:
    ORB_ID = "UnknownORB"

def ORB_init(argv=[], orb_identifier = ORB_ID):
    if _omnipy.need_ORB_init():
        omniORB.poaCache.clear()
        omniORB.orb = ORB(argv, orb_identifier)
        omniORB.rootPOA = None

    return omniORB.orb


class ORB:
    """omnipy ORB object"""

    def __init__(self, argv, orb_identifier):
        self.__release = None
        _omnipy.ORB_init(self, argv, orb_identifier)
        self.__release = _omnipy.orb_func.releaseRef
        self.__context = None

    def __del__(self):
        if self.__release:
            self.__release(self)

    def id(self):
        return ORB_ID

    def string_to_object(self, ior):
        return _omnipy.orb_func.string_to_object(self, ior)

    def object_to_string(self, obj):
        return _omnipy.orb_func.object_to_string(self, obj)

    def register_initial_reference(self, identifier, obj):
        return _omnipy.orb_func.register_initial_reference(self, identifier, obj)

    def list_initial_services(self):
        return _omnipy.orb_func.list_initial_services(self)

    def resolve_initial_references(self, identifier):
        return _omnipy.orb_func.resolve_initial_references(self, identifier)

    def work_pending(self):
        return _omnipy.orb_func.work_pending(self)

    def perform_work(self):
        _omnipy.orb_func.perform_work(self)

    def run(self):
        # We have to use a timeout rather than just blocking in run(),
        # since otherwise ctrl-c is not handled.

        timeout = 0.000001 # 1 usec

        shutdown = _omnipy.orb_func.run_timeout(self, timeout)

        try:
            while not shutdown:
                if timeout < 1.0:
                    timeout = timeout * 1.1

                shutdown = _omnipy.orb_func.run_timeout(self, timeout)

        except BAD_INV_ORDER:
            # If a shutdown races with the timeout occurring, we will
            # call run_timeout() on the shutdown ORB, resulting in a
            # BAD_INV_ORDER exception.
            pass

    def shutdown(self, wait_for_completion):
        omniORB.poaCache.clear()
        _omnipy.orb_func.shutdown(self, wait_for_completion)

    def destroy(self):
        omniORB.poaCache.clear()
        _omnipy.orb_func.destroy(self)


    # TypeCode operations
    def create_struct_tc(self, id, name, members):
        return tcInternal.createStructTC(id, name, members)

    def create_union_tc(self, id, name, discriminator_type, members):
        return tcInternal.createUnionTC(id, name, discriminator_type, members)

    def create_enum_tc(self, id, name, members):
        return tcInternal.createEnumTC(id, name, members)

    def create_alias_tc(self, id, name, original_type):
        return tcInternal.createAliasTC(id, name, original_type)

    def create_exception_tc(self, id, name, members):
        return tcInternal.createExceptionTC(id, name, members)

    def create_interface_tc(self, id, name):
        return tcInternal.createInterfaceTC(id, name)

    def create_string_tc(self, bound):
        return tcInternal.createStringTC(bound)

    def create_wstring_tc(self, bound):
        return tcInternal.createWStringTC(bound)

    def create_fixed_tc(self, digits, scale):
        return tcInternal.createFixedTC(digits, scale)

    def create_sequence_tc(self, bound, element_type):
        return tcInternal.createSequenceTC(bound, element_type)
    
    def create_array_tc(self, length, element_type):
        return tcInternal.createArrayTC(length, element_type)

    def create_value_tc(self, id, name, modifier, base, members):
        return tcInternal.createValueTC(id, name, modifier, base, members)

    def create_value_box_tc(self, id, name, boxed_type):
        return tcInternal.createValueBoxTC(id, name, boxed_type)

    def create_recursive_tc(self, id):
        return tcInternal.createRecursiveTC(id)

    def create_abstract_interface_tc(self, id, name):
        return tcInternal.createAbstractInterfaceTC(id, name)

    def create_local_interface_tc(self, id, name):
        return tcInternal.createLocalInterfaceTC(id, name)

    # Context operation
    def get_default_context(self):
        omniORB.lock.acquire()
        if self.__context is None:
            self.__context = Context("", None)
        omniORB.lock.release()
        return self.__context

    # ValueFactory operations
    def register_value_factory(self, repoId, factory):
        return omniORB.registerValueFactory(repoId, factory)

    def unregister_value_factory(self, repoId):
        omniORB.unregisterValueFactory(repoId)

    def lookup_value_factory(self, repoId):
        return omniORB.findValueFactory(repoId)

    # Policy operation
    def create_policy(self, ptype, val):
        if isinstance(val, Any):
            val = val._v

        import PortableServer
        p = PortableServer._create_policy(ptype, val)
        if p: return p

        import BiDirPolicy
        p = BiDirPolicy._create_policy(ptype, val)
        if p: return p

        raise PolicyError(BAD_POLICY)


    __methods__ = ["string_to_object", "object_to_string",
                   "register_initial_reference",
                   "list_initial_services", "resolve_initial_references",
                   "work_pending", "perform_work", "run",
                   "shutdown", "destroy",
                   "create_struct_tc", "create_union_tc",
                   "create_enum_tc", "create_alias_tc",
                   "create_exception_tc", "create_interface_tc",
                   "create_string_tc", "create_sequence_tc",
                   "create_array_tc", "create_recursive_tc",
                   "create_value_tc", "create_value_box_tc",
                   "create_abstract_interface_tc", "create_local_interface_tc",
                   "get_default_context", "register_value_factory",
                   "lookup_value_factory"]

    class InvalidName (UserException):
        _NP_RepositoryId = "IDL:omg.org/CORBA/ORB/InvalidName:1.0"

    _d_InvalidName  = (omniORB.tcInternal.tv_except, InvalidName,
                       InvalidName._NP_RepositoryId, "InvalidName")
    _tc_InvalidName = omniORB.tcInternal.createTypeCode(_d_InvalidName)
    omniORB.registerType(InvalidName._NP_RepositoryId,
                         _d_InvalidName, _tc_InvalidName)



#############################################################################
#                                                                           #
# OBJECT                                                                    #
#                                                                           #
#############################################################################

class Object:
    """ CORBA::Object base class """

    _NP_RepositoryId = ""

    _nil = None

    def __init__(self):
        self.__release = _omnipy.releaseObjref

    def __del__(self):
        self.__release(self)

    def __getstate__(self):
        return ORB_init().object_to_string(self)

    def __setstate__(self, state):
        o = ORB_init().string_to_object(state)
        self.__dict__.update(o.__dict__)
        def dummy(): pass # Why doesn't dummy want an argument? ***
        o.__del__ = dummy

    def _get_interface(self):
        import omniORB
        if omniORB.orb is None:
            raise BAD_INV_ORDER(omniORB.BAD_INV_ORDER_ORBHasShutdown,
                                COMPLETED_NO)

        omniORB.importIRStubs()

        try:
            return _omnipy.invoke(self, "_interface", _d_Object_interface, ())
        except Exception:
            pass

        ir = omniORB.orb.resolve_initial_references("InterfaceRepository")
        ir = ir and ir._narrow(Repository)
        if ir is None:
            raise INTF_REPOS(omniORB.INTF_REPOS_NotAvailable, COMPLETED_NO)
        interf = ir.lookup_id(self._NP_RepositoryId)
        return interf._narrow(InterfaceDef)
    
    def _is_a(self, repoId):
        return _omnipy.isA(self, repoId)
    
    def _non_existent(self):
        return _omnipy.nonExistent(self)
    
    def _is_equivalent(self, other_object):
        if self == other_object: return TRUE
        return _omnipy.isEquivalent(self, other_object)
    
    def _hash(self, maximum):
        return _omnipy.hash(self, maximum)
    
    def _duplicate(self, obj):
        return self
    
    def _release(self):
        return

    def _narrow(self, dest):
        repoId = dest._NP_RepositoryId
        try:
            dest_objref = omniORB.objrefMapping[repoId]
            if isinstance(self, dest_objref):
                return self
        except KeyError:
            pass
        return _omnipy.narrow(self, repoId, 1)

    def _unchecked_narrow(self, dest):
        repoId = dest._NP_RepositoryId
        try:
            dest_objref = omniORB.objrefMapping[repoId]
            if isinstance(self, dest_objref):
                return self
        except KeyError:
            pass
        return _omnipy.narrow(self, repoId, 0)

    __methods__ = ["_is_a", "_non_existent", "_is_equivalent",
                   "_get_interface", "_hash", "_narrow", "_unchecked_narrow"]

_d_Object  = (omniORB.tcInternal.tv_objref, Object._NP_RepositoryId, "Object")
TC_Object  = _tc_Object = omniORB.tcInternal.createTypeCode(_d_Object)
omniORB.registerType(Object._NP_RepositoryId, _d_Object, _tc_Object)
omniORB.registerObjref(Object._NP_RepositoryId, Object)
other_id = "IDL:omg.org/CORBA/Object:1.0"
omniORB.registerType(other_id, _d_Object, _tc_Object)
omniORB.registerObjref(other_id, Object)
del other_id

#############################################################################
#                                                                           #
# LocalObject                                                               #
#                                                                           #
#############################################################################

class LocalObject (Object):

    _NP_RepositoryId = "IDL:omg.org/CORBA/LocalObject:1.0"

    def __init__(self):
        # Override base Object __init__
        def no_release(obj): pass
        self._Object__release = no_release

_d_LocalObject  = (omniORB.tcInternal.tv_local_interface, LocalObject._NP_RepositoryId, "LocalObject")
TC_LocalObject  = _tc_LocalObject = omniORB.tcInternal.createTypeCode(_d_LocalObject)
omniORB.registerType(LocalObject._NP_RepositoryId, _d_LocalObject, _tc_LocalObject)



#############################################################################
#                                                                           #
# ValueBase                                                                 #
#                                                                           #
#############################################################################

class ValueBase:
    """ CORBA::ValueBase base class """

    _NP_RepositoryId = "IDL:omg.org/CORBA/ValueBase:1.0"
    _NP_ClassName = None
    __in_repr = 0

    def _get_value_def(self):
        import omniORB
        if omniORB.orb is None:
            raise BAD_INV_ORDER(omniORB.BAD_INV_ORDER_ORBHasShutdown,
                                COMPLETED_NO)

        omniORB.importIRStubs()

        ir = omniORB.orb.resolve_initial_references("InterfaceRepository")
        ir = ir and ir._narrow(Repository)
        if ir is None:
            raise INTF_REPOS(omniORB.INTF_REPOS_NotAvailable, COMPLETED_NO)
        interf = ir.lookup_id(self._NP_RepositoryId)
        return interf._narrow(ValueDef)

    def __repr__(self):
        if self.__in_repr:
            return "..."

        self.__in_repr = 1
        try:
            cname = self._NP_ClassName
            if cname is None:
                cname = "%s.%s" % (self.__module__, self.__class__.__name__)

            desc = omniORB.findType(self._NP_RepositoryId)
            if desc is None:
                # Type is not properly registered
                return "<%s instance at 0x%x>" % (cname, builtin_id(self))

            descs = []
            while desc != tcInternal.tv_null:
                descs.append(desc)
                desc = desc[6]
            descs.reverse()

            vals = []

            for desc in descs:
                for i in range(7, len(desc), 3):
                    attr = desc[i]
                    try:
                        val = getattr(self, attr)
                        vals.append("%s=%s" % (attr,repr(val)))
                    except AttributeError:
                        vals.append("%s=<not set>" % attr)

            return "%s(%s)" % (cname, string.join(vals, ", "))
        finally:
            try:
                del self.__in_repr
            except AttributeError:
                pass

    def _narrow(self, dest):
        # Narrow function for abstract interfaces
        if isinstance(self, dest):
            return self

        repoId = dest._NP_RepositoryId
        try:
            dest_skel = omniORB.skeletonMapping[repoId]
            if isinstance(self, dest_skel):
                return self
        except KeyError:
            pass
        return None

    def _NP_postUnmarshal(self):
        """
        _NP_postUnmarshal(self)

        Called when a value has been completely unmarshalled. May
        modify the object state. The return value is used as the
        unmarshalled valuetype, so must return self or a compatible
        object.
        """
        return self


_d_ValueBase = (omniORB.tcInternal.tv_value, ValueBase,
                ValueBase._NP_RepositoryId, "ValueBase", VM_NONE, None,
                omniORB.tcInternal.tv_null)
TC_ValueBase = _tc_ValueBase = omniORB.tcInternal.createTypeCode(_d_ValueBase)
omniORB.registerType(ValueBase._NP_RepositoryId, _d_ValueBase, _tc_ValueBase)
omniORB.registerValueFactory(ValueBase._NP_RepositoryId, ValueBase)

#############################################################################
#                                                                           #
# Policy                                                                    #
#                                                                           #
#############################################################################

class Policy (Object):
    _NP_RepositoryId = "IDL:omg.org/CORBA/Policy:1.0"

    def __init__(self):
        raise RuntimeError("Cannot construct objects of this type.")

    def __del__(self):
        # Override base CORBA.Object
        pass

    def _get_policy_type(self):
        return self._policy_type

    def copy(self):
        return self

    def destroy(self):
        pass

    def _is_a(self, repoId):
        return omniORB.static_is_a(self.__class__, repoId)

    def _non_existent(self):
        return 0

    def _is_equivalent(self, other_object):
        return self == other_object

    def _hash(self, maximum):
        return hash(self) % maximum

    def _narrow(self, dest):
        if self._is_a(dest._NP_RepositoryId):
            return self
        else:
            return None

    __methods__ = ["_get_policy_type", "copy", "destroy"] + Object.__methods__



#############################################################################
#                                                                           #
# Context                                                                   #
#                                                                           #
#############################################################################

class Context (Object):
    _NP_RepositoryId = "IDL:omg.org/CORBA/Context:1.0"

    def __init__(self, name, parent, values=None):
        self.__name   = name
        self.__parent = parent
        if values:
            self.__values = values
        else:
            self.__values = {}

    def __del__(self):
        pass

    def set_one_value(self, name, val):
        if type(name) is not types.StringType or \
           type(val) is not types.StringType:
            raise BAD_PARAM(omniORB.BAD_PARAM_WrongPythonType, COMPLETED_NO)
        
        self.__values[name] = val

    def set_values(self, values):
        if type(values) is not types.DictType:
            raise BAD_PARAM(omniORB.BAD_PARAM_WrongPythonType, COMPLETED_NO)
        
        for k,v in values.items():
            if type(k) is not types.StringType or \
               type(v) is not types.StringType:
                raise BAD_PARAM(omniORB.BAD_PARAM_WrongPythonType,
                                COMPLETED_NO)

        self.__values.update(values)

    def get_values(self, pattern, start_scope=None):
        if type(pattern) is not types.StringType:
            raise BAD_PARAM(omniORB.BAD_PARAM_WrongPythonType, COMPLETED_NO)
        
        ctxt = self
        if start_scope:
            while ctxt and ctxt.__name != start_scope:
                ctxt = ctxt.__parent
            if ctxt is None:
                raise BAD_CONTEXT(omniORB.BAD_CONTEXT_StartingScopeNotFound,
                                  COMPLETED_NO)

        r = ctxt._get_values([pattern])
        if r == {}: raise BAD_CONTEXT(omniORB.BAD_CONTEXT_NoMatchingProperty,
                                      COMPLETED_NO)
        return r

    def delete_values(self, pattern):
        if type(pattern) is not types.StringType or pattern == "":
            raise BAD_PARAM(omniORB.BAD_PARAM_WrongPythonType, COMPLETED_NO)

        found = 0

        try:
            if pattern[-1] == "*":
                # Wildcard
                pattern = pattern[:-1]
                pl = len(pattern)
                for k in self.__values.keys():
                    if k[:pl] == pattern:
                        found = 1
                        del self.__values[k]
            else:
                del self.__values[pattern]
                found = 1

        except KeyError:
            pass
        
        if not found:
            raise BAD_CONTEXT(omniORB.BAD_CONTEXT_NoMatchingProperty,
                              COMPLETED_NO)

    def create_child(self, ctx_name):
        return Context(ctx_name, self)

    #
    # CORBA::Object methods
    #
    def _is_a(self, repoId):
        return omniORB.static_is_a(self.__class__, repoId)

    def _non_existent(self):
        return 0

    def _is_equivalent(self, other_object):
        return self == other_object

    def _hash(self, maximum):
        return hash(self) % maximum

    def _narrow(self, dest):
        if self._is_a(dest._NP_RepositoryId):
            return self
        else:
            return None

    #
    # Internal implementation
    #
    def _get_values(self, patterns, values = None):
        if values is None:
            values = {}

        for pattern in patterns:
            if pattern[-1] == "*":
                # Wildcard
                pattern = pattern[:-1]
                pl = len(pattern)
                for k,v in self.__values.items():
                    if k[:pl] == pattern and not values.has_key(k):
                        values[k] = v
            else:
                # Not a wildcard
                if not values.has_key(pattern):
                    v = self.__values.get(pattern)
                    if v:
                        values[pattern] = v

        if self.__parent:
            self.__parent._get_values(patterns, values)

        return values
        
    __methods__ = ["set_one_value", "set_values", "get_values",
                   "delete_values", "create_child"] + Object.__methods__



#############################################################################
#                                                                           #
# CORBA module functions                                                    #
#                                                                           #
#############################################################################

def is_nil(obj):
    if obj is None:
        return 1
    if isinstance(obj, Object):
        return 0
    raise BAD_PARAM(omniORB.BAD_PARAM_WrongPythonType, COMPLETED_NO)

# Fixed point constructor
fixed = omniORB.fixed


#############################################################################
#                                                                           #
# Wide character things (only for Pythons with Unicode support)             #
#                                                                           #
#############################################################################

try:
    wstr = unichr
    word = ord
except NameError:
    def wstr(c):
        raise NO_IMPLEMENT(omniORB.NO_IMPLEMENT_Unsupported, COMPLETED_NO)
    def word(c):
        raise NO_IMPLEMENT(omniORB.NO_IMPLEMENT_Unsupported, COMPLETED_NO)


#############################################################################
#                                                                           #
# Interface Repository stuff                                                #
#                                                                           #
#############################################################################

# Note that we do not include the majority of the IfR declarations
# here, because that would cause lots of bloat. Call
# omniORB.importIRStubs() to import the full IfR declarations.

# typedef string Identifier
class Identifier:
    _NP_RepositoryId = "IDL:omg.org/CORBA/Identifier:1.0"
    def __init__(self):
        raise RuntimeError("Cannot construct objects of this type.")
_d_Identifier  = (tcInternal.tv_string,0)
_ad_Identifier = (tcInternal.tv_alias, Identifier._NP_RepositoryId, "Identifier", (tcInternal.tv_string,0))
_tc_Identifier = tcInternal.createTypeCode(_ad_Identifier)
omniORB.registerType(Identifier._NP_RepositoryId, _ad_Identifier, _tc_Identifier)


#############################################################################
#                                                                           #
# TypeCodes for System Exceptions                                           #
#                                                                           #
#############################################################################

_d_completion_status  = (tcInternal.tv_enum, completion_status._NP_RepositoryId, "completion_status", completion_status._items)
_tc_completion_status = tcInternal.createTypeCode(_d_completion_status)
omniORB.registerType(completion_status._NP_RepositoryId, _d_completion_status, _tc_completion_status)

# Strings to put in the descriptors, so all descriptors share the same
# strings.
_minor     = "minor"
_completed = "completed"

g = globals()
for exc in _omnipy.system_exceptions:
    r = g[exc]._NP_RepositoryId
    d = (tcInternal.tv_except, g[exc], r, exc, _minor,
         tcInternal.tv_long, _completed, _d_completion_status)
    t = tcInternal.createTypeCode(d)
    g["_d_"  + exc] = d
    g["_tc_" + exc] = t
    omniORB.registerType(r,d,t)

del _minor, _completed, g, r, d, t, exc



#############################################################################
#                                                                           #
# Exceptions defined in CORBA module                                        #
#                                                                           #
#############################################################################

# typedef ... PolicyErrorCode
class PolicyErrorCode:
    _NP_RepositoryId = "IDL:omg.org/CORBA/PolicyErrorCode:1.0"
    def __init__(self, *args, **kw):
        raise RuntimeError("Cannot construct objects of this type.")
_d_PolicyErrorCode  = tcInternal.tv_short
_ad_PolicyErrorCode = (tcInternal.tv_alias, PolicyErrorCode._NP_RepositoryId, "PolicyErrorCode", tcInternal.tv_short)
_tc_PolicyErrorCode = tcInternal.createTypeCode(_ad_PolicyErrorCode)
omniORB.registerType(PolicyErrorCode._NP_RepositoryId, _ad_PolicyErrorCode, _tc_PolicyErrorCode)

BAD_POLICY = 0
UNSUPPORTED_POLICY = 1
BAD_POLICY_TYPE = 2
BAD_POLICY_VALUE = 3
UNSUPPORTED_POLICY_VALUE = 4

# exception PolicyError
class PolicyError (UserException):
    _NP_RepositoryId = "IDL:omg.org/CORBA/PolicyError:1.0"

    def __init__(self, reason):
        UserException.__init__(self, reason)
        self.reason = reason

_d_PolicyError  = (tcInternal.tv_except, PolicyError, PolicyError._NP_RepositoryId, "PolicyError", "reason", omniORB.typeMapping["IDL:omg.org/CORBA/PolicyErrorCode:1.0"])
_tc_PolicyError = tcInternal.createTypeCode(_d_PolicyError)
omniORB.registerType(PolicyError._NP_RepositoryId, _d_PolicyError, _tc_PolicyError)

