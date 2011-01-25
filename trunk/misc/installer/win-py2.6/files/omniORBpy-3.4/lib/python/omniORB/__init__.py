# -*- Mode: Python; -*-
#                            Package   : omniORBpy
# __init__.py                Created on: 1999/07/19
#                            Author    : Duncan Grisby (dpg1)
#
#    Copyright (C) 2002-2008 Apasphere Ltd
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
#    omniORB module -- omniORB specific things


# $Id: __init__.py,v 1.30.2.24 2009/06/18 09:13:47 dgrisby Exp $
# $Log: __init__.py,v $
# Revision 1.30.2.24  2009/06/18 09:13:47  dgrisby
# Track change in Python 2.6's threading module. Thanks Luke Deller.
#
# Revision 1.30.2.23  2009/05/06 16:50:23  dgrisby
# Updated copyright.
#
# Revision 1.30.2.22  2008/08/21 10:53:55  dgrisby
# Hook Thread.__stop instead of __delete. Thanks Luke Deller.
#
# Revision 1.30.2.21  2008/02/01 16:29:16  dgrisby
# Error with implementation of operations with names clashing with
# Python keywords.
#
# Revision 1.30.2.20  2007/10/07 15:30:58  dgrisby
# Problems with modules inside packages. Thanks Fabian Knittel.
#
# Revision 1.30.2.19  2007/05/11 09:37:23  dgrisby
# Ensure hash value of unpickled enum items is the same as that of the
# original item.
#
# Revision 1.30.2.18  2006/09/07 15:29:57  dgrisby
# Use boxes.idl to build standard value boxes.
#
# Revision 1.30.2.17  2006/07/11 13:53:09  dgrisby
# Implement missing TypeCode creation functions.
#
# Revision 1.30.2.16  2006/02/22 13:05:15  dgrisby
# __repr__ and _narrow methods for valuetypes.
#
# Revision 1.30.2.15  2006/01/19 17:28:44  dgrisby
# Merge from omnipy2_develop.
#
# Revision 1.30.2.14  2006/01/17 17:38:20  dgrisby
# Expose omniORB.setClientConnectTimeout function.
#
# Revision 1.30.2.13  2005/11/09 12:33:31  dgrisby
# Support POA LocalObjects.
#
# Revision 1.30.2.12  2005/09/01 15:14:41  dgrisby
# Merge from omnipy3_develop.
#
# Revision 1.30.2.11  2005/07/29 11:21:35  dgrisby
# Fix long-standing problem with module re-opening by #included files.
#
# Revision 1.30.2.10  2005/06/24 17:36:00  dgrisby
# Support for receiving valuetypes inside Anys; relax requirement for
# old style classes in a lot of places.
#
# Revision 1.30.2.9  2005/04/25 18:28:29  dgrisby
# Minor code for TRANSIENT_FailedOnForwarded.
#
# Revision 1.30.2.8  2005/04/14 13:50:45  dgrisby
# New traceTime, traceInvocationReturns functions; removal of omniORB::logf.
#
# Revision 1.30.2.7  2005/03/02 13:39:16  dgrisby
# Another merge from omnipy2_develop.
#
# Revision 1.30.2.6  2005/01/25 11:45:48  dgrisby
# Merge from omnipy2_develop; set RPM version.
#
# Revision 1.30.2.5  2005/01/07 00:22:35  dgrisby
# Big merge from omnipy2_develop.
#
# Revision 1.30.2.4  2003/09/04 14:08:41  dgrisby
# Correct register_value_factory semantics.
#
# Revision 1.30.2.3  2003/07/10 22:13:25  dgrisby
# Abstract interface support.
#
# Revision 1.30.2.2  2003/05/20 17:10:25  dgrisby
# Preliminary valuetype support.
#
# Revision 1.30.2.1  2003/03/23 21:51:43  dgrisby
# New omnipy3_develop branch.
#
# Revision 1.26.2.15  2003/03/12 11:17:49  dgrisby
# Any / TypeCode fixes.
#
# Revision 1.26.2.14  2002/11/27 00:18:25  dgrisby
# Per thread / per objref timeouts.
#
# Revision 1.26.2.13  2002/09/21 23:27:11  dgrisby
# New omniORB.any helper module.
#
# Revision 1.26.2.12  2002/08/16 19:27:36  dgrisby
# Documentation update. Minor ORB updates to match docs.
#
# Revision 1.26.2.11  2002/05/27 01:02:37  dgrisby
# Fix bug with scope lookup in generated code. Fix TypeCode clean-up bug.
#
# Revision 1.26.2.10  2002/03/11 15:40:05  dpg1
# _get_interface support, exception minor codes.
#
# Revision 1.26.2.9  2002/02/25 15:34:26  dpg1
# Get list of keywords from keyword module.
#
# Revision 1.26.2.8  2002/01/18 15:49:45  dpg1
# Context support. New system exception construction. Fix None call problem.
#
# Revision 1.26.2.7  2001/09/20 14:51:26  dpg1
# Allow ORB reinitialisation after destroy(). Clean up use of omni namespace.
#
# Revision 1.26.2.6  2001/08/01 10:12:37  dpg1
# Main thread policy.
#
# Revision 1.26.2.5  2001/06/15 10:59:27  dpg1
# Apply fixes from omnipy1_develop.
#
# Revision 1.26.2.4  2001/05/14 15:22:00  dpg1
# cdrMarshal() / cdrUnmarshal() are back.
#
# Revision 1.26.2.3  2001/04/10 16:35:33  dpg1
# Minor bugs in Any coercion.
#
# Revision 1.26.2.2  2001/04/09 15:22:17  dpg1
# Fixed point support.
#
# Revision 1.26.2.1  2000/10/13 13:55:31  dpg1
# Initial support for omniORB 4.
#
# Revision 1.26  2000/10/02 17:34:58  dpg1
# Merge for 1.2 release
#
# Revision 1.24.2.2  2000/08/23 09:22:07  dpg1
# Fix loading of IfR stubs with "import CORBA"
#
# Revision 1.24.2.1  2000/08/17 08:46:06  dpg1
# Support for omniORB.LOCATION_FORWARD exception
#
# Revision 1.24  2000/07/12 14:33:10  dpg1
# Support for Interface Repository stubs
#
# Revision 1.23  2000/06/28 10:49:07  dpg1
# Incorrect comment removed.
#
# Revision 1.22  2000/06/27 15:09:41  dpg1
# Expanded comment.
#
# Revision 1.21  2000/06/12 15:36:09  dpg1
# Support for exception handler functions. Under omniORB 3, local
# operation dispatch modified so exceptions handlers are run.
#
# Revision 1.20  2000/06/02 14:25:51  dpg1
# orb.run() now properly exits when the ORB is shut down
#
# Revision 1.19  2000/06/01 11:10:30  dme
# add omniORB.WorkerThread create/delete hooks (e.g. for profiling)
#
# Revision 1.18  2000/04/06 09:31:43  dpg1
# newModule() spots if we're trying to re-open the CORBA module, and if
# so uses omniORB.CORBA.
#
# Revision 1.17  2000/03/03 17:41:27  dpg1
# Major reorganisation to support omniORB 3.0 as well as 2.8.
#
# Revision 1.16  2000/01/31 10:51:41  dpg1
# Fix to exception throwing.
#
# Revision 1.15  2000/01/04 16:14:27  dpg1
# Clear out byte-compiled files created by importIDL()
#
# Revision 1.14  2000/01/04 15:29:40  dpg1
# Fixes to modules generated within a package.
#
# Revision 1.13  1999/11/12 17:15:50  dpg1
# Can now specify arguments for omniidl.
#
# Revision 1.12  1999/11/12 16:49:18  dpg1
# Stupid bug introduced with last change.
#
# Revision 1.11  1999/11/12 15:53:48  dpg1
# New functions omniORB.importIDL() and omniORB.importIDLString().
#
# Revision 1.10  1999/10/18 08:25:57  dpg1
# _is_a() now works properly for local objects.
#
# Revision 1.9  1999/09/29 15:46:50  dpg1
# lockWithNewThreadState now creates a dummy threading.Thread object so
# threading doesn't get upset that it's not there. Very dependent on the
# implementation of threading.py.
#
# Revision 1.8  1999/09/27 09:06:37  dpg1
# Friendly error message if there is no thread support.
#
# Revision 1.7  1999/09/24 09:22:01  dpg1
# Added copyright notices.
#
# Revision 1.6  1999/09/23 16:28:16  dpg1
# __doc__ strings now created for existing modules without them.
#
# Revision 1.5  1999/09/22 15:46:11  dpg1
# Fake POA implemented.
#
# Revision 1.4  1999/09/13 15:13:09  dpg1
# Module handling.
# Any coercion (*** not fully tested).
#
# Revision 1.3  1999/08/03 09:03:46  dpg1
# Unions with no default member fixed.
#
# Revision 1.2  1999/07/29 14:16:03  dpg1
# Server side support.
#
# Revision 1.1  1999/07/19 15:53:26  dpg1
# Initial revision
#

"""
omniORB module -- omniORB specific features

"""

import sys, types, string, imp, os, os.path, tempfile, exceptions

try:
    import threading
except ImportError:
    print """
Error: your Python executable was not built with thread support.
       omniORBpy requires threads. Sorry.
"""
    raise ImportError("Python executable has no thread support")

import _omnipy

_coreVersion = _omnipy.coreVersion()


# Make sure _omnipy submodules are in sys.modules, and have not been
# damaged. This can happen if someone has messed with sys.modules, or
# the interpreter has been stopped and restarted.
reinit = 0
for k, v in _omnipy.__dict__.items():
    if k[-5:] == "_func" and isinstance(v, types.ModuleType):
        sub = "_omnipy." + k
        if not sys.modules.has_key(sub):
            reinit = 1
            sys.modules[sub] = v
        del sub
del k, v

if reinit:
    _omnipy.ensureInit()
del reinit


# Add path to COS stubs if need be
_cospath = os.path.join(os.path.dirname(__file__), "COS")
if _cospath not in sys.path:
    sys.path.append(_cospath)
del _cospath


# Public functions

def coreVersion():
    """coreVersion()

Return a string containing the version number of the omniORB core, of
the form major.minor.micro. Versions from 3.0.0 up support the full
POA functionality."""
    return _coreVersion


_omniidl_args = []

def omniidlArguments(args):
    """omniidlArguments(list)

Set default omniidl arguments for importIDL() and importIDLString().
e.g. omniidlArguments(["-I/my/include", "-DMY_DEFINE"])"""

    global _omniidl_args

    if type(args) is not types.ListType:
        raise TypeError("argument must be a list of strings")

    for arg in args:
        if type(arg) is not types.StringType:
            raise TypeError("argument must be a list of strings")

    _omniidl_args = args


# Import an IDL file by forking the IDL compiler and processing the
# output
def importIDL(idlname, args=None, inline=1):
    """importIDL(filename [, args ] [, inline ]) -> tuple

Run the IDL compiler on the specified IDL file, and import the
resulting stubs. If args is present, it must contain a list of strings
used as arguments to omniidl. If args is not present, uses the default
set with omniidlArguments().

Normally imports the definitions for #included files as well as the
main file. Set inline to 0 to only import definitions for the main
file.

Returns a tuple of Python module names corresponding to the IDL module
names declared in the file. The modules can be accessed through
sys.modules."""

    if not os.path.isfile(idlname):
        raise ImportError("File " + idlname + " does not exist")

    if args is None: args = _omniidl_args
    if inline:
        inline_str = "-Wbinline "
    else:
        inline_str = ""

    argstr  = string.join(args, " ")
    modname = string.replace(os.path.basename(idlname), ".", "_")
    pipe    = os.popen("omniidl -q -bpython -Wbstdout " + inline_str + \
                       argstr + " " + idlname)
    try:
        tempname  = tempfile.mktemp()
        tempnamec = tempname + "c"
        while os.path.exists(tempnamec):
            tempname  = tempfile.mktemp()
            tempnamec = tempname + "c"

        m = imp.load_module(modname, pipe, tempname,
                            (".idl", "r", imp.PY_SOURCE))
    finally:
        # Get rid of byte-compiled file
        if os.path.isfile(tempnamec):
            os.remove(tempnamec)

        # Close the pipe
        if pipe.close() is not None:
            del sys.modules[modname]
            raise ImportError("Error spawning omniidl")
    try:
        m.__file__ = idlname
        mods = m._exported_modules

        for mod in mods:
            for m in (mod, skeletonModuleName(mod)):
                if _partialModules.has_key(m):
                    if sys.modules.has_key(m):
                        sys.modules[m].__dict__.update(
                            _partialModules[m].__dict__)
                    else:
                        sys.modules[m] = _partialModules[m]
                    del _partialModules[m]

        return mods

    except (AttributeError, KeyError):
        del sys.modules[modname]
        raise ImportError("Invalid output from omniidl")

def importIDLString(str, args=None, inline=1):
    """importIDLString(string [, args ] [, inline ]) -> tuple

Run the IDL compiler on the given string, and import the resulting
stubs. If args is present, it must contain a list of strings used as
arguments to omniidl. If args is not present, uses the default set
with omniidlArguments().

Normally imports the definitions for #included files as well as the
main file. Set inline to 0 to only import definitions for the main
file.

Returns a tuple of Python module names corresponding to the IDL module
names declared in the file. The modules can be accessed through
sys.modules."""

    tfn = tempfile.mktemp()
    tf  = open(tfn, "w")
    tf.write(str)
    tf.close()
    try:
        ret = importIDL(tfn, args, inline)
    finally:
        os.remove(tfn)
    return ret


def cdrMarshal(tc, data, endian=-1):
    """cdrMarshal(TypeCode, data [,endian]) -> binary string

Marshal data with the given type into a CDR encapsulation. The data
can later be converted back into Python objects with cdrUnmarshal().
The encapsulation is language, platform, and ORB independent.

If the endian boolean is provided, it represents the endianness to
marshal with: True for little endian; false for big endian. The
resulting string in this case is the raw marshalled form, not a CDR
encapsulation. To unmarshal it, the endianness must be known.

Throws CORBA.BAD_PARAM if the data does not match the TypeCode."""

    if not isinstance(tc, CORBA.TypeCode):
        raise TypeError("Argument 1 must be a TypeCode")

    return _omnipy.cdrMarshal(tc._d, data, endian)

def cdrUnmarshal(tc, encap, endian=-1):
    """cdrUnmarshal(TypeCode, string [,endian]) -> data

Unmarshal a CDR stream created with cdrMarshal() or equivalent. The
encapsulation must adhere to the given TypeCode.

If the endian boolean is provided, it represents the endianness to
unmarshal with: True for little endian; false for big endian. In this
case, the string should be the raw marshalled form, not a CDR
encapsulation. If the endianness does not match that used for
marshalling, invalid data may be returned, or exceptions raised.

Throws CORBA.MARSHAL if the binary string does not match the
TypeCode."""

    if not isinstance(tc, CORBA.TypeCode):
        raise TypeError("Argument 1 must be a TypeCode")

    return _omnipy.cdrUnmarshal(tc._d, encap, endian)


WTHREAD_CREATED = 0
WTHREAD_DELETED = 1

def addWThreadHook(hook):
    """addWThreadHook(hook) -> None

Arrange to call "hook(WTHREAD_{CREATED,DELETED}, wt)" on the new thread
whenever the runtime creates or deletes a Python "omniORB.WorkerThread"
"wt" (for instance as a result of a new incoming connection).  There is
no concurrency control: "addWThreadHook()" must be called before the
runtime creates any "WorkerThread"s.
"""
    WorkerThread.hooks.append(hook)


def importIRStubs():
    """importIRStubs() -> None

Make stubs for the Interface Repository appear in the CORBA module"""
    import omniORB.ir_idl
    CORBA._d_Object_interface = ((),(CORBA._d_InterfaceDef,),None)



# Import omniORB API functions. This provides:
#
#   installTransientExceptionHandler()
#   installCommFailureExceptionHandler()
#   installSystemExceptionHandler()
#   traceLevel
#   traceInvocations
#   traceInvocationReturns
#   traceThreadId
#   traceTime
#   log
#   nativeCharCodeSet
#   fixed
#   minorCodeToString
#   setClientCallTimeout
#   setClientThreadCallTimeout
#   setClientConnectTimeout
#   myIPAddresses
#   setPersistentServerIdentifier
#   locationForward

from _omnipy.omni_func import *

# More public things at the end


# Private things

# ORB:
orb      = None
rootPOA  = None
poaCache = {}
lock     = threading.Lock()

# Maps for object reference classes and IDL-defined types:
objrefMapping       = {}
skeletonMapping     = {}
typeMapping         = {}
typeCodeMapping     = {}
valueFactoryMapping = {}


def registerObjref(repoId, objref):
    objrefMapping[repoId] = objref

def registerSkeleton(repoId, skel):
    skeletonMapping[repoId] = skel

def registerType(repoId, desc, tc):
    typeMapping[repoId]     = desc
    typeCodeMapping[repoId] = tc

def findType(repoId):
    return typeMapping.get(repoId)

def findTypeCode(repoId):
    return typeCodeMapping.get(repoId)

def registerValueFactory(repoId, factory):
    old = valueFactoryMapping.get(repoId)
    valueFactoryMapping[repoId] = factory
    return old

def unregisterValueFactory(repoId):
    del valueFactoryMapping[repoId]

def findValueFactory(repoId):
    return valueFactoryMapping.get(repoId)


# Map of partially-opened modules
_partialModules = {}


# Function to return a Python module for the required IDL module name
def openModule(mname, fname=None):
    if mname == "CORBA":
        mod = sys.modules["omniORB.CORBA"]

    elif sys.modules.has_key(mname):
        mod = sys.modules[mname]

        if _partialModules.has_key(mname):
            pmod = _partialModules[mname]
            mod.__dict__.update(pmod.__dict__)
            del _partialModules[mname]
            
    elif _partialModules.has_key(mname):
        mod = _partialModules[mname]

    else:
        mod = newModule(mname)

    if not hasattr(mod, "__doc__") or mod.__doc__ is None:
        mod.__doc__ = "omniORB IDL module " + mname + "\n\n" + \
                      "Generated from:\n\n"

    if fname is not None:
        mod.__doc__ = mod.__doc__ + "  " + fname + "\n"

    return mod

# Function to create a new module, and any parent modules which do not
# already exist
def newModule(mname):
    mlist   = string.split(mname, ".")
    current = ""
    mod     = None

    for name in mlist:
        current = current + name

        if sys.modules.has_key(current):
            mod = sys.modules[current]

        elif _partialModules.has_key(current):
            mod = _partialModules[current]

        else:
            newmod = imp.new_module(current)
            _partialModules[current] = mod = newmod

        current = current + "."

    return mod

# Function to update a module with the partial module store in the
# partial module map
def updateModule(mname):
    if _partialModules.has_key(mname):
        pmod = _partialModules[mname]
        mod  = sys.modules[mname]
        mod.__dict__.update(pmod.__dict__)
        del _partialModules[mname]


def skeletonModuleName(mname):
    l = string.split(mname, ".")
    l[0] = l[0] + "__POA"
    return string.join(l, ".")



# Function to create a new empty class as a scope place-holder
def newEmptyClass():
    class __dummy: pass
    return __dummy

 
# Classes to support IDL type mapping

class EnumItem:
    def __init__(self, name, value):
        self._n = name
        self._v = value
        return

    def __str__(self):
        return self._n

    def __repr__(self):
        return self._n

    def __cmp__(self, other):
        try:
            if isinstance(other, EnumItem):
                if other._parent_id == self._parent_id:
                    return cmp(self._v, other._v)
                else:
                    return cmp(self._parent_id, other._parent_id)
            else:
                return cmp(id(self), id(other))
        except:
            return cmp(id(self), id(other))

    def __hash__(self):
        return hash(self._parent_id + "/" + self._n)

class AnonymousEnumItem (EnumItem):
    def __init__(self, value):
        self._n = ""
        self._v = value
    
    def __repr__(self):
        return "anonymous enum item"


class Enum:
    def __init__(self, repoId, items):
        self._NP_RepositoryId = repoId
        self._items = items
        for i in items:
            i._parent_id = repoId

    def _item(self, n):
        return self._items[n]


class StructBase:
    _NP_RepositoryId = None
    _NP_ClassName = None
    
    def __repr__(self):
        cname = self._NP_ClassName
        if cname is None:
            cname = "%s.%s" % (self.__module__, self.__class__.__name__)

        desc = findType(self._NP_RepositoryId)
        if desc is None:
            # Type is not properly registered
            return "<%s instance at 0x%x>" % (cname, id(self))
        vals = []
        for i in range(4, len(desc), 2):
            attr = desc[i]
            try:
                val = getattr(self, attr)
                vals.append("%s=%s" % (attr,repr(val)))
            except AttributeError:
                vals.append("%s=<not set>" % attr)

        return "%s(%s)" % (cname, string.join(vals, ", "))

    def _tuple(self):
        desc = findType(self._NP_RepositoryId)
        if desc is None:
            # Type is not properly registered
            raise CORBA.BAD_PARAM(BAD_PARAM_IncompletePythonType,
                                  CORBA.COMPLETED_NO)
        vals = []
        for i in range(4, len(desc), 2):
            attr = desc[i]
            vals.append(getattr(self, attr))
        return tuple(vals)


class Union:
    _NP_ClassName = None
    _def_m = None

    def __init__(self, *args, **kw):
        if len(args) == 2:
            self._d = args[0]
            self._v = args[1]
        else:
            ks = kw.keys()
            if len(args) != 0 or len(ks) != 1:
                raise TypeError("require 2 arguments or one keyword argument.")
            k = ks[0]
            self.__setattr__(k, kw[k])

    def __getattr__(self, mem):
        try:
            cmem = self._d_to_m[self._d]
            if mem == cmem:
                return self._v
            else:
                if mem == self._def_m or self._m_to_d.has_key(mem):
                    raise CORBA.BAD_PARAM(BAD_PARAM_WrongUnionMemberSelected,
                                          CORBA.COMPLETED_NO)
                else:
                    raise AttributeError(mem)
        except KeyError:
            if mem == self._def_m:
                return self._v
            else:
                if self._m_to_d.has_key(mem):
                    raise CORBA.BAD_PARAM(BAD_PARAM_WrongUnionMemberSelected,
                                          CORBA.COMPLETED_NO)
                else:
                    raise AttributeError(mem)

    def __setattr__(self, mem, val):
        if mem[0] == "_":
            self.__dict__[mem] = val
        else:
            try:
                disc = self._m_to_d[mem]
                self.__dict__["_d"] = disc
                self.__dict__["_v"] = val
            except KeyError:
                if mem == self._def_m:
                    self.__dict__["_d"] = self._def_d
                    self.__dict__["_v"] = val
                else:
                    raise AttributeError(mem)

    def __repr__(self):
        cname = self._NP_ClassName
        if cname is None:
            cname = "%s.%s" % (self.__module__, self.__class__.__name__)

        try:
            return "%s(%s = %s)" % (cname, self._d_to_m[self._d],
                                    repr(self._v))
        except KeyError:
            return "%s(%s, %s)" % (cname, repr(self._d), repr(self._v))


# Import sub-modules
import CORBA, tcInternal

def createUnknownStruct(repoId, members):

    class UnknownStruct (StructBase):
        def __init__(self, *args):
            if len(args) != len(self._members):
                raise TypeError("__init__() takes exactly %d arguments "
                                "(%d given)" %
                                (len(self._members) + 1, len(args) + 1))

            self._values = args

            for i in range(len(args)):
                if self._members[i] != "":
                    setattr(self, self._members[i], args[i])

        def __repr__(self):
            vals = []
            for i in range(len(self._values)):
                attr = self._members[i]
                val  = self._values[i]
                if attr:
                    vals.append("%s=%s" % (attr, repr(val)))
                else:
                    vals.append(repr(val))

            return "UnknownStruct<%s>(%s)" % (self._NP_RepositoryId,
                                              string.join(vals, ", "))
        def _tuple(self):
            return tuple(self._values)

    UnknownStruct._NP_RepositoryId = repoId
    UnknownStruct._members         = members
    return UnknownStruct

def createUnknownUnion(repoId, def_used, members):

    class UnknownUnion (Union):
        pass

    UnknownUnion._NP_RepositoryId = repoId
    UnknownUnion._NP_ClassName    = "UnknownUnion<%s>" % repoId
    UnknownUnion._d_to_m          = {}
    UnknownUnion._m_to_d          = {}

    for i in range(len(members)):
        if i == def_used:
            UnknownUnion._def_d = members[i][0]
            UnknownUnion._def_m = members[i][1]
        else:
            UnknownUnion._d_to_m[members[i][0]] = members[i][1]
            UnknownUnion._m_to_d[members[i][1]] = members[i][0]

    return UnknownUnion

def createUnknownUserException(repoId, members):

    class UnknownUserException (CORBA.UserException):
        def __init__(self, *args):
            if len(args) != len(self._members):
                raise TypeError("__init__() takes exactly %d arguments "
                                "(%d given)" %
                                (len(self._members) + 1, len(args) + 1))

            self._values = args

            for i in range(len(args)):
                if self._members[i] != "":
                    setattr(self, self._members[i], args[i])

        def __repr__(self):
            vals = []
            for i in range(len(self._values)):
                attr = self._members[i]
                val  = self._values[i]
                if attr:
                    vals.append("%s=%s" % (attr, repr(val)))
                else:
                    vals.append(repr(val))

            return "UnknownUserException<%s>(%s)" % (self._NP_RepositoryId,
                                                     string.join(vals, ", "))

    UnknownUserException._NP_RepositoryId = repoId
    UnknownUserException._members         = members
    return UnknownUserException


class UnknownValueBase (CORBA.ValueBase):
    pass


def createUnknownValue(repoId, base_desc):

    if base_desc == tcInternal.tv_null:
        class UnknownValue (UnknownValueBase):
            pass
    else:
        base_cls = base_desc[1]
        if isinstance(base_cls, UnknownValueBase):
            class UnknownValue (base_cls):
                pass
        else:
            class UnknownValue (UnknownValueBase, base_cls):
                pass

    UnknownValue._NP_RepositoryId = repoId
    return UnknownValue


# Function to coerce an Any value with a partially-specified
# descriptor to a value with an equivalent, fully-specified
# descriptor.

def coerceAny(v, fd, td):
    if fd == td:
        return v

    if not tcInternal.equivalentDescriptors(fd, td):
        return None

    if type(fd) is not types.TupleType or \
       type(td) is not types.TupleType:
        return None

    while fd[0] == tcInternal.tv_alias:
        fd = fd[3]

    while td[0] == tcInternal.tv_alias:
        td = td[3]

    try:
        if fd == td:
            return v

        elif fd[0] == tcInternal.tv_objref:
            return _omnipy.narrow(v, td[1])

        elif fd[0] == tcInternal.tv_struct:
            l = list(v._values)

            # Coerce each member
            for i in range(len(l)):
                l[i] = coerceAny(l[i], fd[i*2 + 5], td[i*2 + 5])
            
            return apply(td[1], l)

        elif fd[0] == tcInternal.tv_union:
            return td[1](v._d, coerceAny(v._v, fd[6][v._d], td[6][v._d]))

        elif fd[0] == tcInternal.tv_enum:
            return td[3][v._v]

        elif fd[0] == tcInternal.tv_sequence:
            l = v[:]
            for i in range(len(l)):
                l[i] = coerceAny(v[i], fd[1], td[1])
            return l

        elif fd[0] == tcInternal.tv_array:
            l = v[:]
            for i in range(len(l)):
                l[i] = coerceAny(v[i], fd[1], td[1])
            return l

        elif fd[0] == tcInternal.tv_except:
            l = list(v._values)

            # Coerce each member
            for i in range(len(l)):
                l[i] = coerceAny(l[i], fd[i*2 + 5], td[i*2 + 5])
            
            return apply(td[1], l)

        elif fd[0] == tcInternal.tv__indirect:
            return coerceAny(v, fd[1][0], td[1][0])

    except:
        return None

    return None


# Support for _is_a()
def static_is_a(cls, repoId):
    if cls._NP_RepositoryId == repoId: return 1
    for b in cls.__bases__:
        if static_is_a(b, repoId): return 1
    return 0


# Fixed point type

class fixedConstructor:
    def __init__(self, repoId, digits, scale):
        self._NP_RepositoryId = repoId
        self.digits           = digits
        self.scale            = scale

    def __call__(self, arg):
        try:
            return fixed(self.digits, self.scale, arg)
        except TypeError:
            raise TypeError("Invalid type for fixed argument")

    def __repr__(self):
        return "omniORB fixed<%d,%d> constructor" % (self.digits, self.scale)


# WorkerThread class used to make the threading module happy during
# operation dispatch.
# *** Depends on threading module internals ***

_thr_init = threading.Thread.__init__
_thr_id   = threading._get_ident
_thr_act  = threading._active
_thr_acq  = threading._active_limbo_lock.acquire
_thr_rel  = threading._active_limbo_lock.release

class WorkerThread(threading.Thread):

    hooks = []

    def __init__(self):
        id = _thr_id()
        _thr_init(self, name="omniORB-%d" % id)
        if hasattr(self._Thread__started, 'set'):
            self._Thread__started.set()
        else:
            self._Thread__started = 1
        self.id = id
        _thr_acq()
        if _thr_act.has_key(id):
            self.add = 0
        else:
            self.add = 1
            _thr_act[id] = self
        _thr_rel()
        if self.add:
            for hook in self.hooks:
                hook(WTHREAD_CREATED, self)

    def delete(self):
        if self.add:
            for hook in self.hooks:
                hook(WTHREAD_DELETED, self)
            _thr_acq()
            try:
                del _thr_act[self.id]
            finally:
                _thr_rel()

    def _set_daemon(self): return 1
    def join(self):        assert 0, "cannot join an omniORB WorkerThread"
    

# omniThreadHook is used to release a dummy omni_thread C++ object
# associated with a threading.Thread object when the thread stops.

class omniThreadHook:
    def __init__(self, target):
        self.target            = target
        self.target_stop       = target._Thread__stop
        target._Thread__stop   = self.omni_thread_stop

    def omni_thread_stop(self):
        try:
            delattr(self.target, "__omni_thread")
            del self.target._Thread__stop
        except AttributeError:
            pass
        self.target_stop()


# System exception mapping.

sysExceptionMapping = {}

cd = CORBA.__dict__
for exc in _omnipy.system_exceptions:
    cls = cd[exc]
    sysExceptionMapping[cls._NP_RepositoryId] = cls

del cd, exc, cls

# Reserved word mapping:

keywordMapping = {}
try:
    import keyword
    for word in keyword.kwlist:
        keywordMapping[word] = "_" + word

    keywordMapping["None"] = "_None"

    del keyword
except ImportError:
    pass

# Exception minor codes. See include/omniORB4/minorCode.h

def omniORBminorCode(c):
    return 0x41540000 | c

def OMGminorCode(c):
    return 0x4f4d0000 | c

from omniORB.minorCodes import *


# More public things, which depend on the CORBA module

# LOCATION_FORWARD exception
class LOCATION_FORWARD (exceptions.Exception):
    """LOCATION_FORWARD(objref, permanent=0)

This exception may be thrown inside any operation implementation. It
causes the ORB the return a LOCATION_FORWARD message to the caller, so
the invocation is retried on the given object reference. If permanent
is set to 1, a permanent location forward is requested."""

    _NP_RepositoryId = "omniORB.LOCATION_FORWARD" # Not really a CORBA type

    def __init__(self, objref, perm=0):
        if not isinstance(objref, CORBA.Object):
            raise CORBA.BAD_PARAM(BAD_PARAM_WrongPythonType,
                                  CORBA.COMPLETED_NO)

        self._forward = objref
        self._perm    = perm

    def __str__(self):
        return "omniORB.LOCATION_FORWARD exception"

# "Static" objects required by the _omnipy module. They are here so
# memory management works correctly if the omniORB modules are
# unloaded.

_emptyTuple      = ()
_ORB_TWIN        = "__omni_orb"
_OBJREF_TWIN     = "__omni_obj"
_SERVANT_TWIN    = "__omni_svt"
_POA_TWIN        = "__omni_poa"
_POAMANAGER_TWIN = "__omni_mgr"
_POACURRENT_TWIN = "__omni_pct"
_NP_RepositoryId = "_NP_RepositoryId"


# Register this module and the threading module with omnipy:
import omniORB, omniORB.PortableServer
_omnipy.registerPyObjects(omniORB)

# Import CORBA module stubs
import corbaidl_idl
import boxes_idl

sys.modules["corbaidl_idl"] = corbaidl_idl
sys.modules["boxes_idl"]    = boxes_idl

# Import the Interface Repository stubs if necessary
if os.environ.has_key("OMNIORBPY_IMPORT_IR_STUBS"):
    importIRStubs()

del omniORB
