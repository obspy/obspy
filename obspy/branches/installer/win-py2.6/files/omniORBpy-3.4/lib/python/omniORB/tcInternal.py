# -*- Mode: Python; -*-
#                            Package   : omniORBpy
# tcInternal.py              Created on: 1999/06/24
#                            Author    : Duncan Grisby (dpg1)
#
#    Copyright (C) 2002-2007 Apasphere Ltd
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
#    TypeCode internal implementation


# $Id: tcInternal.py,v 1.13.2.12 2009/05/06 16:50:22 dgrisby Exp $
# $Log: tcInternal.py,v $
# Revision 1.13.2.12  2009/05/06 16:50:22  dgrisby
# Updated copyright.
#
# Revision 1.13.2.11  2007/06/06 17:59:01  dgrisby
# Allow enums to be used directly in CORBA.TypeCode().
#
# Revision 1.13.2.10  2007/04/26 08:33:38  dgrisby
# Incorrect kind() return for object reference TypeCodes. Thanks
# Andrew Edem.
#
# Revision 1.13.2.9  2006/07/11 13:53:09  dgrisby
# Implement missing TypeCode creation functions.
#
# Revision 1.13.2.8  2006/02/16 22:55:45  dgrisby
# Remove some tab characters that snuck in from a patch.
#
# Revision 1.13.2.7  2006/01/19 17:28:44  dgrisby
# Merge from omnipy2_develop.
#
# Revision 1.13.2.6  2005/11/09 12:33:31  dgrisby
# Support POA LocalObjects.
#
# Revision 1.13.2.5  2005/07/29 11:27:23  dgrisby
# Static map of basic TypeCodes for speed.
#
# Revision 1.13.2.4  2005/01/07 00:22:35  dgrisby
# Big merge from omnipy2_develop.
#
# Revision 1.13.2.3  2003/07/10 22:13:25  dgrisby
# Abstract interface support.
#
# Revision 1.13.2.2  2003/05/20 17:10:26  dgrisby
# Preliminary valuetype support.
#
# Revision 1.13.2.1  2003/03/23 21:51:43  dgrisby
# New omnipy3_develop branch.
#
# Revision 1.10.2.8  2003/03/12 11:17:50  dgrisby
# Any / TypeCode fixes.
#
# Revision 1.10.2.7  2003/03/07 11:56:04  dgrisby
# Missing TypeCode creation functions.
#
# Revision 1.10.2.6  2002/06/11 20:21:31  dgrisby
# Missed out wchar, wstring TypeCodes.
#
# Revision 1.10.2.5  2002/05/27 01:02:37  dgrisby
# Fix bug with scope lookup in generated code. Fix TypeCode clean-up bug.
#
# Revision 1.10.2.4  2002/03/11 15:40:05  dpg1
# _get_interface support, exception minor codes.
#
# Revision 1.10.2.3  2001/05/14 14:49:39  dpg1
# Fix get_compact_typecode() and equivalent()
#
# Revision 1.10.2.2  2001/04/10 11:11:15  dpg1
# TypeCode support and tests for Fixed point.
#
# Revision 1.10.2.1  2000/11/01 15:29:01  dpg1
# Support for forward-declared structs and unions
# RepoIds in indirections are now resolved at the time of use
#
# Revision 1.10  2000/08/21 10:20:19  dpg1
# Merge from omnipy1_develop for 1.1 release
#
# Revision 1.9.2.1  2000/08/07 09:19:24  dpg1
# Long long support
#
# Revision 1.9  2000/01/31 10:51:41  dpg1
# Fix to exception throwing.
#
# Revision 1.8  1999/11/02 10:38:31  dpg1
# Last bug wasn't quite fixed.
#
# Revision 1.7  1999/11/01 20:59:42  dpg1
# Fixed bug in insertIndirections() if the same node is indirected to
# more than once.
#
# Revision 1.6  1999/09/24 09:22:00  dpg1
# Added copyright notices.
#
# Revision 1.5  1999/09/20 15:11:45  dpg1
# Bug in insertIndirections() fixed.
#
# Revision 1.4  1999/09/13 14:52:27  dpg1
# TypeCode equivalence.
#
# Revision 1.3  1999/07/29 14:17:18  dpg1
# TypeCode creation interface.
#
# Revision 1.2  1999/07/19 15:48:40  dpg1
# All sorts of fixes.
#
# Revision 1.1  1999/06/24 15:23:28  dpg1
# Initial revision
#

import omniORB, CORBA
import types

# The TypeCode implementation is based upon descriptors consisting of
# Python tuples, as used by the marshalling code. Although the public
# interface to TypeCodes presents a graph of connected TypeCode
# objects, the graph is actually only fully-stored by the descriptor
# tuple. A root TypeCode object has a reference to the top of the
# descriptor tuple. Non-root TypeCode objects are only created when
# requested.
#
# Recursive descriptors pose a problem for Python's garbage collector.
# To ensure they are properly collected, all non-root TypeCode objects
# keep a reference to the root TypeCode object. The root TypeCode is
# therefore only deleted once all "views" of the TypeCode have gone.
# When the root TypeCode object is deleted, it recursively descends
# its descriptor, and removes any indirections it finds. All loops are
# thus removed from the descriptor, making it a candidate for
# collection.
#
# This approach means that the descriptor becomes invalid once the
# root TypeCode object is deleted, even if there are other active
# references to the descriptor. BEWARE!

# Kinds as numbers:
tv_null               = 0
tv_void               = 1
tv_short              = 2
tv_long               = 3
tv_ushort             = 4
tv_ulong              = 5
tv_float              = 6
tv_double             = 7
tv_boolean            = 8
tv_char               = 9
tv_octet              = 10
tv_any                = 11
tv_TypeCode           = 12
tv_Principal          = 13
tv_objref             = 14
tv_struct             = 15
tv_union              = 16
tv_enum               = 17
tv_string             = 18
tv_sequence           = 19
tv_array              = 20
tv_alias              = 21
tv_except             = 22
tv_longlong           = 23
tv_ulonglong          = 24
tv_longdouble         = 25
tv_wchar              = 26
tv_wstring            = 27
tv_fixed              = 28
tv_value              = 29
tv_value_box          = 30
tv_native             = 31
tv_abstract_interface = 32
tv_local_interface    = 33
tv__indirect          = -1


# Create a TypeCode given a class or a repoId

def typeCodeFromClassOrRepoId(t):
    try:
        t = t._NP_RepositoryId
    except AttributeError:
        pass

    if type(t) is not types.StringType:
        raise TypeError("Argument must be CORBA class or repository id.")

    d = omniORB.findType(t)
    if d is None:
        raise TypeError("Unknown CORBA type.")

    return createTypeCode(d)


# Implementations of public ORB TypeCode creation functions

def createStructTC(id, name, members):
    dlist  = [tv_struct, None, id, name]
    mnames = []
    for m in members:
        mnames.append(m.name)
        dlist.append(m.name)
        dlist.append(m.type._d)
    str = omniORB.createUnknownStruct(id, mnames)
    dlist[1] = str
    d = tuple(dlist)
    return createTypeCode(d)

def createUnionTC(id, name, discriminator_type, members):
    mlist   = []
    count   = 0
    defused = -1
    mmap    = {}
    for m in members:
        val = m.label.value()
        if m.label.typecode().kind() == CORBA.tk_octet and val == 0:
            val     = -1
            defused = count

        tup = (val, m.name, m.type._d)
        if defused != count: mmap[val] = tup
        mlist.append(tup)
        count = count + 1

    union = omniORB.createUnknownUnion(id, defused, mlist)

    if defused >= 0:
        default = mlist[defused]
    else:
        default = None

    d = (tv_union, union, id, name, discriminator_type._k._v,
         defused, tuple(mlist), default, mmap)
    return createTypeCode(d)

def createEnumTC(id, name, members):
    mlist = []
    count = 0

    for m in members:
        mlist.append(omniORB.EnumItem(m, count))
        count = count + 1

    d = (tv_enum, id, name, tuple(mlist))
    return createTypeCode(d)

def createAliasTC(id, name, original_type):
    d = (tv_alias, id, name, original_type._d)
    return createTypeCode(d)

def createExceptionTC(id, name, members):
    dlist  = [tv_except, None, id, name]
    mnames = []
    for m in members:
        mnames.append(m.name)
        dlist.append(m.name)
        dlist.append(m.type._d)
    exc = omniORB.createUnknownException(id, mnames)
    dlist[1] = exc
    d = tuple(dlist)
    return createTypeCode(d)

def createInterfaceTC(id, name):
    d = (tv_objref, id, name)
    return createTypeCode(d)

def createStringTC(bound):
    d = (tv_string, bound)
    return createTypeCode(d)

def createWStringTC(bound):
    d = (tv_wstring, bound)
    return createTypeCode(d)

def createFixedTC(digits, scale):
    if digits < 1 or digits > 31 or scale > digits:
        raise CORBA.BAD_PARAM(omniORB.BAD_PARAM_InvalidFixedPointLimits,
                              CORBA.COMPLETED_NO)

    d = (tv_fixed, digits, scale)
    return createTypeCode(d)

def createSequenceTC(bound, element_type):
    d = (tv_sequence, element_type._d, bound)
    return createTypeCode(d)

def createArrayTC(length, element_type):
    d = (tv_array, element_type._d, length)
    return createTypeCode(d)

def createRecursiveTC(id):
    class recursivePlaceHolder: pass
    recursivePlaceHolder._d = (tv__indirect, [id])
    return recursivePlaceHolder()

def createValueTC(id, name, modifier, base, members):
    base_desc = base._d

    if modifier == CORBA.VM_TRUNCATABLE:
        if base_desc == tv_null:
            raise CORBA.BAD_PARAM(omniORB.BAD_PARAM_InvalidTypeCode,
                                  CORBA.COMPLETED_NO)
        base_ids = base_desc[5]
        if base_ids is None:
            base_ids = (id, base_desc[2])
        else:
            base_ids = (id,) + base_ids
    else:
        base_ids = None

    dlist = [tv_value, omniORB.createUnknownValue(id, base_desc),
             id, name, modifier, base_ids, base_desc]
    for m in members:
        dlist.append(m.name)
        dlist.append(m.type._d)
        dlist.append(m.access)

    return createTypeCode(tuple(dlist))

def createValueBoxTC(id, name, boxed_type):
    d = (tv_value_box, omniORB.createUnknownValue(id, tv_null),
         id, name, boxed_type._d)
    return createTypeCode(d)

def createAbstractInterfaceTC(id, name):
    d = (tv_abstract_interface, id, name)
    return createTypeCode(d)

def createLocalInterfaceTC(id, name):
    d = (tv_local_interface, id, name)
    return createTypeCode(d)





# Function to create a TypeCode object given a descriptor. Returns a
# static (stub generated) TypeCode object if possible.

typeCodeMapping = omniORB.typeCodeMapping

def createTypeCode(d, parent=None):
    try:
        r = basicTypeCodes.get(d)
        if r is not None:
            return r
    except TypeError:
        # Happens if d contains a mutable object
        pass

    if type(d) is not types.TupleType:
        raise CORBA.INTERNAL()

    k = d[0]

    if   k == tv_string:  return TypeCode_string(d)
    elif k == tv_wstring: return TypeCode_wstring(d)
    elif k == tv_fixed:   return TypeCode_fixed(d)

    elif k in [ tv_objref, tv_abstract_interface, tv_local_interface ]:
        tc = typeCodeMapping.get(d[1])
        if tc is None:
            tc = TypeCode_objref(d)
        return tc

    elif k == tv_struct:
        tc = typeCodeMapping.get(d[2])
        if tc is None:
            tc = TypeCode_struct(d, parent)
        return tc
    
    elif k == tv_union:
        tc = typeCodeMapping.get(d[2])
        if tc is None:
            tc = TypeCode_union(d, parent)
        return tc
    
    elif k == tv_enum:
        tc = typeCodeMapping.get(d[1])
        if tc is None:
            tc = TypeCode_enum(d)
        return tc

    elif k == tv_sequence:  return TypeCode_sequence(d, parent)
    elif k == tv_array:     return TypeCode_array(d, parent)

    elif k == tv_alias:
        tc = typeCodeMapping.get(d[1])
        if tc is None:
            tc = TypeCode_alias(d, parent)
        return tc
    
    elif k == tv_except:
        tc = typeCodeMapping.get(d[2])
        if tc is None:
            tc = TypeCode_except(d, parent)
        return tc

    elif k == tv_value:
        tc = typeCodeMapping.get(d[2])
        if tc is None:
            tc = TypeCode_value(d, parent)
        return tc

    elif k == tv_value_box:
        tc = typeCodeMapping.get(d[2])
        if tc is None:
            tc = TypeCode_value_box(d, parent)
        return tc

    elif k == tv__indirect:
        if type(d[1][0]) == types.StringType:
            nd = omniORB.findType(d[1][0])
            if nd is None:
                raise CORBA.BAD_TYPECODE(omniORB.BAD_TYPECODE_InvalidIndirection,
                                         CORBA.COMPLETED_NO)
            d[1][0] = nd
        return createTypeCode(d[1][0], parent)

    raise CORBA.INTERNAL()


# TypeCode base interface

class TypeCode_base (CORBA.TypeCode):
    def __init__(self):
        self._d = 0
        self._k = CORBA.tk_null

    def equal(self, tc):
        try:
            if self._d == tc._d: return CORBA.TRUE
            else:                return CORBA.FALSE
        except AttributeError:
            raise CORBA.BAD_PARAM(omniORB.BAD_PARAM_WrongPythonType,
                                  CORBA.COMPLETED_NO)

    def equivalent(self, tc):
        return self.equal(tc)

    def get_compact_typecode(self):
        return self

    def kind(self):
        return self._k

    # Operations which are only available for some kinds:
    def id(self):                       raise CORBA.TypeCode.BadKind()
    def name(self):                     raise CORBA.TypeCode.BadKind()
    def member_count(self):             raise CORBA.TypeCode.BadKind()
    def member_name(self, index):       raise CORBA.TypeCode.BadKind()
    def member_type(self, index):       raise CORBA.TypeCode.BadKind()
    def member_label(self, index):      raise CORBA.TypeCode.BadKind()

    def discriminator_type(self):       raise CORBA.TypeCode.BadKind()
    def default_index(self):            raise CORBA.TypeCode.BadKind()
    def length(self):                   raise CORBA.TypeCode.BadKind()
    def content_type(self):             raise CORBA.TypeCode.BadKind()

    def fixed_digits(self):             raise CORBA.TypeCode.BadKind()
    def fixed_scale(self):              raise CORBA.TypeCode.BadKind()

    def member_visibility(self, index): raise CORBA.TypeCode.BadKind()
    def type_modifier(self):            raise CORBA.TypeCode.BadKind()
    def concrete_base_type(self):       raise CORBA.TypeCode.BadKind()


# Class for short, long, ushort, ulong, float, double, boolean, char,
# octet, any, TypeCode, Principal, longlong, ulonglong, longdouble:

class TypeCode_empty (TypeCode_base):
    def __init__(self, desc):
        if type(desc) is not types.IntType: raise CORBA.INTERNAL()
        if desc not in [ tv_null, tv_void, tv_short, tv_long, tv_ushort,
                         tv_ulong, tv_float, tv_double, tv_boolean, tv_char,
                         tv_octet, tv_any, tv_TypeCode, tv_Principal,
                         tv_longlong, tv_ulonglong, tv_longdouble, tv_wchar ]:
            raise CORBA.INTERNAL()

        self._d = desc
        self._k = CORBA.TCKind._item(desc)

    def __repr__(self):
        return "CORBA.TC" + str(self._k)[8:]

# string:
class TypeCode_string (TypeCode_base):
    def __init__(self, desc):
        if type(desc) is not types.TupleType or \
           desc[0] != tv_string:
            raise CORBA.INTERNAL()
        self._d = desc
        self._k = CORBA.tk_string

    def length(self):
        return self._d[1]

    def __repr__(self):
        if self._d[1] == 0:
            return "CORBA.TC_string"
        else:
            return "orb.create_string_tc(bound=%d)" % self._d[1]

# wstring:
class TypeCode_wstring (TypeCode_base):
    def __init__(self, desc):
        if type(desc) is not types.TupleType or \
           desc[0] != tv_wstring:
            raise CORBA.INTERNAL()
        self._d = desc
        self._k = CORBA.tk_wstring

    def length(self):
        return self._d[1]

    def __repr__(self):
        if self._d[1] == 0:
            return "CORBA.TC_wstring"
        else:
            return "orb.create_wstring_tc(bound=%d)" % self._d[1]

# fixed:
class TypeCode_fixed (TypeCode_base):
    def __init__(self, desc):
        if type(desc) is not types.TupleType or \
           desc[0] != tv_fixed:
            raise CORBA.INTERNAL()
        self._d = desc
        self._k = CORBA.tk_fixed

    def fixed_digits(self):
        return self._d[1]

    def fixed_scale(self):
        return self._d[2]

    def __repr__(self):
        return "orb.create_fixed_tc(digits=%d,scale=%d)" % (
            self.fixed_digits(), self.fixed_scale())

# objref:
class TypeCode_objref (TypeCode_base):
    def __init__(self, desc):
        if type(desc) is not types.TupleType or \
           desc[0] not in [ tv_objref,
                            tv_abstract_interface,
                            tv_local_interface ]:
            raise CORBA.INTERNAL()
        self._d = desc
        self._k = CORBA.TCKind._items[desc[0]]

    def id(self):
        if self._d[1] is not None:
            return self._d[1]
        else:
            return ""
        
    def name(self): return self._d[2]

    def __repr__(self):
        return 'CORBA.TypeCode("%s")' % self.id()


# struct:
class TypeCode_struct (TypeCode_base):
    def __init__(self, desc, parent):
        if type(desc) is not types.TupleType or \
           desc[0] != tv_struct:
            raise CORBA.INTERNAL()
        self._d = desc
        self._k = CORBA.tk_struct
        self._p = parent

    def __del__(self):
        if self._p is None and removeIndirections is not None:
            removeIndirections(self._d)

    def equivalent(self, tc):
        return equivalentDescriptors(self._d, tc._d)

    def get_compact_typecode(self):
        return TypeCode_struct(getCompactDescriptor(self._d), None)

    def id(self):                 return self._d[2]
    def name(self):               return self._d[3]
    def member_count(self):       return (len(self._d) - 4) / 2
    def member_name(self, index):
        off = index * 2 + 4
        if index < 0 or off >= len(self._d): raise CORBA.TypeCode.Bounds()
        return self._d[off]

    def member_type(self, index):
        off = index * 2 + 5
        if index < 0 or off >= len(self._d): raise CORBA.TypeCode.Bounds()
        if self._p is None and removeIndirections is not None:
            return createTypeCode(self._d[off], self)
        else:
            return createTypeCode(self._d[off], self._p)

    def __repr__(self):
        return 'CORBA.TypeCode("%s")' % self.id()

    
# union:
class TypeCode_union (TypeCode_base):
    def __init__(self, desc, parent):
        if type(desc) is not types.TupleType or \
           desc[0] != tv_union:
            raise CORBA.INTERNAL()
        self._d = desc
        self._k = CORBA.tk_union
        self._p = parent

    def __del__(self):
        if self._p is None and removeIndirections is not None:
            removeIndirections(self._d)

    def equivalent(self, tc):
        return equivalentDescriptors(self._d, tc._d)

    def get_compact_typecode(self):
        return TypeCode_union(getCompactDescriptor(self._d), None)

    def id(self):                  return self._d[2]
    def name(self):                return self._d[3]
    def member_count(self):        return len(self._d[6])

    def member_name(self, index):
        if index < 0 or index >= len(self._d[6]): raise CORBA.TypeCode.Bounds()
        return self._d[6][index][1]
    
    def member_type(self, index):
        if index < 0 or index >= len(self._d[6]): raise CORBA.TypeCode.Bounds()
        if self._p is None and removeIndirections is not None:
            return createTypeCode(self._d[6][index][2], self)
        else:
            return createTypeCode(self._d[6][index][2], self._p)

    def member_label(self, index):
        if index < 0 or index >= len(self._d[6]): raise CORBA.TypeCode.Bounds()
        if index == self._d[5]: return CORBA.Any(CORBA._tc_octet, 0)
        return CORBA.Any(createTypeCode(self._d[4]), self._d[6][index][0])

    def discriminator_type(self): return createTypeCode(self._d[4])

    def default_index(self):
        if self._d[5] >= 0: return self._d[5]
        return -1

    def __repr__(self):
        return 'CORBA.TypeCode("%s")' % self.id()


# enum:
class TypeCode_enum (TypeCode_base):
    def __init__(self, desc):
        if type(desc) is not types.TupleType or \
           desc[0] != tv_enum:
            raise CORBA.INTERNAL()
        self._d = desc
        self._k = CORBA.tk_enum

    def equivalent(self, tc):
        return equivalentDescriptors(self._d, tc._d)

    def get_compact_typecode(self):
        return TypeCode_enum(getCompactDescriptor(self._d))

    def id(self):           return self._d[1]
    def name(self):         return self._d[2]
    def member_count(self): return len(self._d[3])

    def member_name(self, index):
        if index < 0 or index >= len(self._d[3]): raise CORBA.TypeCode.Bounds()
        return self._d[3][index]._n

    def __repr__(self):
        return 'CORBA.TypeCode("%s")' % self.id()

# sequence:
class TypeCode_sequence (TypeCode_base):
    def __init__(self, desc, parent):
        if type(desc) is not types.TupleType or \
           desc[0] != tv_sequence:
            raise CORBA.INTERNAL()
        self._d = desc
        self._k = CORBA.tk_sequence
        self._p = parent

    def __del__(self):
        if self._p is None and removeIndirections is not None:
            removeIndirections(self._d)

    def equivalent(self, tc):
        return equivalentDescriptors(self._d, tc._d)

    def get_compact_typecode(self):
        return TypeCode_sequence(getCompactDescriptor(self._d), None)

    def length(self):       return self._d[2]
    def content_type(self):
        if self._p is None and removeIndirections is not None:
            return createTypeCode(self._d[1], self)
        else:
            return createTypeCode(self._d[1], self._p)

    def __repr__(self):
        return "orb.create_sequence_tc(bound=%d, element_type=%s)" % (
            self.length(), repr(self.content_type()))


# array:
class TypeCode_array (TypeCode_base):
    def __init__(self, desc, parent):
        if type(desc) is not types.TupleType or \
           desc[0] != tv_array:
            raise CORBA.INTERNAL()
        self._d = desc
        self._k = CORBA.tk_array
        self._p = parent

    def __del__(self):
        if self._p is None and removeIndirections is not None:
            removeIndirections(self._d)

    def equivalent(self, tc):
        return equivalentDescriptors(self._d, tc._d)

    def get_compact_typecode(self):
        return TypeCode_sequence(getCompactDescriptor(self._d), None)

    def length(self):       return self._d[2]
    def content_type(self): return createTypeCode(self._d[1])

    def __repr__(self):
        return "orb.create_array_tc(length=%d, element_type=%s)" % (
            self.length(), repr(self.content_type()))

# alias:
class TypeCode_alias (TypeCode_base):
    def __init__(self, desc, parent):
        if type(desc) is not types.TupleType or \
           desc[0] != tv_alias:
            raise CORBA.INTERNAL()
        self._d = desc
        self._k = CORBA.tk_alias
        self._p = parent

    def __del__(self):
        if self._p is None and removeIndirections is not None:
            removeIndirections(self._d)

    def equivalent(self, tc):
        return equivalentDescriptors(self._d, tc._d)

    def get_compact_typecode(self):
        return TypeCode_alias(getCompactDescriptor(self._d), None)

    def id(self):           return self._d[1]
    def name(self):         return self._d[2]
    def content_type(self): return createTypeCode(self._d[3])

    def __repr__(self):
        return 'CORBA.TypeCode("%s")' % self.id()

# except:
class TypeCode_except (TypeCode_base):
    def __init__(self, desc, parent):
        if type(desc) is not types.TupleType or \
           desc[0] != tv_except:
            raise CORBA.INTERNAL()
        self._d = desc
        self._k = CORBA.tk_except
        self._p = parent

    def __del__(self):
        if self._p is None and removeIndirections is not None:
            removeIndirections(self._d)

    def equivalent(self, tc):
        return equivalentDescriptors(self._d, tc._d)

    def get_compact_typecode(self):
        return TypeCode_except(getCompactDescriptor(self._d), None)

    def id(self):                 return self._d[2]
    def name(self):               return self._d[3]
    def member_count(self):       return (len(self._d) - 4) / 2
    def member_name(self, index):
        off = index * 2 + 4
        if index < 0 or off >= len(self._d): raise CORBA.TypeCode.Bounds()
        return self._d[off]

    def member_type(self, index):
        off = index * 2 + 5
        if index < 0 or off >= len(self._d): raise CORBA.TypeCode.Bounds()
        if self._p is None and removeIndirections is not None:
            return createTypeCode(self._d[off], self)
        else:
            return createTypeCode(self._d[off], self._p)

    def __repr__(self):
        return 'CORBA.TypeCode("%s")' % self.id()


# value:
class TypeCode_value (TypeCode_base):
    def __init__(self, desc, parent):
        if type(desc) is not types.TupleType or \
           desc[0] != tv_value:
            raise CORBA.INTERNAL()
        self._d = desc
        self._k = CORBA.tk_value
        self._p = parent

    def __del__(self):
        if self._p is None and removeIndirections is not None:
            removeIndirections(self._d)

    def equivalent(self, tc):
        return equivalentDescriptors(self._d, tc._d)

    def get_compact_typecode(self):
        return TypeCode_value(getCompactDescriptor(self._d), None)

    def id(self):                 return self._d[2]
    def name(self):               return self._d[3]
    def member_count(self):       return (len(self._d) - 7) / 3
    def member_name(self, index):
        off = index * 3 + 7
        if index < 0 or off >= len(self._d): raise CORBA.TypeCode.Bounds()
        return self._d[off]

    def member_type(self, index):
        off = index * 3 + 8
        if index < 0 or off >= len(self._d): raise CORBA.TypeCode.Bounds()
        if self._p is None and removeIndirections is not None:
            return createTypeCode(self._d[off], self)
        else:
            return createTypeCode(self._d[off], self._p)

    def member_visibility(self, index):
        off = index * 3 + 9
        if index < 0 or off >= len(self._d): raise CORBA.TypeCode.Bounds()
        return self._d[off]

    def type_modifier(self):
        return self._d[4]

    def concrete_base_type(self):
        if self._d[6] == tv_null:
            return None
        else:
            return createTypeCode(self._d[6], self._p)

    def __repr__(self):
        return 'CORBA.TypeCode("%s")' % self.id()


# valuebox:
class TypeCode_value_box (TypeCode_base):
    def __init__(self, desc, parent):
        if type(desc) is not types.TupleType or \
           desc[0] != tv_value_box:
            raise CORBA.INTERNAL()
        self._d = desc
        self._k = CORBA.tk_value_box
        self._p = parent

    def __del__(self):
        if self._p is None and removeIndirections is not None:
            removeIndirections(self._d)

    def equivalent(self, tc):
        return equivalentDescriptors(self._d, tc._d)

    def get_compact_typecode(self):
        return TypeCode_alias(getCompactDescriptor(self._d), None)

    def id(self):           return self._d[2]
    def name(self):         return self._d[3]
    def content_type(self): return createTypeCode(self._d[4])

    def __repr__(self):
        return 'CORBA.TypeCode("%s")' % self.id()


# Map of pre-created basic TypeCodes
basicTypeCodes = {
    tv_null:      TypeCode_empty(tv_null),
    tv_void:      TypeCode_empty(tv_void),
    tv_short:     TypeCode_empty(tv_short),
    tv_long:      TypeCode_empty(tv_long),
    tv_ushort:    TypeCode_empty(tv_ushort),
    tv_ulong:     TypeCode_empty(tv_ulong),
    tv_float:     TypeCode_empty(tv_float),
    tv_double:    TypeCode_empty(tv_double),
    tv_boolean:   TypeCode_empty(tv_boolean),
    tv_char:      TypeCode_empty(tv_char),
    tv_octet:     TypeCode_empty(tv_octet),
    tv_any:       TypeCode_empty(tv_any),
    tv_TypeCode:  TypeCode_empty(tv_TypeCode),
    tv_Principal: TypeCode_empty(tv_Principal),
    tv_longlong:  TypeCode_empty(tv_longlong),
    tv_ulonglong: TypeCode_empty(tv_ulonglong),
    tv_longdouble:TypeCode_empty(tv_longdouble),
    tv_wchar:     TypeCode_empty(tv_wchar),

    (tv_string, 0): TypeCode_string ((tv_string,  0)),
    (tv_wstring,0): TypeCode_wstring((tv_wstring, 0)),
}


# Functions to test descriptor equivalence
def equivalentDescriptors(a, b, seen=None, a_ids=None, b_ids=None):

    if seen is None:
        seen  = {}
        a_ids = {}
        b_ids = {}

    try:
        if a == b: return 1

        # If they don't trivially match, they must be tuples:
        if type(a) is not types.TupleType or type(b) is not types.TupleType:
            return 0

        # Follow aliases and indirections
        while (type(a) is types.TupleType and
               (a[0] == tv_alias or a[0] == tv__indirect)):

            if a[0] == tv_alias:
                if a[1] != "": a_ids[a[1]] = a
                a = a[3]
            else:
                if type(a[1][0]) is types.StringType:
                    a = a_ids[a[1][0]]
                else:
                    a = a[1][0]

        while (type(b) is types.TupleType and
               (b[0] == tv_alias or b[0] == tv__indirect)):
            
            if b[0] == tv_alias:
                if b[1] != "": b_ids[b[1]] = b
                b = b[3]
            else:
                if type(b[1][0]) is types.StringType:
                    b = b_ids[b[1][0]]
                else:
                    b = b[1][0]

        # Re-do the trivial checks on the underlying types.
        if a == b: return 1

        if type(a) is not types.TupleType or type(b) is not types.TupleType:
            return 0

        # Handle cycles
        if seen.has_key((id(a),id(b))):
            return 1

        seen[id(a),id(b)] = None

        # Must be same kind
        if a[0] != b[0]:
            return 0

        if a[0] == tv_struct:
            # id
            if a[2] != "": a_ids[a[2]] = a
            if b[2] != "": b_ids[b[2]] = b

            if a[2] != "" and b[2] != "":
                if a[2] == b[2]:
                    return 1
                else:
                    return 0

            # members:
            if len(a) != len(b):
                return 0
            
            for i in range(4, len(a), 2):
                # Member type
                if not equivalentDescriptors(a[i+1], b[i+1],
                                             seen, a_ids, b_ids):
                    return 0
            return 1

        elif a[0] == tv_union:
            # id
            if a[2] != "": a_ids[a[2]] = a
            if b[2] != "": b_ids[b[2]] = b

            if a[2] != "" and b[2] != "":
                if a[2] == b[2]:
                    return 1
                else:
                    return 0

            # discriminant type
            if not equivalentDescriptors(a[4], b[4], seen, a_ids, b_ids):
                return 0

            # default index
            if a[5] != b[5]:
                return 0

            # Members
            if len(a[6]) != len(b[6]):
                return 0

            for i in range(len(a[6])):
                # Member label
                if a[6][i][0] != b[6][i][0]:
                    return 0

                # Member descriptor
                if not equivalentDescriptors(a[6][i][2], b[6][i][2],
                                             seen, a_ids, b_ids):
                    return 0

            return 1

        elif a[0] == tv_enum:
            # id
            if a[1] != "": a_ids[a[1]] = a
            if b[1] != "": b_ids[b[1]] = b

            if a[1] != "" and b[1] != "":
                if a[1] == b[1]:
                    return 1
                else:
                    return 0

            # Members
            if len(a[3]) != len(b[3]):
                return 0

            return 1

        elif a[0] == tv_sequence:
            # Bound
            if a[2] != b[2]:
                return 0

            # Type
            return equivalentDescriptors(a[1], b[1], seen, a_ids, b_ids)

        elif a[0] == tv_array:
            # Length
            if a[2] != b[2]:
                return 0

            # Type
            return equivalentDescriptors(a[1], b[1], seen, a_ids, b_ids)

        elif a[0] == tv_except:
            # id
            if a[2] != "": a_ids[a[2]] = a
            if b[2] != "": b_ids[b[2]] = b

            if a[2] != "" and b[2] != "":
                if a[2] == b[2]:
                    return 1
                else:
                    return 0

                # members:
                if len(a) != len(b):
                    return 0

                for i in range(4, len(self._d), 2):
                    # Member type
                    if not equivalentDescriptors(a[i+1], b[i+1],
                                                 seen, a_ids, b_ids):
                        return 0
            return 1

        elif a[0] == tv_value:
            # id
            if a[2] != "": a_ids[a[2]] = a
            if b[2] != "": b_ids[b[2]] = b

            if a[2] != "" and b[2] != "":
                if a[2] == b[2]:
                    return 1
                else:
                    return 0

            # members
            if len(a) != len(b):
                return 0

            for i in range(7, len(a), 3):
                # Access spec
                if a[i+2] != b[i+2]:
                    return 0
                
                if not equivalentDescriptors(a[i+1], b[i+1],
                                             seen, a_ids, b_ids):
                    return 0

            return 1

        elif a[0] == tv_value_box:
            # id
            if a[2] != "": a_ids[a[2]] = a
            if b[2] != "": b_ids[b[2]] = b

            if a[2] != "" and b[2] != "":
                if a[2] == b[2]:
                    return 1
                else:
                    return 0

            # Boxed type
            if equivalentDescriptors(a[4], b[4], seen, a_ids, b_ids):
                return 1
            else:
                return 0

        return 0

    except AttributeError:
        raise CORBA.BAD_PARAM(BAD_PARAM_WrongPythonType, CORBA.COMPLETED_NO)


# Functions to compact descriptors:
def getCompactDescriptor(d):
    seen = {}
    ind  = []
    r = r_getCompactDescriptor(d, seen, ind)

    # Fix up indirections:
    for i in ind:
        try:
            if (type(i[0]) is types.StringType):
                i[0] = seen[i[0]]
            else:
                i[0] = seen[id(i[0])]
        except KeyError:
            raise CORBA.BAD_TYPECODE(BAD_TYPECODE_InvalidIndirection,
                                     CORBA.COMPLETED_NO)

    return r

def r_getCompactDescriptor(d, seen, ind):
    if type(d) is types.TupleType:
        k = d[0]
    else:
        k = d

    if   k == tv_short:     r = d
    elif k == tv_long:      r = d
    elif k == tv_ushort:    r = d
    elif k == tv_ulong:     r = d
    elif k == tv_float:     r = d
    elif k == tv_double:    r = d
    elif k == tv_boolean:   r = d
    elif k == tv_char:      r = d
    elif k == tv_octet:     r = d
    elif k == tv_any:       r = d
    elif k == tv_TypeCode:  r = d
    elif k == tv_Principal: r = d
    elif k == tv_string:    r = d
    elif k == tv_objref:    r = d
    elif k == tv_longlong:  r = d
    elif k == tv_ulonglong: r = d
    elif k == tv_longdouble:r = d
    
    elif k == tv_struct:
        c = list(d)
        c[3] = ""
        for i in range(4, len(c), 2):
            c[i]   = ""
            c[i+1] = r_getCompactDescriptor(d[i+1], seen, ind)

        r = tuple(c)
        seen[d[2]] = r
        seen[id(d)] = r
    
    elif k == tv_union:
        c = list(d)
        c[3] = ""
        c[4] = r_getCompactDescriptor(d[4], seen, ind)

        m = []
        for u in d[6]:
            m.append((u[0], "", r_getCompactDescriptor(u[2], seen, ind)))

        c[6] = tuple(m)

        if d[7] is not None:
            c[7] = (d[7][0], "", r_getCompactDescriptor(d[7][2], seen, ind))

        r = tuple(c)
        seen[d[2]] = r
        seen[id(d)] = r
        
    elif k == tv_enum:
        m = []
        for e in d[3]:
            m.append(omniORB.AnonymousEnumItem(e._v))
        r = (k, d[1], "", tuple(m))

    elif k == tv_sequence:
        r = (k, r_getCompactDescriptor(d[1], seen, ind), d[2])
        
    elif k == tv_array:
        r = (k, r_getCompactDescriptor(d[1], seen, ind), d[2])

    elif k == tv_alias:
        r = (k, d[1], "", r_getCompactDescriptor(d[3], seen, ind))

    elif k == tv_except:
        c = list(d)
        c[3] = ""
        for i in range(4, len(c), 2):
            c[i]   = ""
            c[i+1] = r_getCompactDescriptor(d[i+1], seen, ind)

        r = tuple(c)

    elif k == tv_value:
        c = list(d)
        c[3] = ""
        for i in range(7, len(c), 3):
            c[i]   = ""
            c[i+1] = r_getCompactDescriptor(d[i+1], seen, ind)

        r = tuple(c)
        seen[d[2]] = r
        seen[id(d)] = r
    
    elif k == tv_value_box:
        c = list(d)
        c[3] = ""
        c[4] = r_getCompactDescriptor(d[4], seen, ind)
        r = tuple(c)
        seen[d[2]] = r
        seen[id(d)] = r

    elif k == tv_abstract_interface:
        r = d

    elif k == tv_local_interface:
        r = d

    elif k == tv__indirect:
        l = [d[1][0]]
        ind.append(l)
        r = (k, l)

    else: raise CORBA.INTERNAL()

    return r


# Function to remove indirections from a descriptor, so it can be
# collected by Python's reference counting garbage collector. Not
# strictly necessary now Python has a cycle collector, but it does no
# harm.

def removeIndirections(desc):
    if type(desc) is not types.TupleType: return

    k = desc[0]

    if k == tv_struct:
        for i in range(5, len(desc), 2):
            removeIndirections(desc[i])

    elif k == tv_union:
        for t in desc[6]:
            removeIndirections(t[2])
        if desc[7] is not None:
            removeIndirections(desc[7][2])

    elif k == tv_sequence:
        removeIndirections(desc[1])

    elif k == tv_array:
        removeIndirections(desc[1])

    elif k == tv_alias:
        removeIndirections(desc[3])

    elif k == tv_except:
        for i in range(5, len(desc), 2):
            removeIndirections(desc[i])

    elif k == tv_value:
        for i in range(8, len(desc), 3):
            removeIndirections(desc[i])

    elif k == tv__indirect:
        desc[1][0] = None
