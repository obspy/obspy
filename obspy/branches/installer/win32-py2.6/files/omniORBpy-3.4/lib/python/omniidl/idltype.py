# -*- python -*-
#                           Package   : omniidl
# idltype.py                Created on: 1999/10/27
#			    Author    : Duncan Grisby (dpg1)
#
#    Copyright (C) 2003-2005 Apasphere Ltd
#    Copyright (C) 1999 AT&T Laboratories Cambridge
#
#  This file is part of omniidl.
#
#  omniidl is free software; you can redistribute it and/or modify it
#  under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
#  02111-1307, USA.
#
# Description:
#   
#   IDL type representation

"""Definitions for IDL type representation

Classes:

  Type     -- abstract base class.
  Base     -- class for CORBA base types.
  String   -- class for string types.
  WString  -- class for wide string types.
  Sequence -- class for sequence types.
  Fixed    -- class for fixed-point types.
  Declared -- class for declared types.

TypeCode kind constants:

  tk_null, tk_void, tk_short, tk_long, tk_ushort, tk_ulong, tk_float,
  tk_double, tk_boolean, tk_char, tk_octet, tk_any, tk_TypeCode,
  tk_Principal, tk_objref, tk_struct, tk_union, tk_enum, tk_string,
  tk_sequence, tk_array, tk_alias, tk_except, tk_longlong,
  tk_ulonglong, tk_longdouble, tk_wchar, tk_wstring, tk_fixed,
  tk_value, tk_value_box, tk_native, tk_abstract_interface"""

import idlutil

tk_null               = 0
tk_void               = 1
tk_short              = 2
tk_long               = 3
tk_ushort             = 4
tk_ulong              = 5
tk_float              = 6
tk_double             = 7
tk_boolean            = 8
tk_char	              = 9
tk_octet              = 10
tk_any	              = 11
tk_TypeCode           = 12
tk_Principal          = 13
tk_objref             = 14
tk_struct             = 15
tk_union              = 16
tk_enum	              = 17
tk_string             = 18
tk_sequence           = 19
tk_array              = 20
tk_alias              = 21
tk_except             = 22
tk_longlong           = 23
tk_ulonglong          = 24
tk_longdouble         = 25
tk_wchar              = 26
tk_wstring            = 27
tk_fixed              = 28
tk_value              = 29
tk_value_box          = 30
tk_native             = 31
tk_abstract_interface = 32
tk_local_interface    = 33

# Non-standard kinds for forward-declared structs and unions
ot_structforward      = 100
ot_unionforward       = 101


class Error:
    """Exception class used by IdlType internals."""

    def __init__(self, err):
        self.err = err

    def __repr__(self):
        return self.err


class Type:
    """Type abstract class.

Function:

  kind()          -- TypeCode kind of type.
  unalias()       -- Return an equivalent Type object with aliases stripped
  accept(visitor) -- visitor pattern accept. See idlvisitor.py."""

    def __init__(self, kind, local):
        self.__kind  = kind
        self.__local = local

    def kind(self):            return self.__kind
    def local(self):           return self.__local

    def unalias(self):
        type = self
        while type.kind() == tk_alias:
            if len(type.decl().sizes()) > 0:
                return type
            type = type.decl().alias().aliasType()
        return type
        
    def accept(self, visitor): pass


# Base types
class Base (Type):
    """Class for CORBA base types. (Type)

No non-inherited functions."""

    def __init__(self, kind):
        if kind not in [tk_null, tk_void, tk_short, tk_long, tk_ushort,
                        tk_ulong, tk_float, tk_double, tk_boolean,
                        tk_char, tk_octet, tk_any, tk_TypeCode,
                        tk_Principal, tk_longlong, tk_ulonglong,
                        tk_longdouble, tk_wchar]:
            
            raise Error("Attempt to create Base type with invalid kind.")

        Type.__init__(self, kind, 0)

    def accept(self, visitor): visitor.visitBaseType(self)


# Strings can be used like base types without a declaration. eg:
#
#   void op(in string<10> s);
#
# therefore, the String type must include its bound here, rather than
# relying on looking at the corresponding declaration

class String (Type):
    """Class for string types (Type)

Function:

  bound() -- bound of bounded string. 0 for unbounded."""

    def __init__(self, bound):
        Type.__init__(self, tk_string, 0)
        self.__bound = bound

    def accept(self, visitor): visitor.visitStringType(self)
    def bound(self): return self.__bound

class WString (Type):
    """Class for wide string types (Type)

Function:

  bound() -- bound of bounded wstring. 0 for unbounded."""

    def __init__(self, bound):
        Type.__init__(self, tk_wstring, 0)
        self.__bound = bound

    def accept(self, visitor): visitor.visitWStringType(self)
    def bound(self): return self.__bound


# Sequences are never declared. They either appear as
#
#   typedef sequence <...> ...
#
# or inside a struct, union, or valuetype

class Sequence (Type):
    """Class for sequence types (Type)

Functions:

  seqType() -- Type this is a sequence of.
  bound()   -- bound of bounded sequence. 0 for unbounded."""

    def __init__(self, seqType, bound, local):
        Type.__init__(self, tk_sequence, local)
        self.__seqType = seqType
        self.__bound   = bound

    def accept(self, visitor): visitor.visitSequenceType(self)
    def seqType(self): return self.__seqType
    def bound(self):   return self.__bound


# Same goes for fixed

class Fixed (Type):
    """Class for fixed point types (Type)

Functions:

  digits() -- digits.
  scale()  -- scale."""

    def __init__(self, digits, scale):
        Type.__init__(self, tk_fixed, 0)
        self.__digits = digits
        self.__scale  = scale

    def accept(self, visitor): visitor.visitFixedType(self)
    def digits(self): return self.__digits
    def scale(self):  return self.__scale


# All other types must be declared, at least implicitly, so they have
# an associated declaration object

class Declared (Type):
    """Class for declared types (Type)

Functions:

  decl()       -- Decl object which corresponds to this type.
  scopedName() -- Fully scoped name of the type as a list of strings.
  name()       -- Simple name of the type."""

    def __init__(self, decl, scopedName, kind, local):
        if kind not in [tk_objref, tk_struct, tk_union, tk_enum,
                        tk_array, tk_alias, tk_except, tk_value,
                        tk_value_box, tk_native, tk_abstract_interface,
                        tk_local_interface, ot_structforward,
                        ot_unionforward]:

            raise Error("Attempt to create Declared type with invalid kind.")

        Type.__init__(self, kind, local)
        self.__decl       = decl
        self.__scopedName = scopedName

    def accept(self, visitor): visitor.visitDeclaredType(self)

    # Decl object where the type was declared.
    def decl(self):       return self.__decl

    # List containing scoped name:
    def scopedName(self): return self.__scopedName

    # Simple name
    def name(self):       return self.__scopedName[-1]


def containsValueType(t, track=None):
    """Returns true if the type contains valuetypes"""

    import idlast

    if track is None:
        track = {}

    if track.has_key(id(t)):
        return 0
    track[id(t)] = None

    if isinstance(t, Sequence):
        return containsValueType(t.seqType(), track)

    if isinstance(t, Declared):
        d = t.decl()

        if isinstance(d, idlast.Declarator):
            alias = d.alias()
            if alias:
                return containsValueType(alias.aliasType(), track)

        if isinstance(d, idlast.Struct):
            for m in d.members():
                if containsValueType(m.memberType(), track):
                    return 1

        if isinstance(d, idlast.Union):
            for c in d.cases():
                if containsValueType(c.caseType(), track):
                    return 1

        if isinstance(d, idlast.ValueAbs):
            return 1

        if isinstance(d, idlast.Value):
            return 1

        if isinstance(d, idlast.ValueBox):
            return 1

        if isinstance(d, idlast.Interface) and d.abstract():
            return 1

    return 0


# Map of singleton Base Type objects
baseTypeMap = {
    tk_null:       Base(tk_null),
    tk_void:       Base(tk_void),
    tk_short:      Base(tk_short),
    tk_long:       Base(tk_long),
    tk_ushort:     Base(tk_ushort),
    tk_ulong:      Base(tk_ulong),
    tk_float:      Base(tk_float),
    tk_double:     Base(tk_double),
    tk_boolean:    Base(tk_boolean),
    tk_char:       Base(tk_char),
    tk_octet:      Base(tk_octet),
    tk_any:        Base(tk_any),
    tk_TypeCode:   Base(tk_TypeCode),
    tk_Principal:  Base(tk_Principal),
    tk_longlong:   Base(tk_longlong),
    tk_ulonglong:  Base(tk_ulonglong),
    tk_longdouble: Base(tk_longdouble),
    tk_wchar:      Base(tk_wchar)
    }

# Maps of String and WString Type objects, indexed by bound
stringTypeMap =  { 0: String(0) }
wstringTypeMap = { 0: WString(0) }

# Map of Sequence Type objects, indexed by (type object,bound)
sequenceTypeMap = {}

# Map of Fixed Type objects, indexed by (digits,scale)
fixedTypeMap = {}

# Map of declared type objects, indexed by stringified scoped name
declaredTypeMap = {}


# Private functions to create or return existing Type objects
def baseType(kind):
    return baseTypeMap[kind]

def stringType(bound):
    try:
        return stringTypeMap[bound]
    except KeyError:
        st = String(bound)
        stringTypeMap[bound] = st
        return st

def wstringType(bound):
    try:
        return wstringTypeMap[bound]
    except KeyError:
        wst = WString(bound)
        wstringTypeMap[bound] = wst
        return wst

def sequenceType(type_spec, bound, local):
    try:
        return sequenceTypeMap[(type_spec,bound)]
    except KeyError:
        st = Sequence(type_spec, bound, local)
        sequenceTypeMap[(type_spec,bound)] = st
        return st

def fixedType(digits, scale):
    try:
        return fixedTypeMap[(digits,scale)]
    except KeyError:
        ft = Fixed(digits, scale)
        fixedTypeMap[(digits,scale)] = ft
        return ft

def declaredType(decl, scopedName, kind, local):
    sname = idlutil.slashName(scopedName)
    if declaredTypeMap.has_key(sname):
        dt = declaredTypeMap[sname]
        if dt.kind() == kind:
            return dt
    dt = Declared(decl, scopedName, kind, local)
    declaredTypeMap[sname] = dt
    return dt

def clear():
    """Clear back-end structures ready for another run"""
    stringTypeMap.clear()
    wstringTypeMap.clear()
    sequenceTypeMap.clear()
    fixedTypeMap.clear()
    declaredTypeMap.clear()
