# -*- Mode: Python; -*-
#                            Package   : omniORBpy
# any.py                     Created on: 2002/09/16
#                            Author    : Duncan Grisby (dgrisby)
#
#    Copyright (C) 2002-2008 Apasphere Ltd
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
#    Utility functions for working with Anys.

"""
omniORB.any module -- Any support functions.

to_any(data)                  -- try to coerce data to an Any.

from_any(any, keep_structs=0) -- return any's contents as plain Python
                                 objects. If keep_structs is true,
                                 CORBA structs are kept as Python class
                                 instances; if false, they are expanded
                                 to dictionaries.
"""

from types import *
import omniORB
import CORBA, tcInternal
import random
import threading

__all__ = ["to_any", "from_any"]


# Counter for generating repoIds. Ideally, should not clash with other
# IDs, but this will do for now...
random.seed()
_idbase  = "%08x" % random.randrange(0, 0x7fffffff)
_idcount = 0
_idlock  = threading.Lock()

# TypeCode kinds that must not be used in struct / sequence members
INVALID_MEMBER_KINDS = [ tcInternal.tv_null,
                         tcInternal.tv_void,
                         tcInternal.tv_except ]


# Fudge things for Pythons without unicode / bool
try:
    UnicodeType
except NameError:
    class UnicodeType:
        pass

try:
    BooleanType
except NameError:
    class BooleanType:
        pass

try:
    bool(1)
except:
    def bool(x): return x


# Fixed type
_f = CORBA.fixed(0)
FixedType = type(_f)


def to_any(data):
    """to_any(data) -- try to return data as a CORBA.Any"""
    tc, val = _to_tc_value(data)
    return CORBA.Any(tc, val)


def _to_tc_value(data):
    """_to_tc_value(data) -- return TypeCode and value for Any insertion"""

    if data is None:
        return CORBA.TC_null, None

    elif isinstance(data, StringType):
        return CORBA.TC_string, data

    elif isinstance(data, UnicodeType):
        return CORBA.TC_wstring, data

    elif isinstance(data, BooleanType):
        return CORBA.TC_boolean, data

    elif isinstance(data, IntType) or isinstance(data, LongType):
        if -2147483648L <= data <= 2147483647L:
            return CORBA.TC_long, int(data)
        elif 0 <= data <= 4294967295L:
            return CORBA.TC_ulong, data
        elif -9223372036854775808L <= data <= 9223372036854775807L:
            return CORBA.TC_longlong, data
        elif 0 <= data <= 18446744073709551615L:
            return CORBA.TC_ulonglong, data
        else:
            raise CORBA.BAD_PARAM(omniORB.BAD_PARAM_PythonValueOutOfRange,
                                  CORBA.COMPLETED_NO)

    elif isinstance(data, FloatType):
        return CORBA.TC_double, data

    elif isinstance(data, ListType):
        if data == []:
            tc = tcInternal.createTypeCode((tcInternal.tv_sequence,
                                            tcInternal.tv_any, 0))
            return tc, data

        d0 = data[0]

        if isinstance(d0, StringType):
            for d in data:
                if not isinstance(d, StringType):
                    break
            else:
                # List of strings
                tc = tcInternal.createTypeCode((tcInternal.tv_sequence,
                                                CORBA.TC_string._d, 0))
                return tc, data

        elif isinstance(d0, UnicodeType):
            for d in data:
                if not isinstance(d, UnicodeType):
                    break
            else:
                # List of wstrings
                tc = tcInternal.createTypeCode((tcInternal.tv_sequence,
                                                CORBA.TC_wstring._d, 0))
                return tc, data

        elif isinstance(d0, BooleanType):
            for d in data:
                if not isinstance(d, BooleanType):
                    break
            else:
                tc = tcInternal.createTypeCode((tcInternal.tv_sequence,
                                                tcInternal.tv_boolean, 0))
                return tc, data

        elif isinstance(d0, IntType) or isinstance(d0, LongType):
            # Numeric. Try to find a numeric type suitable for the whole list
            min_v = max_v = 0
            for d in data:
                if (not (isinstance(d, IntType) or isinstance(d, LongType)) or
                    isinstance(d, BooleanType)):
                    break
                if d < min_v: min_v = d
                if d > max_v: max_v = d
            else:
                if min_v >= -2147483648L and max_v <= 2147483647L:
                    tc = tcInternal.createTypeCode((tcInternal.tv_sequence,
                                                    tcInternal.tv_long, 0))
                    return tc, map(int, data)
                elif min_v >= 0 and max_v <= 4294967295L:
                    tc = tcInternal.createTypeCode((tcInternal.tv_sequence,
                                                    tcInternal.tv_ulong, 0))
                    return tc, map(long, data)
                elif (min_v >= -9223372036854775808L and
                      max_v <= 9223372036854775807L):
                    tc = tcInternal.createTypeCode((tcInternal.tv_sequence,
                                                    tcInternal.tv_longlong, 0))
                    return tc, map(long, data)
                elif min_v >= 0 and max_v <= 18446744073709551615L:
                    tc = tcInternal.createTypeCode((tcInternal.tv_sequence,
                                                    tcInternal.tv_ulonglong,0))
                    return tc, map(long, data)
                else:
                    raise CORBA.BAD_PARAM(
                        omniORB.BAD_PARAM_PythonValueOutOfRange,
                        CORBA.COMPLETED_NO)

        elif isinstance(d0, FloatType):
            for d in data:
                if not isinstance(d, FloatType):
                    break
            else:
                # List of doubles
                tc = tcInternal.createTypeCode((tcInternal.tv_sequence,
                                                tcInternal.tv_double, 0))
                return tc, data

        elif isinstance(d0, CORBA.Any):
            for d in data:
                if not isinstance(d, CORBA.Any):
                    break
            else:
                # List of anys
                tc = tcInternal.createTypeCode((tcInternal.tv_sequence,
                                                tcInternal.tv_any, 0))
                return tc, data

        # Generic list
        tc = tcInternal.createTypeCode((tcInternal.tv_sequence,
                                        tcInternal.tv_any, 0))
        any_list = map(to_any, data)

        atc = any_list[0]._t

        if atc._k._v not in INVALID_MEMBER_KINDS:
            for a in any_list:
                if not a._t.equivalent(atc):
                    break
            else:
                tc = tcInternal.createTypeCode((tcInternal.tv_sequence,
                                                atc._d, 0))
                for i in range(len(any_list)):
                    any_list[i] = any_list[i]._v
            
        return tc, any_list

    elif isinstance(data, TupleType):
        return _to_tc_value(list(data))

    elif isinstance(data, DictType):
        # Represent dictionaries as structs
        global _idcount

        _idlock.acquire()
        try:
            _idcount = _idcount + 1
            id = "omni:%s:%08x" % (_idbase, _idcount)
        finally:
            _idlock.release()

        dl = [tcInternal.tv_struct, None, id, ""]
        ms = []
        svals = []
        items = data.items()
        for (k,v) in items:
            if not isinstance(k, StringType):
                raise CORBA.BAD_PARAM(omniORB.BAD_PARAM_WrongPythonType,
                                      CORBA.COMPLETED_NO)
            t, v = _to_tc_value(v)

            if t._k._v in INVALID_MEMBER_KINDS:
                v = CORBA.Any(t,v)
                t = CORBA.TC_any
            
            ms.append(k)
            dl.append(k)
            dl.append(t._d)
            svals.append(v)
        cls   = omniORB.createUnknownStruct(id, ms)
        dl[1] = cls
        tc    = tcInternal.createTypeCode(tuple(dl))
        value = apply(cls, svals)
        return tc, value

    elif isinstance(data, FixedType):
        if data == CORBA.fixed(0):
            tc = tcInternal.createTypeCode((tcInternal.tv_fixed, 1, 0))
        else:
            tc = tcInternal.createTypeCode((tcInternal.tv_fixed,
                                            data.precision(), data.decimals()))
        return tc, data

    elif isinstance(data, CORBA.Any):
        return data._t, data._v

    elif isinstance(data, omniORB.EnumItem):
        return omniORB.findTypeCode(data._parent_id), data

    elif hasattr(data, "_NP_RepositoryId"):
        return omniORB.findTypeCode(data._NP_RepositoryId), data

    raise CORBA.BAD_PARAM(omniORB.BAD_PARAM_WrongPythonType,
                          CORBA.COMPLETED_NO)


def from_any(any, keep_structs=0):
    """from_any(any, keep_structs=0) -- extract the data from an Any.

    If keep_structs is true, CORBA structs are left as Python
    instances; if false, structs are turned into dictionaries with
    string keys.
    """
    if not isinstance(any, CORBA.Any):
        raise CORBA.BAD_PARAM(omniORB.BAD_PARAM_WrongPythonType,
                              CORBA.COMPLETED_NO)
    return _from_desc_value(any._t._d, any._v, keep_structs)

def _from_desc_value(desc, value, keep_structs=0):
    """_from_desc_value(desc,val,keep_structs) -- de-Any value"""

    if type(desc) is IntType:
        if desc == tcInternal.tv_boolean:
            return bool(value)
        
        elif desc != tcInternal.tv_any:
            # Nothing to do
            return value
        else:
            k = desc
    else:
        k = desc[0]

    while k == tcInternal.tv_alias:
        desc = desc[3]
        if type(desc) is IntType:
            if desc == tcInternal.tv_boolean:
                return bool(value)

            elif desc != tcInternal.tv_any:
                return value
            else:
                k = desc
        else:
            k = desc[0]

    if k == tcInternal.tv_any:
        return _from_desc_value(value._t._d, value._v, keep_structs)

    elif k == tcInternal.tv_struct:
        if keep_structs:
            return value
        d = {}
        for i in range(4, len(desc), 2):
            sm = desc[i]
            sd = desc[i+1]
            d[sm] = _from_desc_value(sd, getattr(value, sm), keep_structs)
        return d

    elif k in [tcInternal.tv_sequence, tcInternal.tv_array]:
        sd = desc[1]
        if type(sd) is IntType and sd != tcInternal.tv_any:
            return value
        else:
            rl = []
            for i in value:
                rl.append(_from_desc_value(sd, i, keep_structs))
            return rl

    return value
