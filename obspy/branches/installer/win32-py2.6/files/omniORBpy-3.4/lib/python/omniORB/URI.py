# -*- Mode: Python; -*-
#                            Package   : omniORBpy
# URI.py                     Created on: 2000/06/26
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
#    URI handling functions

# $Id: URI.py,v 1.2.2.1 2003/03/23 21:51:43 dgrisby Exp $

# $Log: URI.py,v $
# Revision 1.2.2.1  2003/03/23 21:51:43  dgrisby
# New omnipy3_develop branch.
#
# Revision 1.1.4.1  2002/03/11 15:40:05  dpg1
# _get_interface support, exception minor codes.
#
# Revision 1.1  2000/06/27 16:15:46  dpg1
# New omniORB.URI module
#

import types, string, re
import CosNaming
from omniORB import CORBA


__regex = re.compile(r"([/\.\\])")


def stringToName(sname):
    """stringToName(string) -> CosNaming.Name

Convert a stringified name to a CosNaming.Name"""

    # Try to understand this at your peril... :-)
    #
    # It works by splitting the input string into a list. Each item in
    # the list is either a string fragment, or a single "special"
    # character -- ".", "/", or "\". It then walks over the list,
    # building a list of NameComponents, based on the meanings of the
    # special characters.

    global __regex

    if type(sname) is not types.StringType:
        raise CORBA.BAD_PARAM(omniORB.BAD_PARAM_WrongPythonType, COMPLETED_NO)

    if sname == "":
        raise CosNaming.NamingContext.InvalidName()

    parts   = __regex.split(sname)
    name    = [CosNaming.NameComponent("","")]
    dotseen = 0

    parts = filter(None, parts)
    parts.reverse()
    while parts:
        part = parts.pop()

        if part == "\\":
            if not parts:
                raise CosNaming.NamingContext.InvalidName()
            part = parts.pop()
            if part != "\\" and part != "/" and part != ".":
                raise CosNaming.NamingContext.InvalidName()

        elif part == ".":
            if dotseen:
                raise CosNaming.NamingContext.InvalidName()
            dotseen = 1
            continue

        elif part == "/":
            if not parts:
                raise CosNaming.NamingContext.InvalidName()
            
            if dotseen:
                if name[-1].kind == "" and name[-1].id != "":
                    raise CosNaming.NamingContext.InvalidName()
            else:
                if name[-1].id == "":
                    raise CosNaming.NamingContext.InvalidName()

            dotseen = 0
            name.append(CosNaming.NameComponent("",""))
            continue

        if dotseen:
            name[-1].kind = name[-1].kind + part
        else:
            name[-1].id = name[-1].id + part

    return name



def nameToString(name):
    """nameToString(CosNaming.Name) -> string

Convert the CosNaming.Name into its stringified form."""

    global __regex
    parts = []

    if type(name) is not types.ListType and \
       type(name) is not types.TupleType:
        raise CORBA.BAD_PARAM(omniORB.BAD_PARAM_WrongPythonType, COMPLETED_NO)

    if len(name) == 0:
        raise CosNaming.NamingContext.InvalidName()

    try:
        for nc in name:
            if nc.id == "" and nc.kind == "":
                parts.append(".")
            elif nc.kind == "":
                parts.append(__regex.sub(r"\\\1", nc.id))
            else:
                parts.append(__regex.sub(r"\\\1", nc.id) + "." + \
                             __regex.sub(r"\\\1", nc.kind))
    except AttributeError:
        raise CORBA.BAD_PARAM(omniORB.BAD_PARAM_WrongPythonType, COMPLETED_NO)

    return string.join(parts, "/")


def addrAndNameToURI(addr, sname):
    """addrAndNameToURI(addr, sname) -> URI

Create a valid corbaname URI from an address string and a stringified name"""

    # *** Note that this function does not properly check the address
    # string. It should raise InvalidAddress if the address looks
    # invalid.

    import urllib

    if type(addr) is not types.StringType or \
       type(sname) is not types.StringType:
        raise CORBA.BAD_PARAM(omniORB.BAD_PARAM_WrongPythonType, COMPLETED_NO)

    if addr == "":
        raise CosNaming.NamingContextExt.InvalidAddress()

    if sname == "":
        return "corbaname:" + addr
    else:
        stringToName(sname) # This might raise InvalidName
        return "corbaname:" + addr + "#" + urllib.quote(sname)
