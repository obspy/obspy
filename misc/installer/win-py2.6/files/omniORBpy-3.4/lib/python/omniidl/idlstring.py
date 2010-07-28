# -*- python -*-
#                           Package   : omniidl
# idlstring.py              Created on: 2008/11/04
#			    Author    : Duncan Grisby (dgrisby)
#
#    Copyright (C) 2008 Apasphere Ltd.
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
#  Compatibility implementation of old Python string module.

from string import *

try:
    join
except NameError:

    class _Unset:
        pass

    def join(l, sep=" "):
        return sep.join(l)

    def split(s, sep=None, maxplit=None):
        if maxsplit is None:
            return s.split(sep)
        else:
            return s.split(sep, maxsplit)

    def find(s, sub, start=None, end=None):
        return s.find(sub, start, end)

    def replace(s, old, new, maxsplit=None):
        if maxsplit is None:
            return s.replace(old, new)
        else:
            return s.replace(old, new, maxsplit)

    def ljust(s, width):
        return s.ljust(width)

    def zfill(s, width):
        return s.zfill(width)

    
