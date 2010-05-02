# -*- python -*-
#                           Package   : omniidl
# interactive.py            Created on: 2001/10/18
#			    Author    : Duncan Grisby (dpg1)
#
#    Copyright (C) 2001 AT&T Laboratories Cambridge
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
#   Back end that just drops into the interactive interpreter.

"""Interactive back-end."""

import _omniidl
from omniidl import idlast

def run(tree, args):
    idlast.tree = tree
    _omniidl.runInteractiveLoop()
    del idlast.tree
