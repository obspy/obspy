# -*- python -*-
#                           Package   : omniidl
# python.py                 Created on: 1999/10/29
#			    Author    : Duncan Grisby (dpg1)
#
#    Copyright (C) 2002-2008 Apasphere Ltd
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
#   Back-end for Python

# $Id: python.py,v 1.33.2.15 2009/05/06 16:50:25 dgrisby Exp $
# $Log: python.py,v $
# Revision 1.33.2.15  2009/05/06 16:50:25  dgrisby
# Updated copyright.
#
# Revision 1.33.2.14  2008/02/01 16:29:17  dgrisby
# Error with implementation of operations with names clashing with
# Python keywords.
#
# Revision 1.33.2.13  2006/10/11 17:44:14  dgrisby
# None is not a keyword, but it cannot be assigned to.
#
# Revision 1.33.2.12  2006/09/29 16:48:03  dgrisby
# Stub changes broke use of package prefix. Thanks Teemu Torma.
#
# Revision 1.33.2.11  2006/09/07 15:28:57  dgrisby
# Remove obsolete check for presence of omniORB.StructBase.
#
# Revision 1.33.2.10  2006/06/21 14:46:26  dgrisby
# Invalid generated code for structs nested inside valuetypes.
#
# Revision 1.33.2.9  2006/01/19 17:28:44  dgrisby
# Merge from omnipy2_develop.
#
# Revision 1.33.2.8  2006/01/18 19:25:13  dgrisby
# Bug inheriting a valuetype from a typedef.
#
# Revision 1.33.2.7  2005/07/29 11:21:36  dgrisby
# Fix long-standing problem with module re-opening by #included files.
#
# Revision 1.33.2.6  2005/01/07 00:22:34  dgrisby
# Big merge from omnipy2_develop.
#
# Revision 1.33.2.5  2004/03/24 22:28:50  dgrisby
# TypeCodes / truncation for inherited state members were broken.
#
# Revision 1.33.2.4  2004/02/16 10:14:18  dgrisby
# Use stream based copy for local calls.
#
# Revision 1.33.2.3  2003/07/10 22:13:25  dgrisby
# Abstract interface support.
#
# Revision 1.33.2.2  2003/05/20 17:10:24  dgrisby
# Preliminary valuetype support.
#
# Revision 1.33.2.1  2003/03/23 21:51:56  dgrisby
# New omnipy3_develop branch.
#
# Revision 1.29.2.14  2002/11/25 21:31:09  dgrisby
# Friendly error messages with file errors, remove code to kill POA
# modules from pre-1.0.
#
# Revision 1.29.2.13  2002/07/04 13:14:52  dgrisby
# Bug with string escapes in Windows filenames.
#
# Revision 1.29.2.12  2002/05/27 01:02:37  dgrisby
# Fix bug with scope lookup in generated code. Fix TypeCode clean-up bug.
#
# Revision 1.29.2.11  2002/01/18 17:41:17  dpg1
# Support for "docstrings" in IDL.
#
# Revision 1.29.2.10  2002/01/18 15:49:45  dpg1
# Context support. New system exception construction. Fix None call problem.
#
# Revision 1.29.2.9  2001/12/04 12:17:08  dpg1
# Incorrect generated code for fixed constants.
#
# Revision 1.29.2.8  2001/08/29 11:57:16  dpg1
# Const fixes.
#
# Revision 1.29.2.7  2001/06/15 10:59:26  dpg1
# Apply fixes from omnipy1_develop.
#
# Revision 1.29.2.6  2001/06/13 11:29:04  dpg1
# Proper omniidl support for wchar/wstring constants.
#
# Revision 1.29.2.5  2001/05/10 15:16:03  dpg1
# Big update to support new omniORB 4 internals.
#
# Revision 1.29.2.4  2001/04/09 15:22:16  dpg1
# Fixed point support.
#
# Revision 1.29.2.3  2000/11/22 14:43:58  dpg1
# Support code set conversion and wchar/wstring.
#
# Revision 1.29.2.2  2000/11/01 15:29:01  dpg1
# Support for forward-declared structs and unions
# RepoIds in indirections are now resolved at the time of use
#
# Revision 1.29.2.1  2000/10/13 13:55:30  dpg1
# Initial support for omniORB 4.
#
# Revision 1.29  2000/10/02 17:34:58  dpg1
# Merge for 1.2 release
#
# Revision 1.27.2.3  2000/08/22 11:52:28  dpg1
# Generate inherited classes for typedef to struct/union.
#
# Revision 1.27.2.2  2000/08/07 09:19:24  dpg1
# Long long support
#
# Revision 1.27.2.1  2000/07/18 15:31:29  dpg1
# Bug with inheritance from typedef
#
# Revision 1.27  2000/07/12 14:32:13  dpg1
# New no_package option to omniidl backend
#
# Revision 1.26  2000/06/28 12:47:48  dpg1
# Proper error messages for unsupported IDL constructs.
#
# Revision 1.25  2000/06/27 15:01:48  dpg1
# Change from POA_M to M__POA mapping.
# Global module only built if necessary.
#
# Revision 1.24  2000/03/29 10:15:47  dpg1
# Exceptions now more closely follow the interface of
# exceptions.Exception.
#
# Revision 1.23  2000/03/17 12:28:09  dpg1
# Comma missing in nested union descriptor.
#
# Revision 1.22  2000/03/03 17:41:28  dpg1
# Major reorganisation to support omniORB 3.0 as well as 2.8.
#
# Revision 1.21  2000/02/23 10:20:52  dpg1
# Bug in descriptors for single-item enums.
#
# Revision 1.20  2000/01/04 15:29:41  dpg1
# Fixes to modules generated within a package.
#
# Revision 1.19  1999/12/21 16:06:15  dpg1
# DOH!  global= not module= !
#
# Revision 1.18  1999/12/21 16:05:11  dpg1
# New module= option.
#
# Revision 1.17  1999/12/17 11:39:52  dpg1
# New arguments to put modules and stubs in a specified package.
#
# Revision 1.16  1999/12/15 11:32:42  dpg1
# -Wbinline option added.
#
# Revision 1.15  1999/12/09 14:12:55  dpg1
# invokeOp() calls now on a single line. typedef now generates a class
# to be passed to CORBA.id().
#
# Revision 1.14  1999/12/07 15:35:14  dpg1
# Bug in currentScope handling.
#
# Revision 1.13  1999/11/30 10:41:20  dpg1
# Back-ends can now have their own usage string.
#
# Revision 1.12  1999/11/25 11:49:31  dpg1
# Minor version number bumped since server-side _is_a() required an
# incompatible change.
#
# Revision 1.11  1999/11/25 11:21:36  dpg1
# Proper support for server-side _is_a().
#
# Revision 1.10  1999/11/19 11:03:49  dpg1
# Extremely important spelling correction in a comment. :-)
#
# Revision 1.9  1999/11/12 15:53:48  dpg1
# New functions omniORB.importIDL() and omniORB.importIDLString().
#
# Revision 1.8  1999/11/11 15:55:29  dpg1
# Python back-end interface now supports valuetype declarations.
# Back-ends still don't support them, though.
#
# Revision 1.7  1999/11/10 16:08:22  dpg1
# Some types weren't registered properly.
#
# Revision 1.6  1999/11/04 11:46:12  dpg1
# Now uses our own version of the GNU C preprocessor.
#
# Revision 1.5  1999/11/02 12:17:26  dpg1
# Top-level module name now has a prefix of _0_ to avoid clashes with
# names of nested declarations.
#
# Revision 1.4  1999/11/02 10:54:01  dpg1
# Two small bugs in union generation.
#
# Revision 1.3  1999/11/02 10:01:46  dpg1
# Minor fixes.
#
# Revision 1.2  1999/11/01 20:19:55  dpg1
# Support for union switch types declared inside the switch statement.
#
# Revision 1.1  1999/11/01 16:40:11  dpg1
# First revision with new front-end.
#

"""omniORB Python bindings"""

from omniidl import idlast, idltype, idlutil, idlvisitor, output, main
import sys, string, types, os.path, keyword

cpp_args = ["-D__OMNIIDL_PYTHON__"]
usage_string = """\
  -Wbstdout       Send generated stubs to stdout rather than a file
  -Wbinline       Output stubs for #included files in line with the main file
  -Wbfactories    Register value factories for all valuetypes
  -Wbpackage=p    Put both Python modules and stub files in package p
  -Wbmodules=p    Put Python modules in package p
  -Wbstubs=p      Put stub files in package p
  -Wbextern=f:p   Assume Python stub file for file f is in package p.
  -Wbglobal=g     Module to use for global IDL scope (default _GlobalIDL)"""

#""" Uncomment this line to get syntax highlighting on the output strings

# Output strings

pymodule_template = """\
# DO NOT EDIT THIS FILE!
#
# Python module @module@ generated by omniidl

import omniORB
omniORB.updateModule("@package@@module@")

# ** 1. Stub files contributing to this module

# ** 2. Sub-modules

# ** 3. End"""

file_start = """\
# Python stubs generated by omniidl from @filename@

import omniORB, _omnipy
from omniORB import CORBA, PortableServer
_0_CORBA = CORBA

_omnipy.checkVersion(3,0, __file__)
"""

file_end = """\
_exported_modules = ( @export_string@)

# The end."""

module_start = """
#
# Start of module "@sname@"
#
__name__ = "@package@@sname@"
_0_@sname@ = omniORB.openModule("@package@@sname@", r"@filename@")
_0_@s_sname@ = omniORB.openModule("@package@@s_sname@", r"@filename@")
"""

module_end = """
#
# End of module "@sname@"
#
__name__ = "@package@@modname@"
"""

import_idl_file = """\
# #include "@idlfile@"
import @ifilename@"""

open_imported_module_name = """\
_0_@imodname@ = omniORB.openModule("@package@@imodname@")
_0_@s_imodname@ = omniORB.openModule("@package@@s_imodname@")"""

forward_interface = """\

# @abstract@interface @ifid@;
_0_@modname@._d_@ifid@ = (omniORB.tcInternal.@tvkind@, "@repoId@", "@ifid@")
omniORB.typeMapping["@repoId@"] = _0_@modname@._d_@ifid@"""


interface_class = """\

# @abstract@interface @ifid@
_0_@modname@._d_@ifid@ = (omniORB.tcInternal.@tvkind@, "@repoId@", "@ifid@")
omniORB.typeMapping["@repoId@"] = _0_@modname@._d_@ifid@
_0_@modname@.@ifid@ = omniORB.newEmptyClass()
class @ifid@ @inherits@:
    _NP_RepositoryId = _0_@modname@._d_@ifid@[1]

    def __init__(self, *args, **kw):
        raise RuntimeError("Cannot construct objects of this type.")

    _nil = CORBA.Object._nil
"""

interface_descriptor = """
_0_@modname@.@ifid@ = @ifid@
_0_@modname@._tc_@ifid@ = omniORB.tcInternal.createTypeCode(_0_@modname@._d_@ifid@)
omniORB.registerType(@ifid@._NP_RepositoryId, _0_@modname@._d_@ifid@, _0_@modname@._tc_@ifid@)"""

callables_header = """
# @ifid@ operations and attributes"""

attribute_get_descriptor = """\
@ifid@._d__get_@attr@ = ((),(@descr@,),None)"""

attribute_set_descriptor = """\
@ifid@._d__set_@attr@ = ((@descr@,),(),None)"""

operation_descriptor = """\
@ifid@._d_@opname@ = (@inds@, @outds@, @excs@@options@)"""

objref_class = """\

# @ifid@ object reference
class _objref_@ifid@ (@inherits@):
    _NP_RepositoryId = @ifid@._NP_RepositoryId

    def __init__(self):"""

objref_inherit_init = """\
        @inclass@.__init__(self)"""

objref_object_init = """\
        CORBA.Object.__init__(self)"""

objref_attribute_get = """
    def _get_@attr@(self, *args):
        return _omnipy.invoke(self, "_get_@attr@", _0_@modname@.@ifid@._d__get_@attr@, args)"""
objref_attribute_set = """
    def _set_@attr@(self, *args):
        return _omnipy.invoke(self, "_set_@attr@", _0_@modname@.@ifid@._d__set_@attr@, args)"""
objref_operation = """
    def @opname@(self, *args):
        return _omnipy.invoke(self, "@r_opname@", _0_@modname@.@ifid@._d_@opname@, args)"""
objref_methods = """
    __methods__ = @methods@"""

objref_register = """
omniORB.registerObjref(@ifid@._NP_RepositoryId, _objref_@ifid@)
_0_@modname@._objref_@ifid@ = _objref_@ifid@
del @ifid@, _objref_@ifid@"""

skeleton_class = """
# @ifid@ skeleton
__name__ = "@package@@s_modname@"
class @ifid@ (@inherits@):
    _NP_RepositoryId = _0_@modname@.@ifid@._NP_RepositoryId
"""

skeleton_methodmap = """
    _omni_op_d = @methodmap@"""

skeleton_inheritmap = """\
    _omni_op_d.update(@inheritclass@._omni_op_d)"""

skeleton_end = """
@ifid@._omni_skeleton = @ifid@
_0_@s_modname@.@ifid@ = @ifid@
omniORB.registerSkeleton(@ifid@._NP_RepositoryId, @ifid@)
del @ifid@
__name__ = "@package@@modname@"\
"""

skeleton_set_skel = """
@ifid@._omni_skeleton = @ifid@
"""

constant_at_module_scope = """\
_0_@modname@.@cname@ = @value@"""

constant = """\
@cname@ = @value@"""

typedef_header = """\

# typedef ... @tdname@
class @tdname@:
    _NP_RepositoryId = "@repoId@"
    def __init__(self, *args, **kw):
        raise RuntimeError("Cannot construct objects of this type.")"""

typedef_struct_union_header = """\

# typedef ... @tdname@
class @tdname@ (@parent@):
    _NP_RepositoryId = "@repoId@"
"""

typedef_fixed_header = """\
# typedef ... @tdname@
@tdname@ = omniORB.fixedConstructor("@repoId@", @digits@, @scale@)"""

typedef_at_module_scope = """\
_0_@modname@.@tdname@ = @tdname@
_0_@modname@._d_@tdname@  = @desc@
_0_@modname@._ad_@tdname@ = (omniORB.tcInternal.tv_alias, @tdname@._NP_RepositoryId, "@tdname@", @tddesc@)
_0_@modname@._tc_@tdname@ = omniORB.tcInternal.createTypeCode(_0_@modname@._ad_@tdname@)
omniORB.registerType(@tdname@._NP_RepositoryId, _0_@modname@._ad_@tdname@, _0_@modname@._tc_@tdname@)
del @tdname@"""

typedef = """\
_d_@tdname@  = @desc@
_ad_@tdname@ = (omniORB.tcInternal.tv_alias, @tdname@._NP_RepositoryId, "@tdname@", @tddesc@)
_tc_@tdname@ = omniORB.tcInternal.createTypeCode(_ad_@tdname@)
omniORB.registerType(@tdname@._NP_RepositoryId, _ad_@tdname@, _tc_@tdname@)"""

forward_struct_descr_at_module_scope = """
# Forward struct @sname@
_0_@modname@._d_@sname@ = (omniORB.tcInternal.tv__indirect, ["@repoId@"])
omniORB.typeMapping["@repoId@"] = _0_@modname@._d_@sname@"""

forward_struct_descr = """
# Forward struct @sname@
_d_@sname@ = (omniORB.tcInternal.tv__indirect, ["@repoId@"])
omniORB.typeMapping["@repoId@"] = _d_@sname@"""

recursive_struct_descr_at_module_scope = """
# Recursive struct @sname@
_0_@modname@._d_@sname@ = (omniORB.tcInternal.tv__indirect, ["@repoId@"])
omniORB.typeMapping["@repoId@"] = _0_@modname@._d_@sname@"""

recursive_struct_descr = """
# Recursive struct @sname@
_d_@sname@ = (omniORB.tcInternal.tv__indirect, ["@repoId@"])
_0_@scope@._d_@sname@ = _d_@sname@
omniORB.typeMapping["@repoId@"] = _d_@sname@"""

struct_class = """
# struct @sname@
_0_@scopedname@ = omniORB.newEmptyClass()
class @sname@ (omniORB.StructBase):
    _NP_RepositoryId = "@repoId@"
"""

struct_class_name = """\
    _NP_ClassName = "@cname@"
"""

struct_class_init = """\
    def __init__(self@mnames@):"""

struct_init_member = """\
        self.@mname@ = @mname@"""

struct_descriptor_at_module_scope = """\

_0_@modname@.@sname@ = @sname@
_0_@modname@._d_@sname@  = (omniORB.tcInternal.tv_struct, @sname@, @sname@._NP_RepositoryId, "@sname@"@mdescs@)"""

struct_register_at_module_scope = """\
_0_@modname@._tc_@sname@ = omniORB.tcInternal.createTypeCode(_0_@modname@._d_@sname@)
omniORB.registerType(@sname@._NP_RepositoryId, _0_@modname@._d_@sname@, _0_@modname@._tc_@sname@)
del @sname@"""

struct_descriptor = """\

_d_@sname@  = _0_@scope@._d_@sname@ = (omniORB.tcInternal.tv_struct, @sname@, @sname@._NP_RepositoryId, "@sname@"@mdescs@)"""

struct_register = """\
_tc_@sname@ = omniORB.tcInternal.createTypeCode(_d_@sname@)
omniORB.registerType(@sname@._NP_RepositoryId, _d_@sname@, _tc_@sname@)"""

struct_module_descriptors = """
_0_@modname@._d_@sname@  = _d_@sname@
_0_@modname@._tc_@sname@ = _tc_@sname@
del @sname@, _d_@sname@, _tc_@sname@"""

exception_class = """\

# exception @sname@
_0_@scopedname@ = omniORB.newEmptyClass()
class @sname@ (CORBA.UserException):
    _NP_RepositoryId = "@repoId@"
"""

exception_class_init = """\
    def __init__(self@mnames@):
        CORBA.UserException.__init__(self@mnames@)"""

exception_init_member = """\
        self.@mname@ = @mname@"""

exception_descriptor_at_module_scope = """\

_0_@modname@.@sname@ = @sname@
_0_@modname@._d_@sname@  = (omniORB.tcInternal.tv_except, @sname@, @sname@._NP_RepositoryId, "@sname@"@mdescs@)
_0_@modname@._tc_@sname@ = omniORB.tcInternal.createTypeCode(_0_@modname@._d_@sname@)
omniORB.registerType(@sname@._NP_RepositoryId, _0_@modname@._d_@sname@, _0_@modname@._tc_@sname@)
del @sname@"""

exception_descriptor = """\

_d_@sname@  = (omniORB.tcInternal.tv_except, @sname@, @sname@._NP_RepositoryId, "@sname@"@mdescs@)
_tc_@sname@ = omniORB.tcInternal.createTypeCode(_d_@sname@)
omniORB.registerType(@sname@._NP_RepositoryId, _d_@sname@, _tc_@sname@)"""

forward_union_descr_at_module_scope = """
# Forward union @uname@
_0_@modname@._d_@uname@ = (omniORB.tcInternal.tv__indirect, ["@repoId@"])
omniORB.typeMapping["@repoId@"] = _0_@modname@._d_@uname@"""

forward_union_descr = """
# Forward union @uname@
_d_@uname@ = (omniORB.tcInternal.tv__indirect, ["@repoId@"])
omniORB.typeMapping["@repoId@"] = _d_@uname@"""

recursive_union_descr_at_module_scope = """
# Recursive union @uname@
_0_@modname@._d_@uname@ = (omniORB.tcInternal.tv__indirect, ["@repoId@"])
omniORB.typeMapping["@repoId@"] = _0_@modname@._d_@uname@"""

recursive_union_descr = """
# Recursive union @uname@
_d_@uname@ = (omniORB.tcInternal.tv__indirect, ["@repoId@"])
_0_@scope@._d_@uname@ = _d_@uname@
omniORB.typeMapping["@repoId@"] = _d_@uname@"""

union_class = """
# union @uname@
_0_@scopedname@ = omniORB.newEmptyClass()
class @uname@ (omniORB.Union):
    _NP_RepositoryId = "@repoId@"\
"""

union_class_name = """\
    _NP_ClassName = "@cname@"
"""

union_descriptor_at_module_scope = """
_0_@modname@.@uname@ = @uname@

@uname@._m_to_d = {@m_to_d@}
@uname@._d_to_m = {@d_to_m@}
@uname@._def_m  = @def_m@
@uname@._def_d  = @def_d@

_0_@modname@._m_@uname@  = (@m_un@,)
_0_@modname@._d_@uname@  = (omniORB.tcInternal.tv_union, @uname@, @uname@._NP_RepositoryId, "@uname@", @stype@, @defpos@, _0_@modname@._m_@uname@, @m_def@, {@d_map@})"""

union_register_at_module_scope = """\
_0_@modname@._tc_@uname@ = omniORB.tcInternal.createTypeCode(_0_@modname@._d_@uname@)
omniORB.registerType(@uname@._NP_RepositoryId, _0_@modname@._d_@uname@, _0_@modname@._tc_@uname@)
del @uname@"""

union_descriptor = """
@uname@._m_to_d = {@m_to_d@}
@uname@._d_to_m = {@d_to_m@}
@uname@._def_m  = @def_m@
@uname@._def_d  = @def_d@

_m_@uname@  = (@m_un@,)
_d_@uname@  = _0_@scope@._d_@uname@ = (omniORB.tcInternal.tv_union, @uname@, @uname@._NP_RepositoryId, "@uname@", @stype@, @defpos@, _m_@uname@, @m_def@, {@d_map@})"""

union_register = """\
_tc_@uname@ = omniORB.tcInternal.createTypeCode(_d_@uname@)
omniORB.registerType(@uname@._NP_RepositoryId, _d_@uname@, _tc_@uname@)"""


enum_start = """
# enum @ename@\
"""

enum_item_at_module_scope = """\
_0_@modname@.@eitem@ = omniORB.EnumItem("@item@", @eval@)"""

enum_object_and_descriptor_at_module_scope = """\
_0_@modname@.@ename@ = omniORB.Enum("@repoId@", (@eitems@,))

_0_@modname@._d_@ename@  = (omniORB.tcInternal.tv_enum, _0_@modname@.@ename@._NP_RepositoryId, "@ename@", _0_@modname@.@ename@._items)
_0_@modname@._tc_@ename@ = omniORB.tcInternal.createTypeCode(_0_@modname@._d_@ename@)
omniORB.registerType(_0_@modname@.@ename@._NP_RepositoryId, _0_@modname@._d_@ename@, _0_@modname@._tc_@ename@)"""

enum_item = """\
@eitem@ = omniORB.EnumItem("@item@", @eval@)"""

enum_object_and_descriptor = """\
@ename@ = omniORB.Enum("@repoId@", (@eitems@,))

_d_@ename@  = (omniORB.tcInternal.tv_enum, @ename@._NP_RepositoryId, "@ename@", @ename@._items)
_tc_@ename@ = omniORB.tcInternal.createTypeCode(_d_@ename@)
omniORB.registerType(@ename@._NP_RepositoryId, _d_@ename@, _tc_@ename@)"""


value_forward_at_module_scope = """\
# forward valuetype @vname@
_0_@modname@._d_@vname@ = (omniORB.tcInternal.tv__indirect, ["@repoId@"])
omniORB.typeMapping["@repoId@"] = _0_@modname@._d_@vname@
"""

value_class = """
# valuetype @vname@
_0_@modname@._d_@vname@ = (omniORB.tcInternal.tv__indirect, ["@repoId@"])
omniORB.typeMapping["@repoId@"] = _0_@modname@._d_@vname@
_0_@modname@.@vname@ = omniORB.newEmptyClass()

class @vname@ (@inherits@):
    _NP_RepositoryId = "@repoId@"

    def __init__(self, *args, **kwargs):
        if args:
            if len(args) != @arglen@:
                raise TypeError("@vname@() takes @arglen@ argument@s@ "
                                "(%d given)" % len(args))
            @set_args@
        if kwargs:
            self.__dict__.update(kwargs)
"""

valueabs_class = """\
class @vname@ (@inherits@):
    _NP_RepositoryId = "@repoId@"

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Cannot construct objects of this type.")
"""

value_register_factory = """\
omniORB.registerValueFactory(@vname@._NP_RepositoryId, @vname@)
"""

value_descriptor_at_module_scope = """\
_0_@modname@.@vname@ = @vname@
_0_@modname@._d_@vname@  = (omniORB.tcInternal.tv_value, @vname@, @vname@._NP_RepositoryId, "@vname@", @modifier@, @tbaseids@, @basedesc@, @mdescs@)
_0_@modname@._tc_@vname@ = omniORB.tcInternal.createTypeCode(_0_@modname@._d_@vname@)
omniORB.registerType(@vname@._NP_RepositoryId, _0_@modname@._d_@vname@, _0_@modname@._tc_@vname@)
del @vname@
"""

value_objref_register = """
omniORB.registerObjref(@ifid@._NP_RepositoryId, _objref_@ifid@)
_0_@modname@._objref_@ifid@ = _objref_@ifid@
del _objref_@ifid@"""


valuebox = """\

# valuebox @boxname@
class @boxname@:
    _NP_RepositoryId = "@repoId@"
    def __init__(self, *args, **kw):
        raise RuntimeError("Cannot construct objects of this type.")

_0_@modname@.@boxname@ = @boxname@
_0_@modname@._d_@boxname@  = (omniORB.tcInternal.tv_value_box, @boxname@, @boxname@._NP_RepositoryId, "@boxname@", @boxdesc@)
_0_@modname@._tc_@boxname@ = omniORB.tcInternal.createTypeCode(_0_@modname@._d_@boxname@)
omniORB.registerType(@boxname@._NP_RepositoryId, _0_@modname@._d_@boxname@, _0_@modname@._tc_@boxname@)
omniORB.registerValueFactory(@boxname@._NP_RepositoryId, @boxname@)
del @boxname@
"""


example_start = """\
#!/usr/bin/env python

# Python example implementations generated from @filename@

import CORBA, PortableServer

# Import the Python stub modules so type definitions are available.
"""

example_import_skels = """
# Import the Python Skeleton modules so skeleton base classes are available.
"""

example_import = """\
import @module@"""

example_classdef = """

# Implementation of interface @ccname@

class @ifname@_i (@skname@):
@inheritance_note@
    def __init__(self):
        # Initialise member variables here
        pass
"""

example_opdef = """\
    # @signature@
    def @opname@(self@args@):
        raise CORBA.NO_IMPLEMENT(0, CORBA.COMPLETED_NO)
        # *** Implement me
        # Must return: @returnspec@
"""

example_end = """
if __name__ == "__main__":
    import sys
    
    # Initialise the ORB
    orb = CORBA.ORB_init(sys.argv)
    
    # As an example, we activate an object in the Root POA
    poa = orb.resolve_initial_references("RootPOA")

    # Create an instance of a servant class
    servant = @ifname@_i()

    # Activate it in the Root POA
    poa.activate_object(servant)

    # Get the object reference to the object
    objref = servant._this()
    
    # Print a stringified IOR for it
    print orb.object_to_string(objref)

    # Activate the Root POA's manager
    poa._get_the_POAManager().activate()

    # Run the ORB, blocking this thread
    orb.run()
"""


# Global state
imported_files   = {}
exported_modules = {}

# Command line options
output_inline    = 0
global_module    = "_GlobalIDL"
module_package   = ""
stub_package     = ""
stub_directory   = ""
all_factories    = 0
example_impl     = 0
extern_stub_pkgs = {}


def error_exit(message):
    sys.stderr.write(main.cmdname + ": " + message + "\n")
    sys.exit(1)

def run(tree, args):
    global main_idl_file, imported_files, exported_modules, output_inline
    global global_module, module_package, stub_package, stub_directory
    global all_factories, example_impl, extern_stub_pkgs

    imported_files.clear()
    exported_modules.clear()

    # Look at the args:
    use_stdout     = 0
    create_package = 1
    for arg in args:

        if arg == "stdout":
            use_stdout     = 1
            create_package = 0

        elif arg == "no_package":
            create_package = 0

        elif arg == "inline":
            output_inline = 1

        elif arg == "factories":
            all_factories = 1

        elif arg[:8] == "modules=":
            module_package = arg[8:]
            if module_package != "":
                module_package = module_package + "."

        elif arg[:6] == "stubs=":
            stub_package   = arg[6:]
            stub_directory = apply(os.path.join,
                                   string.split(stub_package, "."))
            if stub_package != "":
                stub_package = stub_package + "."

        elif arg[:8] == "package=":
            module_package = stub_package = arg[8:]
            stub_directory = apply(os.path.join,
                                   string.split(stub_package, "."))
            if module_package != "":
                module_package = stub_package = module_package + "."

        elif arg[:7] == "global=":
            global_module = arg[7:]
            if global_module == "":
                error_exit("You may not have an unnamed global module.")

        elif arg == "example":
            example_impl = 1

        elif arg[:7] == "extern=":
            f_p = string.split(arg[7:], ":", 1)
            if len(f_p) == 1:
                extern_stub_pkgs[f_p[0]] = None
            else:
                extern_stub_pkgs[f_p[0]] = f_p[1]

        else:
            sys.stderr.write(main.cmdname + ": Warning: Python " \
                             "back-end does not understand argument: " + \
                             arg + "\n")

    main_idl_file = tree.file()

    outpybasename = outputFileName(main_idl_file)
    outpymodule   = stub_package + outpybasename
    outpyname     = os.path.join(stub_directory, outpybasename + ".py")

    imported_files[outpybasename] = 1

    if create_package:
        checkStubPackage(stub_package)

    if use_stdout:
        st = output.Stream(sys.stdout, 4)
    else:
        try:
            st = output.Stream(open(outpyname, "w"), 4)
        except IOError:
            error_exit('Cannot open "%s" for writing.' % outpyname)

    st.out(file_start, filename=main_idl_file)

    pv = PythonVisitor(st, outpymodule)
    tree.accept(pv)

    dv = DocstringVisitor(st)
    tree.accept(dv)
    dv.output()

    exports = exported_modules.keys()
    exports.sort()
    export_list = map(lambda s: '"' + module_package + s + '"', exports)
    if len(export_list) == 1: export_list.append("")
    export_string = string.join(export_list, ", ")

    st.out(file_end, export_string=export_string)

    if create_package:
        updateModules(exports, outpymodule)

    if example_impl:
        implname = os.path.join(stub_directory, outpybasename + "_example.py")
        exst = output.Stream(open(implname, "w"), 4)
        exst.out(example_start, filename=main_idl_file)
        for mod in exports:
            exst.out(example_import, module=mod)
        exst.out(example_import_skels)
        for mod in exports:
            exst.out(example_import, module=skeletonModuleName(mod))

        ev = ExampleVisitor(exst)
        tree.accept(ev)

        exst.out(example_end, ifname=ev.first)


class PythonVisitor:
    def __init__(self, st, outpymodule):
        self.st          = st
        self.outpymodule = outpymodule

    def handleImported(self, node):
        global imported_files

        if node.mainFile() or output_inline:
            return 0
        else:
            ifilename = outputFileName(node.file())
            if not imported_files.has_key(ifilename):
                imported_files[ifilename] = 1
                ibasename,ext = os.path.splitext(os.path.basename(node.file()))
                if extern_stub_pkgs.has_key(ibasename):
                    ipackage = extern_stub_pkgs[ibasename]
                    if ipackage:
                        fn = ipackage + '.' + ifilename
                    else:
                        fn = ifilename
                else:
                    fn = stub_package + ifilename

                self.st.out(import_idl_file,
                            idlfile=node.file(),
                            ifilename=fn)
            return 1
        
    #
    # The global module
    #
    def visitAST(self, node):
        self.at_module_scope = 1
        self.at_global_scope = 1
        self.currentScope    = ["_0_" + global_module]

        decls_in_global_module = 0

        for n in node.declarations():
            if not isinstance(n, idlast.Module):
                decls_in_global_module = 1
                break

        if decls_in_global_module:
            self.modname = global_module
            self.st.out(module_start,
                        sname=global_module,
                        s_sname=skeletonModuleName(global_module),
                        filename=node.file(), package=module_package)
        else:
            self.modname = self.outpymodule

        for n in node.declarations():
            n.accept(self)

        if decls_in_global_module:
            exported_modules[global_module] = 1
            self.st.out(module_end, modname=self.outpymodule,
                        sname=global_module,
                        package="")

    #
    # Module
    #
    def visitModule(self, node):
        if self.handleImported(node):
            imodname = dotName(node.scopedName())
            ibasename,ext = os.path.splitext(os.path.basename(node.file()))

            if extern_stub_pkgs.has_key(ibasename):
                package = extern_stub_pkgs[ibasename]
                if package is None:
                    package = ""
                else:
                    package = package + "."
            else:
                package = module_package

            self.st.out(open_imported_module_name,
                        imodname=imodname,
                        s_imodname=skeletonModuleName(imodname),
                        package=package)

        assert self.at_module_scope

        sname = dotName(node.scopedName())

        if node.mainFile() or output_inline:
            self.st.out(module_start,
                        sname = sname,
                        s_sname = skeletonModuleName(sname),
                        filename = node.file(), package=module_package)

        parentmodname = self.modname
        self.modname  = dotName(node.scopedName())

        ags = self.at_global_scope
        if ags:
            self.currentScope = ["_0_" + node.identifier()]
        else:
            self.currentScope.append(node.identifier())

        self.at_global_scope = 0

        for n in node.definitions():
            n.accept(self)

        if ags:
            self.currentScope = ["_0_" + global_module]
        else:
            self.currentScope.pop()
        self.at_global_scope = ags
        self.modname         = parentmodname

        if node.mainFile() or output_inline:
            exported_modules[sname] = 1
            self.st.out(module_end, modname=parentmodname, sname=sname,
                        package=module_package)

    #
    # Forward interface
    #
    def visitForward(self, node):
        if self.handleImported(node): return

        assert self.at_module_scope
        ifid   = mangle(node.identifier())
        repoId = node.repoId()
        if node.abstract():
            tvkind = "tv_abstract_interface"
            abstract = "abstract "
        else:
            tvkind = "tv_objref"
            abstract = ""

        self.st.out(forward_interface, ifid=ifid, tvkind=tvkind,
                    repoId=repoId, abstract=abstract, modname=self.modname)

    #
    # Interface
    #
    def visitInterface(self, node):
        if self.handleImported(node): return

        assert self.at_module_scope
        ifid = mangle(node.identifier())

        if len(node.inherits()) > 0:
            inheritl = []
            for i in node.inherits():
                i = i.fullDecl()
                inheritl.append(dotName(fixupScopedName(i.scopedName())))
            
            inherits = "(" + string.join(inheritl, ", ") + ")"
        else:
            inherits = ""

        if node.abstract():
            tvkind = "tv_abstract_interface"
            abstract = "abstract "
        else:
            tvkind = "tv_objref"
            abstract = ""

        # Class header
        self.st.out(interface_class, ifid=ifid, tvkind=tvkind,
                    inherits=inherits, repoId=node.repoId(),
                    abstract=abstract, modname=self.modname)

        # Declarations within the interface
        if len(node.declarations()) > 0:
            self.st.inc_indent()
            self.at_module_scope = 0
            self.currentScope.append(node.identifier())

            for d in node.declarations():
                d.accept(self)

            self.currentScope.pop()
            self.at_module_scope = 1
            self.st.dec_indent()
            self.st.out("")

        # Descriptor and TypeCode for the interface
        self.st.out(interface_descriptor,
                    modname = self.modname, ifid = ifid)

        # Attributes and operations
        # *** Was there a good reason for putting these in the class def?
        if len(node.callables()) > 0:
            self.st.out(callables_header, ifid=ifid)

        for c in node.callables():
            if isinstance(c, idlast.Attribute):

                descr = typeToDescriptor(c.attrType())

                for attr in c.identifiers():

                    self.st.out(attribute_get_descriptor,
                                attr=attr, descr=descr, ifid=ifid)

                    if not c.readonly():

                        self.st.out(attribute_set_descriptor,
                                    attr=attr, descr=descr, ifid=ifid)
            else: # Operation

                inds, outds, excs, ctxts, cv = operationToDescriptors(c)

                options = ""

                if cv:
                    ctxts = ctxts or "None"

                if ctxts:
                    options = ", " + ctxts

                if cv:
                    options = options + ", 1"

                # Output the declaration
                self.st.out(operation_descriptor,
                            opname  = mangle(c.identifier()),
                            inds    = inds,
                            outds   = outds,
                            excs    = excs,
                            options = options,
                            ifid    = ifid)

        # Objref class
        if node.inherits():
            inheritl = []
            for i in node.inherits():
                i = i.fullDecl()
                sn = fixupScopedName(i.scopedName())
                inheritl.append(dotName(sn[:-1] + ["_objref_" + sn[-1]]))
                
            inherits = string.join(inheritl, ", ")
        else:
            inherits = "CORBA.Object"

        self.st.out(objref_class, ifid=ifid, inherits=inherits)

        if node.inherits():
            for inclass in inheritl:
                self.st.out(objref_inherit_init, inclass=inclass)
        else:
            self.st.out(objref_object_init)

        # Operations and attributes
        methodl = []

        for c in node.callables():
            if isinstance(c, idlast.Attribute):

                for attr in c.identifiers():

                    self.st.out(objref_attribute_get,
                                attr    = attr,
                                ifid    = ifid,
                                modname = self.modname)
                    
                    methodl.append('"_get_' + attr + '"')

                    if not c.readonly():

                        self.st.out(objref_attribute_set,
                                    attr    = attr,
                                    ifid    = ifid,
                                    modname = self.modname)
                        
                        methodl.append('"_set_' + attr + '"')

            else: # Operation
                opname = mangle(c.identifier())
                
                self.st.out(objref_operation,
                            opname   = opname,
                            r_opname = c.identifier(),
                            ifid     = ifid,
                            modname  = self.modname)
                
                methodl.append('"' + opname + '"')

        # __methods__ assignment
        methods = "[" + string.join(methodl, ", ") + "]"

        if node.inherits():
            inheritl = []
            for i in node.inherits():
                i = i.fullDecl()
                sn = fixupScopedName(i.scopedName())
                methods = methods + " + " + \
                          dotName(sn[:-1] + ["_objref_" + sn[-1]]) + \
                          ".__methods__"
        else:
            methods = methods + " + CORBA.Object.__methods__"

        self.st.out(objref_methods, methods = methods)

        # registerObjRef()
        self.st.out(objref_register, ifid = ifid, modname = self.modname)

        # Skeleton class
        if node.inherits():
            inheritl = []
            for i in node.inherits():
                i = i.fullDecl()
                fsn = fixupScopedName(i.scopedName())
                dsn = dotName(fsn)
                ssn = skeletonModuleName(dsn)
                inheritl.append(ssn)
                
            inherits = string.join(inheritl, ", ")
        else:
            inherits = "PortableServer.Servant"

        self.st.out(skeleton_class,
                    ifid      = ifid,
                    inherits  = inherits,
                    modname   = self.modname,
                    s_modname = skeletonModuleName(self.modname),
                    package   = module_package)

        # Operations and attributes
        methodl = []

        for c in node.callables():
            if isinstance(c, idlast.Attribute):

                for attr in c.identifiers():

                    methodl.append('"_get_' + attr + '": ' + \
                                   '_0_' + self.modname + '.' + \
                                   ifid + '.' + '_d__get_' + attr)

                    if not c.readonly():

                        methodl.append('"_set_' + attr + '": ' + \
                                       '_0_' + self.modname + '.' + \
                                       ifid + '.' + '_d__set_' + attr)

            else: # Operation
                opname   = c.identifier()
                m_opname = mangle(opname)
                
                methodl.append('"' + opname + '": ' + '_0_' + self.modname +
                               '.' + ifid + '.' + '_d_' + m_opname)

        methodmap = "{" + string.join(methodl, ", ") + "}"

        self.st.out(skeleton_methodmap, methodmap = methodmap)

        if node.inherits():
            for inheritclass in inheritl:
                self.st.out(skeleton_inheritmap, inheritclass = inheritclass)

        self.st.out(skeleton_end,
                    ifid      = ifid,
                    modname   = self.modname,
                    s_modname = skeletonModuleName(self.modname),
                    package   = module_package)

    #
    # Constant
    #
    def visitConst(self, node):
        if self.handleImported(node): return

        cname = mangle(node.identifier())

        if self.at_module_scope:
            value = valueToString(node.value(), node.constKind(), [])
        else:
            value = valueToString(node.value(), node.constKind(),
                                  self.currentScope)
        if self.at_module_scope:
            self.st.out(constant_at_module_scope,
                        cname   = cname,
                        value   = value,
                        modname = self.modname)
        else:
            self.st.out(constant,
                        cname   = cname,
                        value   = value)

    #
    # Typedef
    #
    def visitTypedef(self, node):
        if self.handleImported(node): return

        if node.constrType():
            node.aliasType().decl().accept(self)

        for decl in node.declarators():
            tdname = mangle(decl.identifier())
            if self.at_module_scope:
                desc   = typeAndDeclaratorToDescriptor(node.aliasType(),
                                                       decl, [])
                tddesc = typeAndDeclaratorToDescriptor(node.aliasType(),
                                                       decl, [], 1)

                unaliased_type = node.aliasType().unalias()

                if len(decl.sizes()) == 0 and \
                   unaliased_type.kind() in [idltype.tk_struct,
                                             idltype.tk_union]:

                    parent = dotName(fixupScopedName(unaliased_type.decl().\
                                                     scopedName()))

                    self.st.out(typedef_struct_union_header,
                                tdname = tdname,
                                repoId = decl.repoId(),
                                parent = parent)

                elif len(decl.sizes()) == 0 and\
                     unaliased_type.kind() == idltype.tk_fixed:

                    self.st.out(typedef_fixed_header,
                                tdname = tdname,
                                repoId = decl.repoId(),
                                digits = unaliased_type.digits(),
                                scale  = unaliased_type.scale())
                    
                else:
                    self.st.out(typedef_header,
                                tdname  = tdname,
                                repoId  = decl.repoId())

                self.st.out(typedef_at_module_scope,
                            tdname  = tdname,
                            desc    = desc,
                            tddesc  = tddesc,
                            modname = self.modname)
            else:
                desc   = typeAndDeclaratorToDescriptor(node.aliasType(),
                                                       decl,
                                                       self.currentScope)
                tddesc = typeAndDeclaratorToDescriptor(node.aliasType(),
                                                       decl,
                                                       self.currentScope, 1)

                unaliased_type = node.aliasType().unalias()

                if len(decl.sizes()) == 0 and \
                   unaliased_type.kind() in [idltype.tk_struct,
                                             idltype.tk_union]:

                    psname  = unaliased_type.decl().scopedName()
                    myscope = decl.scopedName()[:-1]

                    # If the struct/union definition is in the same
                    # scope as the typedef, we must use a relative
                    # name to refer to the parent class, since the
                    # enclosing Python class has not yet been fully
                    # defined.

                    if psname[:len(myscope)] == myscope:
                        parent = dotName(psname[len(myscope):])
                    else:
                        parent = dotName(fixupScopedName(psname))

                    self.st.out(typedef_struct_union_header,
                                tdname = tdname,
                                repoId = decl.repoId(),
                                parent = parent)
                else:
                    self.st.out(typedef_header,
                                tdname  = tdname,
                                repoId  = decl.repoId())

                self.st.out(typedef,
                            tdname  = tdname,
                            desc    = desc,
                            tddesc  = tddesc)
    #
    # Struct
    #
    def visitStruct(self, node):
        if self.handleImported(node): return

        sname = mangle(node.identifier())

        fscopedName = fixupScopedName(node.scopedName(), "")

        if node.recursive():
            if self.at_module_scope:
                self.st.out(recursive_struct_descr_at_module_scope,
                            sname   = sname,
                            repoId  = node.repoId(),
                            modname = self.modname)
            else:
                self.st.out(recursive_struct_descr,
                            sname   = sname,
                            repoId  = node.repoId(),
                            scope   = dotName(fscopedName[:-1]))

        self.st.out(struct_class,
                    sname      = sname,
                    repoId     = node.repoId(),
                    scopedname = dotName(fscopedName))

        if not self.at_module_scope:
            self.st.out(struct_class_name, cname = dotName(fscopedName))

        mnamel = []
        mdescl = []
        for mem in node.members():

            # Deal with nested declarations
            if mem.constrType():
                self.st.inc_indent()
                ams = self.at_module_scope
                self.at_module_scope = 0
                self.currentScope.append(node.identifier())
                
                mem.memberType().decl().accept(self)

                self.currentScope.pop()
                self.at_module_scope = ams
                self.st.dec_indent()
                self.st.out("")

            for decl in mem.declarators():
                mnamel.append(mangle(decl.identifier()))
                mdescl.append('"' + mangle(decl.identifier()) + '"')
                
                if self.at_module_scope:
                    mdescl.append(\
                        typeAndDeclaratorToDescriptor(mem.memberType(),
                                                      decl,
                                                      []))
                else:
                    mdescl.append(\
                        typeAndDeclaratorToDescriptor(mem.memberType(),
                                                      decl,
                                                      self.currentScope))
        if len(mnamel) > 0:
            mnames = ", " + string.join(mnamel, ", ")

            self.st.out(struct_class_init, mnames = mnames)

            for mname in mnamel:
                self.st.out(struct_init_member, mname = mname)

        if len(mdescl) > 0:
            mdescs = ", " + string.join(mdescl, ", ")
        else:
            mdescs = ""

        if self.at_module_scope:
            self.st.out(struct_descriptor_at_module_scope,
                        sname   = sname,
                        mdescs  = mdescs,
                        modname = self.modname)
            
            self.st.out(struct_register_at_module_scope,
                        sname   = sname,
                        modname = self.modname)
        else:
            self.st.out(struct_descriptor,
                        sname  = sname,
                        mdescs = mdescs,
                        scope  = dotName(fscopedName[:-1]))

            self.st.out(struct_register, sname = sname)

    #
    # Forward struct
    #
    def visitStructForward(self, node):
        if self.handleImported(node): return

        sname = mangle(node.identifier())

        if self.at_module_scope:
            self.st.out(forward_struct_descr_at_module_scope,
                        sname   = sname,
                        repoId  = node.repoId(),
                        modname = self.modname)
        else:
            self.st.out(forward_struct_descr,
                        sname   = sname,
                        repoId  = node.repoId(),
                        modname = self.modname)

    #
    # Exception
    #
    def visitException(self, node):
        if self.handleImported(node): return

        sname = mangle(node.identifier())
        fscopedName = fixupScopedName(node.scopedName(), "")
        self.st.out(exception_class,
                    sname = sname,
                    repoId = node.repoId(),
                    scopedname = dotName(fscopedName))

        if not self.at_module_scope:
            self.st.out(struct_class_name, cname = dotName(fscopedName))

        mnamel = []
        mdescl = []
        for mem in node.members():

            # Deal with nested declarations
            if mem.constrType():
                self.st.inc_indent()
                ams = self.at_module_scope
                self.at_module_scope = 0
                self.currentScope.append(node.identifier())
                
                mem.memberType().decl().accept(self)

                self.currentScope.pop()
                self.at_module_scope = ams
                self.st.dec_indent()
                self.st.out("")

            for decl in mem.declarators():
                mnamel.append(mangle(decl.identifier()))
                mdescl.append('"' + mangle(decl.identifier()) + '"')

                if self.at_module_scope:
                    mdescl.append(\
                        typeAndDeclaratorToDescriptor(mem.memberType(),
                                                      decl,
                                                      []))
                else:
                    mdescl.append(\
                        typeAndDeclaratorToDescriptor(mem.memberType(),
                                                      decl,
                                                      self.currentScope))

        if len(mnamel) > 0:
            mnames = ", " + string.join(mnamel, ", ")
        else:
            mnames = ""

        self.st.out(exception_class_init, mnames = mnames)

        for mname in mnamel:
            self.st.out(exception_init_member, mname = mname)

        if len(mdescl) > 0:
            mdescs = ", " + string.join(mdescl, ", ")
        else:
            mdescs = ""

        if self.at_module_scope:
            self.st.out(exception_descriptor_at_module_scope,
                        sname = sname, mdescs = mdescs, modname = self.modname)
        else:
            self.st.out(exception_descriptor, sname = sname, mdescs = mdescs)

    #
    # Union
    #
    def visitUnion(self, node):
        if self.handleImported(node): return

        uname = mangle(node.identifier())
        if self.at_module_scope:
            stype = typeToDescriptor(node.switchType(), [])
        else:
            stype = typeToDescriptor(node.switchType(), self.currentScope)

        fscopedName = fixupScopedName(node.scopedName(), "")
        
        if node.recursive():
            if self.at_module_scope:
                self.st.out(recursive_union_descr_at_module_scope,
                            uname   = uname,
                            repoId  = node.repoId(),
                            modname = self.modname)
            else:
                self.st.out(recursive_union_descr,
                            uname   = uname,
                            repoId  = node.repoId(),
                            scope   = dotName(fscopedName[:-1]))

        self.st.out(union_class,
                    uname      = uname,
                    repoId     = node.repoId(),
                    scopedname = dotName(fscopedName))

        if not self.at_module_scope:
            self.st.out(union_class_name, cname = dotName(fscopedName))

        if node.constrType():
            self.st.inc_indent()
            ams = self.at_module_scope
            self.at_module_scope = 0
            self.currentScope.append(node.identifier())
            
            node.switchType().decl().accept(self)

            self.currentScope.pop()
            self.at_module_scope = ams
            self.st.dec_indent()

        def_m    = "None"
        def_d    = "None"
        m_def    = "None"
        defpos   = "-1"
        m_to_d_l = []
        d_to_m_l = []
        m_un_l   = []
        d_map_l  = []

        i = 0
        for case in node.cases():

            # Deal with nested declarations
            if case.constrType():
                self.st.inc_indent()
                ams = self.at_module_scope
                self.at_module_scope = 0
                self.currentScope.append(node.identifier())
                
                case.caseType().decl().accept(self)

                self.currentScope.pop()
                self.at_module_scope = ams
                self.st.dec_indent()
                self.st.out("")

            if self.at_module_scope:
                ctype = typeAndDeclaratorToDescriptor(case.caseType(),
                                                      case.declarator(),
                                                      [])
            else:
                ctype = typeAndDeclaratorToDescriptor(case.caseType(),
                                                      case.declarator(),
                                                      self.currentScope)

            cname = mangle(case.declarator().identifier())

            for label in case.labels():
                if label.default():
                    def_m  = '"' + cname + '"'
                    defpos = str(i)
                    if self.at_module_scope:
                        def_d  = valueToString(label.value(),
                                               label.labelKind(), [])
                        m_def  = "_0_" + self.modname + "._m_" + uname + \
                                 "[" + defpos + "]"
                    else:
                        def_d  = valueToString(label.value(),
                                               label.labelKind(),
                                               self.currentScope)
                        m_def  = "_m_" + uname + "[" + defpos + "]"

                    m_un_l.append('(' + def_d + ', "' + cname + '", ' +\
                                  ctype + ')')
                else:
                    if self.at_module_scope:
                        slabel = valueToString(label.value(),
                                               label.labelKind(), [])
                    else:
                        slabel = valueToString(label.value(),
                                               label.labelKind(),
                                               self.currentScope)

                    m_to_d_l.append('"' + cname + '": ' + slabel)
                    d_to_m_l.append(slabel + ': "' + cname + '"')

                    m_un_l.append('(' + slabel + ', "' + cname + '", ' +\
                                  ctype + ')')

                    if self.at_module_scope:
                        d_map_l.append(slabel + ': ' + '_0_' + self.modname + \
                                       "._m_" + uname + "[" + str(i) + "]")
                    else:
                        d_map_l.append(slabel + ': ' + "_m_" + \
                                       uname + "[" + str(i) + "]")
                i = i + 1

        m_to_d = string.join(m_to_d_l, ", ")
        d_to_m = string.join(d_to_m_l, ", ")
        m_un   = string.join(m_un_l,   ", ")
        d_map  = string.join(d_map_l,  ", ")

        if self.at_module_scope:
            self.st.out(union_descriptor_at_module_scope,
                        uname   = uname,
                        m_to_d  = m_to_d,
                        d_to_m  = d_to_m,
                        def_m   = def_m,
                        def_d   = def_d,
                        m_un    = m_un,
                        stype   = stype,
                        defpos  = defpos,
                        m_def   = m_def,
                        d_map   = d_map,
                        modname = self.modname)
            
            self.st.out(union_register_at_module_scope,
                        uname   = uname,
                        modname = self.modname)
        else:
            self.st.out(union_descriptor,
                        uname   = uname,
                        m_to_d  = m_to_d,
                        d_to_m  = d_to_m,
                        def_m   = def_m,
                        def_d   = def_d,
                        m_un    = m_un,
                        stype   = stype,
                        defpos  = defpos,
                        m_def   = m_def,
                        d_map   = d_map,
                        scope   = dotName(fscopedName[:-1]))
            
            self.st.out(union_register, uname = uname)

    #
    # Forward union
    #
    def visitUnionForward(self, node):
        if self.handleImported(node): return

        uname = mangle(node.identifier())

        if self.at_module_scope:
            self.st.out(forward_union_descr_at_module_scope,
                        uname   = uname,
                        repoId  = node.repoId(),
                        modname = self.modname)
        else:
            self.st.out(forward_union_descr,
                        uname   = uname,
                        repoId  = node.repoId(),
                        modname = self.modname)

    #
    # Enum
    #
    def visitEnum(self, node):
        if self.handleImported(node): return

        ename = mangle(node.identifier())
        self.st.out(enum_start, ename = ename)

        i=0
        elist = []
        for item in node.enumerators():
            eval = str(i)

            if self.at_module_scope:
                self.st.out(enum_item_at_module_scope,
                            item    = item.identifier(),
                            eitem   = mangle(item.identifier()),
                            eval    = eval,
                            modname = self.modname)
            else:
                self.st.out(enum_item,
                            item    = item.identifier(),
                            eitem   = mangle(item.identifier()),
                            eval    = eval)

            if self.at_module_scope:
                elist.append(dotName(fixupScopedName(item.scopedName())))
            else:
                elist.append(mangle(item.identifier()))

            i = i + 1

        eitems = string.join(elist, ", ")

        if self.at_module_scope:
            self.st.out(enum_object_and_descriptor_at_module_scope,
                        ename   = ename,
                        repoId  = node.repoId(),
                        eitems  = eitems,
                        modname = self.modname)
        else:
            self.st.out(enum_object_and_descriptor,
                        ename   = ename,
                        repoId  = node.repoId(),
                        eitems  = eitems)

    def visitNative(self, node):
        if self.handleImported(node): return

        sys.stderr.write(main.cmdname + \
                         ": Warning: ignoring declaration of native " + \
                         node.identifier() + "\n")

    def visitValueForward(self, node):
        if self.handleImported(node): return

        vname = mangle(node.identifier())

        self.st.out(value_forward_at_module_scope,
                    vname=vname, repoId=node.repoId(), modname=self.modname)


    def visitValueBox(self, node):
        if self.handleImported(node): return

        boxname = mangle(node.identifier())
        boxdesc = typeToDescriptor(node.boxedType())

        self.st.out(valuebox, boxname=boxname, repoId=node.repoId(),
                    boxdesc=boxdesc, modname=self.modname)


    def visitValueAbs(self, node):
        if self.handleImported(node): return

        vname = mangle(node.identifier())

        fscopedName = fixupScopedName(node.scopedName(), "")
        scopedname  = dotName(fscopedName)

        if node.inherits():
            inheritl = []
            for i in node.inherits():
                i = i.fullDecl()
                inheritl.append(dotName(fixupScopedName(i.scopedName())))
            
            inherits = string.join(inheritl, ", ")
        else:
            inherits = "_0_CORBA.ValueBase"

        self.st.out(valueabs_class,
                    vname=vname, scopedname=scopedname, repoId=node.repoId(),
                    inherits=inherits, modname=self.modname)

        # Declarations within the value
        if len(node.declarations()) > 0:
            self.st.inc_indent()
            self.at_module_scope = 0
            self.currentScope.append(node.identifier())

            for d in node.declarations():
                d.accept(self)

            self.currentScope.pop()
            self.at_module_scope = 1
            self.st.dec_indent()
            self.st.out("")

        basedesc = "_0_CORBA.tcInternal.tv_null"

        self.st.out(value_descriptor_at_module_scope,
                    vname=vname, modifier="_0_CORBA.VM_ABSTRACT",
                    tbaseids="None", basedesc=basedesc, mdescs="",
                    modname=self.modname)


    def visitValue(self, node):
        if self.handleImported(node): return

        vname = mangle(node.identifier())

        fscopedName = fixupScopedName(node.scopedName(), "")
        scopedname  = dotName(fscopedName)

        if node.inherits():
            inheritl = []
            for i in node.inherits():
                i = i.fullDecl()
                inheritl.append(dotName(fixupScopedName(i.scopedName())))
            
        else:
            inheritl = ["_0_CORBA.ValueBase"]

        skeleton_opl = []
        for i in node.supports():
            i = i.fullDecl()
            sn = fixupScopedName(i.scopedName())
            sn[0] = sn[0] + "__POA"
            dn = dotName(sn)
            inheritl.append(dn)
            skeleton_opl.append(dn)

        inherits = string.join(inheritl, ", ")

        # Go up the chain of inherited interfaces, picking out the
        # state members
        members = []
        ilist   = []
        cnode   = node
        
        while 1:
            cin = cnode.inherits()
            if not cin:
                break
            i = cin[0].fullDecl()
            if not isinstance(i, idlast.Value):
                break
            ilist.append(i)
            cnode = i

        ilist.reverse()
        ilist.append(node)
        
        for i in ilist:
            members.extend(i.statemembers())

        set_argl = []

        for i in range(len(members)):
            member = members[i]
            for d in member.declarators():
                set_argl.append("self.%s = args[%d]" %
                                (mangle(d.identifier()),i))

        if set_argl:
            set_args = string.join(set_argl, "\n")
        else:
            set_args = "pass"

        if len(set_argl) == 1:
            s = ""
        else:
            s = "s"

        self.st.out(value_class,
                    vname=vname, scopedname=scopedname, repoId=node.repoId(),
                    inherits=inherits, set_args=set_args, arglen=len(set_argl),
                    s=s, modname=self.modname)

        # Declarations within the value
        if len(node.declarations()) > 0:
            self.st.inc_indent()
            self.at_module_scope = 0
            self.currentScope.append(node.identifier())

            for d in node.declarations():
                d.accept(self)

            self.currentScope.pop()
            self.at_module_scope = 1
            self.st.dec_indent()
            self.st.out("")

        # Skeleton operation declarations if necessary
        if node.supports():
            self.st.out(skeleton_methodmap, methodmap="{}")
            for i in skeleton_opl:
                self.st.out(skeleton_inheritmap, inheritclass=i)

            self.st.out(skeleton_set_skel, ifid=vname)

        # Register factory if no callables or factories
        register_factory = 1

        if not all_factories:
            cnode = node
            while 1:
                if cnode.callables() or cnode.factories() or cnode.supports():
                    register_factory = 0
                    break
                cin = cnode.inherits()
                if not cin:
                    break
                for n in cin:
                    n = n.fullDecl()
                    if not isinstance(n, idlast.Value):
                        register_factory = 0
                        break
                cnode = cin[0].fullDecl()

        if register_factory:
            self.st.out(value_register_factory, vname=vname)
        
        # If value supports some interfaces, output an objref class for it
        if node.supports():
            inheritl = []
            methodl  = []
            for i in node.supports():
                i = i.fullDecl()
                sn = fixupScopedName(i.scopedName())
                inheritl.append(dotName(sn[:-1] + ["_objref_" + sn[-1]]))
                methodl.append(dotName(sn[:-1] + ["_objref_" + sn[-1]]) +
                               ".__methods__")
                
            inherits = string.join(inheritl, ", ")

            self.st.out(objref_class, ifid=vname, inherits=inherits)

            for inclass in inheritl:
                self.st.out(objref_inherit_init, inclass=inclass)

            methods = string.join(methodl, " + ")
            self.st.out(objref_methods, methods = methods)

            # registerObjRef()
            self.st.out(value_objref_register,
                        ifid=vname, modname=self.modname)

        # Modifier
        if node.custom():
            modifier = "_0_CORBA.VM_CUSTOM"
        elif node.truncatable():
            modifier = "_0_CORBA.VM_TRUNCATABLE"
        else:
            modifier = "_0_CORBA.VM_NONE"

        # Truncatable bases
        tbasel = []
        cnode  = node
        while 1:
            cin = cnode.inherits()
            if not cin:
                break
            i = cin[0]
            i = i.fullDecl()
            if not isinstance(i, idlast.Value):
                break
            if cnode.truncatable():
                sn = fixupScopedName(i.scopedName())
                tbasel.append(dotName(sn) + "._NP_RepositoryId")
            else:
                break
            cnode = i

        if tbasel:
            tbaseids = "(%s._NP_RepositoryId, %s)" % (vname,
                                                     string.join(tbasel, ", "))
        else:
            tbaseids = "None"

        basedesc = None
        if node.inherits():
            i = node.inherits()[0].fullDecl()
            if isinstance(i, idlast.Value):
                sn = i.scopedName()[:]
                sn[-1] = "_d_" + sn[-1]
                basedesc = dotName(fixupScopedName(sn))

        if basedesc is None:
            basedesc = "_0_CORBA.tcInternal.tv_null"

        mlist = []
        for m in node.statemembers():
            for d in m.declarators():
                mlist.append('"%s"' % mangle(d.identifier()))
                mlist.append(typeAndDeclaratorToDescriptor(m.memberType(),
                                                           d, []))
                if m.memberAccess() == 1:
                    mlist.append("_0_CORBA.PRIVATE_MEMBER")
                else:
                    mlist.append("_0_CORBA.PUBLIC_MEMBER")
                    
        mdescs = string.join(mlist, ", ")
        self.st.out(value_descriptor_at_module_scope,
                    vname=vname, modifier=modifier, tbaseids=tbaseids,
                    basedesc=basedesc, mdescs=mdescs, modname=self.modname)



def docConst(node):
    if isinstance(node, idlast.Const)        and \
       node.constKind() == idltype.tk_string and \
       node.identifier()[-7:] == "__doc__":
        return node.identifier()[:-7]
    else:
        return None

def nodeId(node):
    if hasattr(node, "identifier"):
        return node.identifier()
    else:
        return None

def docWarning(node):
    sys.stderr.write(main.cmdname + \
                     ": Warning: Constant '" + node.identifier() + "' looks "
                     "like a Python docstring, but there is no declaration "
                     "named '" + node.identifier()[:-7] + "'.\n")
    
class DocstringVisitor (idlvisitor.AstVisitor):
    def __init__(self, st):
        self.docs = []
        self.st   = st

    def output(self):
        if self.docs:
            self.st.out("""\
#
# Docstrings
#
""")
        for nsn, dsn in self.docs:
            nsn = fixupScopedName(nsn)
            dsn = fixupScopedName(dsn)

            self.st.out("@node@.__doc__ = @doc@",
                        node=dotName(nsn), doc=dotName(dsn))
            
        if self.docs:
            self.st.out("")

    def visitAST(self, node):
        for n in node.declarations():
            if not output_inline and not n.mainFile(): continue
            
            d = docConst(n)
            if d:
                ok = 0
                for o in node.declarations():
                    if nodeId(o) == d:
                        self.docs.append((o.scopedName(), n.scopedName()))
                        if isinstance(o, idlast.Interface):
                            sn = o.scopedName()[:]
                            sn[-1] = "_objref_" + sn[-1]
                            self.docs.append((sn, n.scopedName()))
                        ok = 1
                        break
                if not ok:
                    docWarning(n)
            n.accept(self)

    def visitModule(self, node):
        for n in node.definitions():
            d = docConst(n)
            if d:
                if d == node.identifier():
                    self.docs.append((node.scopedName(), n.scopedName()))
                else:
                    ok = 0
                    for o in node.definitions():
                        if nodeId(o) == d:
                            self.docs.append((o.scopedName(), n.scopedName()))
                            if isinstance(o, idlast.Interface):
                                sn = o.scopedName()[:]
                                sn[-1] = "_objref_" + sn[-1]
                                self.docs.append((sn, n.scopedName()))
                            ok = 1
                            break
                    if not ok:
                        docWarning(n)
            n.accept(self)

    def visitInterface(self, node):
        for n in node.declarations():
            d = docConst(n)
            if d:
                if d == node.identifier():
                    self.docs.append((node.scopedName(), n.scopedName()))
                    sn = node.scopedName()[:]
                    sn[-1] = "_objref_" + sn[-1]
                    self.docs.append((sn, n.scopedName()))
                else:
                    ok = 0
                    for o in node.declarations():
                        if nodeId(o) == d:
                            self.docs.append((o.scopedName(), n.scopedName()))
                            ok = 1
                            break
                                
                    if ok:
                        continue

                    for o in node.callables():
                        self.target_id   = d
                        self.target_node = n
                        self.ok          = 0
                        o.accept(self)
                        if self.ok:
                            break
                    
                    if not self.ok:    
                        docWarning(n)

    def visitOperation(self, node):
        if node.identifier() == self.target_id:
            sn = node.scopedName() + ["im_func"]
            sn[-3] = "_objref_" + sn[-3]
            self.docs.append((sn, self.target_node.scopedName()))
            self.ok = 1

    def visitAttribute(self, node):
        for n in node.declarators():
            if n.identifier() == self.target_id:
                sn = n.scopedName() + ["im_func"]
                sn[-3] = "_objref_" + sn[-3]
                sn[-2] = "_get_"    + sn[-2]
                self.docs.append((sn, self.target_node.scopedName()))
                if not node.readonly():
                    sn = sn[:]
                    sn[-2] = "_set_" + n.identifier()
                    self.docs.append((sn, self.target_node.scopedName()))
                self.ok = 1


class ExampleVisitor (idlvisitor.AstVisitor, idlvisitor.TypeVisitor):
    def __init__(self, st):
        self.st = st
        self.first = None

    def visitAST(self, node):
        for n in node.declarations():
            if not output_inline and not n.mainFile(): continue

            if isinstance(n, idlast.Module) or isinstance(n, idlast.Interface):
                n.accept(self)

    def visitModule(self, node):
        for n in node.definitions():
            if not output_inline and not n.mainFile(): continue

            if isinstance(n, idlast.Module) or isinstance(n, idlast.Interface):
                n.accept(self)

    def visitInterface(self, node):
        ifname = mangle(node.identifier())
        sname  = node.scopedName()
        ccname = idlutil.ccolonName(sname)
        fsname = fixupScopedName(sname, prefix="")
        dname  = dotName(fsname)
        skname = skeletonModuleName(dname)

        if self.first is None:
            self.first = ifname

        if len(node.inherits()) == 1:
            inheritance_note = """
    # Note: this interface inherits from another interface. You must
    # either multiply inherit from the servant class implementing the
    # base interface, or explicitly implement the inherited operations
    # here.
    #
    # Inherited interface:
    #
"""
        elif node.inherits():
            inheritance_note = """
    # Note: this interface inherits from other interfaces. You must either
    # multiply inherit from the servant classes implementing the base
    # interfaces, or explicitly implement the inherited operations here.
    #
    # Inherited interfaces:
    #
"""
        else:
            inheritance_note = ""

        for inh in node.inherits():
            iname = idlutil.ccolonName(inh.fullDecl().scopedName())
            inheritance_note = inheritance_note + "    #   %s\n" % iname
        
        self.st.out(example_classdef, ifname=ifname,
                    ccname=ccname, skname=skname,
                    inheritance_note = inheritance_note)

        for c in node.callables():

            if isinstance(c, idlast.Attribute):

                c.attrType().accept(self)
                attrtype = self.__result_type

                for attr in c.identifiers():

                    signature = "attribute %s %s" % (attrtype, attr)

                    if c.readonly():
                        signature = "readonly " + signature

                    if not c.readonly():
                        self.st.out(example_opdef,
                                    signature = signature,
                                    opname = "_set_" + attr,
                                    args = ", value",
                                    returnspec = "None")

                    self.st.out(example_opdef,
                                signature = signature,
                                opname = "_get_" + attr,
                                args = "",
                                returnspec = "attribute value")
            else:
                # Operation
                innames  = []
                outnames = []
                siglist  = []

                c.returnType().accept(self)
                rettype = self.__result_type

                if c.returnType().kind() != idltype.tk_void:
                    outnames.append("result")

                for p in c.parameters():
                    if p.is_in():
                        innames.append(p.identifier())
                    if p.is_out():
                        outnames.append(p.identifier())

                    direction = {0:"in", 1:"out", 2:"inout"}[p.direction()]

                    p.paramType().accept(self)
                    siglist.append("%s %s %s" % (direction,
                                                 self.__result_type,
                                                 p.identifier()))

                signature = "%s %s(%s)" % (rettype, c.identifier(),
                                           string.join(siglist, ", "))

                if innames:
                    args = ", " + string.join(innames, ", ")
                else:
                    args = ""

                if outnames:
                    returnspec = string.join(outnames, ", ")
                else:
                    returnspec = "None"

                self.st.out(example_opdef,
                            signature = signature,
                            opname = c.identifier(),
                            args = args,
                            returnspec = returnspec)



    ttsMap = {
        idltype.tk_void:       "void",
        idltype.tk_short:      "short",
        idltype.tk_long:       "long",
        idltype.tk_ushort:     "unsigned short",
        idltype.tk_ulong:      "unsigned long",
        idltype.tk_float:      "float",
        idltype.tk_double:     "double",
        idltype.tk_boolean:    "boolean",
        idltype.tk_char:       "char",
        idltype.tk_octet:      "octet",
        idltype.tk_any:        "any",
        idltype.tk_TypeCode:   "CORBA::TypeCode",
        idltype.tk_Principal:  "CORBA::Principal",
        idltype.tk_longlong:   "long long",
        idltype.tk_ulonglong:  "unsigned long long",
        idltype.tk_longdouble: "long double",
        idltype.tk_wchar:      "wchar"
        }

    def visitBaseType(self, type):
        self.__result_type = self.ttsMap[type.kind()]

    def visitStringType(self, type):
        if type.bound() == 0:
            self.__result_type = "string"
        else:
            self.__result_type = "string<" + str(type.bound()) + ">"

    def visitWStringType(self, type):
        if type.bound() == 0:
            self.__result_type = "wstring"
        else:
            self.__result_type = "wstring<" + str(type.bound()) + ">"


    def visitDeclaredType(self, type):
        self.__result_type = idlutil.ccolonName(type.decl().scopedName())





def operationToDescriptors(op):
    """Return the descriptors for an operation.

    Returns a tuple containing strings of (in descriptor, out
    descriptor, exception map, context list, contains values)
    """

    indl  = []
    outdl = []
    cv    = 0

    if op.returnType() is not None and \
       op.returnType().kind() != idltype.tk_void:

        outdl.append(typeToDescriptor(op.returnType()))
        cv = idltype.containsValueType(op.returnType())

    # Make the lists of in and out parameters
    for p in op.parameters():

        if p.is_in():
            indl.append(typeToDescriptor(p.paramType()))
        if p.is_out():
            outdl.append(typeToDescriptor(p.paramType()))

        cv = cv or idltype.containsValueType(p.paramType())

    # Fudge single-item lists so that single item tuples work
    if len(indl)  == 1: indl.append("")
    if len(outdl) == 1: outdl.append("")

    inds = "(" + string.join(indl, ", ") + ")"
    if op.oneway():
        outds = "None"
    else:
        outds = "(" + string.join(outdl, ", ") + ")"

    # Exceptions
    excl = []

    for e in op.raises():
        sn = fixupScopedName(e.scopedName())
        ename = dotName(sn)
        edesc = dotName(sn[:-1] + [ "_d_" + sn[-1]])
        excl.append(ename + "._NP_RepositoryId: " + edesc)

    if len(excl) > 0:
        excs = "{" + string.join(excl, ", ") + "}"
    else:
        excs = "None"

    if op.contexts():
        ctxts = "[" + string.join(map(repr, op.contexts()), ", ") + "]"
    else:
        ctxts = None

    return inds, outds, excs, ctxts, cv



ttdMap = {
    idltype.tk_short:      "omniORB.tcInternal.tv_short",
    idltype.tk_long:       "omniORB.tcInternal.tv_long",
    idltype.tk_ushort:     "omniORB.tcInternal.tv_ushort",
    idltype.tk_ulong:      "omniORB.tcInternal.tv_ulong",
    idltype.tk_float:      "omniORB.tcInternal.tv_float",
    idltype.tk_double:     "omniORB.tcInternal.tv_double",
    idltype.tk_boolean:    "omniORB.tcInternal.tv_boolean",
    idltype.tk_char:       "omniORB.tcInternal.tv_char",
    idltype.tk_octet:      "omniORB.tcInternal.tv_octet",
    idltype.tk_any:        "omniORB.tcInternal.tv_any",
    idltype.tk_TypeCode:   "omniORB.tcInternal.tv_TypeCode",
    idltype.tk_Principal:  "omniORB.tcInternal.tv_Principal",
    idltype.tk_longlong:   "omniORB.tcInternal.tv_longlong",
    idltype.tk_ulonglong:  "omniORB.tcInternal.tv_ulonglong",
    idltype.tk_wchar:      "omniORB.tcInternal.tv_wchar"
}

unsupportedMap = {
    idltype.tk_longdouble: "long double",
}

def typeToDescriptor(tspec, from_scope=[], is_typedef=0):
    if hasattr(tspec, "python_desc"):
        return tspec.python_desc

    if ttdMap.has_key(tspec.kind()):
        tspec.python_desc = ttdMap[tspec.kind()]
        return tspec.python_desc

    if unsupportedMap.has_key(tspec.kind()):
        error_exit("omniORBpy does not support the %s type." %
                   unsupportedMap[tspec.kind()])

    if tspec.kind() == idltype.tk_string:
        ret = "(omniORB.tcInternal.tv_string," + str(tspec.bound()) + ")"

    elif tspec.kind() == idltype.tk_wstring:
        ret = "(omniORB.tcInternal.tv_wstring," + str(tspec.bound()) + ")"

    elif tspec.kind() == idltype.tk_sequence:
        ret = "(omniORB.tcInternal.tv_sequence, " + \
              typeToDescriptor(tspec.seqType(), from_scope) + \
              ", " + str(tspec.bound()) + ")"

    elif tspec.kind() == idltype.tk_fixed:
        ret = "(omniORB.tcInternal.tv_fixed, " + \
              str(tspec.digits()) + ", " + str(tspec.scale()) + ")"

    elif tspec.kind() == idltype.tk_alias:
        sn = fixupScopedName(tspec.scopedName())
        if is_typedef:
            return 'omniORB.typeCodeMapping["%s"]._d' % tspec.decl().repoId()
        else:
            return 'omniORB.typeMapping["%s"]' % tspec.decl().repoId()

    else:
        ret = 'omniORB.typeMapping["%s"]' % tspec.decl().repoId()

    tspec.python_desc = ret
    return ret


def typeAndDeclaratorToDescriptor(tspec, decl, from_scope, is_typedef=0):
    desc = typeToDescriptor(tspec, from_scope, is_typedef)

    if len(decl.sizes()) > 0:
        sizes = decl.sizes()[:]
        sizes.reverse()
        for size in sizes:
            desc = "(omniORB.tcInternal.tv_array, " + \
                   desc + ", " + str(size) + ")"
    return desc

def skeletonModuleName(mname):
    """Convert a scoped name string into the corresponding skeleton
module name. e.g. M1.M2.I -> M1__POA.M2.I"""
    l = string.split(mname, ".")
    l[0] = l[0] + "__POA"
    return string.join(l, ".")

def dotName(scopedName, our_scope=[]):
    if scopedName[:len(our_scope)] == our_scope:
        l = map(mangle, scopedName[len(our_scope):])
    else:
        l = map(mangle, scopedName)
    return string.join(l, ".")

def mangle(name):
    if keyword.iskeyword(name): return "_" + name

    # None is a pseudo-keyword that cannot be assigned to.
    if name == "None": return "_None"

    return name

def fixupScopedName(scopedName, prefix="_0_"):
    """Add a prefix and _GlobalIDL to the front of a ScopedName if necessary"""

    try:
        decl = idlast.findDecl([scopedName[0]])
    except idlast.DeclNotFound:
        decl = None

    if isinstance(decl, idlast.Module):
        scopedName = [prefix + mangle(scopedName[0])] + scopedName[1:]
    else:
        scopedName = [prefix + global_module] + scopedName
    return scopedName

def valueToString(val, kind, scope=[]):
    if kind == idltype.tk_enum:
        return dotName(fixupScopedName(val.scopedName()), scope)

    elif kind in [idltype.tk_string, idltype.tk_char]:
        return '"' + idlutil.escapifyString(val) + '"'

    elif kind == idltype.tk_wstring:
        return 'u"' + idlutil.escapifyWString(val) + '"'

    elif kind == idltype.tk_wchar:
        return 'u"' + idlutil.escapifyWString([val]) + '"'

    elif kind == idltype.tk_long and val == -2147483647 - 1:
        return "-2147483647 - 1"

    elif kind in [idltype.tk_float, idltype.tk_double, idltype.tk_longdouble]:
        return idlutil.reprFloat(val)

    elif kind == idltype.tk_fixed:
        return "CORBA.fixed('" + val + "')"

    else:
        return str(val)

__translate_table = string.maketrans(" -.,", "____")

def outputFileName(idlname):
    global __translate_table
    return string.translate(os.path.basename(idlname), __translate_table)

def checkStubPackage(package):
    """Check the given package name for use as a stub directory

    Make sure all fragments of the package name are directories, or
    create them. Make __init__.py files in all directories."""

    if len(package) == 0:
        return

    if package[-1] == ".":
        package = package[:-1]

    path = ""
    for name in string.split(package, "."):
        path = os.path.join(path, name)
        
        if os.path.exists(path):
            if not os.path.isdir(path):
                error_exit('Output error: "%s" exists and is not '
                           'a directory.' % path)
        else:
            try:
                os.mkdir(path)
            except:
                error_exit('Cannot create directory "%s".\n' % path)

        initfile = os.path.join(path, "__init__.py")

        if os.path.exists(initfile):
            if not os.path.isfile(initfile):
                error_exit('Output error: "%s" exists and is not a file.' %
                           initfile)
        else:
            try:
                open(initfile, "w").write("# omniORB stub directory\n")
            except:
                error_exit('Cannot create "%s".' % initfile)


def updateModules(modules, pymodule):
    """Create or update the Python modules corresponding to the IDL
    module names"""

    checkStubPackage(module_package)

    poamodules = map(skeletonModuleName, modules)

    real_updateModules(modules,    pymodule)
    real_updateModules(poamodules, pymodule)


def real_updateModules(modules, pymodule):

    for module in modules:
        modlist = string.split(module_package, ".") + string.split(module, ".")
        modpath = apply(os.path.join, modlist)
        modfile = os.path.join(modpath, "__init__.py")
        tmpfile = os.path.join(modpath, "new__init__.py")

        if not os.path.exists(modpath):
            try:
                os.makedirs(modpath)
            except:
                error_exit('Cannot create path "%s".' % modpath)

        # Make the __init__.py file if it does not already exist
        if not os.path.exists(modfile):
            try:
                f = open(modfile, "w")
            except:
                error_exit('Cannot create "%s".' % modfile)
            
            st = output.Stream(f, 4)

            st.out(pymodule_template, module=module, package=module_package)

            f.close()
            del f, st

        if not os.path.isfile(modfile):
            error_exit('Output error: "%s" exists but is not a file.' %
                       modfile)

        # Insert the import line for the current IDL file
        try:
            inf = open(modfile, "r")
        except:
            error_exit('Cannot open "%s" for reading.' % modfile)

        try:
            outf = open(tmpfile, "w")
        except:
            error_exit('Cannot open "%s" for writing.' % tmpfile)

        line = ""
        while line[:7] != "# ** 1.":
            line = inf.readline()
            if line == "":
                error_exit('Output error: "%s" ended before I found a '
                           '"# ** 1." tag.\n'
                           'Have you left behind some files from a '
                           'different Python ORB?' % modfile)
                
            outf.write(line)
            
        already    = 0
        outputline = "import " + pymodule + "\n"

        while line != "\n":
            line = inf.readline()
            if line == "":
                error_exit('Output error: "%s" ended while I was '
                           'looking at imports.' % modfile)

            if line != "\n":
                outf.write(line)
                if line == outputline:
                    already = 1

        if not already:
            outf.write(outputline)

        outf.write("\n")

        # Output the rest of the file
        while line != "":
            line = inf.readline()
            outf.write(line)

        inf.close()
        outf.close()

        try:
            os.remove(modfile)
        except:
            error_exit('Cannot remove "%s".' % modfile)
        try:
            os.rename(tmpfile, modfile)
        except:
            error_exit('Cannot rename "%s" to "%s".' % (tmpfile, modfile))

    # Go round again, importing sub-modules from their parent modules
    for module in modules:
        modlist = string.split(module, ".")

        if len(modlist) == 1:
            continue

        modlist = string.split(module_package, ".") + modlist
        submod  = modlist[-1]
        modpath = apply(os.path.join, modlist[:-1])
        modfile = os.path.join(modpath, "__init__.py")
        tmpfile = os.path.join(modpath, "new__init__.py")

        # Insert the import line for the sub-module
        try:
            inf = open(modfile, "r")
        except:
            error_exit('Cannot open "%s" for reading.' % modfile)

        try:
            outf = open(tmpfile, "w")
        except:
            error_exit('Cannot open "%s" for writing.' % tmpfile)

        line = ""
        while line[:7] != "# ** 2.":
            line = inf.readline()
            if line == "":
                error_exit('Output error: "%s" ended before I found a '
                           '"# ** 1." tag.\n'
                           'Have you left behind some files from a '
                           'different Python ORB?' % modfile)
                
            outf.write(line)
            
        already    = 0
        outputline = "import " + submod + "\n"

        while line != "\n":
            line = inf.readline()
            if line == "":
                error_exit('Output error: "%s" ended while I was '
                           'looking at imports.' % modfile)

            if line != "\n":
                outf.write(line)
                if line == outputline:
                    already = 1

        if not already:
            outf.write(outputline)

        outf.write("\n")

        # Output the rest of the file
        while line != "":
            line = inf.readline()
            outf.write(line)

        inf.close()
        outf.close()

        try:
            os.remove(modfile)
        except:
            error_exit('Cannot remove "%s".' % modfile)
        try:
            os.rename(tmpfile, modfile)
        except:
            error_exit('Cannot rename "%s" to "%s".' % (tmpfile, modfile))
