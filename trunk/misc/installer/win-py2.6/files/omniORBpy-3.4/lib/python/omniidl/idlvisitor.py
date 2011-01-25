# -*- python -*-
#                           Package   : omniidl
# idlvisitor.py             Created on: 1999/10/27
#			    Author    : Duncan Grisby (dpg1)
#
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
#   Base classes for Visitors

# $Id: idlvisitor.py,v 1.8.2.1 2003/03/23 21:01:38 dgrisby Exp $
# $Log: idlvisitor.py,v $
# Revision 1.8.2.1  2003/03/23 21:01:38  dgrisby
# Start of omniORB 4.1.x development branch.
#
# Revision 1.5.2.2  2000/11/01 12:46:00  dpg1
# Update to CORBA 2.4 specification.
#
# Revision 1.5.2.1  2000/07/17 10:36:07  sll
# Merged from omni3_develop the diff between omni3_0_0_pre3 and omni3_0_0.
#
# Revision 1.6  2000/07/13 15:25:51  dpg1
# Merge from omni3_develop for 3.0 release.
#
# Revision 1.3  1999/11/15 15:49:23  dpg1
# Documentation strings.
#
# Revision 1.2  1999/10/29 18:20:01  dpg1
# Clean up
#
# Revision 1.1  1999/10/29 15:47:07  dpg1
# First revision.
#

"""Visitor pattern visitors for AST nodes and types

This module declares abstract visitor classes for visiting AST nodes
and type objects. Python's type system means that you do not actually
need to derive from these classes to implement visitors. This module
only really exists to show what functions should exist in visitors.

Classes:

  AstVisitor  -- visitor for classes in idlast.py
  TypeVisitor -- visitor for classes in idltype.py"""

class AstVisitor :
    """Visitor for AST nodes

Functions:

  visitAST(node)
  visitModule(node)
  visitInterface(node)
  visitForward(node)
  visitConst(node)
  visitDeclarator(node)
  visitTypedef(node)
  visitMember(node)
  visitStruct(node)
  visitStructForward(node)
  visitException(node)
  visitCaseLabel(node)
  visitUnionCase(node)
  visitUnion(node)
  visitUnionForward(node)
  visitEnumerator(node)
  visitEnum(node)
  visitAttribute(node)
  visitParameter(node)
  visitOperation(node)
  visitNative(node)
  visitStateMember(node)
  visitFactory(node)
  visitValueForward(node)
  visitValueBox(node)
  visitValueAbs(node)
  visitValue(node)"""

    def visitAST(self, node):           return
    def visitModule(self, node):        return
    def visitInterface(self, node):     return
    def visitForward(self, node):       return
    def visitConst(self, node):         return
    def visitDeclarator(self, node):    return
    def visitTypedef(self, node):       return
    def visitMember(self, node):        return
    def visitStruct(self, node):        return
    def visitStructForward(self, node): return
    def visitException(self, node):     return
    def visitCaseLabel(self, node):     return
    def visitUnionCase(self, node):     return
    def visitUnion(self, node):         return
    def visitUnionForward(self, node):  return
    def visitEnumerator(self, node):    return
    def visitEnum(self, node):          return
    def visitAttribute(self, node):     return
    def visitParameter(self, node):     return
    def visitOperation(self, node):     return
    def visitNative(self, node):        return
    def visitStateMember(self, node):   return
    def visitFactory(self, node):       return
    def visitValueForward(self, node):  return
    def visitValueBox(self, node):      return
    def visitValueAbs(self, node):      return
    def visitValue(self, node):         return


class TypeVisitor:
    """Visitor for Type objects

Functions:

  visitBaseType(type)
  visitStringType(type)
  visitWStringType(type)
  visitSequenceType(type)
  visitFixedType(type)
  visitDeclaredType(type)"""

    def visitBaseType(self, type):     return
    def visitStringType(self, type):   return
    def visitWStringType(self, type):  return
    def visitSequenceType(self, type): return
    def visitFixedType(self, type):    return
    def visitDeclaredType(self, type): return
