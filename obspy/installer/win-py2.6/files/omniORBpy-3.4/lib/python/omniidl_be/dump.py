# -*- python -*-
#                           Package   : omniidl
# idldump.py                Created on: 1999/10/29
#			    Author    : Duncan Grisby (dpg1)
#
#    Copyright (C) 2005 Apasphere Ltd
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
#   Back-end which just dumps the IDL tree

# $Id: dump.py,v 1.12.2.3 2009/05/06 16:14:27 dgrisby Exp $
# $Log: dump.py,v $
# Revision 1.12.2.3  2009/05/06 16:14:27  dgrisby
# Update lots of copyright notices.
#
# Revision 1.12.2.2  2005/06/08 09:40:37  dgrisby
# Update example code, IDL dumping.
#
# Revision 1.12.2.1  2003/03/23 21:01:37  dgrisby
# Start of omniORB 4.1.x development branch.
#
# Revision 1.8.2.7  2002/02/25 15:02:18  dpg1
# Dump wstring constants properly.
#
# Revision 1.8.2.6  2001/08/29 11:54:23  dpg1
# Clean up const handling in IDL compiler.
#
# Revision 1.8.2.5  2001/03/13 10:32:16  dpg1
# Fixed point support.
#
# Revision 1.8.2.4  2000/11/01 15:57:03  dpg1
# More updates for 2.4.
#
# Revision 1.8.2.3  2000/11/01 12:46:00  dpg1
# Update to CORBA 2.4 specification.
#
# Revision 1.8.2.2  2000/10/10 10:18:54  dpg1
# Update omniidl front-end from omni3_develop.
#
# Revision 1.6.2.3  2000/08/14 14:35:14  dpg1
# IDL dumping now properly escapes string and char constants
#
# Revision 1.6.2.2  2000/06/28 14:02:55  dpg1
# Checked in wrong file. Changes backed out.
#
# Revision 1.6.2.1  2000/06/28 13:54:53  dpg1
# Remove dependency on traceback module.
#
# Revision 1.6  1999/11/11 15:55:29  dpg1
# Python back-end interface now supports valuetype declarations.
# Back-ends still don't support them, though.
#
# Revision 1.5  1999/11/02 17:07:23  dpg1
# Changes to compile on Solaris.
#
# Revision 1.4  1999/11/01 20:19:55  dpg1
# Support for union switch types declared inside the switch statement.
#
# Revision 1.3  1999/11/01 16:39:49  dpg1
# Cosmetic change.
#
# Revision 1.2  1999/11/01 10:06:07  dpg1
# Various clean-ups.
#

"""Dumps the IDL tree"""

from omniidl import idlast, idltype, idlutil, idlvisitor, output
import sys, string

class DumpVisitor (idlvisitor.AstVisitor, idlvisitor.TypeVisitor):

    def __init__(self, st):
        self.st = st

    def visitAST(self, node):
        for n in node.declarations():
            n.accept(self)

    def visitModule(self, node):
        self.st.out("""\
module @id@ {""", id = node.identifier())

        self.st.inc_indent()

        for n in node.definitions():
            n.accept(self)

        self.st.dec_indent()

        self.st.out("""\
};""")

    def visitInterface(self, node):
        if len(node.inherits()) != 0:
            inheritl = []
            for i in node.inherits():
                inheritl.append("::" + idlutil.ccolonName(i.scopedName()))

            inherits = ": " + string.join(inheritl, ", ") + " "
        else:
            inherits = ""

        if   node.abstract(): qual = "abstract "
        elif node.local():    qual = "local "
        else:                 qual = ""
        
        self.st.out("""\
@qual@interface @id@ @inherits@{""",
                    id = node.identifier(), inherits=inherits, qual=qual)

        self.st.inc_indent()

        for n in node.contents():
            n.accept(self)

        self.st.dec_indent()
        self.st.out("""\
};""")


    def visitForward(self, node):
        if   node.abstract(): qual = "abstract "
        elif node.local():    qual = "local "
        else:                 qual = ""
        
        self.st.out("""\
@qual@interface @id@;""", id = node.identifier(), qual=qual)


    def visitConst(self, node):
        node.constType().accept(self)
        type = self.__result_type

        if node.constKind() == idltype.tk_enum:
            value = "::" + idlutil.ccolonName(node.value().scopedName())

        elif node.constKind() == idltype.tk_string:
            value = '"' + idlutil.escapifyString(node.value()) + '"'

        elif node.constKind() == idltype.tk_wstring:
            value = 'L"' + idlutil.escapifyWString(node.value()) + '"'

        elif node.constKind() == idltype.tk_wchar:
            value = "L'" + idlutil.escapifyWString([node.value()]) + "'"

        elif node.constKind() in [idltype.tk_float, idltype.tk_double,
                                  idltype.tk_longdouble]:
            value = idlutil.reprFloat(node.value())

        elif node.constKind() == idltype.tk_fixed:
            value = node.value() + "d"
        else:
            value = str(node.value())
        
        self.st.out("""\
const @type@ @id@ = @value@;""",
                    type=type, id=node.identifier(), value=value)


    def visitTypedef(self, node):
        if node.constrType():
            node.aliasType().decl().accept(self)

        node.aliasType().accept(self)
        type  = self.__result_type
        decll = []
        for d in node.declarators():
            d.accept(self)
            decll.append(self.__result_declarator)

        decls = string.join(decll, ", ")

        self.st.out("""\
typedef @type@ @decls@;""",
               type=type, decls=decls)


    def visitStruct(self, node):
        self.st.out("""\
struct @id@ {""",
               id = node.identifier())

        for m in node.members():
            if m.constrType():
                self.st.inc_indent()
                m.memberType().decl().accept(self)
                self.st.dec_indent()

            m.memberType().accept(self)
            type = self.__result_type
            decll = []
            for d in m.declarators():
                d.accept(self)
                decll.append(self.__result_declarator)
            decls = string.join(decll, ", ")

            self.st.out("""\
  @type@ @decls@;""",

                   type=type, decls=decls)

        self.st.out("""\
};""")

    def visitStructForward(self, node):
        self.st.out("""\
struct @id@;""", id = node.identifier())


    def visitException(self, node):
        self.st.out("""\
exception @id@ {""",
               id = node.identifier())

        for m in node.members():
            if m.constrType():
                self.st.inc_indent()
                m.memberType().decl().accept(self)
                self.st.dec_indent()

            m.memberType().accept(self)
            type = self.__result_type
            decll = []
            for d in m.declarators():
                d.accept(self)
                decll.append(self.__result_declarator)
            decls = string.join(decll, ", ")

            self.st.out("""\
  @type@ @decls@;""",

                        type=type, decls=decls)

        self.st.out("""\
};""")


    def visitUnion(self, node):
        if node.constrType():

            self.st.out("""\
union @id@ switch (""",
                        id = node.identifier())
            self.st.inc_indent()
            node.switchType().decl().accept(self)
            self.st.out(""") {""")
            self.st.dec_indent

        else:
            node.switchType().accept(self)
            stype = self.__result_type

            self.st.out("""\
union @id@ switch (@stype@) {""",

                        id=node.identifier(), stype=stype)

        for c in node.cases():
            if c.constrType():
                self.st.inc_indent()
                c.caseType().decl().accept(self)
                self.st.dec_indent()

            for l in c.labels():
                if l.default():
                    self.st.out("""\
  default:""")
                else:
                    if l.labelKind() == idltype.tk_enum:
                        lv = "::" + idlutil.ccolonName(l.value().scopedName())
                    elif l.labelKind() == idltype.tk_char:
                        lv = "'" + repr(l.value())[1:-1] + "'"
                    else:
                        lv = str(l.value())
                        
                    self.st.out("""\
  case @lv@:""",
                           lv=lv)
                    
            c.caseType().accept(self)
            type = self.__result_type
            c.declarator().accept(self)
            decl = self.__result_declarator

            self.st.out("""\
    @type@ @decl@;""",
                   
                   type=type, decl=decl)

        self.st.out("};")


    def visitUnionForward(self, node):
        self.st.out("""\
union @id@;""", id = node.identifier())

    def visitEnum(self, node):
        enuml = []
        for e in node.enumerators(): enuml.append(e.identifier())

        enums = string.join(enuml, ", ")

        self.st.out("""\
enum @id@ {@enums@};""",

               id=node.identifier(), enums=enums)

    def visitAttribute(self, node):
        if node.readonly():
            readonly = "readonly "
        else:
            readonly = ""

        node.attrType().accept(self)
        type = self.__result_type

        ids  = string.join(node.identifiers(), ", ")

        self.st.out("""\
@readonly@attribute @type@ @ids@;""",

               readonly=readonly, type=type, ids=ids)


    def visitOperation(self, node):
        if node.oneway():
            oneway = "oneway "
        else:
            oneway = ""
        
        node.returnType().accept(self)
        rtype = self.__result_type

        paraml = []
        for p in node.parameters():
            if   p.is_in() and p.is_out(): inout = "inout"
            elif p.is_in():                inout = "in"
            else:                          inout = "out"
            p.paramType().accept(self)
            type = self.__result_type
            paraml.append(inout + " " + type + " " + p.identifier())

        params = string.join(paraml, ", ")

        if len(node.raises()) > 0:
            raisel = []
            for r in node.raises():
                ename  = idlutil.ccolonName(r.scopedName())
                raisel.append(ename)

            raises = " raises (" + string.join(raisel, ", ") + ")"
        else:
            raises = ""

        self.st.out("""\
@oneway@@rtype@ @id@(@params@)@raises@;""",
               
               oneway=oneway, rtype=rtype, id=node.identifier(),
               params=params, raises=raises)

    def visitNative(self, node):
        self.st.out("""\
native @id@;""",
                    id=node.identifier())

    def visitValue(self, node):
        if node.inherits():
            inheritl = []
            for i in node.inherits():
                inheritl.append("::" + idlutil.ccolonName(i.scopedName()))

            if node.truncatable():
                truncatable = "truncatable "
            else:
                truncatable = ""

            inherits = ": " + truncatable + string.join(inheritl, ", ") + " "
        else:
            inherits = ""

        if node.supports():
            inheritl = []
            for i in node.supports():
                inheritl.append("::" + idlutil.ccolonName(i.scopedName()))

            inherits = (inherits + "supports " +
                        string.join(inheritl, ", ") + " ")

        if node.custom():
            custom = "custom "
        else:
            custom = ""

        self.st.out("""\
@custom@valuetype @id@ @inherits@{""",
                    id = node.identifier(), inherits=inherits, custom=custom)

        self.st.inc_indent()
        for n in node.contents():
            n.accept(self)

        self.st.dec_indent()
        self.st.out("""\
};""")

    def visitStateMember(self, node):
        access = {0: "public", 1: "private"}[node.memberAccess()]

        if node.constrType():
            node.memberType().decl().accept(self)

        node.memberType().accept(self)
        type = self.__result_type
        decll = []
        for d in node.declarators():
            d.accept(self)
            decll.append(self.__result_declarator)
        decls = string.join(decll, ", ")

        self.st.out("""\
@access@ @type@ @decls@;""",
                    access=access, type=type, decls=decls)

    def visitValueAbs(self, node):
        if node.inherits():
            inheritl = []
            for i in node.inherits():
                inheritl.append("::" + idlutil.ccolonName(i.scopedName()))

            inherits = ": " + string.join(inheritl, ", ") + " "
        else:
            inherits = ""

        if node.supports():
            inheritl = []
            for i in node.supports():
                inheritl.append("::" + idlutil.ccolonName(i.scopedName()))

            inherits = (inherits + "supports " +
                        string.join(inheritl, ", ") + " ")

        self.st.out("""\
abstract valuetype @id@ @inherits@{""",
                    id = node.identifier(), inherits=inherits)

        self.st.inc_indent()
        for n in node.contents():
            n.accept(self)

        self.st.dec_indent()
        self.st.out("""\
};""")

    def visitValueForward(self, node):  return
    def visitValueBox(self, node):      return

    def visitFactory(self, node):       return


    def visitDeclarator(self, node):
        l = [node.identifier()]
        for s in node.sizes():
            l.append("[" + str(s) + "]")

        self.__result_declarator = string.join(l, "")


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

    def visitSequenceType(self, type):
        type.seqType().accept(self)
        if type.bound() == 0:
            self.__result_type = "sequence<" + self.__result_type + ">"
        else:
            self.__result_type = "sequence<" + self.__result_type + ", " +\
                                 str(type.bound()) + ">"

    def visitFixedType(self, type):
        if type.digits() > 0:
            self.__result_type = "fixed<" + str(type.digits()) + "," +\
                                 str(type.scale()) + ">"
        else:
            self.__result_type = "fixed"

    def visitDeclaredType(self, type):
        self.__result_type = "::" + \
                             idlutil.ccolonName(type.decl().scopedName())


def run(tree, args):
    st = output.Stream(sys.stdout, 2)
    dv = DumpVisitor(st)
    tree.accept(dv)
