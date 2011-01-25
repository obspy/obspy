# -*- python -*-
#                           Package   : omniidl
# output.py                 Created on: 1999/10/27
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
#   IDL compiler output functions

"""Output stream

Class:

  Stream -- output stream which outputs templates, performing
            key/value substitution and indentation."""

import idlstring
string = idlstring

def dummy(): pass

StringType = type("")
FuncType   = type(dummy)

class Stream:
    """IDL Compiler output stream class

The output stream takes a template string containing keys enclosed in
'@' characters and replaces the keys with their associated values. It
also provides counted indentation levels.

  eg. Given the template string:

    template = \"\"\"\\
class @id@ {
public:
  @id@(@type@ a) : a_(a) {}

private:
  @type@ a_;
};\"\"\"

  Calling s.out(template, id="foo", type="int") results in:

    class foo {
    public:
      foo(int a) : a_(a) {}

    private:
      int a_;
    };


Functions:

  __init__(file, indent_size)   -- Initialise the stream with the
                                   given file and indent size.
  inc_indent()                  -- Increment the indent level.
  dec_indent()                  -- Decrement the indent level.
  out(template, key=val, ...)   -- Output the given template with
                                   key/value substitution and
                                   indenting.
  niout(template, key=val, ...) -- As out(), but with no indenting."""


    def __init__(self, file, indent_size = 2):
        self.file        = file
        self.indent_size = indent_size
        self.indent      = 0
        self.do_indent   = 1

    def inc_indent(self): self.indent = self.indent + self.indent_size
    def dec_indent(self): self.indent = self.indent - self.indent_size

    def out(self, text, ldict={}, **dict):
        """Output a multi-line string with indentation and @@ substitution."""

        dict.update(ldict)

        pos    = 0
        tlist  = string.split(text, "@")
        ltlist = len(tlist)
        i      = 0
        while i < ltlist:

            # Output plain text
            pos = self.olines(pos, self.indent, tlist[i])

            i = i + 1
            if i == ltlist: break

            # Evaluate @ expression
            try:
                expr = dict[tlist[i]]
            except:
                # If a straight look-up failed, try evaluating it
                if tlist[i] == "":
                    expr = "@"
                else:
                    expr = eval(tlist[i], globals(), dict)

            if type(expr) is StringType:
                pos = self.olines(pos, pos, expr)
            elif type(expr) is FuncType:
                oindent = self.indent
                self.indent = pos
                apply(expr)
                self.indent = oindent
            else:
                pos = self.olines(pos, pos, str(expr))

            i = i + 1

        self.odone()

    def niout(self, text, ldict={}, **dict):
        """Output a multi-line string without indentation."""

        dict.update(ldict)

        pos    = 0
        tlist  = string.split(text, "@")
        ltlist = len(tlist)
        i      = 0
        while i < ltlist:

            # Output plain text
            pos = self.olines(pos, 0, tlist[i])

            i = i + 1
            if i == ltlist: break

            # Evaluate @ expression
            try:
                expr = dict[tlist[i]]
            except:
                # If a straight look-up failed, try evaluating it
                if tlist[i] == "":
                    expr = "@"
                else:
                    expr = eval(tlist[i], globals(), dict)

            if type(expr) is StringType:
                pos = self.olines(pos, pos, expr)
            elif type(expr) is FuncType:
                oindent = self.indent
                self.indent = pos
                apply(expr)
                self.indent = oindent
            else:
                pos = self.olines(pos, pos, str(expr))

            i = i + 1

        self.odone()


    def olines(self, pos, indent, text):
        istr  = " " * indent
        write = self.file.write

        stext = string.split(text, "\n")
        lines = len(stext)
        line  = stext[0]

        if self.do_indent:
            pos = indent
            write(istr)

        write(line)

        for i in range(1, lines):
            line = stext[i]
            write("\n")
            if line:
                pos = indent
                write(istr)
                write(line)

        if lines > 1 and not line: # Newline at end of text
            self.do_indent = 1
            return self.indent

        self.do_indent = 0
        return pos + len(line)

    def odone(self):
        self.file.write("\n")
        self.do_indent = 1


class StringStream(Stream):
    """Writes to a string buffer rather than a file."""
    def __init__(self, indent_size = 2):
        Stream.__init__(self, self, indent_size)
        self.buffer = []

    def write(self, text):
        self.buffer.append(text)

    def __str__(self):
        return string.join(self.buffer, "")
