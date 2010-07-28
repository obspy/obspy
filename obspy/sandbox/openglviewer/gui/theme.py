#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# glydget - an OpenGL widget toolkit
# Copyright (c) 2009 - Nicolas P. Rougier
#
# This file is part of glydget.
#
# glydget is free software: you can  redistribute it and/or modify it under the
# terms of  the GNU General  Public License as  published by the  Free Software
# Foundation, either  version 3 of the  License, or (at your  option) any later
# version.
#
# glydget is  distributed in the hope that  it will be useful,  but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy  of the GNU General Public License along with
# glydget. If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
''' Graphic style '''
from glydget.style import Style
import glydget.state as state

none = [0,0,0,0]

# --- Debug ---
debug = Style()
debug.font_size = 12
debug.halign = 0.5
debug.colors[state.default]              = [255,255,255,255]
debug.foreground_colors[state.default]   = [255,255,255,50]*8
debug.background_colors[state.default]   = [255,255,255,50]*4
debug.colors[state.focused]              = [255,255,255,255]
debug.foreground_colors[state.focused]   = [255,255,255,100]*8
debug.background_colors[state.focused]   = [255,255,255,100]*4
debug.colors[state.activated]            = [255,255,255,255]
debug.foreground_colors[state.activated] = [255,255,255,150]*8
debug.background_colors[state.activated] = [255,255,255,150]*4


# --- Default ---
default = Style()
default.colors[state.default]      = [255,255,255,255]
default.foreground_colors[state.default]   = [255,255,255,50]*8
default.background_colors[state.default]   = [255,255,255,50]*4
default.colors[state.focused]      = [255,255,255,255]
default.foreground_colors[state.focused]   = [255,255,255,100]*8
default.background_colors[state.focused]   = [255,255,255,100]*4
default.colors[state.activated]    = [255,255,255,255]
default.foreground_colors[state.activated] = [255,255,255,150]*8
default.background_colors[state.activated] = [255,255,255,150]*4

# --- ObsPy  Database viewer---
database = Style()
database.halign = 0.5
database.font_size = 10
database.colors[state.default]      = [0,0,0,255]
database.foreground_colors[state.default]   = [155,155,155,255]*8
#database.background_colors[state.default]   = [195,195,195,255]*4
database.background_colors[state.default]   = [100,100,100,255,
                                               100,100,100,255,130,130,130,255,130,130,130,255]
database.colors[state.focused]      = [50,50,50,255]
database.foreground_colors[state.focused]   = [155,155,155,100]*8
database.background_colors[state.focused]   = [200,200,200,255]*4
database.colors[state.activated]    = [155,155,155,255]
database.foreground_colors[state.activated] = [155,155,155,150]*8
database.background_colors[state.activated] = [100,100,100,255]*4

# --- Window ---
Window = Style()
Window.colors[state.default]    = [0,0,0,155]
Window.foreground_colors[state.default] = [0,0,0,255]*8
Window.background_colors[state.default] = [127,127,127,127]*4
Window.padding = [4,4,4,4]
Window.child = Style()
Window.child.padding = [1,1,1,1]
#Window.child.background_colors[state.default] = [0,0,0,127]*4
#Window.child.foreground_colors[state.default] = [0,0,0,127]*8
Window.title = Style()
Window.title.bold = True


# --- Label ---
Label = Style()

# --- Container ---
Container = Style()
Container.padding = [0,0,0,0]


# --- Folder ---
Folder = Style()
Folder.padding = [0,0,0,0]
Folder.child = Style()
Folder.child.padding = [0,0,12,0]
Folder.child.halign = 0
Folder.title = Style()
Folder.title.bold = True
Folder.title.background_colors[state.focused] = [255,255,255,50]*4


# --- Entry ---
Entry = Style()
Entry.colors[state.default]      = [255,255,100,255]
Entry.colors[state.focused]      = [255,255,100,255]
Entry.colors[state.activated]    = [255,255,100,255]
Entry.background_colors[state.default]   = [0,0,0,50]*4
Entry.background_colors[state.focused]   = [0,0,0,75]*4
Entry.background_colors[state.activated] = [0,0,0,100]*4
Entry.foreground_colors[state.default]   = none*8
Entry.foreground_colors[state.focused]   = none*8
Entry.foreground_colors[state.activated] = none*8

# --- Entry ---
BoolEntry = Style()
BoolEntry.colors[state.default]      = [255,255,100,255]
BoolEntry.colors[state.focused]      = [255,255,100,255]
BoolEntry.colors[state.activated]    = [255,255,100,255]
BoolEntry.background_colors[state.default]   = [0,0,0,50]*4
BoolEntry.background_colors[state.focused]   = [0,0,0,75]*4
BoolEntry.background_colors[state.activated] = [0,0,0,50]*4
BoolEntry.foreground_colors[state.default]   = none*8
BoolEntry.foreground_colors[state.focused]   = none*8
BoolEntry.foreground_colors[state.activated] = none*8


# --- Button ---
Button = Style()
Button.foreground_colors[state.default]    = [0,0,0,150]*8
Button.background_colors[state.default]    = [0,0,0,50]*4
Button.foreground_colors[state.focused]    = [0,0,0,150]*8
Button.background_colors[state.focused]    = [0,0,0,75]*4
Button.foreground_colors[state.activated]  = [0,0,0,150]*8
Button.background_colors[state.activated]  = [0,0,0,100]*4
Button.halign = 0.5


# --- Arrow Button ---
ArrowButton = Style()
ArrowButton.font_size = 7
ArrowButton.padding = [-1,-2,5,5]
ArrowButton.halign = 0.5
ArrowButton.colors[state.default]      = [255,255,100,255]
ArrowButton.colors[state.focused]      = [255,255,100,255]
ArrowButton.colors[state.activated]    = [255,255,100,255]
ArrowButton.background_colors[state.default]   = [0,0,0,50]*4
ArrowButton.background_colors[state.focused]   = [0,0,0,75]*4
ArrowButton.background_colors[state.activated] = [0,0,0,100]*4


# --- Arrow Button ---
Spacer = Style()
Spacer.padding = [5,5,5,5]
