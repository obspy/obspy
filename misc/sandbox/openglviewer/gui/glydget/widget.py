#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# glydget - an OpenGL widget toolkit
# Copyright (c) 2009 - Nicolas P. Rougier
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
''' Widget class. '''
import pyglet
import rectangle, theme, state


class Widget(object):
    '''
    The Widget class is the base class for all glydget widgets. It provides the
    common set of method for the widgets including:

        * selection methods
        * methods to hide and and show widgets
        * methods to manage size allocation and requests
        * methods to deal with the widget's place in the widget hierarchy
        * event management methods
        * methods to modify the style settings

    Widget introduces style properties - these are basically object properties
    that are stored not on the object, but in the style object associated to the
    widget.
 '''

    _activated = None
    _focused = None

    def __init__(self):
        '''
        '''
        self._parent = None
        self._batch = None
        self._bg_group = None
        self._fg_group = None
        self._background = rectangle.Rectangle(filled=True)
        self._foreground = rectangle.Rectangle(filled=False)
        self._focusable = False
        self._activable = False
        self._style = theme.default
        self._state = state.default
        self._minimum_size = [0,0]
        self._size_request = [0,0]
        self._size_allocation = [0,0]
        self._expand = [True,True]
        self._position = [0,0]
        self._deleted = True
        

    def show(self, batch=None, group=None):
        ''' Show widget on screen.

        :Parameters:

        `batch` : pyglet.graphics.Batch
             Optional batch to add the object to
        `group` : pyglet.graphics.Group
             Optional group to use
        '''

        self._build(batch, group)
        self._update_state()
        self._update_style()
        self._update_size()
        if not self.parent:
            self._allocate(self.size_request[0],self.size_request[1])


    def hide(self):
        ''' Hide widget. '''

        self._delete()
    


    def activate(self):
        ''' Activate widget. '''

        self.state = state.activated



    def deactivate(self):
        ''' Deactivate widget. '''

        self.state = state.default



    def focus(self):
        ''' Focus widget. '''

        self.state = state.focused



    def unfocus(self):
        ''' Unfocus widget. '''

        if not self.activated:
            self.state = state.default



    def move(self, x, y):
        '''Move widget to (x,y).

        If widget is a toplevel widget, the `move` method actually moves the
        widget to the specified location. If widget is not a toplevel widget,
        the `move` method will move the whole hierarchy to a new location that
        will result in this widget being at the specified location.

        :Parameters:
            `x` : int
                X coordinate of widget top-left corner
            `y` : int
                Y coordinate of widget top-left corner
        '''

        if self.parent:
            dx, dy = x-self.x, y-self.y
            self.parent._move(self.parent.x+dx, self.parent.y+dy)
        else:
            self._move(x,y)



    def _move(self, x, y):
        '''Moves widget to (x,y).

        :Parameters:
            `x` : int
                X coordinate of widget top-left corner
            `y` : int
                Y coordinate of widget top-left corner
        '''

        self._position = [x,y]
        self._foreground._position = [self._position[0], self._position[1]]
        self._foreground._update()
        self._background._position = [self._position[0], self._position[1]]
        self._background._update()



    def resize(self, width, height):
        ''' Resize widget to (width, height).

        :Parameters:
            `width` : int
                Width of the object in pixels
            `height` : int
                Height of the object in pixels

        '''

        self.size_request = [width,height]
        if not self.parent and not self._deleted:
            self._allocate(self.size_request[0],self.size_request[1])



    def _build(self, batch=None, group=None):
        ''' Build widget and add it to batch and group.

        :Parameters:
            `batch` : pyglet.graphics.Batch
                Optional batch to add the object to
            `group` : pyglet.graphics.Group_
                Optional group to use
        '''

        self._batch = batch or self._batch or pyglet.graphics.Batch()
        self._bg_group = group or self._bg_group
        if not self._bg_group:
            self._bg_group = pyglet.graphics.OrderedGroup(0,group)
        group = self._bg_group.parent
        if not self._fg_group:
            self._fg_group = pyglet.graphics.OrderedGroup(1,group)
        self._background.delete()
        self._background.build(self._batch, self._bg_group)
        self._foreground.delete()
        self._foreground.build(self._batch, self._fg_group)
        self._deleted = False



    def _delete(self):
        self._foreground.delete()
        self._background.delete()
        self._deleted = True



    def _update_style(self):
        pass



    def _update_state(self):
        self._foreground._colors = self._style.foreground_colors[self._state]
        self._foreground._update_colors()
        self._background._colors = self._style.background_colors[self._state]
        self._background._update_colors()



    def _update_size(self, propagate=False):
        p = self._style.padding
        self._minimum_size = [p[2]+p[3]+1, p[0]+p[1]+1]
        if self.parent and propagate:
            self.parent._update_size(propagate)
        elif propagate:
            self._allocate(self.size_request[0],
                                     self.size_request[1])


    def _allocate(self, width, height):
        self._size_allocation = [width,height]
        self._foreground._position = [self._position[0], self._position[1]]
        self._foreground._size = [self._size_allocation[0],self._size_allocation[1]]
        self._foreground._update()
        self._background._position = [self._position[0], self._position[1]]
        self._background._size = [self._size_allocation[0],self._size_allocation[1]]
        self._background._update()


    def _hit(self, x, y):
        ''' Tests whether (x,y) is in widget area. '''
        return (not self._deleted and 
                (0 < x-self._position[0] < self._size_allocation[0]) and
                (0 < self._position[1]-y < self._size_allocation[1]))



    def on_mouse_press(self, x, y, button, modifiers):
        ''' Default mouse press handler. '''
        if self._hit(x,y) and self._activable:
            self.activate()
            return True
        else:
            self.deactivate()



    def on_mouse_release(self, x, y, button, modifiers):
        ''' Default mouse release handler. '''
        if self._hit(x,y) and self._state == state.default:
            self.focus()
            return True



    def on_mouse_motion(self, x, y, dx, dy):
        ''' Default mouse motion handler. '''
        if self._hit(x,y):
            self.focus()
            return True
        else:
            self.unfocus()



    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        ''' Default mouse scroll handler. '''
        return



    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        ''' Default mouse drag handler. '''
        return



    def on_key_press(self, symbol, modifiers):
        ''' Default key press handler. '''
        return



    def on_key_release(self, symbol, modifiers):
        ''' Default key release handler. '''
        return



    def on_text(self, text):
        ''' Default text handler '''
        return



    def on_text_motion(self, motion):
        ''' Default text motion handler '''
        return



    def on_text_motion_select(self, motion, select):
        ''' Default text motion select handler '''
        return



    def _set_size_request(self, request):
        min_size = self._minimum_size
        if request[0] > 0:
            self._size_request[0] = max(request[0], min_size[0])
        else:
            self._size_request[0] = request[0]
        if request[1] > 0:
            self._size_request[1] = max(request[1], min_size[1])
        else:
            self._size_request[1] = request[1]

    def _get_size_request(self):
        if self._size_request[0] <= 0:
            width = self._minimum_size[0]
        else:
            width = max(self._size_request[0], self._minimum_size[0])
        if self._size_request[1] <= 0:
            height = self._minimum_size[1]
        else:
            height = max(self._size_request[1], self._minimum_size[1])
        return [width, height]

    size_request = property(_get_size_request, _set_size_request,
                            doc = 'Widget size request')




    def _get_state(self):
        return self._state

    def _set_state(self, state_):
        if state_ == state.activated and self._activable:
            if Widget._activated == self:
                return
            if Widget._activated:
                Widget._activated.deactivate()
                Widget._activated = None
            if Widget._focused:
                Widget._focused.unfocus()
                Widget._focused = None
            if self._state != state.activated:
                self._state = state.activated
                self._update_state()
                Widget._activated = self
        elif state_ == state.focused:
            if Widget._focused == self:
                return
            if Widget._focused:
                Widget._focused.unfocus()
                Widget._focused = None
            if self._state != state.activated and self._focusable:
                self._state = state.focused
                self._update_state()
                Widget._focused = self
        elif state_ == state.default:
            self._state = state.default
            self._update_state()
            if Widget._focused == self:
                Widget._focused = None
            if Widget._activated == self:
                Widget._activated = None

    state = property(_get_state, _set_state,
                     doc = '''
     Widget current state.

     One of:
       * glydget.state.default     : default widget state.
       * glydget.state.activated   : widget receives keyboard and mouse events.
       * glydget.state.focused     : widget receives mouse events.
       * glydget.state.selected    : widget has been selected.
       * glydget.state.insensitive : widget cannot be activated, selected or focused.
    ''')



    def _get_focused(self):
        return self._state == state.focused

    focused = property(_get_focused,
                       doc = '''
    Indicate whether widget is currently focused.

    :type: bool, read-only.
    ''')



    def _get_activated(self):
        return self._state == state.activated

    activated = property(_get_activated,
                         doc = '''
    Indicate whether widget is currently activated.

    :type: bool, read-only.
    ''')



    def _get_focusable(self):
        return self._focusable

    def _set_focusable(self, focusable):
        self._focusable = focusable
        if not focusable and Widget._focused == self:
            self.unfocus()

    focusable = property(_get_focusable, _set_focusable,
                         doc = '''
    Indicate whether widget can be focused.

    :type: bool, read-write.
    ''')



    def _get_activable(self):
        return self._activable

    def _set_activable(self, activable):
        self._activable = activable
        if not activable and Widget._activated == self:
            self.deactivate()

    activable = property(_get_activable, _set_activable,
                         doc = '''
    Indicate whether widget can be activated.

    :type: bool, read-write.
    ''')


    def _get_expand(self):
        return self._expand

    def _set_expand(self, expand):
        self._expand = expand

    expand = property(_get_expand, _set_expand,
                       doc = '''
    Widget propension to expand itself when possible.

    :type: [bool,bool] read-write.
    ''')


    def _set_style(self, style):
        self._style = style
        self._update_style()
        self._update_state()

    def _get_style(self):
        return self._style

    style = property(_get_style, _set_style,
                 doc='''
    Style associated to the widget.

    :type: :class:`glydget.style.Style`, read-write.
    ''')



    def _get_parent(self):
        return self._parent

    parent = property(_get_parent,
                    doc = '''
    Parent container or None if the widget has no parent.

    :type: :class:`glydget.widget.Widget`, read-only.
    ''')


    def _get_root(self):
        root = self
        while root._parent:
            root = root._parent
        return root

    root = property(_get_root,
                    doc = '''
    Toplevel container or self if the widget has no parent.

    :type: :class:`glydget.widget.Widget`, read-only.
    ''')


    def _get_x(self):
        return self._position[0]

    x = property(_get_x,
                 doc='''
    X coordinate of top-left corner.

    :type: int, read-only.
    ''')



    def _get_y(self):
        return self._position[1]

    y = property(_get_y,
                 doc='''
    Y coordinate of top-left corner.

    :type: int, read-only.
    ''')



    def _get_width(self):
        return self._size_allocation[0]

    width = property(_get_width,
                     doc='''
    Width in pixels.

    :type: int, read-only.
    ''')



    def _get_height(self):
        return self._size_allocation[1]

    height = property(_get_height,
                      doc='''
    Height in pixels.

    :type: int, read-only.
    ''')



    def _get_batch(self):
        return self._batch

    batch = property(_get_batch,
                      doc='''
    Graphic batch where widget is currently in.

    :type: pyglet.graphics.Batch, read-only.
    ''')



    def _get_group(self):
        return self._group

    group = property(_get_group,
                      doc='''
    Graphic group this widget belongs to.

    :type: pyglet.graphics.Group, read-only.
    ''')


