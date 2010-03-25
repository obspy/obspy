#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gui_element import GUIElement
import theme
import glydget
import pyglet

class Dialog(GUIElement):
    """
    Creates a Dialog Base Class. It will automatically push all handler when
    created and pop all once it is deleted.
    """
    def __init__(self, *args, **kwargs):
        super(Dialog, self).__init__(self, **kwargs)
        # Two groups on top of all other groups are needed.
        self.group1 = pyglet.graphics.OrderedGroup(0, self.group)
        self.group2 = pyglet.graphics.OrderedGroup(8, self.group)
        self.pushHandler()
        self.createMenu()
        self.darkenWindow()

    def pushHandler(self):
        """
        Pushes all handlers so that all user interaction will be not be passed
        to the underlaying objects.
        """
        def on_key_press(symbol, modifiers): pass
        def on_key_release(symbol, modifiers): pass
        def on_mouse_motion(x, y, dx, dy): return True
        def on_mouse_press(x, y, button, modifiers): return True
        def on_mouse_release(x, y, button, modifiers): return True
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers): return True
        def on_mouse_enter(x, y): return True
        def on_mouse_leave(x, y): return True
        def on_mouse_scroll(x, y, scroll_x, scroll_y): return True
        # Actually push the handlers to the stack.
        self.win.window.push_handlers(on_key_press, on_key_release,
                on_mouse_motion, on_mouse_press, on_mouse_release,
                on_mouse_drag, on_mouse_enter, on_mouse_leave, on_mouse_scroll)

    def darkenWindow(self):
        """
        Creates a semi-transparent box that covers the whole view.
        """
        self.darkBox = glydget.Rectangle(0,self.win.window.height, self.win.window.width,
                                         self.win.window.height,
                                         (0,0,0,155))
        self.darkBox.build(batch = self.batch, group = self.group1)
        self.win.object_list.append(self)

    def createMenu(self):
        """
        Creates very basic menu.
        """
        buttons = [glydget.Button('Apply', self.apply),
                   glydget.Button('Cancel', self.cancel),
                   glydget.Button('OK', self.ok)]
        self.buttons = glydget.HBox(buttons, homogeneous = False)
        self.dialog_box = glydget.VBox([self.buttons], homogeneous = False)
        self.dialog_box.move(self.win.window.width/2.0, self.win.window.height/2.0)
        self.dialog_box.show(batch = self.batch, group = self.group2)
        self.light = glydget.Rectangle(100,100, 100,
                                         100,
                                         (255,255,255,255))
        self.light.build(batch = self.batch, group = self.group2)
        self.win.window.push_handlers(self.dialog_box)  

    def resize(self, width, height):
        """
        Handles all resizing aspekts of the dialog box.
        """
        self.darkBox.begin_update()
        self.darkBox.resize(width, height)
        self.darkBox.move(0,height)
        self.darkBox.end_update()

    def apply(self, button):
        print 'Pressed Apply'

    def cancel(self, button):
        self.win.window.pop_handlers()
        self.win.window.pop_handlers()
        self.darkBox.delete()
        self.light.delete()
        self.dialog_box._delete()
        del self

    def ok(self, button):
        print 'Pressed OK'
