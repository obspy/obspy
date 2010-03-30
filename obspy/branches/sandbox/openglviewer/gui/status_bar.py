#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gui_element import GUIElement
import glydget
import pyglet

class StatusBar(GUIElement):
    """
    Creates a Status Bar at the Bottom of the screen and positions text on it.

    This has two main attributes to work with:
        self.text for all messages that should be displayed in the status bar.
        self.error_text for all error messages.
    """
    def __init__(self, *args, **kwargs):
        super(StatusBar, self).__init__(self, **kwargs)
        self.height = kwargs.get('height', 16)
        self.createBar()
        self.text = ''
        self.error = kwargs.get('error', '')
        self.createMessages()

    def createBar(self):
        """
        Create the status bar and add it to self.batch.
        """
        self.bar = glydget.Rectangle(0, self.height-1,
                        self.win.window.width, self.height-1,
                        [255, 255, 255, 250, 255, 255, 255, 240,
                        255, 255, 255, 230, 255, 255, 255, 215])
        # Add line to separate the status bar.
        self.line  = self.batch.add(2, pyglet.gl.GL_LINES,
                                  self.group,
                                  ('v2i', (0, self.height,
                                           self.win.window.width, self.height)),
                                  ('c3B', (0, 0, 0, 0, 0, 0)))
        # Add to batch.
        self.bar.build(batch = self.batch, group = self.group)
        # Add to object_list.
        self.win.object_list.append(self)

    def createMessages(self):
        """
        Creates the two layout types. One for normal messages and one for error
        messages.
        """
        # Error Messages.
        error_document = pyglet.text.decode_text(self.error)
        error_document.set_style(0, 2, dict(font_name='arial',
                                 bold=True, font_size=10, color=(200,0,0,255)))
        self.error_layout = pyglet.text.DocumentLabel(document = error_document,
                          x = self.win.window.width - 2, y = 3,
                          batch = self.batch, anchor_x = 'right',
                          group = self.group)
        self.error_text = self.error_layout.text
        # Standart log messages.
        text_document = pyglet.text.decode_text(self.text)
        text_document.set_style(0, 2, dict(font_name='arial',
                                 bold=True, font_size=10, color=(0,0,0,255)))
        self.text_layout = pyglet.text.DocumentLabel(document = text_document,
                          x = 2, y = 3, batch = self.batch, anchor_x = 'left',
                          group = self.group)
        self.text = self.text_layout.text
        # Server Messages.
        server_document = pyglet.text.decode_text(self.error)
        server_document.set_style(0, 2, dict(font_name='arial',
                                 bold=True, font_size=10, color=(50,50,50,255)))
        self.server_layout = pyglet.text.DocumentLabel(document = server_document,
                          x = self.win.window.width/2.0, y = 3,
                          batch = self.batch, anchor_x = 'center',
                          group = self.group)
        self.server_text = self.server_layout.text

    def setText(self, text):
        """
        Sets the text.
        """
        self.text = text
        self.text_layout.begin_update()
        self.text_layout.text = self.text
        self.text_layout.end_update()

    def setError(self, text):
        """
        Sets the text.
        """
        self.error = text
        self.error_layout.begin_update()
        self.error_layout.text = self.error
        self.error_layout.end_update()

    def setServer(self, text):
        """
        Sets the text.
        """
        self.server = text
        self.server_layout.begin_update()
        self.server_layout.text = self.server
        self.server_layout.end_update()

    def resize(self, width, height):
        """
        Should get called on all resize events.
        """
        self.bar.begin_update()
        self.bar.resize(width,self.height-1)
        self.bar.end_update()
        self.error_layout.begin_update()
        self.error_layout.x = width - 2
        self.error_layout.text = self.error_text
        self.error_layout.end_update()
        self.text_layout.begin_update()
        self.text_layout.x = 2
        self.text_layout.text = self.text
        self.text_layout.end_update()
        self.line.vertices = (0, self.height, self.win.window.width,
                              self.height)
