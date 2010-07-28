#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gui_element import GUIElement
import glydget

class Background(GUIElement):
    """
    A simple gradient background.
    """
    def __init__(self, *args, **kwargs):
        super(Background, self).__init__(self, *args, **kwargs)
        # Background colors. Can take 4 full colors for color gradient.
        self.bg_colors = kwargs.get('bg_color', [155,155,155,255])
        # Just four values result in a uniform background.
        if len(self.bg_colors) == 4:
            self.bg_colors = self.bg_colors*4
        # Eight colors result in a simple gradient from up to down.
        elif len(self.bg_colors) == 8:
            color1 = self.bg_colors[0:4]
            color2 = self.bg_colors[4:8]
            temp = []
            temp.extend(color1)
            temp.extend(color1)
            temp.extend(color2)
            temp.extend(color2)
            #self.bg_colors = self.bg_colors[0:4].append(self.bg_colors[4:8])
            self.bg_colors = temp
        self.create()

    def create(self):
        """
        Creates the background.
        """
        self.background = glydget.Rectangle(0, self.win.window.height,
                            self.win.window.width, self.win.window.height,
                            colors = self.bg_colors, filled = True)
        # Add to batch and safe in the windows batch list to keep track of
        # objects.
        self.win.object_list.append(self)
        self.background.build(batch = self.batch, group = self.group)

    def resize(self, width, height):
        """
        Should get called on all resize events.
        """
        # Make sure the background always fills the whole window.
        self.background.begin_update()
        self.background.move(0,height)
        self.background.resize(width,height)
        self.background.end_update()

