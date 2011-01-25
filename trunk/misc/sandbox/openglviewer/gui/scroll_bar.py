#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gui_element import GUIElement
import glydget

class ScrollBar(GUIElement):
    """
    Creates and updates a vertical Scrollbar.
    """
    def __init__(self, *args, **kwargs):
        super(ScrollBar, self).__init__(self, **kwargs)
        self.active_color = [255, 255, 255, 250, 255, 255, 255, 240,
                             255, 255, 255, 230, 255, 255, 255, 215]
        self.inactive_color = [150, 150, 150, 250, 150, 150, 150, 240,
                             150, 150, 150, 230, 150, 150, 150, 215]
        self.createScrollBar()

    def createScrollBar(self):
        """
        Creates a scroll bar on the right side.
        """
        top = self.win.window.height
        bottom = self.win.status_bar.height
        x_start = self.win.window.width - self.win.geometry.scroll_bar_width
        self.bar = glydget.Rectangle(x_start, top,
                                     self.win.geometry.scroll_bar_width,
                        top - bottom, self.active_color)
        # Create little box to show the current position. This will currently
        # just create a grey box because the real box can only be shown after
        # the rest of all the GUI Elements has been rendered.
        self.box = glydget.Rectangle(x_start, top,
                                     self.win.geometry.scroll_bar_width,
                        top - bottom, self.inactive_color)
        # Add to batch.
        self.bar.build(batch = self.batch, group = self.group)
        self.box.build(batch = self.batch, group = self.group)
        # Add to object_list.
        self.win.object_list.append(self)

    def changePosition(self):
        """
        Changes the position of the little box in the scroll bar.
        """
        # Account for the extra_vertical space at the bottom that is not
        # included in the max viewable area because each Waveform Element has
        # an offset from the y-axis.
        # XXX: Need to make dynamic or make offset dynamic. The current solution
        # is not really clean.
        view_area = self.win.max_viewable + 10 + self.win.geometry.time_scale
        if view_area < self.win.current_view_span:
            view_area = self.win.current_view_span
        width = self.win.window.width
        height = self.win.window.height
        box_top = -self.win.waveform_offset
        box_bottom = box_top + self.win.current_view_span
        box_height = box_bottom - box_top
        box_vspan = int(round(box_height/float(view_area) *\
                    self.win.current_view_span))
        top = int(round(box_top/float(view_area) *\
              self.win.current_view_span))
        top = height - top
        # Actually handle the updates.
        self.box.begin_update()
        self.box.move(width - self.win.geometry.scroll_bar_width, top)
        self.box.resize(self.win.geometry.scroll_bar_width ,box_vspan)
        self.box.end_update()

    def resize(self, width, height):
        """
        Gets called on resize.
        """
        bottom = self.win.status_bar.height
        x_start = width - self.win.geometry.scroll_bar_width
        v_span = height - self.win.status_bar.height
        self.bar.begin_update()
        self.bar.move(width - self.win.geometry.scroll_bar_width, height)
        self.bar.resize(self.win.geometry.scroll_bar_width ,v_span)
        self.bar.end_update()
        # Update the box. If max_viewable is not set yet create grey box on top
        # of the other one.
        #if self.win.max_viewable == 0:
        #    pass
        #else:
        self.changePosition()
