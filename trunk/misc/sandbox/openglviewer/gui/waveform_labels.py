#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gui_element import GUIElement
from obspy.core import UTCDateTime
import pyglet

class WaveformLabel(GUIElement):
    """
    Handles the calculation and drawing of the labels.
    """
    def __init__(self, *args, **kwargs):
        """
        Usual init method.
        """
        super(WaveformLabel, self).__init__(self, **kwargs)

    def calculateTicks(self):
        """
        Calculates the ticks.
        """
        self.starttime = parent.starttime
        self.endtime = parent.endtime
        self.time_range = endtime - starttime
        # Decide what type of spacing.
        if self.time_range < 86400 * 2:
            self.scale = 'hour'
        elif self.time_range < 86400 * 45:
            self.scale = 'day'
        elif self.time_range < 86400 * 365 * 2:
            self.scale = 'month'
        else:
            self.scale = 'year'

    def _getNextMonth(self, datetime):
        """
        Little helper routine that will return a UTCDateTime object with the
        beginning of the next month of the given UTCDateTime object.
        """
        year = datetime.year
        month = datetime.month
        next_month = month + 1
        if next_month != 12:
            next_month = next_month % 12
        if next_month == 1:
            year += 1
        return UTCDateTime(year, next_month, 1)

    def _getBeginningOfMonth(self, datetime):
        """
        Same as _getNextMonth but this one will return the beginning of the
        month as a UTCDateTime object.
        """
        return UTCDateTime(datetime.year, datetime.month, 1)

    def _getRelativePosition(self, datetime):
        """
        Returns the relative position of datetime within the graph in respect
        to self.starttime and self.time_range.
        """
        return (datetime - self.starttime) / self.time_range *\
               parent.graph_width

    def _calculateMonthlyTicks(self):
        """
        Calculates the tick positions for the months in relative units, e.g. 0
        is at the left border of the graph and 1 at the right border.
        """
        first_tick = self._getNextMonth(self.starttime)
        last_tick = self._getBeginningOfMonth(self.endtime)
        self.ticks = [self._getRelativePosition(first_tick)]
        # Loop and get the relative positions.
        while first_tick < last_tick:
            first_tick = self._getNextMonth(first_tick)
            self.ticks.append(self._getRelativePosition(first_tick))

