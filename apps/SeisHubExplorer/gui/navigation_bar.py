#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt4 import QtCore, QtGui

from obspy.core import UTCDateTime

class NavigationBar(QtGui.QFrame):
    """
    The navigation bar.
    """
    def __init__(self, env, parent = None, *args, **kwargs):
        super(NavigationBar, self).__init__()
        self.env = env
        # Style sheet for the buttons. They take too much space without a style
        # sheet.
        #self.button_style_sheet = 'border: 1px solid #AAAAAA; border-radius:5px'
        self.button_style_sheet = ''
        # Remove the border to save more space.
        #self.setStyleSheet('border:0px;')
        # Define the layout.
        self.layout = QtGui.QHBoxLayout()
        # Remove the margins.
        self.layout.setMargin(0)
        self.setLayout(self.layout)
        # Define the buttons.
        # Goes back 100 percent.
        self.way_back_button = QtGui.QPushButton('<<')
        #self.way_back_button.setStyleSheet(self.button_style_sheet)
        self.way_back_button.setToolTip('Go back 100 percent.')
        self.layout.addWidget(self.way_back_button)
        # Goes back 50 percent.
        self.step_back_button = QtGui.QPushButton('<')
        self.step_back_button.setStyleSheet(self.button_style_sheet)
        self.step_back_button.setToolTip('Go back 50 percent.')
        self.layout.addWidget(self.step_back_button)
        # Zooms out to show 31 days.
        self.zoom_month_button = QtGui.QPushButton('Full Month')
        self.zoom_month_button.setStyleSheet(self.button_style_sheet)
        self.zoom_month_button.setToolTip('Zoom to 31 days centered around the current view.')
        self.layout.addWidget(self.zoom_month_button)
        # Zooms out to show 7 days.
        self.zoom_full_week_button = QtGui.QPushButton('Full Week')
        self.zoom_full_week_button.setStyleSheet(self.button_style_sheet)
        self.zoom_full_week_button.setToolTip('Zoom to 7 days centered around the current view.')
        self.layout.addWidget(self.zoom_full_week_button)
        # Shows just today.
        self.zoom_today_button = QtGui.QPushButton('Today')
        self.zoom_today_button.setStyleSheet(self.button_style_sheet)
        self.zoom_today_button.setToolTip('Show the current day.')
        self.layout.addWidget(self.zoom_today_button)
        # Show today and 7 days back.
        self.zoom_last_week_button = QtGui.QPushButton('Last Week')
        self.zoom_last_week_button.setStyleSheet(self.button_style_sheet)
        self.zoom_last_week_button.setToolTip('Show last week.')
        self.layout.addWidget(self.zoom_last_week_button)
        # Show today and 31 days back.
        self.zoom_last_month_button = QtGui.QPushButton('Last Month')
        self.zoom_last_month_button.setStyleSheet(self.button_style_sheet)
        self.zoom_last_month_button.setToolTip('Show last month.')
        self.layout.addWidget(self.zoom_last_month_button)
        # Goes forward 50 percent.
        self.step_forward_button = QtGui.QPushButton('>')
        self.step_forward_button.setStyleSheet(self.button_style_sheet)
        self.step_forward_button.setToolTip('Go forward 50 percent.')
        self.layout.addWidget(self.step_forward_button)
        # Goes forward 100 percent.
        self.way_forward_button = QtGui.QPushButton('>>')
        self.way_forward_button.setStyleSheet(self.button_style_sheet)
        self.way_forward_button.setToolTip('Go forward 100 percent.')
        self.layout.addWidget(self.way_forward_button)
        # Finally connect the buttons to do something.
        self.connectSignals()

    def connectSignals(self):
        """
        Connects all necessary signals with the corresponding methods.
        """
        QtCore.QObject.connect(self.zoom_month_button,
                               QtCore.SIGNAL("clicked()"), self.zoom_month)
        QtCore.QObject.connect(self.zoom_full_week_button,
                               QtCore.SIGNAL("clicked()"), self.zoom_week)
        QtCore.QObject.connect(self.zoom_last_week_button,
                               QtCore.SIGNAL("clicked()"), self.zoom_last_week)
        QtCore.QObject.connect(self.zoom_last_month_button,
                               QtCore.SIGNAL("clicked()"), self.zoom_last_month)
        QtCore.QObject.connect(self.zoom_today_button,
                               QtCore.SIGNAL("clicked()"), self.zoom_today)
        QtCore.QObject.connect(self.way_back_button,
                               QtCore.SIGNAL("clicked()"), self.zoom_way_back)
        QtCore.QObject.connect(self.step_back_button,
                               QtCore.SIGNAL("clicked()"), self.zoom_step_back)
        QtCore.QObject.connect(self.way_forward_button,
                               QtCore.SIGNAL("clicked()"), self.zoom_way_forward)
        QtCore.QObject.connect(self.step_forward_button,
                               QtCore.SIGNAL("clicked()"), self.zoom_step_forward)

    def zoom_way_back(self):
        """
        Zooms 100 percent back.
        """
        self.env.main_window.changeTimes(\
                    self.env.starttime - self.env.time_range,
                    self.env.endtime - self.env.time_range)

    def zoom_step_back(self):
        """
        Zooms 50 percent back.
        """
        self.env.main_window.changeTimes(\
                    self.env.starttime - 0.5*self.env.time_range,
                    self.env.endtime - 0.5*self.env.time_range)

    def zoom_way_forward(self):
        """
        Zooms 100 percent forward.
        """
        self.env.main_window.changeTimes(\
                    self.env.starttime + self.env.time_range,
                    self.env.endtime + self.env.time_range)

    def zoom_step_forward(self):
        """
        Zooms 50 percent forward.
        """
        self.env.main_window.changeTimes(\
                    self.env.starttime + 0.5*self.env.time_range,
                    self.env.endtime + 0.5*self.env.time_range)

    def zoom_month(self):
        """
        Zoom to the full month. If more than one month is on the screen, it
        will zoom to the month in the middle.
        """
        middle = self.env.starttime + 0.5 * self.env.time_range
        starttime = UTCDateTime(middle.year, middle.month, 1)
        if middle.month == 12:
            next_month = 1
            next_year = middle.year + 1
        else:
            next_month = middle.month + 1
            next_year = middle.year
        endtime = UTCDateTime(next_year, next_month, 1) - 1
        self.env.main_window.changeTimes(starttime, endtime)

    def zoom_week(self):
        """
        Zooms to one week centered on the current view.
        """
        middle = self.env.starttime + 0.5 * self.env.time_range
        middle = UTCDateTime(middle.year, middle.month, middle.day)
        starttime = middle - 3*86400
        endtime = middle + 4*86400 -1
        self.env.main_window.changeTimes(starttime, endtime)

    def zoom_today(self):
        """
        Zooms to the current day.
        """
        now = UTCDateTime()
        starttime = UTCDateTime(now.year, now.month, now.day)
        endtime = starttime + 86400 - 1 
        self.env.main_window.changeTimes(starttime, endtime)

    def zoom_last_week(self):
        """
        Zooms to the last week including today.
        """
        now = UTCDateTime()
        starttime = UTCDateTime(now.year, now.month, now.day)
        endtime = starttime + 86400 - 1 
        starttime -= 6*86400
        self.env.main_window.changeTimes(starttime, endtime)

    def zoom_last_month(self):
        """
        Zooms to the last 31 days including today.
        """
        now = UTCDateTime()
        starttime = UTCDateTime(now.year, now.month, now.day)
        endtime = starttime + 86400 - 1 
        starttime -= 31*86400
        self.env.main_window.changeTimes(starttime, endtime)
