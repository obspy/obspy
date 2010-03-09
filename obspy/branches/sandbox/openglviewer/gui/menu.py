#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gui_element import GUIElement
from obspy.core import UTCDateTime
from waveform_plot import WaveformPlot
import theme
import glydget
from utils import add_plot

class Menu(GUIElement):
    """
    Creates and updates the Menu.
    """
    def __init__(self, *args, **kwargs):
        super(Menu, self).__init__(self, **kwargs)
        # Vertical margin.
        self.vert_margin = 5
        # Total space available for the menu.
        self.width = kwargs.get('width', 140)
        self.createMenu()

    def createMenu(self):
        """
        Creates the Menu.
        """
        self._getFolder()
        # Various calls to show and position the menu.
        self.menu.show()
        x_pos = self.win.window.width
        self.menu.move(x_pos - 150, self.win.window.height-5)
        self.win.window.push_handlers(self.menu)  
        # Add to object list to handle the resizing.
        self.win.object_list.append(self)

    def change_scale(self, button):
        """
        Changes the scale of the plots.
        """
        text = str(button.text)
        if text == 'normal':
            scale = False
            # Toggle the correct buttons.
            self.normalScaleButton._active = True
            self.normalScaleButton._update_state()
            self.logScaleButton._active = False
            self.logScaleButton._update_state()
        else:
            scale = 1.00000001
            # Toggle the correct buttons.
            self.normalScaleButton._active = False
            self.normalScaleButton._update_state()
            self.logScaleButton._active = True
            self.logScaleButton._update_state()
        self.win.log_scale = scale
        for waveform in self.win.waveforms:
            waveform.replot()

    def change_times(self,button):
        """
        Change time scale.
        """
        starttime = UTCDateTime(str(button.parent.children[1].text))
        endtime = UTCDateTime(str(button.parent.children[2].text))
        self.win.starttime = UTCDateTime(starttime)
        self.win.endttime = UTCDateTime(endtime)
        for waveform in self.win.waveforms:
            waveform.replot()

    def change_detail(self, button):
        """
        Change the detail of the plot.
        """
        detail = int(button.parent.children[1].text)
        if detail > 2000:
            detail = 2000
        elif detail < 100:
            detail = 100
        # Update Text if necessary.
        button.parent.children[1].text = str(detail)
        self.win.detail = detail
        for waveform in self.win.waveforms:
            waveform.replot()

    def _getFolder(self):
        """
        Builds the folders.
        """
        paths = self.env.paths
        folders = []
        # Get networks.
        networks = paths.keys()
        networks.sort()
        network_list = []

        def all_channels(button):
            """
            Selects all certain channels in a network.
            """
            channel = button.text[-3:] + '.D'
            network = button.parent.parent.parent.title.text.split()[-1]
            add_plot(self.win, network, '*', channel)

        def on_toggle(button):
            """
            Inline Function for creating plots.
            """
            channel = button.parent.children[0].text + '.D'
            station = button.parent.parent.parent.title.text.split()[-1]
            networkt =\
            button.parent.parent.parent.parent.parent.title.text.split()[-1]
            if channel == 'EH*.D':
                for sub_channel in button.parent.parent.children:
                    chan = sub_channel.children[0].text
                    if chan == 'EHE' or chan == 'EHN' or chan == 'EHZ':
                        add_plot(self.win, network, station, chan + '.D')
            else:
                add_plot(self.win, network, station, channel)

        def delete_all(button):
            """
            Deletes the plots.
            XXX: Needs to be better and safer.
            """
            for waveform in self.win.waveforms:
                waveform._delete()
                del waveform
            self.win.waveforms = []
            for item in self.win.object_list:
                if type(item) == WaveformPlot:
                    del item
            self.win.status_bar.setText('0 Traces')

        # Loop over networks.
        for network in networks:
            stations = paths[network].keys()
            stations.sort()
            station_list = []
            # Add Buttons to quickly select all similar channels.
            all_channels = glydget.HBox([\
                            glydget.Button('*.EHE', all_channels),
                            glydget.Button('*.EHN', all_channels),
                            glydget.Button('*.EHZ', all_channels)],
                            homogeneous = False)
            station_list.append(all_channels)
            # Loop over all stations.
            for station in stations:
                channels = paths[network][station]
                channels.sort()
                channels_list = []
                # Button to select all EH* stations.
                all_stations = glydget.ToggleButton('EH*',
                                                    False, on_toggle)
                channels_list.append(glydget.HBox([all_stations],
                                     homogeneous = False))
                for channel in channels:
                    button = glydget.ToggleButton(channel.split('.')[0], False, on_toggle)
                    channels_list.append(glydget.HBox([button],
                                         homogeneous = False))
                box = glydget.Folder(station,
                      glydget.VBox(channels_list, homogeneous = False), active\
                                    = False)
                station_list.append(box)
            network_box = glydget.Folder(network,
                          glydget.VBox(station_list, homogeneous = False))
            network_list.append(network_box)
        # Menu to select the times.
        start = glydget.Entry(str(self.win.starttime))
        end = glydget.Entry(str(self.win.endtime))
        time_button = glydget.Button('OK', self.change_times)
        times = glydget.VBox([glydget.Label('Select Timeframe:'), start,
                      end, time_button], homogeneous=False)
        # Add buttons to change the scale.
        self.normalScaleButton = glydget.ToggleButton('normal', False,
                                     self.change_scale)
        self.logScaleButton = glydget.ToggleButton('log', False,
                                     self.change_scale)
        # Read the scaling of the axis and toggle the correct Button.
        if self.win.log_scale:
            self.normalScaleButton._active = False
            self.normalScaleButton._update_state()
            self.logScaleButton._active = True
            self.logScaleButton._update_state()
        else:
            self.normalScaleButton._active = True
            self.normalScaleButton._update_state()
            self.logScaleButton._active = False
            self.logScaleButton._update_state()
        # Change the styling of the buttons.
        self.normalScaleButton.style = theme.database
        self.logScaleButton.style = theme.database
        scaleButtons = glydget.HBox([glydget.Label('Scale:'),
                                     self.normalScaleButton,
                                     self.logScaleButton],
                                     homogeneous = False)
        deleteButton = glydget.Button('Delete all', delete_all)
        # Button and input field to change the detail.
        detailBox = glydget.HBox([glydget.Label('Detail:'),
                                  glydget.Entry(str(self.win.detail)),
                                  glydget.Button('OK', self.change_detail)],
                                  homogeneous = False)
        seperator1 = \
                glydget.Label('----------------------------------------------')
        seperator2 = \
                glydget.Label('----------------------------------------------')
        options = glydget.Label('Options:')
        items = [times, seperator1, deleteButton]
        items.extend(network_list)
        items.extend([seperator2, options, scaleButtons])
        items.append(detailBox)
        self.menu = glydget.VBox(items, homogeneous = False)
        self.menu.style = glydget.theme.debug
        # Fixed width and variable height.
        self.menu.resize(self.width - 2 * self.vert_margin,1)

    def resize(self, width, height):
        """
        Handles the resizing.
        """
        x_pos = self.win.window.width
        self.menu.move(x_pos - self.width - 2 * self.vert_margin, self.win.window.height-5)
