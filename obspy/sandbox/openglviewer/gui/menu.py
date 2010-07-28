#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gui_element import GUIElement
from obspy.core import UTCDateTime
from waveform_plot import WaveformPlot
from dialog import Dialog
import theme
import glydget

class Menu(GUIElement):
    """
    Creates and updates the Menu.
    """
    def __init__(self, *args, **kwargs):
        super(Menu, self).__init__(self, **kwargs)
        self.createMenu()

    def createMenu(self):
        """
        Creates the Menu.
        """
        # Some helper variables.
        geo = self.win.geometry
        self.available_heigth = self.win.window.height - 2 *\
            geo.vertical_margin - geo.status_bar_height
        self.y_start = self.win.window.height - self.win.geometry.vertical_margin
        # Get the Index from the Server.
        self._getIndex()
        # Various calls to show and position the menu.
        self.menu.show(batch = self.batch, group = self.group)
        x_pos = self.win.window.width
        self.menu.move(x_pos - 150, self.win.window.height-5)
        self.win.window.push_handlers(self.menu)  
        # Add to object list to handle the resizing.
        self.win.object_list.append(self)

    def openOptionsMenu(self, button):
        """
        Creates a new option screen.
        """
        dia = Dialog(parent = self, group =999)

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
        self.win.utils.changeTimes(starttime, endtime)
        # Update text of the items.
        button.parent.children[1].text = str(self.win.starttime)
        button.parent.children[2].text = str(self.win.endtime)

    def change_detail(self, button):
        """
        Change the detail of the plot.
        """
        detail = int(button.parent.children[1].text)
        if detail > 2000:
            detail = 2000
        elif detail < 100:
            detail = 100
        #Update Text if necessary.
        button.parent.children[1].text = str(detail)
        self.win.detail = detail
        for waveform in self.win.waveforms:
            waveform.replot()

    def _getIndex(self):
        """
        Builds the folders.
        """
        self.networks = self.env.networks
        folders = []
        # Get networks.
        networks = self.networks.keys()
        networks.sort()
        network_list = []

        def all_channels(button):
            """
            Selects all certain channels in a network.
            """
            channel = button.text[-3:] + '.D'
            network = button.parent.parent.parent.title.text.split()[-1]
            self.win.utils.add_plot(network, '*', channel)

        def on_toggle(button):
            """
            Inline Function for creating plots.
            """
            channel = button.parent.children[0].text
            station = button.parent.parent.parent.title.text.split()[-1]
            network =\
            button.parent.parent.parent.parent.parent.title.text.split()[-1]
            # Split station if necessary.
            temp = station.split('.')
            if len(temp) > 1:
                station = temp[0]
                location = temp[1]
            else:
                location = ''
            if channel == 'EH*':
                for sub_channel in button.parent.parent.children:
                    chan = sub_channel.children[0].text
                    if chan == 'EHE' or chan == 'EHN' or chan == 'EHZ':
                        self.win.utils.add_plot(network, station, location, chan)
            else:
                self.win.utils.add_plot(network, station, location, channel)

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
            stations = self.networks[network].keys()
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
                locations = self.networks[network][station].keys()
                locations.sort()
                channels_list = []
                for location in locations:
                    # Button to select all EH* stations.
                    all_stations = glydget.ToggleButton('EH*',
                                                        False, on_toggle)
                    channels_list.append(glydget.HBox([all_stations],
                                         homogeneous = False))
                    channels = self.networks[network][station][location]
                    channels.sort()
                    for channel in channels[0]:
                        button = glydget.ToggleButton(channel, False, on_toggle)
                        button.style = theme.database
                        channels_list.append(glydget.HBox([button],
                                             homogeneous = False))
                    if location:
                        station_name = '%s.%s' % (station, location)
                    else:
                        station_name = station
                    box = glydget.Folder(station_name,
                          glydget.VBox(channels_list, homogeneous = False), active\
                                        = False)
                    box.title.style.font_size = 8
                    box.style = theme.database
                    station_list.append(box)
            network_box = glydget.Folder(network,
                          glydget.VBox(station_list, homogeneous = False),
                                         active = False)
            network_box.style = theme.database
            network_list.append(network_box)
        # Menu to select the times.
        self.starttime = glydget.Entry(str(self.win.starttime))
        self.endtime = glydget.Entry(str(self.win.endtime))
        self.starttime.style = theme.database
        self.endtime.style = theme.database
        time_button = glydget.Button('OK', self.change_times)
        time_button.style = theme.database
        time_label =glydget.Label('Select Timeframe:')
        time_label.style = theme.database
        times = glydget.VBox([time_label, self.starttime, self.endtime, time_button], homogeneous=False)
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
        scaleButtons.style = theme.database
        deleteButton = glydget.Button('Delete all', delete_all)
        deleteButton.style = theme.database
        # Button and input field to change the detail.
        detail_label = glydget.Label('Detail:')
        detail_box = glydget.Entry(str(self.win.detail))
        detail_button = glydget.Button('OK', self.change_detail)
        detail_label.style = theme.database
        detail_box.style = theme.database
        detail_button.style = theme.database
        detailBox = glydget.HBox([detail_label, detail_box, detail_button],
                                  homogeneous = False)
        seperator1 = \
                glydget.Label('----------------------------------------------')
        seperator1.style = theme.database
        seperator2 = \
                glydget.Label('----------------------------------------------')
        seperator2.style = theme.database
        options = glydget.Label('Options:')
        seperator2.style = theme.database
        items = [times, seperator1, deleteButton]
        items.extend(network_list)
        items.extend([seperator2, options, scaleButtons])
        items.append(detailBox)
        # Button to enter the options menu.
        optionsButton = glydget.Button('Options', self.openOptionsMenu)
        optionsButton.style = theme.database
        items.append(optionsButton)
        self.menu = glydget.VBox(items, homogeneous = False)
        self.menu.style = theme.database
        # Fixed width and variable height.
        self.menu.resize(self.win.geometry.menu_width, 1)

    def scrollMenu(self, scroll_y):
        """
        Scrolls the Menu.
        """
        geo = self.win.geometry
        new_y_pos = self.menu.y - 3 * scroll_y
        total_height = self.menu.height + geo.vertical_margin + \
                       geo.status_bar_height
        if new_y_pos > total_height:
            new_y_pos = total_height
        if self.menu.height <= self.available_heigth\
                or new_y_pos < self.y_start:
            new_y_pos = self.y_start
        self.menu.move(self.menu.x, new_y_pos)

    def resize(self, width, height):
        """
        Handles the resizing.
        """
        # Some helper variables for the scrolling of the menu.
        geo = self.win.geometry
        self.available_heigth = self.win.window.height - 2 *\
            geo.vertical_margin - geo.status_bar_height
        self.y_start = self.win.window.height - self.win.geometry.vertical_margin
        x_pos = self.win.window.width
        self.menu.move(x_pos - self.win.geometry.menu_width -\
                       self.win.geometry.horizontal_margin -\
                       self.win.geometry.scroll_bar_width,
                       self.win.window.height - \
                       self.win.geometry.vertical_margin)
