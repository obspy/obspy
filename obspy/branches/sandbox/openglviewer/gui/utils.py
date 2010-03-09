#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from waveform_plot import WaveformPlot

def get_channels(base_path):
    """
    XXX: Need better and more generic version.
    
    Only works with certain directory structures. Returns a dictionary.
    """
    directories = {}

    for root, dirs, files in os.walk(base_path):
        if not len(dirs):
            continue
        root = root[len(base_path):]
        root = root.split(os.path.sep)

        if root[0] == '':
            for dir in dirs:
                directories[dir] = {}

        elif root[0] != '' and len(root) == 1:
            for dir in dirs:
                directories[root[0]][dir] = []

        elif len(root) == 2:
            directories[root[0]][root[1]].extend(dirs)

    return directories

def add_plot(window, network, station, channel):
    """
    XXX: Just for testing purposes.
    """
    network = str(network)
    station = str(station)
    channel = str(channel)
    if network == '*' or station == '*' or channel == '*':
        # XXX: Only works for station wildcard.
        stations = window.env.paths[network]
        for station_id in stations.keys():
            cur_station = stations[station_id]
            for cur_channel in cur_station:
                if cur_channel != channel:
                    continue
                path = window.env.path + network + os.path.sep + station_id + os.path.sep + \
                       cur_channel + os.path.sep
                # XXX: Need to add check if it already exists.
                WaveformPlot(parent = window, group = 2, dir = path)
    else:
        # Check if already there.
        id = network + '.' + station + '..' + channel
        for waveform in window.waveforms:
            if waveform.header + '.D'  == id:
                return
        path = window.env.path + network + os.path.sep + station + os.path.sep + \
               channel + os.path.sep
        WaveformPlot(parent = window, group = 2, dir = path)
