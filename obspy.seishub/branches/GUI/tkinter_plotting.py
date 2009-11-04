from matplotlib import use as matplotlibuse
matplotlibuse('TkAgg')
from Tkinter import *
import tkFileDialog
import tkColorChooser
import obspy
from obspy.imaging.waveform import _getMinMaxList as minmaxlist
from obspy.seishub import Client
from obspy.core import UTCDateTime
import pickle
import inspect
import os
import numpy as np
from matplotlib import figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

class Seishub(object):
    def __init__(self):
        self.networks = {}
        self.client = Client()
        self.pickle_file = os.path.join(
                    os.path.dirname(inspect.getsourcefile(self.__class__)),
                    'seishub_dict.pickle')
        self.networks = None

    def get_infos(self):
        try:
            file = open(self.pickle_file, 'rb')
            self.networks = pickle.load(file)
            file.close()
        except:
            self.reload_infos()

    def reload_infos(self):
        self.networks = {}
        networks = self.client.waveform.getNetworkIds()
        # Get stations.
        for key in networks:
            if not key:
                continue
            self.networks[key] = {}
            stations = self.client.waveform.getStationIds(network_id=key)
            for station in stations:
                if not station:
                    continue
                self.networks[key][station] = {}
                # Get locations.
                locations = self.client.waveform.getLocationIds(network_id=key,
                                                        station_id=station)
                for location in locations:
                    channels = self.client.waveform.getChannelIds(\
                        network_id=key , station_id=station,
                        location_id=location)
                    self.networks[key][station][location] = [channels]
        # Add current date to Dictionary.
        self.networks['Date'] = UTCDateTime()
        # Open file.
        file = open(self.pickle_file, 'wb')
        pickle.dump(self.networks, file)
        file.close()


class NeededVars(object):
    def __init__(self):
        self.color = 'black'
        self.stream = None
        self.network = None
        self.station = None
        self.location = None

def main():
    """
    GUI Mainloop.
    """
    NV = NeededVars()
    # Get Seishub informations.
    SH = Seishub()
    SH.get_infos()
    # Get the seishub stuff.
    # color chooser dialogue.
    def choosecolor(*arg, **kwargs):
        """
        Set color and update graph if neccessary.
        """
        NV.color = tkColorChooser.askcolor()[1]
        colorcanvas.itemconfigure(colorcanvas.find_all()[1], fill=NV.color)
        if NV.stream:
            create_graph()

    def openfile():
        """
        Opens the file, reads it and calculates a minmax list.
        """
        filename = tkFileDialog.askopenfilename()
        if not filename:
            return
        # Read the file.
        NV.st = obspy.read(filename)
        st = NV.st
        # Get minmaxlist.
        NV.minmax = minmaxlist(st, 799, st[0].stats.starttime.timestamp,
                            st[0].stats.endtime.timestamp)[2]
        del NV.minmax[-1]
        create_graph()

    def create_graph():
        """
        Creates the graph.
        """
        print NV.stream
        #import pdb;pdb.set_trace()
        a.cla()
        length = len(NV.stream[0].data)
        a.set_ylim(NV.stream[0].data.min(), NV.stream[0].data.max())
        a.set_xlim(0, length)
        a.set_yticks([])
        a.plot(NV.stream[0].data, color = NV.color)
        a.set_xticks([1,length/2, length-1])
        starttime = NV.stream[0].stats.starttime
        endtime = NV.stream[0].stats.endtime
        midtime = starttime + (endtime - starttime)/2
        a.set_xticklabels([starttime.strftime('%H:%M:%S'),
                           midtime.strftime('%H:%M:%S'),
                           endtime.strftime('%H:%M:%S')])
        canvas.show()

    def changeStationList(*args, **kwargs):
        # Delete all old items in station_box.
        station_box.delete(0, station_box.size())
        # Also delete location and channel box.
        location_box.delete(0, location_box.size())
        channel_box.delete(0, channel_box.size())
        network = network_box.get(network_box.curselection()[0])
        NV.network = network
        for station in SH.networks[network].keys():
            station_box.insert(END, station)

    def changeLocationList(*args, **kwargs):
        try:
            station = station_box.get(station_box.curselection()[0])
        except:
            return
        NV.station = station
        # Delete all old items in location_box.
        location_box.delete(0, station_box.size())
        # Also delete all items in channel_box.
        channel_box.delete(0, channel_box.size())
        station = station_box.get(station_box.curselection()[0])
        NV.station = station
        for location in SH.networks[NV.network][station].keys():
            if not location:
                location = 'EMPTY'
            location_box.insert(END, location)

    def changeChannelList(*args, **kwargs):
        try:
            location = location_box.get(location_box.curselection()[0])
        except:
            return
        # Delete all old items in channel_box.
        channel_box.delete(0, channel_box.size())
        location = location_box.get(location_box.curselection()[0])
        NV.location = location
        if location == 'EMPTY':
            location = ''
        for channel in SH.networks[NV.network][NV.station][location][0]:
            channel_box.insert(END, channel)

    def getWaveform(*args, **kwargs):
        """
        """
        try:
            channel = channel_box.get(channel_box.curselection()[0])
        except:
            return
        network = NV.network
        station = NV.station
        location = NV.location
        if location == 'EMPTY':
            location = ''
        # Read the waveform
        start = UTCDateTime(2009, 8, 20, 6, 35, 0, 0)
        end = start + 60

        st = SH.client.waveform.getWaveform(network, station, location,
                                            channel, start, end)
        NV.stream = st
        # Merge the Stream and replace all masked values with NaNs.
        for trace in NV.stream:
            if np.ma.is_masked(trace):
                trace = trace.data[trace._mask] = np.NaN
        create_graph()


    # Root window.
    root = Tk()
    root.title('ObsPy Plotting')
    # Listboxes to select the channel.
    network_box = Listbox(root)
    network_box.grid(row=0, column=0)
    for network in SH.networks.keys():
        if network == 'Date':
            continue
        network_box.insert(END, network)
    station_box = Listbox(root)
    station_box.grid(row=0, column=1)
    location_box = Listbox(root)
    location_box.grid(row=0, column=2)
    channel_box = Listbox(root)
    channel_box.grid(row=0, column=3)

    f = figure.Figure(figsize=(9,4), dpi=100)
    a = f.add_subplot(111)
    a.set_yticks([])
    a.set_xticks([])
    
    # a tk.DrawingArea
    canvas = FigureCanvasTkAgg(f, master=root)
    #canvas.show()
    canvas.get_tk_widget().grid(row=1, column=0, columnspan=4)
    #canvas._tkcanvas.grid(row=1, column=0, columnspan=4)

    # Color picker canvas.
    colorcanvas = Canvas(root, width=50, height=50, bg='white')
    colorcanvas.grid()


    ## Colorpicker canvas.
    colorcanvas.create_rectangle(5, 5, 45, 45, fill='white', outline='black')
    colorcanvas.create_rectangle(8, 8, 43, 43, fill=NV.color, outline='')

    colorcanvas.bind("<Button-1>", choosecolor)
    network_box.bind('<ButtonRelease-1>', changeStationList)
    station_box.bind('<ButtonRelease-1>', changeLocationList)
    location_box.bind('<ButtonRelease-1>', changeChannelList)
    channel_box.bind('<ButtonRelease-1>', getWaveform)
    root.mainloop()

main()
