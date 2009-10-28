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
        self.minmax = None
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
        if NV.minmax:
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
        # Erase any old graphs.
        for _i in canvas.find_all():
            canvas.delete(_i)
        # Draw a white rectangle and erase any previous plot.
        #canvas.create_rectangle(0,0,900,400, fill = 'white', outline = 'white')
        # Create the outline again.
        canvas.create_line(50, 350, 850, 350, width=1)
        canvas.create_line(50, 350, 50, 50, width=1)
        canvas.create_line(50, 50, 850, 50, width=1)
        canvas.create_line(850, 50, 850, 350, width=1)
        minmax = NV.minmax
        # Figure out the range.
        xmin = min([_i[0] for _i in minmax])
        xmax = max([_i[1] for _i in minmax])
        x_range = float(300) / (xmax - xmin)
        # Loop over the list and draw the lines.
        start_value = 51
        for _i in xrange(len(minmax)):
            min_point = (350 - x_range * (minmax[_i][0] - xmin))
            max_point = int(min_point - x_range * (minmax[_i][1] - minmax[_i][0]))
            canvas.create_line(start_value, min_point, start_value, max_point, \
                               width=1, fill=NV.color)
            start_value += 1

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
        end = start + 60 * 30

        st = SH.client.waveform.getWaveform(network, station, location,
                                            channel, start, end)
        NV.st = st
        # Get minmaxlist.
        NV.minmax = minmaxlist(st, 799, st[0].stats.starttime.timestamp,
                               st[0].stats.endtime.timestamp)[2]
        del NV.minmax[-1]
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

    # Size of the window and set background to white.
    canvas = Canvas(root, width=900, height=400, bg='white')
    canvas.grid(row=1, column=0, columnspan=4)#    # Color picker canvas.
    colorcanvas = Canvas(root, width=50, height=50, bg='white')
    colorcanvas.grid()
    Button(root, text='Open File', command=openfile).grid()
    Button(root, text='Quit', command=root.quit).grid(row=3, column=1)
    # Layout of the canvas.
    # ------------------------------
    # |(50,50)             (850,50)|
    # |                            |
    # |                            |
    # |(50,350)           (850,350)|
    # ------------------------------
    canvas.create_line(50, 350, 850, 350, width=1)
    canvas.create_line(50, 350, 50, 50, width=1)
    canvas.create_line(50, 50, 850, 50, width=1)
    canvas.create_line(850, 50, 850, 350, width=1)

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
