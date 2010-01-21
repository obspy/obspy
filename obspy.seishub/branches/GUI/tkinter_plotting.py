from Tkinter import *
from matplotlib import figure, use as matplotlibuse, rc as matplotlibrc
from obspy.core.utcdatetime import UTCDateTime
matplotlibuse('TkAgg')
matplotlibrc('figure.subplot', right=0.95, bottom=0.07, top=0.97) # set default
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, \
    NavigationToolbar2TkAgg
from obspy.core import UTCDateTime, Stream, Trace
#from obspy.imaging.waveform import _getMinMaxList as minmaxlist
from obspy.seishub import Client
import inspect
import numpy as np
from obspy.core import read
import os
import pickle
import sys
import time
import tkColorChooser
import tkFileDialog

################################################################################
## GLOBAL CONFIGURATION VARIABLES
# Maximum time span allowed to be retrieved via Seishub in seconds.
MAX_SPAN = 60 * 60

SERVER = "http://teide:8080"
################################################################################


class Seishub(object):
    def __init__(self):
        self.networks = {}
        self.client = Client(base_url=SERVER)
        self.pickle_file = os.path.join(
                    os.path.dirname(inspect.getsourcefile(self.__class__)),
                    'seishub_dict.pickle')
        self.networks = None

    def get_infos(self):
        try:
            file = open(self.pickle_file, 'rb')
            self.networks = pickle.load(file)
            file.close()
            try:
                self.networks['Server']
            except:
                self.reload_infos()
            if self.networks['Server'] == self.client.base_url:
                return
            self.networks = None
            self.reload_infos()
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
        # Also add the server to it.
        self.networks['Server'] = self.client.base_url
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
        self.selected_list = []
        self.jump_location = None
        self.starttime = None
        self.endtime = None

def main():
    """
    GUI Mainloop.
    """
    NV = NeededVars()
    # Get Seishub informations.
    SH = Seishub()
    try:
        SH.get_infos()
    except:
        msg = 'No connection to SeisHub server %s' % SH.client.base_url
        raise Exception(msg)
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

#    def openfile():
#        """
#        Opens the file, reads it and calculates a minmax list.
#        """
#        filename = tkFileDialog.askopenfilename()
#        if not filename:
#            return
#        # Read the file.
#        NV.st = read(filename)
#        st = NV.st
#        # Get minmaxlist.
#        NV.minmax = minmaxlist(st, 799, st[0].stats.starttime.timestamp,
#                            st[0].stats.endtime.timestamp)[2]
#        del NV.minmax[-1]
#        create_graph()

    def create_graph():
        """
        Creates the graph.
        """
        a.cla()
        if not NV.stream:
            a.set_xticks([])
            a.set_yticks([])
            canvas.show()
            return
        print NV.stream
        status_bar.configure(text='Plotting data...', foreground='black')
        status_bar.update_idletasks()
        length = len(NV.stream[0].data)
        a.set_xlim(0, 999)
        for _i in xrange(len(NV.stream)):
            _j = len(NV.stream) - 1 - _i
            # Normalize to length 1000.
            factor = np.arange(NV.stream[_i].stats.npts) / \
                    float(NV.stream[_i].stats.npts - 1) * 1000.00
            if len(NV.stream[_i].data):
                a.plot(factor, NV.stream[_i].data + (1 + _j * 3), color=NV.color)
            else:
                a.text(500, 1 + _j * 3, 'No data available.',
                       horizontalalignment='center',
                       verticalalignment='center', size='x-small')
        a.set_xticks([0, 500, 999])
        starttime = NV.stream[0].stats.starttime
        endtime = NV.stream[0].stats.endtime
        midtime = starttime + (endtime - starttime) / 2
        a.set_title(midtime.strftime('%Y-%m-%d'))
        a.set_xticklabels([starttime.strftime('%H:%M:%S'),
                           midtime.strftime('%H:%M:%S'),
                           endtime.strftime('%H:%M:%S')], size='x-small')
        a.set_yticks([1 + _i * 3 for _i in range(len(NV.stream))])
        ticklabels = ['%s.%s.%s.%s' % (trace.stats.network, trace.stats.station, trace.stats.location, trace.stats.channel) for trace in NV.stream]
        ticklabels.reverse()
        a.set_yticklabels(ticklabels, size='x-small')
        a.set_ylim(-1, 3 + (len(NV.stream) - 1) * 3)
        canvas.show()
        status_bar.configure(text='')
        status_bar.update_idletasks()

    def changeStationList(*args, **kwargs):
        # Delete all old items in station_box.
        station_box.delete(0, station_box.size())
        # Also delete location and channel box.
        location_box.delete(0, location_box.size())
        channel_box.delete(0, channel_box.size())
        network = network_box.get(network_box.curselection()[0])
        NV.network = network
        sorted_stations = SH.networks[network].keys()
        sorted_stations.sort()
        for station in sorted_stations:
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
        sorted_locations = SH.networks[NV.network][station].keys()
        sorted_locations.sort()
        for location in sorted_locations:
            if not location and len(sorted_locations) == 1:
                location = 'NOT AVAILABLE'
                NV.jump_location = location
                changeChannelList()
            location_box.insert(END, location)
            if len(sorted_locations) == 1 and location == 'NOT AVAILABLE':
                location_box.itemconfig(0, fg='#AAAAAA',
                                        selectforeground='#DCDCDC',
                                        selectbackground='white')

    def changeChannelList(*args, **kwargs):
        if not NV.jump_location:
            try:
                cur_location = location_box.get(location_box.curselection()[0])
            except:
                return
        else:
            cur_location = NV.jump_location
            NV.jump_location = None
        # Delete all old items in channel_box.
        channel_box.delete(0, channel_box.size())
        NV.location = cur_location
        if cur_location == 'NOT AVAILABLE':
            cur_location = ''
        sorted_channels = SH.networks[NV.network][NV.station][cur_location][0]
        sorted_channels.sort()
        channel_box.insert(END, '*')
        for channel in sorted_channels:
            channel_box.insert(END, channel)

    def addToSelectedList(*args, **kwargs):
        try:
            channel = channel_box.get(channel_box.curselection()[0])
        except:
            return
        updateSelectedList(channel)

    def clearList(*args, **kwargs):
        NV.selected_list = []
        selected_box.delete(0, selected_box.size())

    def refreshIndex(*args, **kwargs):
        status_bar.configure(text='Refreshing index...', foreground='black')
        status_bar.update_idletasks()
        SH.reload_infos()
        status_bar.configure(text='', foreground='black')
        status_bar.update_idletasks()
        info_label.configure(\
        text='Current Server: %s\nIndex last updated %s' \
            % (SH.networks['Server'],
              SH.networks['Date'].strftime('%Y-%m-%dT%H:%M:%S')))
        info_label.update_idletasks()

    def removeFromList(*args, **kwargs):
        try:
            del NV.selected_list[int(selected_box.curselection()[0])]
        except:
            return
        # Delete all old items in selected_box.
        selected_box.delete(0, selected_box.size())
        for item in NV.selected_list:
            selected_box.insert(END, item)

    def changeTime(*args, **kwargs):
        """
        Change the times of the plot.
        """
        timedict = {'-1 h' :-60 * 60, '-10 min' :-10 * 60,
                    'Current': 'NOW',
                    '+10 min': 10 * 60,
                    '+1 h': 60 * 60}
        timechange = timedict[args[0].widget.cget("text")]
        if isinstance(timechange, int):
            start = UTCDateTime(NV.starttime.get()) + timechange
            end = UTCDateTime(NV.endtime.get()) + timechange
        elif timechange == 'NOW':
            end = UTCDateTime()
            start = UTCDateTime() - 10 * 60
        else:
            import pdb;pdb.set_trace()
        NV.starttime.set(start.strftime('%Y-%m-%dT%H:%M:%S'))
        NV.endtime.set(end.strftime('%Y-%m-%dT%H:%M:%S'))
        getWaveform()

    def updateSelectedList(channel, *args, **kwargs):
        if NV.location == 'NOT AVAILABLE':
            cur_location = ''
        else:
            cur_location = NV.location
        if channel == '*':
            # Delete all old items in selected_box.
            selected_box.delete(0, selected_box.size())
            sorted_channels = SH.networks[NV.network][NV.station][cur_location][0]
            sorted_channels.sort()
            NV.selected_list.extend(['%s.%s.%s.%s' % (NV.network, NV.station,
                cur_location, _i) for _i in sorted_channels])
            # Remove duplicates.
            NV.selected_list = list(set(NV.selected_list))
            NV.selected_list.sort()
            for item in NV.selected_list:
                selected_box.insert(END, item)
            return
        cur_channel = '%s.%s.%s.%s' % \
                    (NV.network, NV.station, cur_location, channel)
        if cur_channel in NV.selected_list:
            return
        # Delete all old items in selected_box.
        selected_box.delete(0, channel_box.size())
        NV.selected_list.append(cur_channel)
        NV.selected_list.sort()
        for item in NV.selected_list:
            selected_box.insert(END, item)

    def getWaveform(*args, **kwargs):
        """
        Retrieves the waveforms and normalizes the graphs
        """
        # Check the two dates.
        try:
            st = UTCDateTime(NV.starttime.get())
        except:
            status_bar.configure(text='Please enter a valid start time.', foreground='red')
            status_bar.update_idletasks()
            return
        try:
            ed = UTCDateTime(NV.endtime.get())
        except:
            status_bar.configure(text='Please enter a valid end time.', foreground='red')
            status_bar.update_idletasks()
            return
        if ed - st <= 0:
            status_bar.configure(text='Start time need to be smaller than end time.', foreground='red')
            status_bar.update_idletasks()
            return
        now = UTCDateTime()
        if now < st:
            status_bar.configure(text='You cannot plot the future...', foreground='red')
            status_bar.update_idletasks()
            return
        if ed - st > MAX_SPAN:
            status_bar.configure(text='Timeframe too large. Maximal %s seconds allowed.' % MAX_SPAN, foreground='red')
            status_bar.update_idletasks()
            return
        stream_list = []
        if len(NV.selected_list) == 0:
            NV.stream = None
            create_graph()
            return
        status_bar.configure(text='Retrieving data...', foreground='black')
        status_bar.update_idletasks()
        for channel in NV.selected_list:
            # Read the waveform
            start = UTCDateTime(NV.starttime.get())
            end = UTCDateTime(NV.endtime.get())
            splitted = channel.split('.')
            network = splitted[0]
            station = splitted[1]
            location = splitted[2]
            channel = splitted[3]
            try:
                st = SH.client.waveform.getWaveform(network, station, location,
                                                channel, start, end)
            except:
                trace = Trace(header={'network' : network,
                    'station' : station, 'location' : location,
                    'channel' : channel, 'starttime': start,
                    'endtime' : end, 'npts' : 0, 'sampling_rate' : 1.0})
                st = Stream(traces=[trace])
            st.merge()
            st.trim(start, end)
            stream_list.append(st)
        st = stream_list[0]
        for _i in xrange(1, len(stream_list)):
            st += stream_list[_i]
        # Merge the Stream and replace all masked values with NaNs.
        st.merge()
        st.sort()
        # Normalize all traces and throw out traces with no data.
        try:
            max_diff = max([trace.data.max() - trace.data.min() for trace in st \
                        if len(trace.data) > 0])
        except:
            pass
        for trace in st:
            if (np.ma.is_masked(trace.data) and not False in trace.data._mask)or\
                len(trace.data) == 0:
                trace.data = np.array([])
            else:
                trace.data = trace.data - trace.data.mean()
                trace.data = trace.data / (max_diff / 2)
        NV.stream = st
        # Get the min. starttime and the max. endtime.
        starttime = UTCDateTime(NV.starttime.get())
        endtime = UTCDateTime(NV.endtime.get())
        for trace in NV.stream:
            if np.ma.is_masked(trace):
                trace = trace.data[trace._mask] = np.NaN
        # Loop over all traces again and fill with NaNs.
        for trace in NV.stream:
            startgaps = int(round((trace.stats.starttime - starttime) * \
                                trace.stats.sampling_rate))
            endgaps = int(round((endtime - trace.stats.endtime) * \
                                trace.stats.sampling_rate))
            print endgaps
            if startgaps or endgaps:
                if startgaps > 0:
                    start = np.empty(startgaps)
                    start[:] = np.NaN
                else:
                    start = []
                if endgaps > 0:
                    end = np.empty(endgaps)
                    end[:] = np.NaN
                else:
                    end = []
                trace.data = np.concatenate([start, trace.data, end])
                trace.stats.npts = trace.data.size
                trace.stats.starttime = UTCDateTime(NV.starttime.get())
                #trace.stats.endtime = UTCDateTime(NV.endtime.get())
        status_bar.configure(text='')
        status_bar.update_idletasks()
        create_graph()


    # Root window.
    root = Tk()
    NV.starttime = StringVar()
    NV.endtime = StringVar()
    NV.starttime.set((UTCDateTime() - 10 * 60).strftime('%Y-%m-%dT%H:%M:%S'))
    NV.endtime.set((UTCDateTime(NV.starttime.get()) + 10 * 60).strftime('%Y-%m-%dT%H:%M:%S'))
    root.title('SeisHub Waveform Demo')
    # Boxframe
    boxframe = Frame(root)
    boxframe.grid(row=1, column=0, columnspan=5)
    # Listboxes to select the channel.
    network_container = LabelFrame(boxframe, text='Network')
    network_container.grid(row=1, column=0, padx=20)
    network_box = Listbox(network_container)
    network_box.grid()
    sorted_network_list = SH.networks.keys()
    sorted_network_list.sort()
    for network in sorted_network_list:
        if network == 'Date' or network == 'Server':
            continue
        network_box.insert(END, network)
    station_container = LabelFrame(boxframe, text='Station')
    station_container.grid(row=1, column=1, padx=20)
    station_box = Listbox(station_container)
    station_box.grid()
    location_container = LabelFrame(boxframe, text='Location')
    location_container.grid(row=1, column=2, padx=20)
    location_box = Listbox(location_container)
    location_box.grid()
    channel_container = LabelFrame(boxframe, text='Channel')
    channel_container.grid(row=1, column=3, padx=20)
    channel_box = Listbox(channel_container)
    channel_box.grid()
    selected_container = LabelFrame(boxframe, text='Selected Channels')
    selected_container.grid(row=1, column=4, padx=20)
    selected_box = Listbox(selected_container)
    selected_box.grid()

    f = figure.Figure(figsize=(9.5, 4), dpi=100)
    a = f.add_subplot(111)
    a.set_yticks([])
    a.set_xticks([])

    # Entry widgets for date/time entry.
    time_container = LabelFrame(root, text='Timeframe')
    time_container.grid(row=2, column=0, columnspan=4, pady=4,
                        padx=8, sticky=W)

    starttime = Entry(time_container, textvariable=NV.starttime, width=18)
    starttime.grid(row=0, column=0)
    dash = Label(time_container, text='-')
    dash.grid(row=0, column=1)
    endtime = Entry(time_container, textvariable=NV.endtime, width=18)
    endtime.grid(row=0, column=2)
    bbackward_button = Button(time_container, text='-1 h')
    bbackward_button.grid(row=0, column=3)
    backward_button = Button(time_container, text='-10 min')
    backward_button.grid(row=0, column=4)
    now_button = Button(time_container, text='Current')
    now_button.grid(row=0, column=5)
    forward_button = Button(time_container, text='+10 min')
    forward_button.grid(row=0, column=6)
    fforward_button = Button(time_container, text='+1 h')
    fforward_button.grid(row=0, column=7)


    # Button frame.
    button_frame = Frame(root)
    button_frame.grid(row=2, column=4, sticky=E, pady=4, padx=7)
    # Clear list button
    clear_button = Button(button_frame, text='Clear List')
    clear_button.grid(row=0, column=0)
    # A plotting button.
    plot_button = Button(button_frame, text='Plot')
    plot_button.grid(row=0, column=1)


    # a tk.DrawingArea
    canvas = FigureCanvasTkAgg(f, master=root)
    #canvas.show()
    canvas.get_tk_widget().grid(row=3, column=0, columnspan=5)
    #canvas._tkcanvas.grid(row=1, column=0, columnspan=4)

    #toolbar = NavigationToolbar2TkAgg(canvas, root)
    #toolbar.grid(row=4, column=0, columnspan=5)


    # Color picker canvas.
    colorcanvas = Canvas(root, width=50, height=50, bg='white')
    colorcanvas.grid(row=5, column=0)

    bar_frame = Frame(root)
    bar_frame.grid(row=5, column=1, columnspan=2, pady=4)
    status_bar = Label(bar_frame, height=1, text='')
    status_bar.grid(row=0, column=0, columnspan=2, sticky=E)

    info_label = Label(bar_frame,
        text='Current Server: %s -- Index last updated %s' \
            % (SH.networks['Server'],
              SH.networks['Date'].strftime('%Y-%m-%dT%H:%M:%S')),
        foreground='#BBBBBB', anchor=E, justify=RIGHT)
    info_label.grid(row=1, column=0, columnspan=2, sticky=E)


    ## Colorpicker canvas.
    colorcanvas.create_rectangle(5, 5, 45, 45, fill='white', outline='black')
    colorcanvas.create_rectangle(8, 8, 43, 43, fill=NV.color, outline='')

    # Button frame.
    button_frame2 = Frame(root)
    button_frame2.grid(row=5, column=3, columnspan=2,
                       sticky=E, pady=4, padx=7)
    # Update index button
    update_button = Button(button_frame2, text='Update Index')
    update_button.grid(row=0, column=0)

    quit_button = Button(button_frame2, text='Quit', command=root.quit)
    quit_button.grid(row=0, column=1)
    # Check whether the servers exists.
    try:
        SH.client.waveform.getNetworkIds()
    except:
        status_bar.configure(text='NO CONNECTION TO SERVER',
                             foreground='RED')

    colorcanvas.bind("<Button-1>", choosecolor)
    plot_button.bind("<Button-1>", getWaveform)
    clear_button.bind("<Button-1>", clearList)
    update_button.bind("<Button-1>", refreshIndex)
    network_box.bind('<ButtonRelease-1>', changeStationList)
    station_box.bind('<ButtonRelease-1>', changeLocationList)
    location_box.bind('<ButtonRelease-1>', changeChannelList)
    channel_box.bind('<ButtonRelease-1>', addToSelectedList)
    selected_box.bind('<ButtonRelease-1>', removeFromList)
    # Bind time buttons.
    bbackward_button.bind("<Button-1>", changeTime)
    backward_button.bind("<Button-1>", changeTime)
    now_button.bind("<Button-1>", changeTime)
    forward_button.bind("<Button-1>", changeTime)
    fforward_button.bind("<Button-1>", changeTime)
    root.mainloop()

main()
