#-------------------------------------------------------------------
# Filename: util.py
#  Purpose: Helper functions for ObsPyck
#   Author: Tobias Megies, Lion Krischer
#    Email: megies@geophysik.uni-muenchen.de
#  License: GPLv2
#
# Copyright (C) 2010 Tobias Megies, Lion Krischer
#---------------------------------------------------------------------

import os
import sys
import platform
import shutil
import subprocess
import copy
import tempfile
import glob
import fnmatch

import PyQt4
import numpy as np
import matplotlib as mpl
from matplotlib.colors import ColorConverter
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as QFigureCanvas
from matplotlib.widgets import MultiCursor as MplMultiCursor

from obspy.core import UTCDateTime


mpl.rc('figure.subplot', left=0.05, right=0.98, bottom=0.10, top=0.92,
       hspace=0.28)
mpl.rcParams['font.size'] = 10


COMMANDLINE_OPTIONS = (
        # XXX wasn't working as expected
        #(("--debug"), {'dest': "debug", 'action': "store_true",
        #        'default': False,
        #        'help': "Switch on Ipython debugging in case of exception"}),
        (("-t", "--time"), {'dest': "time",
                'help': "Starttime of seismogram to retrieve. It takes a "
                "string which UTCDateTime can convert. E.g. "
                "'2010-01-10T05:00:00'"}),
        (("-d", "--duration"), {'type': "float", 'dest': "duration",
                'help': "Duration of seismogram in seconds"}),
        (("-f", "--files"), {'type': "string", 'dest': "files",
                'help': "Local files containing waveform data. List of "
                "absolute paths separated by commas"}),
        (("--dataless",), {'type': "string", 'dest': "dataless",
                'help': "Local Dataless SEED files to look up metadata for "
                "local waveform files. List of absolute paths separated by "
                "commas"}),
        (("-i", "--seishub-ids"), {'dest': "seishub_ids", 'default': "",
                'help': "Ids to retrieve from SeisHub. Star for channel and "
                "wildcards for stations are allowed, e.g. "
                "'BW.RJOB..EH*,BW.RM?*..EH*'"}),
        (("--seishub-servername",), {'dest': "seishub_servername",
                'default': 'teide',
                'help': "Servername of the SeisHub server"}),
        (("--seishub-port",), {'type': "int", 'dest': "seishub_port",
                'default': 8080, 'help': "Port of the SeisHub server"}),
        (("--seishub-user",), {'dest': "seishub_user", 'default': 'obspyck',
                'help': "Username for SeisHub server"}),
        (("--seishub-password",), {'dest': "seishub_password",
                'default': 'obspyck', 'help': "Password for SeisHub server"}),
        (("--seishub-timeout",), {'dest': "seishub_timeout", 'type': "int",
                'default': 10, 'help': "Timeout for SeisHub server"}),
        (("-k", "--keys"), {'action': "store_true", 'dest': "keybindings",
                'default': False, 'help': "Show keybindings and quit"}),
        (("--lowpass",), {'type': "float", 'dest': "lowpass", 'default': 20.0,
                'help': "Frequency for Lowpass-Slider"}),
        (("--highpass",), {'type': "float", 'dest': "highpass", 'default': 1.0,
                'help': "Frequency for Highpass-Slider"}),
        (("--nozeromean",), {'action': "store_true", 'dest': "nozeromean",
                'default': False,
                'help': "Deactivate offset removal of traces"}),
        (("--nonormalization",), {'action': "store_true",
                'dest': "nonormalization", 'default': False,
                'help': "Deactivate normalization to nm/s for plotting " + \
                "using overall sensitivity (tr.stats.paz.sensitivity)"}),
        (("--pluginpath",), {'dest': "pluginpath",
                'default': "/baysoft/obspyck/",
                'help': "Path to local directory containing the folders with "
                "the files for the external programs. Large files/folders "
                "should only be linked in this directory as the contents are "
                "copied to a temporary directory (links are preserved)."}),
        (("-o", "--starttime-offset"), {'type': "float", 'dest': "starttime_offset",
                'default': 0.0, 'help': "Offset to add to specified starttime "
                "in seconds. Thus a time from an automatic picker can be used "
                "with a specified offset for the starttime. E.g. to request a "
                "waveform starting 30 seconds earlier than the specified time "
                "use -30."}),
        (("-m", "--merge"), {'type': "choice", 'dest': "merge", 'default': "",
                'choices': ("", "safe", "overwrite"),
                'help': "After fetching the streams run a merge "
                "operation on every stream. If not done, streams with gaps "
                "and therefore more traces per channel get discarded.\nTwo "
                "methods are supported (see http://docs.obspy.org/packages/"
                "auto/obspy.core.trace.Trace.__add__.html  for details)\n  "
                "\"safe\": overlaps are discarded "
                "completely\n  \"overwrite\": the second trace is used for "
                "overlapping parts of the trace"}),
        (("--arclink-ids",), {'dest': "arclink_ids", 'default': '',
                'help': "Ids to retrieve via arclink, star for channel "
                "is allowed, e.g. 'BW.RJOB..EH*,BW.ROTZ..EH*'"}),
        (("--arclink-servername",), {'dest': "arclink_servername",
                'default': 'webdc.eu',
                'help': "Servername of the arclink server"}),
        (("--arclink-port",), {'type': "int", 'dest': "arclink_port",
                'default': 18001, 'help': "Port of the arclink server"}),
        (("--arclink-user",), {'dest': "arclink_user", 'default': 'Anonymous',
                'help': "Username for arclink server"}),
        (("--arclink-password",), {'dest': "arclink_password", 'default': '',
                'help': "Password for arclink server"}),
        (("--arclink-institution",), {'dest': "arclink_institution",
                'default': 'Anonymous',
                'help': "Password for arclink server"}),
        (("--arclink-timeout",), {'dest': "arclink_timeout", 'type': "int",
                'default': 20, 'help': "Timeout for arclink server"}),
        (("--fissures-ids",), {'dest': "fissures_ids", 'default': '',
                'help': "Ids to retrieve via Fissures, star for component "
                "is allowed, e.g. 'GE.APE..BH*,GR.GRA1..BH*'"}),
        (("--fissures-network_dc",), {'dest': "fissures_network_dc",
                'default': ("/edu/iris/dmc", "IRIS_NetworkDC"),
                'help': "Tuple containing Fissures dns and NetworkDC name."}),
        (("--fissures-seismogram_dc",), {'dest': "fissures_seismogram_dc",
                'default': ("/edu/iris/dmc", "IRIS_DataCenter"),
                'help': "Tuple containing Fissures dns and DataCenter name."}),
        (("--fissures-name_service",), {'dest': "fissures_name_service",
                'default': "dmc.iris.washington.edu:6371/NameService",
                'help': "String containing the Fissures name service."}))
PROGRAMS = {
        'nlloc': {'filenames': {'exe': "NLLoc", 'phases': "nlloc.obs",
                                'summary': "nlloc.hyp",
                                'scatter': "nlloc.scat"}},
        'hyp_2000': {'filenames': {'exe': "hyp2000",'control': "bay2000.inp",
                                   'phases': "hyp2000.pha",
                                   'stations': "stations.dat",
                                   'summary': "hypo.prt"}},
        'focmec': {'filenames': {'exe': "rfocmec", 'phases': "focmec.dat",
                                 'stdout': "focmec.stdout",
                                 'summary': "focmec.out"}},
        '3dloc': {'filenames': {'exe': "3dloc_pitsa", 'out': "3dloc-out",
                                'in': "3dloc-in"}}}
SEISMIC_PHASES = ('P', 'S')
PHASE_COLORS = {'P': "red", 'S': "blue", 'Psynth': "black", 'Ssynth': "black",
        'Mag': "green", 'PErr1': "red", 'PErr2': "red", 'SErr1': "blue",
        'SErr2': "blue"}
PHASE_LINESTYLES = {'P': "-", 'S': "-", 'Psynth': "--", 'Ssynth': "--",
        'PErr1': "-", 'PErr2': "-", 'SErr1': "-", 'SErr2': "-"}
PHASE_LINEHEIGHT_PERC = {'P': 1, 'S': 1, 'Psynth': 1, 'Ssynth': 1,
        'PErr1': 0.75, 'PErr2': 0.75, 'SErr1': 0.75, 'SErr2': 0.75}
KEY_FULLNAMES = {'P': "P pick", 'Psynth': "synthetic P pick",
        'PWeight': "P pick weight", 'PPol': "P pick polarity",
        'POnset': "P pick onset", 'PErr1': "left P error pick",
        'PErr2': "right P error pick", 'S': "S pick",
        'Ssynth': "synthetic S pick", 'SWeight': "S pick weight",
        'SPol': "S pick polarity", 'SOnset': "S pick onset",
        'SErr1': "left S error pick", 'SErr2': "right S error pick",
        'MagMin1': "Magnitude minimum estimation pick",
        'MagMax1': "Magnitude maximum estimation pick",
        'MagMin2': "Magnitude minimum estimation pick",
        'MagMax2': "Magnitude maximum estimation pick"}
WIDGET_NAMES = ("qToolButton_clearAll", "qToolButton_clearOrigMag", "qToolButton_clearFocMec",
        "qToolButton_doHyp2000", "qToolButton_do3dloc", "qToolButton_doNlloc",
        "qComboBox_nllocModel", "qToolButton_calcMag", "qToolButton_doFocMec",
        "qToolButton_showMap", "qToolButton_showFocMec", "qToolButton_nextFocMec",
        "qToolButton_showWadati", "qToolButton_getNextEvent",
        "qToolButton_updateEventList", "qToolButton_sendEvent", "qCheckBox_publishEvent",
        "qToolButton_deleteEvent", "qCheckBox_sysop", "qLineEdit_sysopPassword",
        "qToolButton_previousStream", "qLabel_streamNumber", "qComboBox_streamName",
        "qToolButton_nextStream", "qToolButton_overview",
        "qComboBox_phaseType", "qToolButton_filter", "qComboBox_filterType",
        "qCheckBox_zerophase", "qLabel_highpass", "qDoubleSpinBox_highpass",
        "qLabel_lowpass", "qDoubleSpinBox_lowpass", "qToolButton_spectrogram",
        "qCheckBox_spectrogramLog", "qPlainTextEdit_stdout", "qPlainTextEdit_stderr")
#Estimating the maximum/minimum in a sample-window around click
MAG_PICKWINDOW = 10
MAG_MARKER = {'marker': "x", 'edgewidth': 1.8, 'size': 20}
AXVLINEWIDTH = 1.2
# dictionary for key-bindings.
KEYS = {'setPick': "a", 'setPickError': "s", 'delPick': "q",
        'setMagMin': "a", 'setMagMax': "s", 'delMagMinMax': "q",
        'switchPhase': "control",
        'prevStream': "y", 'nextStream': "x", 'switchWheelZoomAxis': "shift",
        'setWeight': {'0': 0, '1': 1, '2': 2, '3': 3},
        'setPol': {'u': "up", 'd': "down", '+': "poorup", '-': "poordown"},
        'setOnset': {'i': "impulsive", 'e': "emergent"}}
# XXX Qt:
#KEYS = {'setPick': "Key_A", 'setPickError': "Key_S", 'delPick': "Key_Q",
#        'setMagMin': "Key_A", 'setMagMax': "Key_S", 'delMagMinMax': "Key_Q",
#        'switchPhase': "Key_Control",
#        'prevStream': "Key_Y", 'nextStream': "Key_X", 'switchWheelZoomAxis': "Key_Shift",
#        'setWeight': {'Key_0': 0, 'Key_1': 1, 'Key_2': 2, 'Key_3': 3},
#        'setPol': {'Key_U': "up", 'Key_D': "down", 'Key_Plus': "poorup", 'Key_Minus': "poordown"},
#        'setOnset': {'Key_I': "impulsive", 'Key_E': "emergent"}}
# the following dicts' keys should be all lower case, we use "".lower() later
POLARITY_CHARS = {'up': "U", 'down': "D", 'poorup': "+", 'poordown': "-"}
ONSET_CHARS = {'impulsive': "I", 'emergent': "E",
               'implusive': "I"} # XXX some old events have a typo there... =)


class QMplCanvas(QFigureCanvas):
    """
    Class to represent the FigureCanvas widget.
    """
    def __init__(self, parent=None):
        # Standard Matplotlib code to generate the plot
        self.fig = Figure()
        # initialize the canvas where the Figure renders into
        QFigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

def matplotlib_color_to_rgb(color):
    """
    Converts matplotlib colors to rgb.
    """
    rgb = ColorConverter().to_rgb(color)
    return [int(_i*255) for _i in rgb]

def check_keybinding_conflicts(keys):
    """
    check for conflicting keybindings. 
    we have to check twice, because keys for setting picks and magnitudes
    are allowed to interfere...
    """
    for ignored_key_list in [['setMagMin', 'setMagMax', 'delMagMinMax'],
                             ['setPick', 'setPickError', 'delPick']]:
        tmp_keys = copy.deepcopy(keys)
        tmp_keys2 = {}
        for ignored_key in ignored_key_list:
            tmp_keys.pop(ignored_key)
        while tmp_keys:
            key, item = tmp_keys.popitem()
            if isinstance(item, dict):
                while item:
                    k, v = item.popitem()
                    tmp_keys2["_".join([key, str(v)])] = k
            else:
                tmp_keys2[key] = item
        if len(set(tmp_keys2.keys())) != len(set(tmp_keys2.values())):
            err = "Interfering keybindings. Please check variable KEYS"
            raise Exception(err)

def fetch_waveforms_with_metadata(options):
    """
    Sets up obspy clients and fetches waveforms and metadata according to command
    line options.
    Now also fetches data via arclink (fissures) if --arclink-ids
    (--fissures-ids) is used.
    XXX Notes: XXX
     - there is a problem in the arclink client with duplicate traces in
       fetched streams. therefore at the moment it might be necessary to use
       "-m overwrite" option.

    :returns: (dictionary with clients,
               list(:class:`obspy.core.stream.Stream`s))
    """
    t1 = UTCDateTime(options.time) + options.starttime_offset
    t2 = t1 + options.duration
    streams = []
    clients = {}
    sta_fetched = set()
    # Local files:
    if options.files and options.dataless:
        from obspy.core import read
        from obspy.xseed import Parser
        print "=" * 80
        print "Reading local files:"
        print "-" * 80
        parsers = []
        for file in options.dataless.split(","):
            print file
            parsers.append(Parser(file))
        for file in options.files.split(","):
            print file
            st = read(file, starttime=t1, endtime=t2)
            for tr in st:
                for parser in parsers:
                    try:
                        tr.stats.paz = parser.getPAZ(tr.id, tr.stats.starttime)
                        tr.stats.coordinates = parser.getCoordinates(tr.id, tr.stats.starttime)
                        break
                    except:
                        continue
                    print "found no metadata for %s!!!" % file
            streams.append(st)
    # SeisHub
    if options.seishub_ids:
        from obspy.seishub import Client
        print "=" * 80
        print "Fetching waveforms and metadata from SeisHub:"
        print "-" * 80
        baseurl = "http://" + options.seishub_servername + ":%i" % options.seishub_port
        client = Client(base_url=baseurl, user=options.seishub_user,
                        password=options.seishub_password, timeout=options.seishub_timeout)
        for id in options.seishub_ids.split(","):
            net, sta_wildcard, loc, cha = id.split(".")
            stations_to_fetch = []
            if "?" in sta_wildcard or "*" in sta_wildcard:
                for sta in sorted(client.waveform.getStationIds(network=net)):
                    if fnmatch.fnmatch(sta, sta_wildcard):
                        stations_to_fetch.append(sta)
            else:
                stations_to_fetch = [sta_wildcard]
            for sta in stations_to_fetch:
                # make sure we dont fetch a single station of
                # one network twice (could happen with wildcards)
                net_sta = "%s.%s" % (net, sta)
                if net_sta in sta_fetched:
                    print "%s skipped! (Was already retrieved)" % net_sta.ljust(8)
                    continue
                try:
                    sys.stdout.write("\r%s ..." % net_sta.ljust(8))
                    sys.stdout.flush()
                    st = client.waveform.getWaveform(net, sta, loc, cha, t1,
                            t2, apply_filter=True, getPAZ=True,
                            getCoordinates=True)
                    sta_fetched.add(net_sta)
                    sys.stdout.write("\r%s fetched.\n" % net_sta.ljust(8))
                    sys.stdout.flush()
                except Exception, e:
                    sys.stdout.write("\r%s skipped! (Server replied: %s)\n" % (net_sta.ljust(8), e))
                    sys.stdout.flush()
                    continue
                for tr in st:
                    tr.stats['_format'] = "SeisHub"
                streams.append(st)
        clients['SeisHub'] = client
    # ArcLink
    if options.arclink_ids:
        from obspy.arclink import Client
        print "=" * 80
        print "Fetching waveforms and metadata via ArcLink:"
        print "-" * 80
        client = Client(host=options.arclink_servername,
                        port=options.arclink_port,
                        timeout=options.arclink_timeout,
                        user=options.arclink_user,
                        password=options.arclink_password,
                        institution=options.arclink_institution)
        for id in options.arclink_ids.split(","):
            net, sta, loc, cha = id.split(".")
            net_sta = "%s.%s" % (net, sta)
            if net_sta in sta_fetched:
                print "%s skipped! (Was already retrieved)" % net_sta.ljust(8)
                continue
            try:
                sys.stdout.write("\r%s ..." % net_sta.ljust(8))
                sys.stdout.flush()
                st = client.getWaveform(network=net, station=sta,
                                        location=loc, channel=cha,
                                        starttime=t1, endtime=t2,
                                        getPAZ=True, getCoordinates=True)
                sta_fetched.add(net_sta)
                sys.stdout.write("\r%s fetched.\n" % net_sta.ljust(8))
                sys.stdout.flush()
            except Exception, e:
                sys.stdout.write("\r%s skipped! (Server replied: %s)\n" % (net_sta.ljust(8), e))
                sys.stdout.flush()
                continue
            for tr in st:
                tr.stats['_format'] = "ArcLink"
            streams.append(st)
        clients['ArcLink'] = client
    # Fissures
    if options.fissures_ids:
        from obspy.fissures import Client
        print "=" * 80
        print "Fetching waveforms and metadata via Fissures:"
        print "-" * 80
        client = Client(network_dc=options.fissures_network_dc,
                        seismogram_dc=options.fissures_seismogram_dc,
                        name_service=options.fissures_name_service)
        for id in options.fissures_ids.split(","):
            net, sta, loc, cha = id.split(".")
            net_sta = "%s.%s" % (net, sta)
            if net_sta in sta_fetched:
                print "%s skipped! (Was already retrieved)" % net_sta.ljust(8)
                continue
            try:
                sys.stdout.write("\r%s ..." % net_sta.ljust(8))
                sys.stdout.flush()
                st = client.getWaveform(network=net, station=sta,
                                        location=loc, channel=cha,
                                        starttime=t1, endtime=t2,
                                        getPAZ=True, getCoordinates=True)
                sta_fetched.add(net_sta)
                sys.stdout.write("\r%s fetched.\n" % net_sta.ljust(8))
                sys.stdout.flush()
            except Exception, e:
                sys.stdout.write("\r%s skipped! (Server replied: %s)\n" % (net_sta.ljust(8), e))
                sys.stdout.flush()
                continue
            for tr in st:
                tr.stats['_format'] = "Fissures"
            streams.append(st)
        clients['Fissures'] = client
    print "=" * 80
    return (clients, streams)

def merge_check_and_cleanup_streams(streams, options):
    """
    Cleanup given list of streams so that they conform with what ObsPyck
    expects.

    Conditions:
    - either one Z or three ZNE traces
    - no two streams for any station (of same network)
    - no streams with traces of different stations

    :returns: (warn_msg, merge_msg, list(:class:`obspy.core.stream.Stream`s))
    """
    # Merge on every stream if this option is passed on command line:
    if options.merge:
        if options.merge.lower() == "safe":
            for st in streams:
                st.merge(method=0)
        elif options.merge.lower() == "overwrite":
            for st in streams:
                st.merge(method=1)
        else:
            err = "Unrecognized option for merging traces. Try " + \
                  "\"safe\" or \"overwrite\"."
            raise Exception(err)

    # Sort streams again, if there was a merge this could be necessary 
    for st in streams:
        st.sort()
        st.reverse()
    sta_list = set()
    # we need to go through streams/dicts backwards in order not to get
    # problems because of the pop() statement
    warn_msg = ""
    merge_msg = ""
    # XXX we need the list() because otherwise the iterator gets garbled if
    # XXX removing streams inside the for loop!!
    for st in list(streams):
        # check for streams with mixed stations/networks and remove them
        if len(st) != len(st.select(network=st[0].stats.network,
                                    station=st[0].stats.station)):
            msg = "Warning: Stream with a mix of stations/networks. " + \
                  "Discarding stream."
            print msg
            warn_msg += msg + "\n"
            streams.remove(st)
            continue
        net_sta = "%s.%s" % (st[0].stats.network.strip(),
                             st[0].stats.station.strip())
        # Here we make sure that a station/network combination is not
        # present with two streams.
        if net_sta in sta_list:
            msg = "Warning: Station/Network combination \"%s\" " + \
                  "already in stream list. Discarding stream." % net_sta
            print msg
            warn_msg += msg + "\n"
            streams.remove(st)
            continue
        if len(st) not in [1, 3]:
            msg = 'Warning: All streams must have either one Z trace ' + \
                  'or a set of three ZNE traces.'
            print msg
            warn_msg += msg + "\n"
            # remove all unknown channels ending with something other than
            # Z/N/E and try again...
            removed_channels = ""
            for tr in st:
                if tr.stats.channel[-1] not in ["Z", "N", "E"]:
                    removed_channels += " " + tr.stats.channel
                    st.remove(tr)
            if len(st.traces) in [1, 3]:
                msg = 'Warning: deleted some unknown channels in ' + \
                      'stream %s.%s' % (net_sta, removed_channels)
                print msg
                warn_msg += msg + "\n"
                continue
            else:
                msg = 'Stream %s discarded.\n' % net_sta + \
                      'Reason: Number of traces != (1 or 3)'
                print msg
                warn_msg += msg + "\n"
                #for j, tr in enumerate(st.traces):
                #    msg = 'Trace no. %i in Stream: %s\n%s' % \
                #            (j + 1, tr.stats.channel, tr.stats)
                msg = str(st)
                print msg
                warn_msg += msg + "\n"
                streams.remove(st)
                merge_msg = '\nIMPORTANT:\nYou can try the command line ' + \
                        'option merge (-m safe or -m overwrite) to ' + \
                        'avoid losing streams due gaps/overlaps.'
                continue
        if len(st) == 1 and st[0].stats.channel[-1] != 'Z':
            msg = 'Warning: All streams must have either one Z trace ' + \
                  'or a set of three ZNE traces.'
            msg += 'Stream %s discarded. Reason: ' % net_sta + \
                   'Exactly one trace present but this is no Z trace'
            print msg
            warn_msg += msg + "\n"
            #for j, tr in enumerate(st.traces):
            #    msg = 'Trace no. %i in Stream: %s\n%s' % \
            #            (j + 1, tr.stats.channel, tr.stats)
            msg = str(st)
            print msg
            warn_msg += msg + "\n"
            streams.remove(st)
            continue
        if len(st) == 3 and (st[0].stats.channel[-1] != 'Z' or
                             st[1].stats.channel[-1] != 'N' or
                             st[2].stats.channel[-1] != 'E'):
            msg = 'Warning: All streams must have either one Z trace ' + \
                  'or a set of three ZNE traces.'
            msg += 'Stream %s discarded. Reason: ' % net_sta + \
                   'Exactly three traces present but they are not ZNE'
            print msg
            warn_msg += msg + "\n"
            #for j, tr in enumerate(st.traces):
            #    msg = 'Trace no. %i in Stream: %s\n%s' % \
            #            (j + 1, tr.stats.channel, tr.stats)
            msg = str(st)
            print msg
            warn_msg += msg + "\n"
            streams.remove(st)
            continue
        sta_list.add(net_sta)
    # demean traces if not explicitly deactivated on command line
    if not options.nozeromean:
        for st in streams:
            for tr in st:
                tr.data = tr.data - tr.data.mean()
    return (warn_msg, merge_msg, streams)

def setup_dicts(streams):
    """
    Function to set up the list of dictionaries that is used alongside the
    streams list.
    Also removes streams that do not provide the necessary metadata.

    :returns: (list(:class:`obspy.core.stream.Stream`s),
               list(dict))
    """
    #set up a list of dictionaries to store all picking data
    # set all station magnitude use-flags False
    dicts = []
    for i in xrange(len(streams)):
        dicts.append({})
    # we need to go through streams/dicts backwards in order not to get
    # problems because of the pop() statement
    for i in range(len(streams))[::-1]:
        dict = dicts[i]
        st = streams[i]
        trZ = st.select(component="Z")[0]
        if len(st) == 3:
            trN = st.select(component="N")[0]
            trE = st.select(component="E")[0]
        dict['MagUse'] = True
        sta = trZ.stats.station.strip()
        dict['Station'] = sta
        #XXX not used: dictsMap[sta] = dict
        # XXX should not be necessary
        #if net == '':
        #    net = 'BW'
        #    print "Warning: Got no network information, setting to " + \
        #          "default: BW"
        try:
            dict['StaLon'] = trZ.stats.coordinates.longitude
            dict['StaLat'] = trZ.stats.coordinates.latitude
            dict['StaEle'] = trZ.stats.coordinates.elevation / 1000. # all depths in km!
            dict['pazZ'] = trZ.stats.paz
            if len(st) == 3:
                dict['pazN'] = trN.stats.paz
                dict['pazE'] = trE.stats.paz
        except:
            net = trZ.stats.network.strip()
            print 'Error: Missing metadata for %s. Discarding stream.' \
                    % (":".join([net, sta]))
            streams.pop(i)
            dicts.pop(i)
            continue
    return streams, dicts

def setup_external_programs(options):
    """
    Sets up temdir, copies program files, fills in PROGRAMS dict, sets up
    system calls for programs.
    Depends on command line options, returns temporary directory.

    :param options: Command line options of ObsPyck
    :type options: options as returned by :meth:`optparse.OptionParser.parse_args`
    :returns: String representation of temporary directory with program files.
    """
    tmp_dir = tempfile.mkdtemp()
    # set binary names to use depending on architecture and platform...
    env = os.environ
    architecture = platform.architecture()[0]
    system = platform.system()
    global SHELL
    if system == "Windows":
        SHELL = True
    else:
        SHELL = False
    # Setup external programs #############################################
    for prog_basename, prog_dict in PROGRAMS.iteritems():
        prog_srcpath = os.path.join(options.pluginpath, prog_basename)
        prog_tmpdir = os.path.join(tmp_dir, prog_basename)
        prog_dict['dir'] = prog_tmpdir
        shutil.copytree(prog_srcpath, prog_tmpdir, symlinks=True)
        prog_dict['files'] = {}
        for key, filename in prog_dict['filenames'].iteritems():
            prog_dict['files'][key] = os.path.join(prog_tmpdir, filename)
        prog_dict['files']['exe'] = "__".join(\
                [prog_dict['filenames']['exe'], system, architecture])
        # setup clean environment
        prog_dict['env'] = {}
        prog_dict['env']['PATH'] = prog_dict['dir'] + os.pathsep + env['PATH']
        if 'SystemRoot' in env:
            prog_dict['env']['SystemRoot'] = env['SystemRoot']
    # 3dloc ###############################################################
    prog_dict = PROGRAMS['3dloc']
    prog_dict['env']['D3_VELOCITY'] = \
            os.path.join(prog_dict['dir'], 'D3_VELOCITY') + os.sep
    prog_dict['env']['D3_VELOCITY_2'] = \
            os.path.join(prog_dict['dir'], 'D3_VELOCITY_2') + os.sep
    def tmp(prog_dict):
        files = prog_dict['files']
        for file in [files['out'], files['in']]:
            if os.path.isfile(file):
                os.remove(file)
        return
    prog_dict['PreCall'] = tmp
    def tmp(prog_dict):
        sub = subprocess.Popen(prog_dict['files']['exe'], shell=SHELL,
                cwd=prog_dict['dir'], env=prog_dict['env'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if system == "Darwin":
            returncode = sub.returncode
        else:
            returncode = sub.wait()
        msg = "".join(sub.stdout.readlines())
        err = "".join(sub.stderr.readlines())
        return (msg, err, returncode)
    prog_dict['Call'] = tmp
    # Hyp2000 #############################################################
    prog_dict = PROGRAMS['hyp_2000']
    prog_dict['env']['HYP2000_DATA'] = prog_dict['dir'] + os.sep
    def tmp(prog_dict):
        files = prog_dict['files']
        for file in [files['phases'], files['stations'], files['summary']]:
            if os.path.isfile(file):
                os.remove(file)
        return
    prog_dict['PreCall'] = tmp
    def tmp(prog_dict):
        sub = subprocess.Popen(prog_dict['files']['exe'], shell=SHELL,
                cwd=prog_dict['dir'], env=prog_dict['env'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        input = open(prog_dict['files']['control'], "rt").read()
        (msg, err) = sub.communicate(input)
        if system == "Darwin":
            returncode = sub.returncode
        else:
            returncode = sub.wait()
        return (msg, err, returncode)
    prog_dict['Call'] = tmp
    # NLLoc ###############################################################
    prog_dict = PROGRAMS['nlloc']
    def tmp(prog_dict):
        filepattern = os.path.join(prog_dict['dir'], "nlloc*")
        print filepattern
        for file in glob.glob(filepattern):
            os.remove(file)
        return
    prog_dict['PreCall'] = tmp
    def tmp(prog_dict, controlfilename):
        sub = subprocess.Popen([prog_dict['files']['exe'], controlfilename],
                cwd=prog_dict['dir'], env=prog_dict['env'], shell=SHELL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if system == "Darwin":
            returncode = sub.returncode
        else:
            returncode = sub.wait()
        msg = "".join(sub.stdout.readlines())
        err = "".join(sub.stderr.readlines())
        for pattern, key in [("nlloc.*.*.*.loc.scat", 'scatter'),
                             ("nlloc.*.*.*.loc.hyp", 'summary')]:
            pattern = os.path.join(prog_dict['dir'], pattern)
            newname = os.path.join(prog_dict['dir'], prog_dict['files'][key])
            for file in glob.glob(pattern):
                os.rename(file, newname)
        return (msg, err, returncode)
    prog_dict['Call'] = tmp
    # focmec ##############################################################
    prog_dict = PROGRAMS['focmec']
    def tmp(prog_dict):
        sub = subprocess.Popen(prog_dict['files']['exe'], shell=SHELL,
                cwd=prog_dict['dir'], env=prog_dict['env'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if system == "Darwin":
            returncode = sub.returncode
        else:
            returncode = sub.wait()
        msg = "".join(sub.stdout.readlines())
        err = "".join(sub.stderr.readlines())
        return (msg, err, returncode)
    prog_dict['Call'] = tmp
    #######################################################################
    return tmp_dir

#Monkey patch (need to remember the ids of the mpl_connect-statements to remove them later)
#See source: http://matplotlib.sourcearchive.com/documentation/0.98.1/widgets_8py-source.html
class MultiCursor(MplMultiCursor):
    def __init__(self, canvas, axes, useblit=True, **lineprops):
        self.canvas = canvas
        self.axes = axes
        xmin, xmax = axes[-1].get_xlim()
        xmid = 0.5*(xmin+xmax)
        self.lines = [ax.axvline(xmid, visible=False, **lineprops) for ax in axes]
        self.visible = True
        self.useblit = useblit
        self.background = None
        self.needclear = False
        self.id1=self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.id2=self.canvas.mpl_connect('draw_event', self.clear)
    
   
def gk2lonlat(x, y, m_to_km=True):
    """
    This function converts X/Y Gauss-Krueger coordinates (zone 4, central
    meridian 12 deg) to Longitude/Latitude in WGS84 reference ellipsoid.
    We do this using pyproj (python bindings for proj4) which can be installed
    using 'easy_install pyproj' from pypi.python.org.
    Input can be single coordinates or coordinate lists/arrays.
    
    Useful Links:
    http://pyproj.googlecode.com/svn/trunk/README.html
    http://trac.osgeo.org/proj/
    http://www.epsg-registry.org/
    """
    import pyproj

    proj_wgs84 = pyproj.Proj(init="epsg:4326")
    proj_gk4 = pyproj.Proj(init="epsg:31468")
    # convert to meters first
    if m_to_km:
        x = x * 1000.
        y = y * 1000.
    lon, lat = pyproj.transform(proj_gk4, proj_wgs84, x, y)
    return (lon, lat)

def readNLLocScatter(scat_filename, textviewStdErrImproved):
    """
    This function reads location and values of pdf scatter samples from the
    specified NLLoc *.scat binary file (type "<f4", 4 header values, then 4
    floats per sample: x, y, z, pdf value) and converts X/Y Gauss-Krueger
    coordinates (zone 4, central meridian 12 deg) to Longitude/Latitude in
    WGS84 reference ellipsoid.
    We do this using the Linux command line tool cs2cs.
    Messages on stderr are written to specified GUI textview.
    Returns an array of xy pairs.
    """
    # read data, omit the first 4 values (header information) and reshape
    data = np.fromfile(scat_filename, dtype="<f4").astype("float")[4:]
    data = data.reshape((len(data)/4, 4)).swapaxes(0, 1)
    lon, lat = gk2lonlat(data[0], data[1])
    return np.vstack((lon, lat, data[2]))

def errorEllipsoid2CartesianErrors(azimuth1, dip1, len1, azimuth2, dip2, len2,
                                   len3):
    """
    This method converts the location error of NLLoc given as the 3D error
    ellipsoid (two azimuths, two dips and three axis lengths) to a cartesian
    representation.
    We calculate the cartesian representation of each of the ellipsoids three
    eigenvectors and use the maximum of these vectors components on every axis.
    """
    z = len1 * np.sin(np.radians(dip1))
    xy = len1 * np.cos(np.radians(dip1))
    x = xy * np.sin(np.radians(azimuth1))
    y = xy * np.cos(np.radians(azimuth1))
    v1 = np.array([x, y, z])

    z = len2 * np.sin(np.radians(dip2))
    xy = len2 * np.cos(np.radians(dip2))
    x = xy * np.sin(np.radians(azimuth2))
    y = xy * np.cos(np.radians(azimuth2))
    v2 = np.array([x, y, z])

    v3 = np.cross(v1, v2)
    v3 /= np.sqrt(np.dot(v3, v3))
    v3 *= len3

    v1 = np.abs(v1)
    v2 = np.abs(v2)
    v3 = np.abs(v3)

    error_x = max([v1[0], v2[0], v3[0]])
    error_y = max([v1[1], v2[1], v3[1]])
    error_z = max([v1[2], v2[2], v3[2]])
    
    return (error_x, error_y, error_z)

def formatXTicklabels(x, *pos):
    """
    Make a nice formatting for y axis ticklabels: minutes:seconds.microsec
    """
    # x is of type numpy.float64, the string representation of that float
    # strips of all tailing zeros
    # pos returns the position of x on the axis while zooming, None otherwise
    min = int(x / 60.)
    if min > 0:
        sec = x % 60
        return "%i:%06.3f" % (min, sec)
    else:
        return "%.3f" % x

def map_qKeys(key_dict):
    """
    Map Dictionary of form {'functionality': "Qt_Key_name"} to
    {'functionality': Qt_Key_Code} for use in check against event Keys.

    >>> KEYS = {'delMagMinMax': 'Key_Escape',
    ...         'delPick': 'Key_Escape',
    ...         'nextStream': 'Key_X',
    ...         'prevStream': 'Key_Y',
    ...         'setPol': {'Key_D': 'down',
    ...                    'Key_Minus': 'poordown',
    ...                    'Key_Plus': 'poorup',
    ...                    'Key_U': 'up'},
    ...         'setWeight': {'Key_0': 0, 'Key_1': 1, 'Key_2': 2, 'Key_3': 3},
    ...         'switchWheelZoomAxis': 'Key_Shift'}
    >>> map_qKeys(KEYS)
    {'delMagMinMax': 16777216,
     'delPick': 16777216,
     'nextStream': 88,
     'prevStream': 89,
     'setPol': {43: 'poorup', 45: 'poordown', 68: 'down', 85: 'up'},
     'setWeight': {48: 0, 49: 1, 50: 2, 51: 3},
     'switchWheelZoomAxis': 16777248}
    """
    Qt = PyQt4.QtCore.Qt
    for functionality, key_name in key_dict.iteritems():
        if isinstance(key_name, str):
            key_dict[functionality] = getattr(Qt, key_name)
        # sometimes we get a nested dictionary (e.g. "setWeight")...
        elif isinstance(key_name, dict):
            nested_dict = key_name
            new = {}
            for key_name, value in nested_dict.iteritems():
                new[getattr(Qt, key_name)] = value
            key_dict[functionality] = new
    return key_dict

class SplitWriter():
    """
    Implements a write method that writes a given message on all children
    """
    def __init__(self, *objects):
        """
        Remember provided objects as children.
        """
        self.children = objects

    def write(self, msg):
        """
        Sends msg to all childrens write method.
        """
        for obj in self.children:
            if isinstance(obj, PyQt4.QtGui.QPlainTextEdit):
                if msg == '\n':
                    return
                obj.appendPlainText(msg)
            else:
                obj.write(msg)
