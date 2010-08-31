COMMANDLINE_OPTIONS = [
        [["-t", "--time"], {'dest': "time", 'default': '2009-07-21T04:33:00',
                'help': "Starttime of seismogram to retrieve. It takes a "
                "string which UTCDateTime can convert. E.g. "
                "'2010-01-10T05:00:00'"}],
        [["-d", "--duration"], {'type': "float", 'dest': "duration",
                'default': 120, 'help': "Duration of seismogram in seconds"}],
        [["-i", "--ids"], {'dest': "ids",
                'default': 'BW.RJOB..EH*,BW.RMOA..EH*',
                'help': "Ids to retrieve, star for channel and wildcards for "
                "stations are allowed, e.g. 'BW.RJOB..EH*,BW.RM?*..EH*'"}],
        [["-s", "--servername"], {'dest': "servername", 'default': 'teide',
                'help': "Servername of the seishub server"}],
        [["-p", "--port"], {'type': "int", 'dest': "port", 'default': 8080,
                'help': "Port of the seishub server"}],
        [["--user"], {'dest': "user", 'default': 'obspyck',
                'help': "Username for seishub server"}],
        [["--password"], {'dest': "password", 'default': 'obspyck',
                'help': "Password for seishub server"}],
        [["--timeout"], {'dest': "timeout", 'type': "int", 'default': 10,
                'help': "Timeout for seishub server"}],
        [["-k", "--keys"], {'action': "store_true", 'dest': "keybindings",
                'default': False, 'help': "Show keybindings and quit"}],
        [["--lowpass"], {'type': "float", 'dest': "lowpass", 'default': 20.0,
                'help': "Frequency for Lowpass-Slider"}],
        [["--highpass"], {'type': "float", 'dest': "highpass", 'default': 1.0,
                'help': "Frequency for Highpass-Slider"}],
        [["--nozeromean"], {'action': "store_true", 'dest': "nozeromean",
                'default': False,
                'help': "Deactivate offset removal of traces"}],
        [["--pluginpath"], {'dest': "pluginpath",
                'default': "/baysoft/obspyck/",
                'help': "Path to local directory containing the folders with "
                "the files for the external programs"}],
        [["--starttime-offset"], {'type': "float", 'dest': "starttime_offset",
                'default': 0.0, 'help': "Offset to add to specified starttime "
                "in seconds. Thus a time from an automatic picker can be used "
                "with a specified offset for the starttime. E.g. to request a "
                "waveform starting 30 seconds earlier than the specified time "
                "use -30."}],
        [["-m", "--merge"], {'type': "string", 'dest': "merge", 'default': "",
                'help': "After fetching the streams from seishub run a merge "
                "operation on every stream. If not done, streams with gaps "
                "and therefore more traces per channel get discarded.\nTwo "
                "methods are supported (see http://svn.geophysik.uni-muenchen"
                ".de/obspy/docs/packages/auto/obspy.core.trace.Trace.__add__"
                ".html for details)\n  \"safe\": overlaps are discarded "
                "completely\n  \"overwrite\": the second trace is used for "
                "overlapping parts of the trace"}]]
PROGRAMS = {
        'nlloc': {'filenames': {'exe': "NLLoc", 'phases': "nlloc.obs", 'summary': "nlloc.hyp", 'scatter': "nlloc.scat"}},
        'hyp_2000': {'filenames': {'exe': "hyp2000",'control': "bay2000.inp", 'phases': "hyp2000.pha", 'stations': "stations.dat", 'summary': "hypo.prt"}},
        'focmec': {'filenames': {'exe': "rfocmec", 'phases': "focmec.dat", 'stdout': "focmec.stdout", 'summary': "focmec.out"}},
        '3dloc': {'filenames': {'exe': "3dloc_pitsa", 'out': "3dloc-out", 'in': "3dloc-in"}}}
SEISMIC_PHASES = ['P', 'S']
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

#Estimating the maximum/minimum in a sample-window around click
MAG_PICKWINDOW = 10
MAG_MARKER = {'marker': "x", 'edgewidth': 1.8, 'size': 20}
AXVLINEWIDTH = 1.2
#dictionary for key-bindings
KEYS = {'setPick': 'alt', 'setPickError': ' ', 'delPick': 'escape',
        'setMagMin': 'alt', 'setMagMax': ' ', 'delMagMinMax': 'escape',
        'switchPhase': 'control', 'switchPan': 'p',
        'prevStream': 'y', 'nextStream': 'x', 'switchWheelZoomAxis': 'shift',
        'setWeight': {'0': 0, '1': 1, '2': 2, '3': 3},
        'setPol': {'u': "up", 'd': "down", '+': "poorup", '-': "poordown"},
        'setOnset': {'i': "impulsive", 'e': "emergent"}}
# the following dicts' keys should be all lower case, we use "".lower() later
POLARITY_CHARS = {'up': "U", 'down': "D", 'poorup': "+", 'poordown': "-"}
ONSET_CHARS = {'impulsive': "I", 'emergent': "E",
               'implusive': "I"} # XXX some old events have a typo there... =)


