
def fetch_waveforms_metadata(options):
    """
    Sets up a client and fetches waveforms and metadata according to command
    line options.

    :returns: (:class:`obspy.seishub.client.Client`,
               list(:class:`obspy.core.stream.Stream`s))
    """
    t = UTCDateTime(options.time)
    t = t + options.starttime_offset
    baseurl = "http://" + options.servername + ":%i" % options.port
    client = Client(base_url=baseurl, user=options.user,
                    password=options.password, timeout=options.timeout)
    streams = []
    sta_fetched = set()
    print "Fetching waveforms and metadata from seishub:"
    for id in options.ids.split(","):
        net, sta_wildcard, loc, cha = id.split(".")
        for sta in client.waveform.getStationIds(network_id=net):
            if not fnmatch.fnmatch(sta, sta_wildcard):
                continue
            # make sure we dont fetch a single station of
            # one network twice (could happen with wildcards)
            net_sta = "%s:%s" % (net, sta)
            if net_sta in sta_fetched:
                print "%s skipped! (Was already retrieved)" % net_sta
                continue
            try:
                sys.stdout.write("\r%s ..." % net_sta)
                sys.stdout.flush()
                st = client.waveform.getWaveform(net, sta, loc, cha, t,
                        t + options.duration, apply_filter=True,
                        getPAZ=True, getCoordinates=True)
                sys.stdout.write("\r%s fetched.\n" % net_sta)
                sys.stdout.flush()
                sta_fetched.add(net_sta)
            except Exception, e:
                sys.stdout.write("\r%s skipped! (Server replied: %s)\n" % (net_sta, e))
                sys.stdout.flush()
                continue
            streams.append(st)
    return (client, streams)

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
        net_sta = "%s:%s" % (st[0].stats.network.strip(),
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
                      'stream %s:%s' % (net_sta, removed_channels)
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
    :type options: options as returned by optparse.OptionParser.parse_args()
    :returns: String representation of temporary directory with program files.
    """
    tmp_dir = tempfile.mkdtemp()
    # set binary names to use depending on architecture and platform...
    architecture = platform.architecture()[0]
    system = platform.system()
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
    # 3dloc ###############################################################
    prog_dict = PROGRAMS['3dloc']
    prog_dict['D3_VELOCITY'] = os.path.join(prog_dict['dir'], 'D3_VELOCITY')
    prog_dict['D3_VELOCITY_2'] = os.path.join(prog_dict['dir'], 'D3_VELOCITY_2')
    prog_dict['PreCall'] = 'rm %s %s &> /dev/null' \
            % (prog_dict['files']['out'], prog_dict['files']['in'])
    prog_dict['Call'] = 'export D3_VELOCITY=%s/;' % \
            PROGRAMS['3dloc']['D3_VELOCITY'] + \
            'export D3_VELOCITY_2=%s/;' % \
            PROGRAMS['3dloc']['D3_VELOCITY_2'] + \
            'cd %s; ./%s' % (prog_dict['dir'], prog_dict['files']['exe'])
    # Hyp2000 #############################################################
    prog_dict = PROGRAMS['hyp_2000']
    prog_dict['PreCall'] = 'rm %s %s %s &> /dev/null' \
            % (prog_dict['files']['phases'], prog_dict['files']['stations'],
               prog_dict['files']['summary'])
    prog_dict['Call'] = 'export HYP2000_DATA=%s;' % (prog_dict['dir']) + \
            'cd $HYP2000_DATA; ./%s < %s &> /dev/null' % \
            (prog_dict['files']['exe'], prog_dict['filenames']['control'])
    # NLLoc ###############################################################
    prog_dict = PROGRAMS['nlloc']
    prog_dict['PreCall'] = 'rm %s/nlloc* &> /dev/null' % prog_dict['dir']
    prog_dict['Call'] = 'cd %s; ./%s %%s' % (prog_dict['dir'],
                                          prog_dict['files']['exe']) + \
            '; mv nlloc.*.*.*.loc.hyp %s' % prog_dict['files']['summary'] + \
            '; mv nlloc.*.*.*.loc.scat %s' % prog_dict['files']['scatter']
    # focmec ##############################################################
    prog_dict = PROGRAMS['focmec']
    prog_dict['Call'] = 'cd %s; ./%s' % (prog_dict['dir'], prog_dict['files']['exe'])
    #######################################################################
    return tmp_dir
   
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
    data = data.reshape((len(data)/4, 4))

    # convert km to m (*1000), use x/y/z values (columns 0/1/2,
    # write a file to pipe to cs2cs later
    tmp_file = tempfile.mkstemp(suffix='.scat',
                                dir=os.path.dirname(scat_filename))[1]
    file_lonlat = scat_filename + ".lonlat"
    np.savetxt(tmp_file, (data[:,:3]*1000.), fmt='%.3f')

    command = "cs2cs +init=epsg:31468 +to +init=epsg:4326 -E -f %.6f " + \
              "< %s > %s" % (tmp_file, file_lonlat)

    sub = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)

    err = "".join(sub.stderr.readlines())
    textviewStdErrImproved.write(err)
    
    data = np.loadtxt(file_lonlat, usecols=[3,4,2])

    return data

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

def formatXTicklabels(x, pos):
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

def check_keybinding_conflicts(keys):
    """
    check for conflicting keybindings. 
    we have to check twice, because keys for setting picks and magnitudes
    are allowed to interfere...
    """
    for ignored_key_list in [['setMagMin', 'setMagMax', 'delMagMinMax'],
                             ['setPick', 'setPickError', 'delPick']]:
        tmp_keys = keys.copy()
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

