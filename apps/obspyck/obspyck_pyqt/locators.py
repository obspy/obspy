#####
## NLLoc stuff.
####

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

