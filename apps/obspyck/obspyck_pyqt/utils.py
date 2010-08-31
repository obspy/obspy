def getCoord(client, network, station):
    """
    Returns longitude, latitude and elevation of given station from given
    client instance
    """
    coord = []
    resource = "dataless.seed.%s_%s.xml" % (network, station)
    xml = fromstring(client.station.getResource(resource, format='metadata'))
    for attrib in [u'Longitude (\xb0)', u'Latitude (\xb0)',  u'Elevation (m)']:
        node =  xml.xpath(u".//item[@title='%s']" % attrib)[0]
        value = float(node.getchildren()[0].attrib['text'])
        coord.append(value)
    return coord

def gk2lonlat(x, y):
    """
    This function converts X/Y Gauss-Krueger coordinates (zone 4, central
    meridian 12 deg) to Longitude/Latitude in WGS84 reference ellipsoid.
    We do this using the Linux command line tool cs2cs.
    """
    # convert to meters first
    x *= 1000.
    y *= 1000.
    command = "echo \"%.3f %.3f\"" % (x, y)
    command += " | cs2cs +init=epsg:31468 +to +init=epsg:4326 -f %.4f"
    sub = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    line = sub.stdout.readline().split()
    lon = float(line[0])
    lat = float(line[1])
    return (lon, lat)

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

