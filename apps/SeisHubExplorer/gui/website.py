import os

class Website(object):
    """
    Creates the website.
    """
    def __init__(self, env, *args, **kwargs):
        self.env = env
        self.map_file = os.path.join(self.env.temp_res_dir, 'map.html')
        # Read in the html template.
        f = open(os.path.join(self.env.res_dir, 'upperPart.html'))
        self.upperPart = f.read()
        f.close()
        f = open(os.path.join(self.env.res_dir, 'lowerPart.html'))
        self.lowerPart = f.read()
        f.close()

    def startup(self):
        # Preprocess stations.
        self.processStations()
        # Get the boundaries.
        self.getBoundaries()
        # Create the middle part of the html.
        self.createMiddle()
        self.createHtml()

    def getBoundaries(self):
        lats = []
        longs = []
        for station in self.stations:
            info = station[2]
            lats.append(info['latitude'])
            longs.append(info['longitude'])
        # XXX: Ugly hack only for Tobis Computer. Needs a real workaround but
        # should work so far. Maybe change the order of loading the submodules
        # during the startup phase?
        try:
            self.bounds = (min(longs), max(longs), min(lats), max(lats))
        except ValueError:
            self.bounds = (47.0, 49.0, 11.0, 12.0)

    def createHtml(self):
        html = self.upperPart + self.middlePart + self.lowerPart
        f = open(self.map_file, 'w')
        f.write(html)
        f.close()

    def processStations(self):
        nw_dict = self.env.seishub.networks
        self.stations = []
        self.omitted_stations = []
        networks = nw_dict.keys()
        for network in networks:
            if network == 'Server' or network == 'Date':
                continue
            st_dict = nw_dict[network]
            stations = st_dict.keys()
            for station in stations:
                info = st_dict[station]['info']
                if not len(info):
                    self.omitted_stations.append('%s.%s' % (network, station))
                    continue
                self.stations.append([network, station, info])

    def createMiddle(self):
        center_long = (self.bounds[0] + self.bounds[1])/2.0
        center_lat = (self.bounds[2] + self.bounds[3])/2.0
        part = 'map.setCenter(new GLatLng(%f, %f), 6);' % (center_lat,
                                                            center_long)
        part += 'map.setUIToDefault();'

        for _i, station in enumerate(self.stations):
            info = station[2]
            lat = info['latitude']
            long = info['longitude']
            station_name = info['station_name']
            station_id = '%s' % station[1]
            elevation = info['elevation']
            path = '%s' % station[1]
            part += """
            var point%i = new GPoint(%f,%f);
            var html%i = "<h2>%s</h2> <p>%s<br>Elevation: %f</p>";
            var iconOptions = {};
            iconOptions.width = 40;
            iconOptions.height = 32;
            iconOptions.primaryColor = "#FF0000";
            iconOptions.label = "%s";
            iconOptions.labelSize = 0;
            iconOptions.labelColor = "#000000";
            iconOptions.shape = "roundrect";
            var newIcon = MapIconMaker.createFlatIcon(iconOptions);
            var beck%i = new GMarker(point%i, {icon: newIcon});
            GEvent.addListener(beck%i, "click", function()
                                 {beck%i.openInfoWindowHtml(html%i)});
            map.addOverlay(beck%i);
            """ % (_i, long, lat, _i, path, station_name,
                   elevation,station_id, _i, _i, _i,
                    _i, _i, _i)
        self.middlePart = part

                

