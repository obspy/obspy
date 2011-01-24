# -*- coding: utf-8 -*-

import numpy as np
import lxml.etree
from lxml.etree import SubElement as Sub
from util import UniqueList, raise_locked_warning, str_or_None, \
        set_attribute
from obspy.core import AttribDict, UTCDateTime
from pick import Pick, PICK_DICT_SEISHUB

ORIGIN_DICT_SEISHUB = {'program': (".//program", None),
        'time': (".//time/value", UTCDateTime),
        'latitude': (".//latitude/value", float),
        'longitude': (".//longitude/value", float),
        'longitude_error': (".//longitude/uncertainty", float),
        'latitude_error': (".//latitude/uncertainty", float),
        'depth': (".//depth/value", float),
        'depth_error': (".//depth/uncertainty", float),
        'depth_type': (".//depth_type", None),
        'earth_model': (".//earth_mod", None),
        'used_p_count': (".//originQuality/P_usedPhaseCount", int),
        'used_s_count': (".//originQuality/S_usedPhaseCount", int),
        'used_phase_count': (".//originQuality/usedPhaseCount", int),
        'used_station_count': (".//originQuality/usedStationCount", int),
        'associated_phase_count': (".//originQuality/associatedPhaseCount", int),
        'associated_station_count': (".//originQuality/associatedStationCount", int),
        'depth_phase_count': (".//originQuality/depthPhaseCount", int),
        'standarderror': (".//originQuality/standardError", float),
        'azimuthal_gap': (".//originQuality/azimuthalGap", float),
        'ground_truth_level': (".//originQuality/groundTruthLevel", float),
        'minimum_distance': (".//originQuality/minimumDistance", float),
        'maximum_distance': (".//originQuality/maximumDistance", float),
        'median_distance': (".//originQuality/medianDistance", float)}

MAGNITUDE_DICT_SEISHUB = {'program': (".//program", None),
        'value': (".//mag/value", float),
        'uncertainty': (".//mag/uncertainty", float),
        'type': (".//type", None),
        'station_count': (".//stationCount", int)}


class Event(object):
    """
    Class that handles picks and location for events.
    """
    def __init__(self):
        # Status variable that determine whether the object and its picks can
        # be edited.
        # The base class needs to be called explicitly.
        object.__setattr__(self, '_Event__locked', False)

        # All picks will be stored in a unique list to avoid duplicate entries.
        # It also provides some convenience functions.
        self.picks = UniqueList()
        self.origin = AttribDict()
        self.magnitude = AttribDict()
        self.focal_mechanism = AttribDict()

        # Sets some common attributes meaningful for all events.
        self._setImportantAttributes()

    def __setattr__(self, name, value):
        """
        Attribute access only when the object is not locked.
        """
        if self.__locked:
            raise_locked_warning()
            return
        # Call the base class and set the value.
        object.__setattr__(self, name, value)

    def _setImportantAttributes(self):
        """
        Sets some important attributes that are meaningful for all events.
        """
        self.id = None
        self.type = None
        self.account = None
        self.user = None
        self.public = None

    def _lock(self):
        """
        Locks the object and all subobjects.
        """
        self.__locked = True
        # Lock every pick.
        for pick in self.picks:
            pick._lock()
        # Lock the pick list.
        self.picks._lock()

    def _unlock(self):
        """
        Unlocks the object.
        """
        # The base class needs to be called explicitly.
        object.__setattr__(self, '_Event__locked', False)
        # Unlock every pick.
        for pick in self.picks:
            pick._unlock()
        # Lock the pick list.
        self.picks._unlock()

    def locate(self):
        """
        Dummy function that currently just prevents any further changes to the
        object event and all picks.
        """
        self._lock()

    def readSeishubXML(self, filename):
        """
        Read a SeisHub style XML from file.
        """
        self.__init__()
        resource_xml = lxml.etree.parse(filename)
        try:
            self.__setattr__('id', resource_xml.xpath(u".//event_id/value")[0].text)
        except:
            self.__setattr__('id', None)
        try:
            self.__setattr__('type', resource_xml.xpath(u".//event_type/value")[0].text)
        except:
            self.__setattr__('type', None)
        for key in ['account', 'user', 'public']:
            try:
                self.__setattr__(key, resource_xml.xpath(u".//event_type/" + key)[0].text)
            except:
                self.__setattr__(key, None)

        #analyze picks:
        picks = UniqueList()
        for pick in resource_xml.xpath(u".//pick"):
            p = Pick()
            # attributes
            id = pick.find("waveform").attrib
            p.network = id["networkCode"]
            p.station = id["stationCode"]
            p.location = id["locationCode"]
            p.channel = id['channelCode']
            # values
            p.time = pick.xpath(".//time/value")[0].text
            p.uncertainty = pick.xpath(".//time/uncertainty")[0].text
            p.phasehint = pick.xpath(".//phaseHint")[0].text
            for key, value in PICK_DICT_SEISHUB.iteritems():
                set_attribute(p, pick, key, value[0], value[1]) 
            picks.append(p)
        self.picks = picks
        #analyze origin:
        o = self.origin
        try:
            origin = resource_xml.xpath(u".//origin")[0]
            for key, value in ORIGIN_DICT_SEISHUB.iteritems():
                set_attribute(o, origin, key, value[0], value[1], do_pass=True)
        except IndexError:
            pass
        #analyze magnitude:
        m = self.magnitude
        try:
            magnitude = resource_xml.xpath(u".//magnitude")[0]
            for key, value in MAGNITUDE_DICT_SEISHUB.iteritems():
                set_attribute(m, magnitude, key, value[0], value[1], do_pass=True)
        except IndexError:
            pass
        #analyze stationmagnitudes:
        self.magnitude.stationmagnitudes = UniqueList()
        for stamag in resource_xml.xpath(u".//stationMagnitude"):
            sm = AttribDict()
            sm.station = stamag.xpath(".//station")[0].text
            # values
            try:
                sm.uncertainty = stamag.xpath('.//mag/uncertainty')[0].text
            except:
                sm.uncertainty = None
            sm.value = float(stamag.xpath(".//mag/value")[0].text)
            sm.channels = stamag.xpath(".//channels")[0].text
            sm.weight = float(stamag.xpath(".//weight")[0].text)
            if sm.weight == 0:
                sm.used = False
            else:
                sm.used = True
            self.magnitude.stationmagnitudes.append(sm)
        
        #analyze focal mechanism:
        f = self.focal_mechanism
        try:
            focmec = resource_xml.xpath(u".//focalMechanism")[0]
            try:
                dF['Program'] = focmec.xpath(".//program")[0].text
            except:
                pass
            try:
                strike = focmec.xpath(".//nodalPlanes/nodalPlane1/strike/value")[0].text
                f.strike = float(strike)
            except:
                pass
            try:
                dip = focmec.xpath(".//nodalPlanes/nodalPlane1/dip/value")[0].text
                f.dip = float(dip)
            except:
                pass
            try:
                rake = focmec.xpath(".//nodalPlanes/nodalPlane1/rake/value")[0].text
                f.rake = float(rake)
            except:
                pass
            try:
                staPolCount = focmec.xpath(".//stationPolarityCount")[0].text
                f.station_polarity_count = int(staPolCount)
            except:
                pass
            try:
                staPolErrCount = focmec.xpath(".//stationPolarityErrorCount")[0].text
                f.errors = int(staPolErrCount)
            except:
                pass
        except:
            pass

    def writeSeishubXML(self, filename):
        """
        Write a SeisHub style XML to a file.
        """
        xml =  lxml.etree.Element("event")
        Sub(Sub(xml, "event_id"), "value").text = self.id
        event_type = Sub(xml, "event_type")
        Sub(event_type, "value").text = self.type
        Sub(event_type, "account").text = self.account
        Sub(event_type, "user").text = self.user
        Sub(event_type, "public").text = self.public
        
        for pick in self.picks:
            p = Sub(xml, "pick")
            wave = Sub(p, "waveform")
            wave.set("networkCode", pick.get("network"))
            wave.set("stationCode", pick.get("station"))
            wave.set("channelCode", pick.get("channel"))
            wave.set("locationCode", pick.get("location"))
            time = Sub(p, "time")
            Sub(time, "value").text = pick.get("time")
            Sub(time, "uncertainty").text = pick.get("uncertainty")
            Sub(p, "phaseHint").text = pick.get("phasehint")
            Sub(p, "onset").text = pick.get("onset")
            Sub(p, "polarity").text = pick.get("polarity")
            weight = pick.get("weight")
            Sub(p, "weight").text = str_or_None(weight)
            Sub(Sub(p, "min_amp"), "value") #XXX what is min_amp???
            # XXX the mapping for phase_compu should go to pick in the fashion
            # XXX of stats.endtime as a derived attribute...
            onset_map = {'impulsive': "I", 'emergent': "E"}
            onset_map['implusive'] = "I" # XXX old xmls have typo
            pol_map = {'up': "U", 'poorup': "+", 'down': "D", 'poordown': "-"}
            phase_compu = onset_map.get(pick.get("onset"), "?")
            phase_compu += pick.phasehint
            phase_compu += pol_map.get(pick.get("polarity"), "?")
            phase_compu += (weight is None) and "?" or str(weight)
            Sub(p, "phase_compu").text = phase_compu
            for key in ["phase_res", 'phase_weight', "phase_delay", "azimuth",
                        "incident", "epi_dist", "hyp_dist"]:
                Sub(Sub(p, key), "value").text = str_or_None(pick.get(key))

        #origin output
        o = self.origin
        origin = Sub(xml, "origin")
        Sub(origin, "program").text = o.program
        date = Sub(origin, "time")
        Sub(date, "value").text = o.time.isoformat()
        Sub(date, "uncertainty")
        lat = Sub(origin, "latitude")
        Sub(lat, "value").text = str(o.latitude)
        Sub(lat, "uncertainty").text = str(o.latitude_error) #XXX Lat Error in km!!
        lon = Sub(origin, "longitude")
        Sub(lon, "value").text = str(o.longitude)
        Sub(lon, "uncertainty").text = str(o.longitude_error) #XXX Lon Error in km!!
        depth = Sub(origin, "depth")
        Sub(depth, "value").text = str(o.depth)
        Sub(depth, "uncertainty").text = str(o.depth_error)
        if 'depth_type' in o:
            Sub(origin, "depth_type").text = str(o.depth_type)
        else:
            Sub(origin, "depth_type")
        if 'earth_model' in o:
            Sub(origin, "earth_mod").text = o.earth_model
        else:
            Sub(origin, "earth_mod")
        if o.program == "hyp2000":
            uncertainty = Sub(origin, "originUncertainty")
            Sub(uncertainty, "preferredDescription").text = "uncertainty ellipse"
            Sub(uncertainty, "horizontalUncertainty")
            Sub(uncertainty, "minHorizontalUncertainty")
            Sub(uncertainty, "maxHorizontalUncertainty")
            Sub(uncertainty, "azimuthMaxHorizontalUncertainty")
        else:
            Sub(origin, "originUncertainty")
        quality = Sub(origin, "originQuality")
        Sub(quality, "P_usedPhaseCount").text = '%i' % o.used_p_count
        Sub(quality, "S_usedPhaseCount").text = '%i' % o.used_s_count
        Sub(quality, "usedPhaseCount").text = '%i' % o.used_phase_count
        Sub(quality, "usedStationCount").text = '%i' % o.used_station_count
        Sub(quality, "associatedPhaseCount").text = '%i' % o.associated_phase_count
        Sub(quality, "associatedStationCount").text = '%i' % o.associated_station_count
        Sub(quality, "depthPhaseCount").text = str(o.depth_phase_count)
        Sub(quality, "standardError").text = str(o.standarderror)
        Sub(quality, "azimuthalGap").text = str(o.azimuthal_gap)
        try:
            Sub(quality, "groundTruthLevel").text = str(o.ground_truth_level)
        except:
            Sub(quality, "groundTruthLevel")
        Sub(quality, "minimumDistance").text = str(o.minimum_distance)
        Sub(quality, "maximumDistance").text = str(o.maximum_distance)
        Sub(quality, "medianDistance").text = str(o.median_distance)
        
        #magnitude output
        m = self.magnitude
        magnitude = Sub(xml, "magnitude")
        Sub(magnitude, "program").text = m.program
        mag = Sub(magnitude, "mag")
        if np.isnan(m.value):
            Sub(mag, "value")
            Sub(mag, "uncertainty")
        else:
            Sub(mag, "value").text = str(m.value)
            Sub(mag, "uncertainty").text = str(m.uncertainty)
        if "type" in m:
            Sub(magnitude, "type").text = m.type
        else:
            Sub(magnitude, "type")
        Sub(magnitude, "stationCount").text = '%i' % m.station_count
        for sm in self.magnitude.stationmagnitudes:
            stationMagnitude = Sub(xml, "stationMagnitude")
            mag = Sub(stationMagnitude, 'mag')
            Sub(mag, 'value').text = str(sm.value)
            if sm.uncertainty:
                Sub(mag, 'uncertainty').text = str(sm.uncertainty)
            else:
                Sub(mag, 'uncertainty')
            Sub(stationMagnitude, 'station').text = sm.station
            Sub(stationMagnitude, 'weight').text = str(sm.weight)
            Sub(stationMagnitude, 'channels').text = sm.channels
        
        # XXX deactivate focal mechanism for the moment
        ##focal mechanism output
        #f = self.focal_mechanism
        #focmec = Sub(xml, "focalMechanism")
        #Sub(focmec, "program").text = f.program
        #nodplanes = Sub(focmec, "nodalPlanes")
        #nodplanes.set("preferredPlane", "1")
        #nodplane1 = Sub(nodplanes, "nodalPlane1")
        #strike = Sub(nodplane1, "strike")
        #Sub(strike, "value").text = str(f.strike)
        #Sub(strike, "uncertainty")
        #dip = Sub(nodplane1, "dip")
        #Sub(dip, "value").text = str(f.dip)
        #Sub(dip, "uncertainty")
        #rake = Sub(nodplane1, "rake")
        #Sub(rake, "value").text = str(f.rake)
        #Sub(rake, "uncertainty")
        #Sub(focmec, "stationPolarityCount").text = "%i" % \
        #        f.station_polarity_count
        #Sub(focmec, "stationPolarityErrorCount").text = "%i" % f.errors
        #Sub(focmec, "possibleSolutionCount").text = "%i" % \
        #        f.possible_solution_count

        xml = lxml.etree.tostring(xml, pretty_print=True, encoding="utf-8", xml_declaration=True)
        open(filename, "wt").write(xml)
        return

