#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides the SiteCharacterization class.

:copyright:
    ORFEUS, 2025
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.util.base import ComparingObject
#from obspy.core.util.obspy_types import (ObsPyException, ZeroSamplingRate,
#                                         FloatWithUncertaintiesAndUnit)
#from obspy.core.event import ResourceIdentifier
from obspy.io.sitexml.util import (TopographySchemaA, TopographySchemaB, EC8Class, 
                                   ResonanceFrequencyMethod, VelocityS30Method,
                                    _sitexml_check_type, _sitexml_check_enum, _pretty_str,
                                    _wrapped_property)
from obspy.core.inventory.util import (Latitude, Longitude, Distance, ExternalReference)

class SERASite(ComparingObject):
    """
    This is the parent class for the siteXML object tree.
    """
    def __init__(self, station_code=None, site_owner=None, site_description=None, site_characterization_parameters=None, overall_quality_index=None):
        """
        :type station_code: str
        :param station_code: Not used in SiteXML, but is needed in order to correlate with the Station Object
        :type site_description: :class:`~obspy.core.io.sitexml.SiteDescription`
        :param site_description: The site description parameters (H800, Bedrock depth, EC8 class, geological unit, morphology, topology)
        :type site_characterization_parameters: :class:`~obspy.core.io.sitexml.SiteCharacterizationParameters`
        :param site_characterization_parameters: The site characterization parameters (VS30, resonance frequency, velocity profiles)
        :type overall_quality_index: float
        :param overall_quality_index: The overall quality index of the site characterization parameters.
        """
        self.station_code = station_code

        self.site_owner = site_owner
        
        self.site_description = _sitexml_check_type(
            site_description, SiteDescription, "site_description", True)

        self.site_characterization_parameters = _sitexml_check_type(
            site_characterization_parameters, SiteCharacterizationParameters, "site_characterization_parameters", True)
        
        # TO CHECK: If this one is calculated it should be removed from the parameters 
        self.overall_quality_index = overall_quality_index
       
class SiteDescription(ComparingObject):
    def __init__(self, latitude, longitude, altitude=None, min_distance_from_station=None, max_distance_from_station=None, 
                 ec8=None, bedrock_depth=None, h800=None, geological_unit=None, morphology=None, topologyA=None, topologyB=None):
        """
        :type latitude: :class:`~obspy.core.inventory.util.Latitude`
        :param latitude: The latitude of the site
        :type longitude: :class:`~obspy.core.inventory.util.Longitude`
        :param longitude: The longitude of the site
        :type altitude: :class:`~obspy.core.inventory.util.Distance`
        :param altitude: Elevation of ground with respect to sea level (m)
        :type min_distance_from_station: :class:`~obspy.core.inventory.util.Distance`
        :param min_distance_from_station: Minimum distance between the permanent seismological station and 
            site characterization measurement. Should be used only when representative latitude and longitude 
            of site characterization measurements cannot be provided.
        :type max_distance_from_station: :class:`~obspy.core.inventory.util.Distance`
        :param max_distance_from_station: Maximum distance between the permanent seismological station and 
            site characterization measurement. Should be used only when representative latitude and longitude 
            of site characterization measurements cannot be provided.
        :type ec8: :class:`~obspy.io.sitexml.core.EC8`
        :param ec8: Ground type according to Eurocode 8, based on the velocity S30 value and geotechnical description
        :type h800: :class:`~obspy.io.sitexml.core.H800`
        :param h800: Engineering depth. Depth beyond which the shear-wave velocity Vs exceeds 800 m/s.
        :type bedrock_depth: :class:`~obspy.io.sitexml.core.BedrockDepth`
        :param bedrock_depth: Seismological bedrock depth
        :type geological_unit: :class:`~obspy.io.sitexml.core.GeologicalUnit`
        :param geological_unit: Brief description of the surface geology (free text)
        :type morphology: str
        :param morphology: Qualitative description of the shape of the earth's surface (free text)
        :type topologyA: str
        :param topologyA: Quantitative description of the surface according to the Italian Code 
            (detailed description of the scheme in SERA Deliverable D7.1 - Appendix I).
            See :class:`~obspy.io.sitexml.util.TopographySchemaA` for allowed values.
        :type topologyB: str
        :param topologyB: Quantitative description of the shape of the earth's surface according to 
            Burjanek et al, 2014 (detailed description of the scheme in SERA Deliverable D7.1 - Appendix I). 
            See :class:`~obspy.io.sitexml.util.TopographySchemaB` for allowed values.
        """
        # Topology and topography have the same meaning ?
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.min_distance_from_station = min_distance_from_station
        self.max_distance_from_station = max_distance_from_station

        self.ec8 = ec8
        self.h800 = h800
        self.bedrock_depth = bedrock_depth
        self.geological_unit = geological_unit
        
        self.morphology = morphology
        self.topologyA = topologyA
        self.topologyB = topologyB
    
    @property
    def longitude(self):
        return self._longitude

    @longitude.setter
    def longitude(self, value):
        if isinstance(value, Longitude):
            self._longitude = value
        else:
            self._longitude = Longitude(value)

    @property
    def latitude(self):
        return self._latitude

    @latitude.setter
    def latitude(self, value):
        if isinstance(value, Latitude):
            self._latitude = value
        else:
            self._latitude = Latitude(value)

    @property
    def altitude(self):
        return self._altitude

    @altitude.setter
    def altitude(self, value):
        if value is None:
            self._altitude = None
        elif isinstance(value, Distance):
            self._altitude = value
        else:
            self._altitude = Distance(value)

    @property
    def min_distance_from_station(self):
        return self._min_distance_from_station

    @min_distance_from_station.setter
    def min_distance_from_station(self, value):
        if value is None:
            self._min_distance_from_station = None
        elif isinstance(value, Distance):
            self._min_distance_from_station = value
        else:
            self._min_distance_from_station = Distance(value)

    @property
    def max_distance_from_station(self):
        return self._max_distance_from_station

    @max_distance_from_station.setter
    def max_distance_from_station(self, value):
        if value is None:
            self._max_distance_from_station = None
        elif isinstance(value, Distance):
            self._max_distance_from_station = value
        else:
            self._max_distance_from_station = Distance(value)

    @property
    def ec8(self):
        return self._ec8

    @ec8.setter
    def ec8(self, value):
        if value is None:
            self._ec8 = None
        elif isinstance(value, EC8):
            self._ec8 = value
        else:
            self._ec8 = EC8(value)

    @property
    def h800(self):
        return self._h800

    @h800.setter
    def h800(self, value):
        if value is None:
            self._h800 = None
        elif isinstance(value, H800):
            self._h800 = value
        else:
            self._h800 = H800(value)

    @property
    def bedrock_depth(self):
        return self._bedrock_depth

    @bedrock_depth.setter
    def bedrock_depth(self, value):
        if value is None:
            self._bedrock_depth = None
        elif isinstance(value, BedrockDepth):
            self._bedrock_depth = value
        else:
            self._bedrock_depth = BedrockDepth(value)

    @property
    def geological_unit(self):
        return self._geological_unit

    @geological_unit.setter
    def geological_unit(self, value):
        if value is None:
            self._geological_unit = None
        elif isinstance(value, GeologicalUnit):
            self._geological_unit = value
        else:
            self._geological_unit = GeologicalUnit(value)

    @property
    def topologyA(self):
        return self._topologyA

    @topologyA.setter
    def topologyA(self, value):
        if value is None:
            self._topologyA = None
        elif value in TopographySchemaA:
            self._topologyA = value
        else:
            valid_values = [e for e in TopographySchemaA]  # Get all valid Enum names
            raise ValueError(f"\nInvalid value for 'topologyA'. Expected one of {valid_values}, but got '{value}'.")    

    @property
    def topologyB(self):
        return self._topologyB

    @topologyB.setter
    def topologyB(self, value):
        if value is None:
            self._topologyB = None
        elif value in TopographySchemaB:
            self._topologyB = value
        else:
            valid_values = [e for e in TopographySchemaB]  # Get all valid Enum names
            raise ValueError(f"\nInvalid value for 'topologyB'. Expected one of {valid_values}, but got '{value}'.")    

    def __str__(self):
        ret = ("Site Description parameters:\n"
               "\tLatitude {lat:.4f}, Longitude: {lng:.4f}, Altitude {alt} m,\n"
               "\tMorphology: {morphology},\n"
               "\tTopology A: {topoA},\n"
               "\tTopology B: {topoB},\n"
               "\tEC8 class: {ec8},\n"
               "\tH800: {h800} m,\n"
               "\tBedrock depth: {bdepth} m,\n"
               "\tGeological Unit: {gunit}\n")
        ret = ret.format(
            lat=self.latitude, lng=self.longitude, alt=self.altitude,
            morphology = self.morphology,
            topoA = self.topologyA, topoB = self.topologyB,
            ec8=self.ec8.value if self.ec8 else "None",
            h800=self.h800.value if self.h800 else "None",
            bdepth=self.bedrock_depth.value if self.bedrock_depth else "None",
            gunit=self.geological_unit.value if self.geological_unit else "None")
        return ret

class SiteCharacterizationParameters(ComparingObject):
    def __init__(self, publicID=None, analysis_publicID=None, resonance_frequency=None, velocity_s30=None, 
                 velocity_profile_count=None, velocity_profile=None, velocity_profile_qindex=None, 
                 velocity_profile_reference=None, spt_logs_count=None, cpt_logs_count=None, 
                 borehole_logs_count=None):
        """
        :type publicID: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
        :param publicID: All channels belonging to this station.
        :type analysis_publicID: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
        :param analysis_publicID: The lexical description of the site
        :type resonance_frequency: :class:`~obspy.io.sitexml.core.ResonanceFrequency`
        :param resonance_frequency: The Resonance frequency of the soil column 
        :type velocity_s30: :class:`~obspy.io.sitexml.core.velocityS30`
        :param velocity_s30: Average shear-wave velocity between 0 and 30 meters depth
        :type velocity_profile_count: int
        :param velocity_profile_count: Number of available velocity profiles
        :type velocity_profile: :class:`~obspy.io.sitexml.core.VelocityProfile`
        :param velocity_profile: Velocity Profile
        :type spt_logs_count: int
        :param spt_logs_count: Number of available SPT profile(s)
        :type cpt_logs_count: int
        :param cpt_logs_count: Number of available CPT profile(s)
        :type borehole_logs_count: int
        :param borehole_logs_count: Number of available borehole log profile(s)
       """
        self.publicID = publicID        
        self.analysis_publicID = analysis_publicID   
        self.resonance_frequency = resonance_frequency
        self.velocity_s30 = velocity_s30
        self.spt_logs_count = spt_logs_count
        self.cpt_logs_count = cpt_logs_count
        self.borehole_logs_count = borehole_logs_count
        self.velocity_profile_count = velocity_profile_count
        self.velocity_profile = velocity_profile

    @property
    def resonance_frequency(self):
        return self._resonance_frequency

    @resonance_frequency.setter
    def resonance_frequency(self, value):
        if value is None:
            self._resonance_frequency = None
        elif isinstance(value, ResonanceFrequency):
            self._resonance_frequency = value
        else:
            self._resonance_frequency = ResonanceFrequency(value)

    @property
    def velocity_s30(self):
        return self._velocity_s30

    @velocity_s30.setter
    def velocity_s30(self, value):
        if value is None:
            self._velocity_s30 = None
        elif isinstance(value, VelocityS30):
            self._velocity_s30 = value
        else:
            self._velocity_s30 = VelocityS30(value)
    
    @property
    def velocity_profile(self):
        return self._velocity_profile

    @velocity_profile.setter
    def velocity_profile(self, value):
        if value is None:
            self._velocity_profile = None
        elif isinstance(value, VelocityProfile):
            self._velocity_profile = value
        else:
            self._velocity_profile = VelocityProfile(value)

    def __str__(self):
        ret = ("Site Characterization parameters:\n"
               "\tResonance Frequency {rfreq},\n"
               "\tVelocity S30: {vs30},\n"
               "\tVelocity Profiles count: {vp_count},\n"
               "\tSPT Logs count: {spt_logs_count},\n"
               "\tCPT Logs count: {cpt_logs_count},\n"
               "\tBorehole Logs count: {bh_logs_count} \n")
        ret = ret.format(
            rfreq = self.resonance_frequency.value if self.resonance_frequency else "None",
            vs30 = self.velocity_s30.value if self.velocity_s30 else "None",
            vp_count = self.velocity_profile_count, 
            spt_logs_count = self.spt_logs_count, 
            cpt_logs_count = self.cpt_logs_count, 
            bh_logs_count = self.borehole_logs_count)
        return ret

class SiteIndicator(ComparingObject):
    def __init__(self, name, value, uncertainty=None, methods=None, quality_index=None, literature_source=None, file_resource=None):
        """
        :type name: str
        :param name: Indicator type. One of: "ec8", "h800", "bedrock_depth", "geological_unit", "velocity_s30", "resonance_frequency"
        :type value: str / int / float
        :param value: Value of the indicator
        :type uncertainty: int / float
        :param uncertainty: Uncertainty related with the provided site indicator value
        :type methods: list of str
        :param methods: Methods used for the estimation / calculation of the site indicator
        :type quality_index: float
        :param quality_index: Quality index of the site indicator. Takes values between 0 and 1.
            Calculated according to the guidelines of the SERA D7.2 Deliverable.
        :type literature_source: :class:`~obspy.io.sitexml.core.LiteratureSource`
        :param literature_source: The literature source related with the provided site indicator value
        :type file_resource: :class:`~obspy.core.inventory.util.ExternalReference` ????
        :param file_resource: A public URL for the literature_source
        """
        self.name = name
        self.value = value
        self.uncertainty = uncertainty
        self.methods = methods or []
        self.quality_index = quality_index  # Maybe this is internal only??
        self.literature_source = literature_source
        self.file_resource = file_resource

    @property
    def file_resource(self):
        return self._file_resource

    @file_resource.setter
    def file_resource(self, value):
        if value is None:
            self._file_resource = None
        elif isinstance(value, ExternalReference):
            self._file_resource = value
        else:
            self._file_resource = ExternalReference(value, None)

    @property
    def literature_source(self):
        return self._literature_source

    @literature_source.setter
    def literature_source(self, value):
        if value is None:
            self._literature_source = None
        elif isinstance(value, LiteratureSource):
            self._literature_source = value
        else:
            self._literature_source = LiteratureSource(value)

    # This needs more work if value is allowed to be <0 
    def _validate_value_uncertainty(self, valid_type):
        if not isinstance(self.value, valid_type) or self.value <= 0:
            raise ValueError(f"Value of {self.name} must be a positive {valid_type}")
        
        if self.uncertainty is not None:
            if not isinstance(self.uncertainty, valid_type) or self.uncertainty <= 0:
                raise ValueError(f"Uncertainty of {self.name} must be a positive {valid_type} or None")

    def __str__(self):
        ret = ("{name} parameters:\n"
               "\t{name} value: {value},\n"
               "\tUncertainty: {uncertainty},\n"
               "\tMethods: {methods},\n"
               "\tQuality index: {qindex},\n"
               "\tLiterature source: {lit_source},\n"
               "\tFile resource: {fresource},\n")
        ret = ret.format(
            name=self.name, 
            value=self.value if self.name != "VP" else "None",
            uncertainty=self.uncertainty,
            methods = self.methods,     # iterate over methods for printing
            qindex = self.quality_index,
            lit_source=self.literature_source if self.literature_source else "None",
            fresource=_pretty_str(self.file_resource) if self.file_resource else "None")
        return ret

class EC8(SiteIndicator):
    def __init__(self, value, quality_index=None, literature_source=None, file_resource=None):
        """
        :type value: Enum of type :class:`~obspy.io.sitexml.core.EC8Class`
        :param value: EC8 class
        :type quality_index: float
        :param quality_index: Quality index of the site indicator. Takes values between 0 and 1.
            Calculated according to the guidelines of the SERA D7.2 Deliverable.
        :type literature_source: :class:`~obspy.io.sitexml.core.LiteratureSource`
        :param literature_source: The literature source related with the provided site indicator value
        :type file_resource: :class:`~obspy.core.inventory.util.ExternalReference` ????
        :param file_resource: A public URL for the literature_source
        """
        # Maybe here I should also use setter / getter
        if ( _sitexml_check_enum(value, EC8Class, "EC8") ):
            super(EC8, self).__init__(
                name="ec8", value=value, uncertainty=None, methods=None, 
                quality_index=quality_index, literature_source=literature_source, 
                file_resource=file_resource)

class H800(SiteIndicator):
    def __init__(self, value, uncertainty=None, quality_index=None, literature_source=None, 
                 file_resource=None):
        """
        :type value: int        
        :param value: Engineering depth. Depth beyond which the shear-wave velocity Vs exceeds 800 m/s.
        :type uncertainty: int
        :param uncertainty: Uncertainty related with the provided site indicator value
        :type quality_index: float
        :param quality_index: Quality index of the site indicator. Takes values between 0 and 1.
            Calculated according to the guidelines of the SERA D7.2 Deliverable.
        :type literature_source: :class:`~obspy.io.sitexml.core.LiteratureSource`
        :param literature_source: The literature source related with the provided site indicator value
        :type file_resource: :class:`~obspy.core.inventory.util.ExternalReference` ????
        :param file_resource: A public URL for the literature_source
        """
        super(H800, self).__init__(
                name="h800", value=value, uncertainty=uncertainty, methods=None, 
                quality_index=quality_index, literature_source=literature_source, 
                file_resource=file_resource)
        
        self._validate_value_uncertainty(int)

class BedrockDepth(SiteIndicator):
    def __init__(self, value, uncertainty=None, quality_index=None, literature_source=None, 
                 file_resource=None):
        """
        :type value: int        
        :param value: Seismological bedrock depth.
        :type uncertainty: int
        :param uncertainty: Uncertainty related with the provided site indicator value
        :type quality_index: float
        :param quality_index: Quality index of the site indicator. Takes values between 0 and 1.
            Calculated according to the guidelines of the SERA D7.2 Deliverable.
        :type literature_source: :class:`~obspy.io.sitexml.core.LiteratureSource`
        :param literature_source: The literature source related with the provided site indicator value
        :type file_resource: :class:`~obspy.core.inventory.util.ExternalReference` ????
        :param file_resource: A public URL for the literature_source
        """
        super(BedrockDepth, self).__init__(
            name="bedrock_depth", value=value, uncertainty=uncertainty, methods=None, 
            quality_index=quality_index, literature_source=literature_source, 
            file_resource=file_resource)
        
        self._validate_value_uncertainty(int)

class GeologicalUnit(SiteIndicator):
    def __init__(self, value, quality_index=None, geological_map_scale=None, 
                 geological_unit_OGE=None, literature_source=None, file_resource=None):
        """
        :type value: str
        :param value: Brief description of the surface geology (free text)
        :type quality_index: float
        :param quality_index: Quality index of the site indicator. Takes values between 0 and 1.
            Calculated according to the guidelines of the SERA D7.2 Deliverable.
        :type geological_map_scale: str
        :param geological_map_scale: Scale of geological map used for the description of surface geology
        :type geological_unit_OGE: str
        :param geological_unit_OGE: Description of the surface geology according to a Unified, Pan- European Map
        :type literature_source: :class:`~obspy.io.sitexml.core.LiteratureSource`
        :param literature_source: The literature source related with the provided site indicator value
        :type file_resource: :class:`~obspy.core.inventory.util.ExternalReference` ????
        :param file_resource: A public URL for the literature_source
        """
        self.geological_map_scale = geological_map_scale
        self.geological_unit_OGE = geological_unit_OGE
        super(GeologicalUnit, self).__init__(
            name="geological_unit", value=value, quality_index=quality_index, 
                literature_source=literature_source, file_resource=file_resource)
        
class ResonanceFrequency(SiteIndicator):
    def __init__(self, value, uncertainty=None, methods=None, 
                 quality_index=None, literature_source=None, 
                 file_resource=None):
        super(ResonanceFrequency, self).__init__(
            name="resonance_frequency", value=value, uncertainty=uncertainty, methods=methods, 
            quality_index=quality_index, literature_source=literature_source, 
            file_resource=file_resource)
        
class VelocityS30(SiteIndicator):
    def __init__(self, value, uncertainty=None, methods=None, 
                 quality_index=None, literature_source=None, 
                 file_resource=None, method_combined_quality_index=None, manual_quality_index=None):
        self.method_combined_quality_index = method_combined_quality_index
        self.manual_quality_index = manual_quality_index
        super(VelocityS30, self).__init__(
            name="velocity_s30", value=value, uncertainty=uncertainty, methods=methods, 
            quality_index=quality_index, literature_source=literature_source, 
            file_resource=file_resource)

"""
class VelocityProfile(SiteIndicator): 
"""
class VelocityProfile(SiteIndicator):
    def __init__(self, velocity_profile_data=None, quality_index=None, 
                 literature_source=None, file_resource=None):
        """
        :type velocity_profile_data: :class:`~obspy.io.sitexml.core.VelocityProfileData`
        :param velocity_profile_data: List of Velocity Profiles.
        """
        self.velocity_profile_data = velocity_profile_data  # triggers setter/validation
        super().__init__(
            name="VP", value=self.velocity_profile_data,
            quality_index=quality_index,
            literature_source=literature_source,
            file_resource=file_resource)

    def __str__(self):
        output=[]
        output.append(super().__str__())
        for i in range(0, len(self.velocity_profile_data)):
            output.append("\nVelocity Profile # " + str(i) + "\n")
            output.append(self.velocity_profile_data[i].__str__())
        return "\n".join(output) 
    
    @property
    def velocity_profile_data(self):
        return self._velocity_profile_data

    @velocity_profile_data.setter
    def velocity_profile_data(self, value):
        if value is None:
            self._velocity_profile_data = []
            return
        if not hasattr(value, "__iter__"):
            raise ValueError("velocity_profile_data must be iterable (e.g., a list).")
        vp_data = list(value)  # ensure we evaluate any generator
        if any(not isinstance(x, VelocityProfileData) for x in vp_data):
            raise ValueError(
                f"velocity_profile_data must contain only VelocityProfileData instances. Got: {[type(x) for x in vp_data]}"
            )
        self._velocity_profile_data = vp_data

class VelocityProfileData(ComparingObject):
    def __init__(self, layer_count, density=None, velocityP=None, velocityS=None, 
                 top_depth=None, bottom_depth=None):
        """
        :type layer_count: int
        :param layer_count: Number of layers in velocity profile.
        :type vp_layer_data: :class:`~obspy.io.sitexml.core.VelocityProfileLayer`
        :param vp_layer_data: List of Velocity Profiles.
        """
        self.layer_count = layer_count
        self.density = density or []
        self.velocityP = velocityP or []
        self.velocityS = velocityS or []
        self.top_depth = top_depth or []
        self.bottom_depth = bottom_depth or []
        
    def __str__(self):
        def format_vwu(obj):
            if obj is None or obj.value is None:
                return "N/A"
            if obj.uncertainty is not None:
                return f"{obj.value:.2f} ± {obj.uncertainty:.2f}"
            else:
                return f"{obj.value:.2f}"

        headers = ["Layer", "Density", "Velocity P", "Velocity S", "Top Depth", "Bottom Depth"]
        rows = []
        for i in range(self.layer_count):
            row = [
                str(i + 1),
                format_vwu(self.density[i]) if i < len(self.density) else "N/A",
                format_vwu(self.velocityP[i]) if i < len(self.velocityP) else "N/A",
                format_vwu(self.velocityS[i]) if i < len(self.velocityS) else "N/A",
                format_vwu(self.top_depth[i]) if i < len(self.top_depth) else "N/A",
                format_vwu(self.bottom_depth[i]) if i < len(self.bottom_depth) else "N/A"
            ]
            rows.append(row)

        # Calculate column widths for formatting
        col_widths = [
            max(len(str(item)) for item in [header] + [row[i] for row in rows])
            for i, header in enumerate(headers)
        ]

        def format_row(row):
            return " | ".join(f"{cell:<{col_widths[i]}}" for i, cell in enumerate(row))

        lines = [
            format_row(headers),
            "-+-".join("-" * width for width in col_widths),
        ] + [format_row(row) for row in rows]

        return "\n".join(lines)

    def _validate_list_of_vwu(self, name, value):
        """
        Validates and standardizes a list of ValueWithUncertainty objects.
        Converts numbers to ValueWithUncertainty, keeps None, raises on bad types.
        """
        if value is None:
            return []

        if not hasattr(value, "__iter__") or isinstance(value, (str, bytes)):
            raise ValueError(f"{name} must be an iterable (e.g., a list of floats or ValueWithUncertainty).")

        validated = []
        for i, item in enumerate(value):
            if item is None:
                validated.append(None)
            elif isinstance(item, ValueWithUncertainty):
                validated.append(item)
            elif isinstance(item, (int, float)):
                validated.append(ValueWithUncertainty(item))
            else:
                raise TypeError(f"{name}[{i}] is not a valid type (expected float, ValueWithUncertainty, or None): {item}")
        
        return validated

    @property
    def layer_count(self):
        return self._layer_count

    @layer_count.setter
    def layer_count(self, value):
        if value is not None and value > 0:
            self._layer_count = value
        else:
            raise ValueError("layer_count must be a positive value.")

    @property
    def density(self):
        return self._density
    @density.setter
    def density(self, value):
        self._density = self._validate_list_of_vwu("density", value)

    @property
    def velocityP(self):
        return self._velocityP
    @velocityP.setter
    def velocityP(self, value):
        self._velocityP = self._validate_list_of_vwu("velocityP", value)

    @property
    def velocityS(self):
        return self._velocityS
    @velocityS.setter
    def velocityS(self, value):
        self._velocityS = self._validate_list_of_vwu("velocityS", value)

    @property
    def top_depth(self):
        return self._top_depth
    @top_depth.setter
    def velocityS(self, value):
        self._top_depth = self._validate_list_of_vwu("top_depth", value)

    @property
    def bottom_depth(self):
        return self._bottom_depth
    @bottom_depth.setter
    def velocityS(self, value):
        self._bottom_depth = self._validate_list_of_vwu("bottom_depth", value)
 
class ValueWithUncertainty():
    def __init__(self, value, uncertainty=None):
        self.value = value
        self.uncertainty = uncertainty

class LiteratureSource(ComparingObject):
    def __init__(self, title, firstAuthor=None, secondaryAuthors=None, year=None, booktitle=None, language=None, doi=None):
        self.title = title
        self.firstAuthor = firstAuthor
        self.secondaryAuthors = secondaryAuthors
        self.year = year
        self.booktitle = booktitle
        self.language = language
        self.doi = doi
   
    def __str__(self):
        return _pretty_str(self)
       
class SERASiteOwner(ComparingObject):
    def __init__(self, owner_codename, owner_fullname, ownerID=None, 
                 person_firstname=None, person_lastname=None, person_mbox=None, person_homepage=None, personID=None, 
                 institution_name=None, institution_mbox=None, institution_phone=None, institution_homepage=None, institution_ID=None,
                 address_street=None, address_locality=None, address_postal_code=None, address_country=None, address_country_code=None,
                 affiliation_department=None, affiliation_function=None):
        self.owner_codename = owner_codename 
        self.owner_fullname = owner_fullname
        self.ownerID = ownerID 
    
        self.person_firstname = person_firstname
        self.person_lastname = person_lastname
        self.person_mbox = person_mbox
        self.person_homepage = person_homepage
        self.personID = personID
    
        self.institution_name = institution_name
        self.institution_mbox = institution_mbox
        self.institution_phone = institution_phone
        self.institution_homepage = institution_homepage
        self.institution_ID = institution_ID
                 
        self.address_street = address_street
        self.address_locality = address_locality
        self.address_postal_code = address_postal_code
        self.address_country = address_country
        self.address_country_code = address_country_code
    
        self.affiliation_department = affiliation_department
        self.affiliation_function = affiliation_function

    def __str__(self):
        ret = ("Site owner information:\n"
               "\t{code}, {name},\n"
               "\tContact: {firstname} {lastname}, {mbox}, {homepage}\n"
               "\tContact affiliation:\n"
               "\t\tInstitution: {ins_name}, {ins_mbox}, {ins_phone}, {ins_homepage}\n"
               "\t\tInstitution address: {street}, {postal_code}, {country}\n"
               "\t\tDepartment: {department}\n"
               "\t\tPosition: {position}\n")
        
        ret = ret.format(
            code=self.owner_codename, name=self.owner_fullname, 

            firstname=self.person_firstname, lastname=self.person_lastname,
            mbox=self.person_mbox, homepage=self.person_homepage,
            
            ins_name=self.institution_name, ins_mbox=self.institution_mbox,
            ins_phone=self.institution_phone, ins_homepage=self.institution_homepage,
            
            street=self.address_street, postal_code=self.address_postal_code,
            country=self.address_country,
            
            department=self.affiliation_department, position=self.affiliation_function)
        
        return ret
    
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
