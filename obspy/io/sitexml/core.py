#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Provides the SERASite class.

:copyright:
    ORFEUS, 2025
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

from obspy.core.util.base import ComparingObject
from obspy.core.event import ResourceIdentifier
from obspy.core.inventory.util import (Latitude, Longitude, Distance, ExternalReference)
from obspy.io.sitexml.util import (TopographySchemaA, TopographySchemaB, EC8Class, 
                                    ResonanceFrequencyMethod, VelocityS30Method,
                                _pretty_str, _add_property, _enum_property, _add_iterable_property,
                                _validate_enum_list)

# Update Site indicator so that some indicatos have a simple / str value
# and others have valuewithuncertainty

class ValueWithUncertainty:
    def __init__(self, value, uncertainty=None, valid_type=float):
        """
        :param value: int or float, the main value.
        :param uncertainty: int, float, or None, representing uncertainty.
        :param indicator_name: str, for meaningful error messages.
        :param valid_type: type, expected numeric type (e.g., float, int).
        """
        self.valid_type = valid_type
        #self.indicator_name = indicator_name
        self.value = value
        self.uncertainty = uncertainty

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        try:
            val = self.valid_type(val)
        except (ValueError, TypeError):
            raise ValueError(f"Value must be convertible to {self.valid_type.__name__}")
        #if val <= 0:
        #    raise ValueError(f"Value of {self.indicator_name} must be positive.")
        self._value = val

    @property
    def uncertainty(self):
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, val):
        if val is None:
            self._uncertainty = None
            return
        try:
            val = self.valid_type(val)
        except (ValueError, TypeError):
            raise ValueError(f"Uncertainty must be convertible to {self.valid_type.__name__} or None")
        #if val <= 0:
        #    raise ValueError(f"Uncertainty of {self.indicator_name} must be positive.")
        self._uncertainty = val

    def __str__(self):
        if self is None or self.value is None:
            return "N/A"
        if self.uncertainty is not None:
            return f"{self.value:.2f} ± {self.uncertainty:.2f}"
        else:
            return f"{self.value:.2f}"

class LiteratureSource(ComparingObject):
    
    def __init__(self, title, first_author=None, secondary_authors=None, 
                 year=None, booktitle=None, language=None, doi=None):
        self.title = title
        self.first_author = first_author
        self.secondary_authors = secondary_authors
        self.year = year
        self.booktitle = booktitle
        self.language = language
        self.doi = doi
   
    def __str__(self):
        return _pretty_str(self)

class SiteIndicator(ComparingObject):
    literature_source = _add_property("literature_source", LiteratureSource)
    #quality_index = _add_property("quality_index", float)
    #file_resource = _add_property("file_resource", ExternalReference)

    def __init__(self, name, value, methods=None, 
                 quality_index=None, literature_source=None, file_resource=None):
        """
        :type name: str
        :param name: Indicator type. One of: "siteClassEC8", "h800", "bedrockDepth", 
                    "geologicalUnit", "velocityS30", "resonanceFrequency", "velocityProfile"
        :type value: str / ValueWithUncertainty / VelocityProfileData
        :param value: Value of the indicator. Type depends on the indicator.
        :type methods: list of str, optional
        :param methods: Methods used for the estimation / calculation of the site indicator
        :type quality_index: float, optional
        :param quality_index: Quality index of the site indicator. Takes values between 0 and 1.
            Calculated according to the guidelines of the SERA D7.2 Deliverable.
        :type literature_source: :class:`~obspy.io.sitexml.core.LiteratureSource`, optional
        :param literature_source: The literature source related with the provided site indicator value
        :type file_resource: :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param file_resource: A public URL for the literature_source
        """
        self.name = name
        self.value = value
        self.methods = methods or []
        self.quality_index = quality_index 
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

    def __str__(self):
        ret = ("{name} parameters:\n"
               "\t{name} value: {value},\n"
               "\tMethods: {methods},\n"
               "\tQuality index: {qindex},\n"
               "\tLiterature source: {lit_source},\n"
               "\tFile resource: {fresource},\n")
        ret = ret.format(
            name=self.name, 
            value = self.value if self.name != "VelocityProfile" else "None",
            methods = self.methods,     # iterate over methods for printing
            qindex = self.quality_index,
            lit_source=self.literature_source if self.literature_source else "None",
            fresource=_pretty_str(self.file_resource) if self.file_resource else "None")
        return ret

class EC8(SiteIndicator):
    value = _enum_property("value", EC8Class)
    
    def __init__(self, value, quality_index=None, literature_source=None, file_resource=None):
        """
        :type value: Enum of type :class:`~obspy.io.sitexml.util.EC8Class`, required.
        :param value: EC8 class
        :type quality_index: float, optional
        :param quality_index: Quality index of the site indicator. Takes values between 0 and 1.
            Calculated according to the guidelines of the SERA D7.2 Deliverable.
        :type literature_source: :class:`~obspy.io.sitexml.core.LiteratureSource`, optional.
        :param literature_source: The literature source related with the provided site indicator value
        :type file_resource: :class:`~obspy.core.inventory.util.ExternalReference` ????
        :param file_resource: A public URL for the literature_source
        """
        super(EC8, self).__init__(
                name="siteClassEC8", value=value, quality_index=quality_index, 
                literature_source=literature_source, file_resource=file_resource)

class H800(SiteIndicator):
    value = _add_property("value", ValueWithUncertainty)

    def __init__(self, value, quality_index=None, literature_source=None, 
                 file_resource=None):
        """
        :type value: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, required.   
        :param value: Engineering depth. Depth beyond which the shear-wave 
                        velocity Vs exceeds 800 m/s. Expecting Integer value.
        :type quality_index: float, optional
        :param quality_index: Quality index of the site indicator. Takes values between 0 and 1.
            Calculated according to the guidelines of the SERA D7.2 Deliverable.
        :type literature_source: :class:`~obspy.io.sitexml.core.LiteratureSource`, optional.
        :param literature_source: The literature source related with the provided site indicator value
        :type file_resource: :class:`~obspy.core.inventory.util.ExternalReference` ????
        :param file_resource: A public URL for the literature_source
        """
        super(H800, self).__init__(
                name="h800", value=value, quality_index=quality_index, 
                literature_source=literature_source, file_resource=file_resource)

class BedrockDepth(SiteIndicator):
    value = _add_property("value", ValueWithUncertainty)

    def __init__(self, value, quality_index=None, literature_source=None, 
                 file_resource=None):
        """
        :type value: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, required.      
        :param value: Seismological bedrock depth. Expecting Integer values.
        :type quality_index: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, optional
        :param quality_index: Quality index of the site indicator. Takes values between 0 and 1.
            Calculated according to the guidelines of the SERA D7.2 Deliverable.
        :type literature_source: :class:`~obspy.io.sitexml.core.LiteratureSource`, optional.
        :param literature_source: The literature source related with the provided site indicator value
        :type file_resource: :class:`~obspy.core.inventory.util.ExternalReference` ????
        :param file_resource: A public URL for the literature_source
        """
        super(BedrockDepth, self).__init__(
            name="bedrockDepth", value=value, quality_index=quality_index, 
            literature_source=literature_source, file_resource=file_resource)

class GeologicalUnit(SiteIndicator):
    def __init__(self, value, quality_index=None, geological_map_scale=None, 
                 geological_unit_OGE=None, literature_source=None, file_resource=None):
        """
        :type value: str, required.
        :param value: Brief description of the surface geology (free text)
        :type quality_index: float, optional
        :param quality_index: Quality index of the site indicator. Takes values between 0 and 1.
            Calculated according to the guidelines of the SERA D7.2 Deliverable.
        :type geological_map_scale: str, optional.
        :param geological_map_scale: Scale of geological map used for the description of surface geology
        :type geological_unit_OGE: str, optional.
        :param geological_unit_OGE: Description of the surface geology according to a Unified, Pan- European Map
        :type literature_source: :class:`~obspy.io.sitexml.core.LiteratureSource`, optional.
        :param literature_source: The literature source related with the provided site indicator value
        :type file_resource: :class:`~obspy.core.inventory.util.ExternalReference` ????
        :param file_resource: A public URL for the literature_source
        """
        self.geological_map_scale = geological_map_scale
        self.geological_unit_OGE = geological_unit_OGE
        super(GeologicalUnit, self).__init__(
            name="geologicalUnit", value=value, quality_index=quality_index, 
                literature_source=literature_source, file_resource=file_resource)
        
class ResonanceFrequency(SiteIndicator):
    value = _add_property("value", ValueWithUncertainty)
    methods = _validate_enum_list("methods", ResonanceFrequencyMethod)

    def __init__(self, value, quality_index=None, methods=None, 
                 literature_source=None, file_resource=None):
        """
        :type value: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, required.           
        :param value: Resonance Frequency (f0). Expecting float values.
        :type quality_index: float, optional
        :param quality_index: Quality index of the site indicator. Takes values between 0 and 1.
            Calculated according to the guidelines of the SERA D7.2 Deliverable.
        :type methods: List of Enum type :class:`~obspy.io.sitexml.util.ResonanceFrequencyMethod`
        :param methods: Methods used for the estimation of ResonanceFrequency
        :type literature_source: :class:`~obspy.io.sitexml.core.LiteratureSource`, optional.
        :param literature_source: The literature source related with the provided site indicator value
        :type file_resource: :class:`~obspy.core.inventory.util.ExternalReference` ????
        :param file_resource: A public URL for the literature_source
        """
        super(ResonanceFrequency, self).__init__(
            name="resonanceFrequency", value=value, methods=methods, 
            quality_index=quality_index, literature_source=literature_source, 
            file_resource=file_resource)
        
class VelocityS30(SiteIndicator):
    value = _add_property("value", ValueWithUncertainty)
    methods = _validate_enum_list("methods", VelocityS30Method)
    
    def __init__(self, value, quality_index=None, methods=None,
                 method_combined_quality_index=None, manual_quality_index=None, 
                 literature_source=None, file_resource=None):
        """
        :type value: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, required.         
        :param value: Velocity S30. Expecting float values.
        :type quality_index: float, optional
        :param quality_index: Quality index of the site indicator. Takes values between 0 and 1.
            Calculated according to the guidelines of the SERA D7.2 Deliverable.
        :type methods: List of Enum type :class:`~obspy.io.sitexml.util.VelocityS30Method`
        :param methods: Methods used for the estimation of Velocity S30
        :type method_combined_quality_index: float
        :param method_combined_quality_index: 
        :type manual_quality_index: float
        :param manual_quality_index: 
        :type literature_source: :class:`~obspy.io.sitexml.core.LiteratureSource`, optional.
        :param literature_source: The literature source related with the provided site indicator value
        :type file_resource: :class:`~obspy.core.inventory.util.ExternalReference` ????
        :param file_resource: A public URL for the literature_source
        """

        self.method_combined_quality_index = method_combined_quality_index
        self.manual_quality_index = manual_quality_index
        super(VelocityS30, self).__init__(
            name="velocityS30", 
            value=value, 
            quality_index=quality_index, 
            methods=methods, 
            literature_source=literature_source, 
            file_resource=file_resource)
        
class VelocityProfileSurvey(SiteIndicator):
    
    def __init__(self, velocity_profiles=None, quality_index=None, 
                 literature_source=None, file_resource=None):
        """
        :type velocity_profiles: list of :class:`~obspy.io.sitexml.core.VelocityProfile`
        :param velocity_profiles: List of Velocity Profiles.
        :type quality_index: float, optional
        :param quality_index: Quality index of the site indicator. Takes values between 0 and 1.
            Calculated according to the guidelines of the SERA D7.2 Deliverable.
        :type literature_source: :class:`~obspy.io.sitexml.core.LiteratureSource`, optional
        :param literature_source: The literature source related with the provided site indicator value
        :type file_resource: :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param file_resource: A public URL for the literature_source
        """
        self.velocity_profiles = velocity_profiles  # triggers setter/validation
        super(VelocityProfileSurvey, self).__init__(
            name="velocityProfile", 
            value=self.velocity_profiles,
            quality_index=quality_index,
            literature_source=literature_source,
            file_resource=file_resource)

    def __str__(self):
        output=[]
        output.append(super().__str__())
        if self.velocity_profiles:
            for i in range(0, len(self.velocity_profiles)):
                output.append("\nVelocity Profile # " + str(i) + "\n")
                output.append(self.velocity_profiles[i].__str__())
        return "\n".join(output) 

class VelocityProfileData(ComparingObject):
    density = _add_property("density", ValueWithUncertainty)
    velocityP = _add_property("velocityP", ValueWithUncertainty)
    velocityS = _add_property("velocityS", ValueWithUncertainty)
    top_depth = _add_property("top_depth", ValueWithUncertainty)
    bottom_depth = _add_property("bottom_depth", ValueWithUncertainty)

    def __init__(self, density=None, velocityP=None, velocityS=None, 
                 top_depth=None, bottom_depth=None):
        """
        :type density: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`
        :param density: Layer density
        :type velocityP: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`
        :param velocityP: Layer velocityP value
        :type velocityS: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`
        :param velocityS: Layer velocityS value
        :type top_depth: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`
        :param top_depth: Layer top depth 
        :type bottom_depth: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`
        :param bottom_depth: Layer bottom depth
        """
        self.density = density 
        self.velocityP = velocityP 
        self.velocityS = velocityS 
        self.top_depth = top_depth
        self.bottom_depth = bottom_depth

class VelocityProfile(ComparingObject):
    velocity_profile_data = _add_iterable_property("velocity_profile_data", VelocityProfileData)

    def __init__(self, layer_count, resource_id=None, velocity_profile_data=None):
        """
        :type resource_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
        :param resource_id: Unique Velocity Profile Resource ID
        :type layer_count: Positive int, required.
        :param layer_count: Number of layers in velocity profile.
        :type velocity_profile_data: :class:`~obspy.io.sitexml.core.VelocityProfileData`
        :param velocity_profile_data: An array of velocity profile data for all layers.
                            Length of array should be equal to layer_count.
        """
        self.resource_id = resource_id 
        self.layer_count = layer_count 
        self.velocity_profile_data = velocity_profile_data 

    @property
    def layer_count(self):
        return self._layer_count

    @layer_count.setter
    def layer_count(self, value):
        if value is not None and value > 0:
            self._layer_count = value
        else:
            raise ValueError("layer_count must be a positive value.")
    
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
                format_vwu(self.velocity_profile_data[i].density),
                format_vwu(self.velocity_profile_data[i].velocityP),
                format_vwu(self.velocity_profile_data[i].velocityS),
                format_vwu(self.velocity_profile_data[i].top_depth),
                format_vwu(self.velocity_profile_data[i].bottom_depth) 
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
            "Resource_ID: " + (self.resource_id if self.resource_id else "N/A"),
            "Layer Count: " + str(self.layer_count) + "\n",
            format_row(headers),
            "-+-".join("-" * width for width in col_widths),
        ] + [format_row(row) for row in rows]
        return "\n".join(lines)
    
class SERASiteOwner(ComparingObject):
    def __init__(self, owner_codename=None, owner_fullname=None, ownerID=None, 
                 person_firstname=None, person_lastname=None, person_mbox=None, person_homepage=None, personID=None, 
                 institution_name=None, institution_mbox=None, institution_phone=None, institution_homepage=None, institutionID=None,
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
        self.institutionID = institutionID
                 
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
    
class SiteDescription(ComparingObject):
    station_code = _add_property("station_code", str)
    latitude = _add_property("latitude", Latitude)
    longitude = _add_property("longitude", Longitude)
    altitude = _add_property("altitude", Distance)
    min_distance_from_station = _add_property("min_distance_from_station", Distance)
    max_distance_from_station = _add_property("max_distance_from_station", Distance)
    bedrock_depth = _add_property("bedrock_depth", BedrockDepth)
    h800 = _add_property("h800", H800)
    ec8 = _add_property("ec8", EC8)
    geological_unit = _add_property("geological_unit", GeologicalUnit)
    topographyA = _enum_property("topographyA", TopographySchemaA)
    topographyB = _enum_property("topographyB", TopographySchemaB)

    def __init__(self, resource_id, station_code=None, latitude=0, longitude=0, altitude=None, 
                 min_distance_from_station=None, max_distance_from_station=None, 
                 ec8=None, bedrock_depth=None, h800=None, geological_unit=None, 
                 morphology=None, topographyA=None, topographyB=None, 
                 preferred_site_analysisID=None, preferred_velocity_profileID=None):
        """
        :type resource_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`, required
        :param resource_id: Unique Site Description Resource ID
        :type station_code: str, optional.
        :param station_code: Not used in SiteXML, but is needed in order to 
                            correlate with the Station Object. 
        :type latitude: :class:`~obspy.core.inventory.util.Latitude`, required
        :param latitude: The latitude of the site.
        :type longitude: :class:`~obspy.core.inventory.util.Longitude`, required
        :param longitude: The longitude of the site. 
        :type altitude: :class:`~obspy.core.inventory.util.Distance`, optional.
        :param altitude: Elevation of ground with respect to sea level (m).
        :type min_distance_from_station: :class:`~obspy.core.inventory.util.Distance`, optional.
        :param min_distance_from_station: Minimum distance between the permanent seismological station and 
            site characterization measurement. Should be used only when representative latitude and longitude 
            of site characterization measurements cannot be provided. 
        :type max_distance_from_station: :class:`~obspy.core.inventory.util.Distance`, optional.
        :param max_distance_from_station: Maximum distance between the permanent seismological station and 
            site characterization measurement. Should be used only when representative latitude and longitude 
            of site characterization measurements cannot be provided. 
        :type ec8: :class:`~obspy.io.sitexml.core.EC8`, optional.
        :param ec8: Ground type according to Eurocode 8, based on the velocity S30 value and geotechnical description. 
        :type h800: :class:`~obspy.io.sitexml.core.H800`, optional.
        :param h800: Engineering depth. Depth beyond which the shear-wave velocity Vs exceeds 800 m/s.
        :type bedrock_depth: :class:`~obspy.io.sitexml.core.BedrockDepth`, optional.
        :param bedrock_depth: Seismological bedrock depth. 
        :type geological_unit: :class:`~obspy.io.sitexml.core.GeologicalUnit`, optional.
        :param geological_unit: Brief description of the surface geology (free text). 
        :type morphology: str, optional.
        :param morphology: Qualitative description of the shape of the earth's surface (free text). 
        :type topographyA: str, optional.
        :param topographyA: Quantitative description of the surface according to the Italian Code 
            (detailed description of the scheme in SERA Deliverable D7.1 - Appendix I).
            See :class:`~obspy.io.sitexml.util.TopographySchemaA` for allowed values. 
        :type topographyB: str, optional.
        :param topographyB: Quantitative description of the shape of the earth's surface according to 
            Burjanek et al, 2014 (detailed description of the scheme in SERA Deliverable D7.1 - Appendix I). 
            See :class:`~obspy.io.sitexml.util.TopographySchemaB` for allowed values. 
        :type preferred_site_analysisID: :class:`~obspy.core.event.resourceid.ResourceIdentifier`.
        :param preferred_site_analysisID: Preferred Site Analysis ID. If you provide one or more
                analysis for this site you should use this field to designate the prefered analysis.
        :type preferred_velocity_profileID: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
        :param preferred_velocity_profileID: Preferred Velocity Profile ID. If you provide one or more
                velocity profiles for this site you should use this field to designate the prefered VP.
        """

        self.resource_id = resource_id
        self.station_code = station_code
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
        self.topographyA = topographyA
        self.topographyB = topographyB

        self.preferred_site_analysisID = preferred_site_analysisID
        self.preferred_velocity_profileID = preferred_velocity_profileID
    
    def __str__(self):
        ret = ("Site Description parameters:\n"
               "\tresource_id: {id},\n"
               "\tStation: {station},\n"
               "\tLatitude {lat:.4f}, Longitude: {lng:.4f}, Altitude {alt} m,\n"
               "\tMorphology: {morphology},\n"
               "\tTopography A: {topoA},\n"
               "\tTopography B: {topoB},\n"
               "\tEC8 class: {ec8},\n"
               "\tH800: {h800} m,\n"
               "\tBedrock depth: {bdepth} m,\n"
               "\tGeological Unit: {gunit}\n"
               "\n\tPreferred Analysis: {analysis_id}\n"
               "\tPreferred Velocity Profile: {vp_id}\n")
        ret = ret.format(
            id=self.resource_id,
            station=self.station_code,
            lat=self.latitude, lng=self.longitude, alt=self.altitude,
            morphology = self.morphology,
            topoA = self.topographyA, topoB = self.topographyB,
            ec8=self.ec8.value if self.ec8 else "None",
            h800=self.h800.value if self.h800 else "None",
            bdepth=self.bedrock_depth.value if self.bedrock_depth else "None",
            gunit=self.geological_unit.value if self.geological_unit else "None",
            analysis_id = self.preferred_site_analysisID,
            vp_id = self.preferred_velocity_profileID)
        return ret

class Analysis(ComparingObject):
    resonance_frequency = _add_property("resonance_frequency", ResonanceFrequency)
    velocity_s30 = _add_property("velocity_s30", VelocityS30)
    velocity_profile_survey = _add_property("velocity_profile_survey", VelocityProfileSurvey)
    velocity_profile_count = _add_property("velocity_profile_count", int)
    spt_logs_count = _add_property("spt_logs_count", int)
    cpt_logs_count = _add_property("cpt_logs_count", int)
    borehole_logs_count = _add_property("borehole_logs_count", int)

    # TODOS
    # Check counter type if value > 0 (like layer count)

    def __init__(self, resource_id=None, site_descriptionID=None, creation_date=None, 
                 resonance_frequency=None, velocity_s30=None, 
                 velocity_profile_survey=None, velocity_profile_count=None, 
                 spt_logs_count=None, cpt_logs_count=None, borehole_logs_count=None):
        """
        :type resource_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`, required.
        :param resource_id: Analysis resource ID. 
        :type site_descriptionID: :class:`~obspy.core.event.resourceid.ResourceIdentifier`, required.
        :param site_descriptionID: The Site Description object this analysis refers to. 
        :type creation_date: datetime, optional.
        :param creation_date: Date that this analysis was published.
        :type resonance_frequency: :class:`~obspy.io.sitexml.core.ResonanceFrequency`, optional.
        :param resonance_frequency: The Resonance frequency of the soil column. 
        :type velocity_s30: :class:`~obspy.io.sitexml.core.VelocityS30`, optional.
        :param velocity_s30: Average shear-wave velocity between 0 and 30 meters depth. 
        :type velocity_profile_survey: :class:`~obspy.io.sitexml.core.VelocityProfileSurvey`, optional.
        :param velocity_profile_survey: Velocity Profile Survey.
            Parent object for Velocity Profiles. 
        :type velocity_profile_count: int, optional.
        :param velocity_profile_count: Number of available velocity profiles. If this analysis 
            includes velocity profiles, you should use this field to provide the number of VPs.
        :type spt_logs_count: int, optional.
        :param spt_logs_count: Number of available SPT profile(s). 
        :type cpt_logs_count: int, optional.
        :param cpt_logs_count: Number of available CPT profile(s). 
        :type borehole_logs_count: int, optional.
        :param borehole_logs_count: Number of available borehole log profile(s). 
       """
        self.resource_id = resource_id        
        self.site_descriptionID = site_descriptionID   
        self.resonance_frequency = resonance_frequency
        self.velocity_s30 = velocity_s30
        self.velocity_profile_count = velocity_profile_count
        self.spt_logs_count = spt_logs_count
        self.cpt_logs_count = cpt_logs_count
        self.borehole_logs_count = borehole_logs_count
        self.velocity_profile_survey = velocity_profile_survey
            
    def __str__(self):
        ret = ("Analysis:\n"
               "\tResource ID: {analysis_id},\n"
               "\tSite Description ID: {sd_id},\n"
               "\tResonance Frequency: {rfreq},\n"
               "\tVelocity S30: {vs30},\n"
               "\tVelocity Profiles count: {vp_count},\n"
               "\tSPT Logs count: {spt_logs_count},\n"
               "\tCPT Logs count: {cpt_logs_count},\n"
               "\tBorehole Logs count: {bh_logs_count} \n")
        ret = ret.format(
            analysis_id = self.resource_id, 
            sd_id = self.site_descriptionID,
            rfreq = self.resonance_frequency.value if self.resonance_frequency else "None",
            vs30 = self.velocity_s30.value if self.velocity_s30 else "None",
            vp_count = self.velocity_profile_count, 
            spt_logs_count = self.spt_logs_count, 
            cpt_logs_count = self.cpt_logs_count, 
            bh_logs_count = self.borehole_logs_count)
        return ret
     
class SERASite(ComparingObject):
    """
    This is the parent class for the siteXML object tree.
    """
    site_owner = _add_property("site_owner", SERASiteOwner)
    site_description = _add_property("site_description", SiteDescription)
    external_references = _add_iterable_property("external_references", ExternalReference)
    analysis = _add_iterable_property("analysis", Analysis)
    resource_id = _add_property("resource_id", ResourceIdentifier)
    
    def __init__(self, resource_id, site_owner, site_description, 
                 analysis=None, overall_quality_index=None, 
                 created=None, external_references=None):
        """
        :type resource_id: :class:`~obspy.core.event.resourceid.ResourceIdentifier`
        :param resource_id: SERA SiteXML Unique Identifier (siteID).
        :type site_owner: :class:`~obspy.core.io.sitexml.SERASiteOwner`, required.
        :param site_owner: The site owner metadata. 
        :type site_description: :class:`~obspy.core.io.sitexml.SiteDescription`, required.
        :param site_description: The site description parameters (H800, Bedrock depth, 
                            EC8 class, geological unit, morphology, topography).
        :type analysis: list of :class:`~obspy.io.sitexml.core.Analysis`, optional.
        :param analysis: The site characterization parameters 
                            (VS30, resonance frequency, velocity profiles).
        :type overall_quality_index: float, optional.
        :param overall_quality_index: The overall quality index of the site 
                            characterization parameters.
        :type created: :class:`~obspy.UTCDateTime`
        :param created: DateTime the SiteXML file was generated
        :type external_references: List of :class:`~obspy.core.inventory.util.ExternalReference`, optional.
        :param external_references: Additional resources with site characterization metadata. 
        """
        self.resource_id = resource_id
        self.site_owner = site_owner
        self.site_description = site_description
        self.analysis = analysis
        self.created = created
        self.external_references = external_references
        # TO CHECK: If this one is calculated it should be removed from the parameters 
        self.overall_quality_index = overall_quality_index
    
        """
        # From ObsPy event for resource id
        # Automatically bind a resource id to the parent object
        # if value is a resource id bind or unbind the resource_id
            if isinstance(value, ResourceIdentifier):
                if name == "resource_id":  # bind the resource_id to self
                    self.resource_id.set_referred_object(self, warn=False)
                else:  # else unbind to allow event scoping later
                    value._parent_key = None
        """
    def __str__(self):
        output=["\n#################\n"]
        if self.site_description.station_code:
            title = "Site Metadata (station: " + self.site_description.station_code + ")"
        else:
            title = "Site Metadata"
        output.append(title)
        output.append("\n#################\n")
       
        #output.append("resource_id: " + self.resource_id + "\n")
        output.append(self.site_owner.__str__())
        output.append(self.site_description.__str__())
        
        if self.analysis:
            for i in range(0, len(self.analysis)):
                output.append("\nAnalysis # " + str(i) + "\n")
                output.append(self.analysis[i].__str__())
        return "\n".join(output) 

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
