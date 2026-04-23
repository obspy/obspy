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
from collections.abc import Iterable

import obspy
from obspy.core.inventory.util import (Latitude, Longitude, Distance, 
                                       ExternalReference)
from .util import (BaseNode, SiteXMLValidationError,
                    TopographySchemaA, TopographySchemaB, EC8Class, 
                    ResonanceFrequencyMethod, VelocityS30Method,
                    Vs30MethodCombined, Vs30ManualIndex,
                    _pretty_str, scalar_property, resource_id_property,
                    wrapped_property, enum_property, wrapped_list_property,
                    enum_list_property)
    
class ValueWithUncertainty(BaseNode):
    """
    Numeric SiteXML value with an optional uncertainty of the same type.
    """

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
            raise SiteXMLValidationError(
                f"Value must be convertible to {self.valid_type.__name__}"
            )
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
            raise SiteXMLValidationError(
                f"Uncertainty must be convertible to "
                f"{self.valid_type.__name__} or None"
            )
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

class LiteratureSource(BaseNode):
    """
    Bibliographic source metadata used by SiteXML indicator references.
    """

    title = scalar_property("title", allow_none=False, allow_empty=False)
    first_author = scalar_property(
        "first_author", allow_none=False, allow_empty=False)

    def __init__(self, title, first_author, secondary_authors=None,
                 year=None, booktitle=None, language=None, doi=None):
        """
        :type title: str
        :param title: Title of the publication. Required.
        :type first_author: str
        :param first_author: Main author of the publication. Required.
        :type secondary_authors: str, optional
        :param secondary_authors: Comma-separated list of secondary authors.
        :type year: str or int, optional
        :param year: Four-digit publication year. Stored as a string.
        :type booktitle: str, optional
        :param booktitle: Journal, book, or proceedings title.
        :type language: str, optional
        :param language: Two-letter lowercase ISO 639-1 language code.
        :type doi: str, optional
        :param doi: Digital Object Identifier (DOI) of the publication.
        """
        self.title = title
        self.first_author = first_author
        self.secondary_authors = secondary_authors
        self.year = year
        self.booktitle = booktitle
        self.language = language
        self.doi = doi
   
    def __str__(self):
        return _pretty_str(self)

    @property
    def year(self):
        return self._year

    @year.setter
    def year(self, value):
        if value is None:
            self._year = None
            return
        if isinstance(value, float):
            if not value.is_integer():
                raise SiteXMLValidationError(
                    f"year must be a four-digit string, got {value!r}"
                )
            value = int(value)
        value = str(value)
        if not value.isdigit() or len(value) != 4:
            raise SiteXMLValidationError(
                f"year must be a four-digit string, got {value!r}"
            )
        self._year = value

class SiteIndicator(BaseNode):
    """
    Base class for SiteXML site-characterization indicator objects.
    """

    literature_source = wrapped_property("literature_source", LiteratureSource)
    #quality_index = wrapped_property("quality_index", float)

    def __init__(self, name, value, methods=None, 
                 quality_index=None, literature_source=None, external_reference=None):
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
        :type external_reference: :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param external_reference: An external URI and description for this indicator.
        """
        self.name = name
        self.value = value
        self.methods = methods or []
        self.quality_index = quality_index 
        self.literature_source = literature_source
        self.external_reference = external_reference

    @property
    def external_reference(self):
        return self._external_reference

    @external_reference.setter
    def external_reference(self, value):
        if value is None:
            self._external_reference = None
        elif isinstance(value, ExternalReference):
            self._external_reference = value
        else:
            self._external_reference = ExternalReference(value, None)

    def __str__(self):
        ret = ("{name} parameters:\n"
               "\t{name} value: {value},\n"
               "\tMethods: {methods},\n"
               "\tQuality index: {qindex},\n"
               "\tLiterature source: {lit_source},\n"
               "\tExternal reference: {external_ref},\n")
        ret = ret.format(
            name=self.name, 
            value = self.value if self.name != "VelocityProfile" else "None",
            methods = self.methods,     # iterate over methods for printing
            qindex = self.quality_index,
            lit_source=self.literature_source if self.literature_source else "None",
            external_ref=_pretty_str(self.external_reference) if self.external_reference else "None")
        return ret

class EC8(SiteIndicator):
    """
    Eurocode 8 ground type indicator.
    """

    value = enum_property("value", EC8Class)
    
    def __init__(self, value, quality_index=None, literature_source=None, external_reference=None):
        """
        :type value: Enum of type :class:`~obspy.io.sitexml.util.EC8Class`, required.
        :param value: EC8 class
        :type quality_index: float, optional
        :param quality_index: Quality index of the site indicator. Takes values between 0 and 1.
            Calculated according to the guidelines of the SERA D7.2 Deliverable.
        :type literature_source: :class:`~obspy.io.sitexml.core.LiteratureSource`, optional.
        :param literature_source: The literature source related with the provided site indicator value
        :type external_reference: :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param external_reference: An external URI and description for this indicator.
        """
        super(EC8, self).__init__(
                name="siteClassEC8", value=value, quality_index=quality_index, 
                literature_source=literature_source, external_reference=external_reference)

class H800(SiteIndicator):
    """
    Engineering bedrock depth indicator for Vs greater than 800 m/s.
    """

    value = wrapped_property("value", ValueWithUncertainty)

    def __init__(self, value, quality_index=None, literature_source=None, 
                 external_reference=None):
        """
        :type value: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, required.   
        :param value: Engineering depth. Depth beyond which the shear-wave 
                        velocity Vs exceeds 800 m/s. Expecting Integer value.
        :type quality_index: float, optional
        :param quality_index: Quality index of the site indicator. Takes values between 0 and 1.
            Calculated according to the guidelines of the SERA D7.2 Deliverable.
        :type literature_source: :class:`~obspy.io.sitexml.core.LiteratureSource`, optional.
        :param literature_source: The literature source related with the provided site indicator value
        :type external_reference: :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param external_reference: An external URI and description for this indicator.
        """
        super(H800, self).__init__(
                name="h800", value=value, quality_index=quality_index, 
                literature_source=literature_source, external_reference=external_reference)

class BedrockDepth(SiteIndicator):
    """
    Seismological bedrock depth indicator.
    """

    value = wrapped_property("value", ValueWithUncertainty)

    def __init__(self, value, quality_index=None, literature_source=None, 
                 external_reference=None):
        """
        :type value: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, required.      
        :param value: Seismological bedrock depth. Expecting Integer values.
        :type quality_index: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, optional
        :param quality_index: Quality index of the site indicator. Takes values between 0 and 1.
            Calculated according to the guidelines of the SERA D7.2 Deliverable.
        :type literature_source: :class:`~obspy.io.sitexml.core.LiteratureSource`, optional.
        :param literature_source: The literature source related with the provided site indicator value
        :type external_reference: :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param external_reference: An external URI and description for this indicator.
        """
        super(BedrockDepth, self).__init__(
            name="bedrockDepth", value=value, quality_index=quality_index, 
            literature_source=literature_source, external_reference=external_reference)

class GeologicalUnit(SiteIndicator):
    """
    Surface geology indicator with optional map-scale metadata.
    """

    def __init__(self, value, quality_index=None, geological_map_scale=None, 
                 geological_unit_OGE=None, literature_source=None, external_reference=None):
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
        :type external_reference: :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param external_reference: An external URI and description for this indicator.
        """
        self.geological_map_scale = geological_map_scale
        self.geological_unit_OGE = geological_unit_OGE
        super(GeologicalUnit, self).__init__(
            name="geologicalUnit", value=value, quality_index=quality_index, 
                literature_source=literature_source, external_reference=external_reference)
        
class ResonanceFrequency(SiteIndicator):
    """
    Site resonance-frequency indicator.
    """

    value = wrapped_property("value", ValueWithUncertainty)
    methods = enum_list_property("methods", ResonanceFrequencyMethod)

    def __init__(self, value, quality_index=None, methods=None, 
                 literature_source=None, external_reference=None):
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
        :type external_reference: :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param external_reference: An external URI and description for this indicator.
        """
        super(ResonanceFrequency, self).__init__(
            name="resonanceFrequency", value=value, methods=methods, 
            quality_index=quality_index, literature_source=literature_source, 
            external_reference=external_reference)
        
class VelocityS30(SiteIndicator):
    """
    Time-averaged shear-wave velocity over the upper 30 meters.
    """

    value = wrapped_property("value", ValueWithUncertainty)
    methods = enum_list_property("methods", VelocityS30Method)
    method_combined_qindex = enum_property("velocityS30MethodCombIndex", Vs30MethodCombined)
    manual_qindex = enum_property("velocityS30ManualIndex", Vs30ManualIndex)
    
    def __init__(self, value, quality_index=None, methods=None,
                 method_combined_qindex=None, manual_qindex=None, 
                 literature_source=None, external_reference=None):
        """
        :type value: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, required.         
        :param value: Velocity S30. Expecting float values.
        :type quality_index: float, optional
        :param quality_index: Quality index of the site indicator. Takes values between 0 and 1.
            Calculated according to the guidelines of the SERA D7.2 Deliverable.
        :type methods: List of Enum type :class:`~obspy.io.sitexml.util.VelocityS30Method`
        :param methods: Methods used for the estimation of Velocity S30
        :type method_combined_qindex: Enum of type :class:`~obspy.io.sitexml.util.Vs30MethodCombined`, optional
        :param method_combined_qindex: Carries the information on whether a combination of 
            two or more methodshas been applied to estimate the Vs30 value.
        :type manual_qindex: Enum of type :class:`~obspy.io.sitexml.util.Vs30ManualIndex`, optional
        :param manual_qindex: Overall qualitative factor on the knowledge of the 
            maximum depth of Vs measurements
        :type literature_source: :class:`~obspy.io.sitexml.core.LiteratureSource`, optional.
        :param literature_source: The literature source related with the provided site indicator value
        :type external_reference: :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param external_reference: An external URI and description for this indicator.
        """

        self.method_combined_qindex = method_combined_qindex
        self.manual_qindex = manual_qindex
        super(VelocityS30, self).__init__(
            name="velocityS30", 
            value=value, 
            quality_index=quality_index, 
            methods=methods, 
            literature_source=literature_source, 
            external_reference=external_reference)

    def __str__(self):
        output = [super().__str__()]
        output.append(
            "\tMethod Combined Qindex : " + str(self.method_combined_qindex)
        )
        output.append(
            "\tManual Qindex : " + str(self.manual_qindex)
        )
        return "\n".join(output)
        
class VelocityProfileSurvey(SiteIndicator):
    """
    Site indicator containing one or more velocity profiles.
    """

    def __init__(self, velocity_profiles=None, quality_index=None, 
                 literature_source=None, external_reference=None):
        """
        :type velocity_profiles: list of :class:`~obspy.io.sitexml.core.VelocityProfile`
        :param velocity_profiles: List of Velocity Profiles.
        :type quality_index: float, optional
        :param quality_index: Quality index of the site indicator. Takes values between 0 and 1.
            Calculated according to the guidelines of the SERA D7.2 Deliverable.
        :type literature_source: :class:`~obspy.io.sitexml.core.LiteratureSource`, optional
        :param literature_source: The literature source related with the provided site indicator value
        :type external_reference: :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param external_reference: An external URI and description for this indicator.
        """
        self.velocity_profiles = velocity_profiles  # triggers setter/validation
        super(VelocityProfileSurvey, self).__init__(
            name="velocityProfile", 
            value=self.velocity_profiles,
            quality_index=quality_index,
            literature_source=literature_source,
            external_reference=external_reference)

    def __str__(self):
        output=[]
        output.append(super().__str__())
        if self.velocity_profiles:
            for i in range(0, len(self.velocity_profiles)):
                output.append("\nVelocity Profile # " + str(i) + "\n")
                output.append(self.velocity_profiles[i].__str__())
        return "\n".join(output) 


class VelocityProfile(BaseNode):
    """
    Layered velocity profile associated with an analysis.
    """

    resource_id = resource_id_property(
        "resource_id", allow_none=False, allow_empty=False)

    def __init__(self, resource_id, velocity_profile_data, layer_count=None):
        """
        :type resource_id: str or :class:`~obspy.core.event.resourceid.ResourceIdentifier`
        :param resource_id: Unique Velocity Profile Resource ID.
        :type velocity_profile_data: :class:`~obspy.io.sitexml.core.VelocityProfileData`
        :param velocity_profile_data: An array of velocity profile data for all
            layers. Must contain at least one layer.
        :type layer_count: Positive int, optional.
        :param layer_count: Number of layers in velocity profile. If omitted,
            it is derived from ``velocity_profile_data``.
        """
        self.resource_id = resource_id
        self.velocity_profile_data = velocity_profile_data
        self.layer_count = layer_count

    @property
    def layer_count(self):
        if self._layer_count is None:
            return len(self.velocity_profile_data)
        return self._layer_count

    @layer_count.setter
    def layer_count(self, value):
        if value is None:
            self._layer_count = None
            return
        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise SiteXMLValidationError(
                f"Could not convert {value} to int: {exc}"
            )
        if value <= 0:
            raise SiteXMLValidationError(
                "layer_count must be a positive integer."
            )
        if hasattr(self, "_velocity_profile_data") and \
                self._velocity_profile_data is not None and \
                value != len(self._velocity_profile_data):
            raise SiteXMLValidationError(
                "Number of velocity profile data layers does not match "
                "the layer_count value."
            )
        self._layer_count = value

    @property
    def velocity_profile_data(self):
        return self._velocity_profile_data

    @velocity_profile_data.setter
    def velocity_profile_data(self, values):
        if values is None:
            raise SiteXMLValidationError("velocity_profile_data is required.")

        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            raise SiteXMLValidationError(
                "velocity_profile_data must be an iterable"
            )

        wrapped_items = []
        for value in values:
            if isinstance(value, VelocityProfileData):
                wrapped_items.append(value)
            else:
                try:
                    wrapped_items.append(VelocityProfileData(value))
                except Exception as exc:
                    raise SiteXMLValidationError(
                        "Could not convert element "
                        f"{value} to VelocityProfileData: {exc}"
                    )

        if not wrapped_items:
            raise SiteXMLValidationError(
                "velocity_profile_data must contain at least one layer."
            )

        if hasattr(self, "_layer_count") and self._layer_count is not None and \
                self._layer_count != len(wrapped_items):
            raise SiteXMLValidationError(
                "Number of velocity profile data layers does not match "
                "the layer_count value."
            )

        self._velocity_profile_data = wrapped_items
    
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

class VelocityProfileData(BaseNode):
    """
    Physical properties for a single velocity-profile layer.
    """

    top_depth = wrapped_property("top_depth", ValueWithUncertainty,
                                 allow_none=False)
    bottom_depth = wrapped_property("bottom_depth", ValueWithUncertainty)
    density = wrapped_property("density", ValueWithUncertainty)
    velocityP = wrapped_property("velocityP", ValueWithUncertainty)
    velocityS = wrapped_property("velocityS", ValueWithUncertainty)

    def __init__(self, top_depth, bottom_depth=None, density=None, 
                velocityP=None, velocityS=None):
        """
        :type top_depth: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`
        :param top_depth: Layer top depth, required.
        :type bottom_depth: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`
        :param bottom_depth: Layer bottom depth, optional.
        :type density: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`
        :param density: Layer density, optional
        :type velocityP: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`
        :param velocityP: Layer velocityP value, optional
        :type velocityS: :class:`~obspy.io.sitexml.core.ValueWithUncertainty`
        :param velocityS: Layer velocityS value, optional
        """
        self.top_depth = top_depth
        self.bottom_depth = bottom_depth
        self.density = density 
        self.velocityP = velocityP 
        self.velocityS = velocityS 
        
class SERASiteOwner(BaseNode):
    """
    Site owner and required contact-person metadata.
    """

    owner_codename = scalar_property(
        "owner_codename", allow_none=False, allow_empty=False)
    owner_fullname = scalar_property(
        "owner_fullname", allow_none=False, allow_empty=False)
    person_firstname = scalar_property(
        "person_firstname", allow_none=False, allow_empty=False)
    person_lastname = scalar_property(
        "person_lastname", allow_none=False, allow_empty=False)
    person_mbox = scalar_property(
        "person_mbox", allow_none=False, allow_empty=False)
    ownerID = resource_id_property("ownerID")
    personID = resource_id_property("personID")
    institutionID = resource_id_property("institutionID")

    def __init__(self, owner_codename, owner_fullname,
                 person_firstname, person_lastname, person_mbox, ownerID=None,
                 person_homepage=None, personID=None,
                 institution_name=None, institution_mbox=None, institution_phone=None, institution_homepage=None, institutionID=None,
                 address_street=None, address_locality=None, address_postal_code=None, address_country=None, address_country_code=None,
                 affiliation_department=None, affiliation_function=None):
        """
        :type owner_codename: str
        :param owner_codename: Short code name for the site owner. Required.
        :type owner_fullname: str
        :param owner_fullname: Full name of the site owner. Required.
        :type person_firstname: str
        :param person_firstname: First name of the contact person. Required.
        :type person_lastname: str
        :param person_lastname: Last name of the contact person. Required.
        :type person_mbox: str
        :param person_mbox: Email address of the contact person. Required.
        :type ownerID: str, optional
        :param ownerID: Public identifier for the owner.
        :type person_homepage: str, optional
        :param person_homepage: Homepage URL for the contact person.
        :type personID: str, optional
        :param personID: Public identifier for the contact person.
        :type institution_name: str, optional
        :param institution_name: Name of the contact person's institution.
        :type institution_mbox: str, optional
        :param institution_mbox: Email address of the institution.
        :type institution_phone: str, optional
        :param institution_phone: Phone number of the institution.
        :type institution_homepage: str, optional
        :param institution_homepage: Homepage URL of the institution.
        :type institutionID: str, optional
        :param institutionID: Public identifier for the institution.
        :type address_street: str, optional
        :param address_street: Street address of the institution.
        :type address_locality: str, optional
        :param address_locality: Locality of the institution address.
        :type address_postal_code: str, optional
        :param address_postal_code: Postal code of the institution address.
        :type address_country: str, optional
        :param address_country: Country name of the institution address.
        :type address_country_code: str, optional
        :param address_country_code: Country code of the institution address.
        :type affiliation_department: str, optional
        :param affiliation_department: Department of the contact person.
        :type affiliation_function: str, optional
        :param affiliation_function: Function or position of the contact person.
        """
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
    
class SiteDescription(BaseNode):
    """
    Location, morphology, and near-surface description for a SiteXML site.
    """

    resource_id = resource_id_property(
        "resource_id", allow_none=False, allow_empty=False)
    preferred_site_analysisID = resource_id_property(
        "preferred_site_analysisID")
    preferred_velocity_profileID = resource_id_property(
        "preferred_velocity_profileID")
    latitude = wrapped_property("latitude", Latitude, allow_none=False)
    longitude = wrapped_property("longitude", Longitude, allow_none=False)
    altitude = wrapped_property("altitude", Distance)
    min_distance_from_station = wrapped_property("min_distance_from_station", Distance)
    max_distance_from_station = wrapped_property("max_distance_from_station", Distance)
    bedrock_depth = wrapped_property("bedrock_depth", BedrockDepth)
    h800 = wrapped_property("h800", H800)
    ec8 = wrapped_property("ec8", EC8)
    geological_unit = wrapped_property("geological_unit", GeologicalUnit)
    topographyA = enum_property("topographyA", TopographySchemaA)
    topographyB = enum_property("topographyB", TopographySchemaB)

    def __init__(self, resource_id, latitude, longitude, altitude=None,
                 min_distance_from_station=None, max_distance_from_station=None, 
                 station_code=None, 
                 ec8=None, bedrock_depth=None, h800=None, geological_unit=None, 
                 morphology=None, topographyA=None, topographyB=None, 
                 preferred_site_analysisID=None, preferred_velocity_profileID=None,
                 overall_quality_index=None):
        """
        :type resource_id: str or :class:`~obspy.core.event.resourceid.ResourceIdentifier`, required
        :param resource_id: Unique Site Description Resource ID
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
        :type station_code: str, optional.
        :param station_code: The seismological station code installed in the site (if any). 
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
        :type preferred_site_analysisID: str or :class:`~obspy.core.event.resourceid.ResourceIdentifier`
        :param preferred_site_analysisID: Preferred Site Analysis ID. If you provide one or more
                analysis for this site you should use this field to designate the prefered analysis.
        :type preferred_velocity_profileID: str or :class:`~obspy.core.event.resourceid.ResourceIdentifier`
        :param preferred_velocity_profileID: Preferred Velocity Profile ID. If you provide one or more
                velocity profiles for this site you should use this field to designate the prefered VP.
        :type overall_quality_index: float, optional.
        :param overall_quality_index: The overall quality index of the site 
                            characterization parameters.
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

        self.overall_quality_index = overall_quality_index
    
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

    # TODOs: Check station_code against rules
    #
    @property
    def station_code(self):
        return self._station_code

    @station_code.setter
    def station_code(self, value):
        if value is None:
            self._station_code = None
            return
        if not isinstance(value, str):
            raise SiteXMLValidationError(
                "station_code must be a string or None"
            )
        self._station_code = value

class Analysis(BaseNode):
    """
    Site-characterization analysis and related indicator metadata.
    """

    resource_id = resource_id_property(
        "resource_id", allow_none=False, allow_empty=False)
    site_descriptionID = resource_id_property(
        "site_descriptionID", allow_none=False, allow_empty=False)
    resonance_frequency = wrapped_property("resonance_frequency", ResonanceFrequency)
    velocity_s30 = wrapped_property("velocity_s30", VelocityS30)
    velocity_profile_survey = wrapped_property("velocity_profile_survey", VelocityProfileSurvey)

    def __init__(self, resource_id, site_descriptionID, creation_date=None,
                 resonance_frequency=None, velocity_s30=None, 
                 velocity_profile_survey=None, spt_logs_count=None,
                 cpt_logs_count=None, borehole_logs_count=None):
        """
        :type resource_id: str or :class:`~obspy.core.event.resourceid.ResourceIdentifier`, required.
        :param resource_id: Analysis resource ID. 
        :type site_descriptionID: str or :class:`~obspy.core.event.resourceid.ResourceIdentifier`, required.
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
        :type spt_logs_count: Non-negative int, optional.
        :param spt_logs_count: Number of available SPT profile(s). 
        :type cpt_logs_count: Non-negative int, optional.
        :param cpt_logs_count: Number of available CPT profile(s). 
        :type borehole_logs_count: Non-negative int, optional.
        :param borehole_logs_count: Number of available borehole log profile(s). 
        """
        self.resource_id = resource_id        
        self.site_descriptionID = site_descriptionID
        self.creation_date = creation_date
        self.resonance_frequency = resonance_frequency
        self.velocity_s30 = velocity_s30
        self.spt_logs_count = spt_logs_count
        self.cpt_logs_count = cpt_logs_count
        self.borehole_logs_count = borehole_logs_count
        self.velocity_profile_survey = velocity_profile_survey
            
    def __str__(self):
        ret = ("Analysis:\n"
               "\tResource ID: {analysis_id},\n"
               "\tSite Description ID: {sd_id},\n"
               "\tCreation Date: {dt},\n"
               "\tResonance Frequency: {rfreq},\n"
               "\tVelocity S30: {vs30},\n"
               "\tSPT Logs count: {spt_logs_count},\n"
               "\tCPT Logs count: {cpt_logs_count},\n"
               "\tBorehole Logs count: {bh_logs_count} \n")
        ret = ret.format(
            analysis_id = self.resource_id, 
            sd_id = self.site_descriptionID,
            dt = self.creation_date,
            rfreq = self.resonance_frequency.value if self.resonance_frequency else "None",
            vs30 = self.velocity_s30.value if self.velocity_s30 else "None",
            spt_logs_count = self.spt_logs_count, 
            cpt_logs_count = self.cpt_logs_count, 
            bh_logs_count = self.borehole_logs_count)
        return ret

    @staticmethod
    def _coerce_non_negative_int(attr_name, value):
        """
        Return ``value`` as a non-negative integer or ``None``.

        :rtype: int or None
        """
        if value is None:
            return None
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise SiteXMLValidationError(
                f"Could not convert {value} to int: {exc}"
            )
        if coerced < 0:
            raise SiteXMLValidationError(
                f"{attr_name} must be non-negative"
            )
        return coerced

    @property
    def spt_logs_count(self):
        return self._spt_logs_count

    @spt_logs_count.setter
    def spt_logs_count(self, value):
        self._spt_logs_count = self._coerce_non_negative_int(
            "spt_logs_count", value)

    @property
    def cpt_logs_count(self):
        return self._cpt_logs_count

    @cpt_logs_count.setter
    def cpt_logs_count(self, value):
        self._cpt_logs_count = self._coerce_non_negative_int(
            "cpt_logs_count", value)

    @property
    def borehole_logs_count(self):
        return self._borehole_logs_count

    @borehole_logs_count.setter
    def borehole_logs_count(self, value):
        self._borehole_logs_count = self._coerce_non_negative_int(
            "borehole_logs_count", value)
     
class SERASite(BaseNode):
    """
    This is the parent class for the siteXML object tree.
    """
    resource_id = resource_id_property(
        "resource_id", allow_none=False, allow_empty=False)
    site_owner = wrapped_property("site_owner", SERASiteOwner)
    site_description = wrapped_property("site_description", SiteDescription)
    external_references = wrapped_list_property("external_references", ExternalReference)
    analysis = wrapped_list_property("analysis", Analysis)
    created = wrapped_property("created", obspy.UTCDateTime)
    
    def __init__(self, resource_id, site_owner, site_description, 
                 analysis=None, created=None, external_references=None):
        """
        :type resource_id: str or :class:`~obspy.core.event.resourceid.ResourceIdentifier`
        :param resource_id: SERA SiteXML Unique Identifier (siteID).
        :type site_owner: :class:`~obspy.core.io.sitexml.SERASiteOwner`, required.
        :param site_owner: The site owner metadata. 
        :type site_description: :class:`~obspy.core.io.sitexml.SiteDescription`, required.
        :param site_description: The site description parameters (H800, Bedrock depth, 
                            EC8 class, geological unit, morphology, topography).
        :type analysis: list of :class:`~obspy.io.sitexml.core.Analysis`, optional.
        :param analysis: The site characterization parameters 
                            (VS30, resonance frequency, velocity profiles).
        :type created: :class:`~obspy.UTCDateTime`
        :param created: Root-level SiteXML document creation time. This
            value is serialization metadata for the XML document itself, not
            the creation time of the underlying site metadata. When
            :func:`~obspy.io.sitexml.sitexml.write_sitexml` serializes a
            ``SERASite`` object, it replaces this value with the current write
            time and writes that timestamp to the root ``<creationTime>``
            element.
        :type external_references: List of :class:`~obspy.core.inventory.util.ExternalReference`, optional.
        :param external_references: Additional resources with site characterization metadata. 
        """
        self.resource_id = resource_id
        self.site_owner = site_owner
        self.site_description = site_description
        self.analysis = analysis
        self.created = created
        self.external_references = external_references

    def validate_references(self):
        """
        Validate internal SiteXML object-graph references for one site.

        :rtype: None
        """
        analysis_ids = set()
        velocity_profile_ids = set()
        site_description_id = self.site_description.resource_id

        for analysis in self.analysis or []:
            if analysis.site_descriptionID != site_description_id:
                raise SiteXMLValidationError(
                    "Analysis site_descriptionID does not match the parent "
                    "SiteDescription resource_id."
                )

            if analysis.resource_id in analysis_ids:
                raise SiteXMLValidationError(
                    f"Duplicate analysis resource_id: {analysis.resource_id}"
                )
            analysis_ids.add(analysis.resource_id)

            survey = analysis.velocity_profile_survey
            if survey is None or not survey.velocity_profiles:
                continue

            for velocity_profile in survey.velocity_profiles:
                if velocity_profile.resource_id in velocity_profile_ids:
                    raise SiteXMLValidationError(
                        "Duplicate velocity profile resource_id: "
                        f"{velocity_profile.resource_id}"
                    )
                velocity_profile_ids.add(velocity_profile.resource_id)

        preferred_analysis_id = self.site_description.preferred_site_analysisID
        if preferred_analysis_id is not None and \
                preferred_analysis_id not in analysis_ids:
            raise SiteXMLValidationError(
                "preferred_site_analysisID does not match any attached "
                "analysis resource_id."
            )

        preferred_velocity_profile_id = \
            self.site_description.preferred_velocity_profileID
        if preferred_velocity_profile_id is not None and \
                preferred_velocity_profile_id not in velocity_profile_ids:
            raise SiteXMLValidationError(
                "preferred_velocity_profileID does not match any attached "
                "velocity profile resource_id."
            )

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
