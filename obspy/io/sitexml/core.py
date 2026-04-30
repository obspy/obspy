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
import re
import math

import obspy
from obspy.core.event import ResourceIdentifier
from obspy.core.inventory.util import (Latitude, Longitude, Distance, 
                                       ExternalReference)
from .util import (BaseNode, SiteXMLValidationError,
                    TopographySchemaA, TopographySchemaB, EC8Class, 
                    ResonanceFrequencyMethod, VelocityS30Method,
                    Vs30MethodCombined, Vs30ManualIndex,
                    _pretty_str, _scalar_property, _resource_id_property,
                    _wrapped_property, _enum_property, _wrapped_list_property,
                    _enum_list_property, _split_station_code)
    
class ValueWithUncertainty(BaseNode):
    """
    Numeric SiteXML value with an optional uncertainty of the same type.

    SiteXML stores a single symmetric ``uncertainty`` value next to the main
    ``value``. ObsPy's
    :class:`~obspy.core.util.obspy_types.FloatWithUncertainties` stores lower
    and upper uncertainty values instead. This class intentionally keeps the
    SiteXML representation simple and schema-shaped, while the conversion
    helpers below make future ObsPy interoperability explicit.
    """

    def __init__(self, value, uncertainty=None, valid_type=float):
        """
        :param value: int or float, the main value.
        :param uncertainty: int, float, or None, representing uncertainty.
        :param valid_type: type, expected numeric type (e.g., float, int).
        """
        self.valid_type = valid_type
        self.value = value
        self.uncertainty = uncertainty

    @classmethod
    def from_float_with_uncertainties(cls, value, valid_type=float):
        """
        Convert an ObsPy ``FloatWithUncertainties`` to SiteXML form.

        SiteXML can only represent symmetric uncertainty. Values with different
        lower and upper uncertainties are rejected to avoid silent data loss.
        The ObsPy ``measurement_method`` metadata is not represented in
        SiteXML's value/uncertainty pair and is therefore intentionally
        ignored.

        .. rubric:: Example

        >>> from obspy.core.util.obspy_types import FloatWithUncertainties
        >>> value = FloatWithUncertainties(
        ...     18.2, lower_uncertainty=0.5, upper_uncertainty=0.5)
        >>> site_value = ValueWithUncertainty.from_float_with_uncertainties(
        ...     value)
        >>> site_value.value
        18.2
        >>> site_value.uncertainty
        0.5
        """
        lower = value.lower_uncertainty
        upper = value.upper_uncertainty
        if lower != upper:
            raise SiteXMLValidationError(
                "SiteXML value/uncertainty pairs only support symmetric "
                "uncertainty"
            )
        return cls(valid_type(value), lower, valid_type=valid_type)

    def to_float_with_uncertainties(self):
        """
        Convert this value to ObsPy's ``FloatWithUncertainties`` type.

        The SiteXML uncertainty, when present, is mapped to both lower and
        upper ObsPy uncertainties.

        .. rubric:: Example

        >>> site_value = ValueWithUncertainty(18.2, uncertainty=0.5)
        >>> obspy_value = site_value.to_float_with_uncertainties()
        >>> float(obspy_value)
        18.2
        >>> obspy_value.lower_uncertainty
        0.5
        >>> obspy_value.upper_uncertainty
        0.5
        """
        from obspy.core.util.obspy_types import FloatWithUncertainties

        return FloatWithUncertainties(
            float(self.value),
            lower_uncertainty=self.uncertainty,
            upper_uncertainty=self.uncertainty)

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

    title = _scalar_property("title", allow_none=False, allow_empty=False)
    first_author = _scalar_property(
        "first_author", allow_none=False, allow_empty=False)

    def __init__(self, title, first_author, secondary_authors=None,
                 year=None, booktitle=None, language=None, doi=None):
        """
        :type title: str, required
        :param title: Title of the publication.
        :type first_author: str, required
        :param first_author: Main author of the publication.
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

    literature_source = _wrapped_property("literature_source", LiteratureSource)
    external_references = _wrapped_list_property("external_references", ExternalReference)

    def __init__(self, name, value, methods=None, quality_index=None,
                 literature_source=None, external_references=None):
        """
        :type name: str, required
        :param name: Indicator type. One of: "siteClassEC8", "h800",
            "bedrockDepth", "geologicalUnit", "velocityS30",
            "resonanceFrequency", "velocityProfile".
        :type value: str or :class:`~obspy.io.sitexml.core.ValueWithUncertainty`
            or :class:`~obspy.io.sitexml.core.VelocityProfileData`, required
        :param value: Value of the indicator. Type depends on the indicator.
        :type methods: list of str, optional
        :param methods: Methods used for the estimation/calculation of the
            site indicator.
        :type quality_index: float, optional
        :param quality_index: Quality index of the site indicator. Takes
            values between 0 and 1. Calculated according to the guidelines of
            the SERA D7.2 Deliverable.
        :type literature_source:
            :class:`~obspy.io.sitexml.core.LiteratureSource`, optional
        :param literature_source: Literature source related to the provided
            site indicator value.
        :type external_references: list of
            :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param external_references: External URIs and descriptions for this
            indicator.
        """
        self.name = name
        self.value = value
        self.methods = methods or []
        self.quality_index = quality_index 
        self.literature_source = literature_source
        self.external_references = external_references

    @property
    def quality_index(self):
        return self._quality_index

    @quality_index.setter
    def quality_index(self, value):
        if value is None:
            self._quality_index = None
            return
        if isinstance(value, bool):
            raise SiteXMLValidationError(
                "quality_index must be a number between 0 and 1."
            )
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise SiteXMLValidationError(
                "quality_index must be a number between 0 and 1."
            ) from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SiteXMLValidationError(
                "quality_index must be a number between 0 and 1."
            )
        self._quality_index = value

    def calculate_quality_index1(
            self, method=None, evaluation=None, reliability=None,
            report=None, assign=True):
        """
        Calculate Q_Index1 for this site indicator.

        The input criteria are not stored in SiteXML. If ``assign`` is true,
        store the calculated value in ``self.quality_index``.

        See :func:`obspy.io.sitexml.quality_index.quality_index1` for the formula
        and accepted criterion values.

        :rtype: float
        """
        from .quality_index import quality_index1

        value = quality_index1(
            method=method,
            evaluation=evaluation,
            reliability=reliability,
            report=report)

        if assign:
            self.quality_index = value

        return value

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
            external_ref=_pretty_str(self.external_references) if self.external_references else "None")
        return ret

class EC8(SiteIndicator):
    """
    Eurocode 8 ground type indicator.
    """

    value = _enum_property("value", EC8Class)
    
    def __init__(self, value, quality_index=None, literature_source=None,
                 external_references=None):
        """
        :type value: Enum of type
            :data:`~obspy.io.sitexml.util.EC8Class`, required
        :param value: EC8 class
        :type quality_index: float, optional
        :param quality_index: Quality index of the site indicator. Takes
            values between 0 and 1. Calculated according to the guidelines of
            the SERA D7.2 Deliverable.
        :type literature_source:
            :class:`~obspy.io.sitexml.core.LiteratureSource`, optional
        :param literature_source: Literature source related to the provided
            site indicator value.
        :type external_references: list of
            :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param external_references: External URIs and descriptions for this
            indicator.
        """
        super(EC8, self).__init__(
                name="siteClassEC8", value=value, quality_index=quality_index, 
                literature_source=literature_source, external_references=external_references)

class H800(SiteIndicator):
    """
    Engineering bedrock depth indicator for Vs greater than 800 m/s.
    """

    value = _wrapped_property("value", ValueWithUncertainty)

    def __init__(self, value, quality_index=None, literature_source=None, 
                 external_references=None):
        """
        :type value:
            :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, required
        :param value: Engineering depth. Depth beyond which the shear-wave
            velocity Vs exceeds 800 m/s. Expecting Integer value.
        :type quality_index: float, optional
        :param quality_index: Quality index of the site indicator. Takes
            values between 0 and 1. Calculated according to the guidelines of
            the SERA D7.2 Deliverable.
        :type literature_source:
            :class:`~obspy.io.sitexml.core.LiteratureSource`, optional
        :param literature_source: Literature source related to the provided
            site indicator value.
        :type external_references: list of
            :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param external_references: External URIs and descriptions for this
            indicator.
        """
        super(H800, self).__init__(
                name="h800", value=value, 
                quality_index=quality_index, 
                literature_source=literature_source, 
                external_references=external_references)

class BedrockDepth(SiteIndicator):
    """
    Seismological bedrock depth indicator.
    """

    value = _wrapped_property("value", ValueWithUncertainty)

    def __init__(self, value, quality_index=None, literature_source=None, 
                 external_references=None):
        """
        :type value:
            :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, required
        :param value: Seismological bedrock depth. Expecting Integer values.
        :type quality_index:
            :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, optional
        :param quality_index: Quality index of the site indicator. Takes
            values between 0 and 1. Calculated according to the guidelines of
            the SERA D7.2 Deliverable.
        :type literature_source:
            :class:`~obspy.io.sitexml.core.LiteratureSource`, optional
        :param literature_source: Literature source related to the provided
            site indicator value.
        :type external_references: list of
            :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param external_references: External URIs and descriptions for this
            indicator.
        """
        super(BedrockDepth, self).__init__(
            name="bedrockDepth", value=value, 
            quality_index=quality_index, 
            literature_source=literature_source, 
            external_references=external_references)

class GeologicalUnit(SiteIndicator):
    """
    Surface geology indicator with optional map-scale metadata.
    """

    def __init__(self, value, quality_index=None, 
                geological_map_scale=None, geological_unit_OGE=None, 
                literature_source=None, external_references=None):
        """
        :type value: str, required
        :param value: Brief description of the surface geology (free text)
        :type quality_index: float, optional
        :param quality_index: Quality index of the site indicator. Takes
            values between 0 and 1. Calculated according to the guidelines of
            the SERA D7.2 Deliverable.
        :type geological_map_scale: str, optional
        :param geological_map_scale: Scale of geological map used for the
            description of surface geology.
        :type geological_unit_OGE: str, optional
        :param geological_unit_OGE: Description of the surface geology
            according to a Unified, Pan-European Map.
        :type literature_source:
            :class:`~obspy.io.sitexml.core.LiteratureSource`, optional
        :param literature_source: Literature source related to the provided
            site indicator value.
        :type external_references: list of
            :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param external_references: External URIs and descriptions for this
            indicator.
        """
        self.geological_map_scale = geological_map_scale
        self.geological_unit_OGE = geological_unit_OGE
        super(GeologicalUnit, self).__init__(
            name="geologicalUnit", value=value, 
                quality_index=quality_index, 
                literature_source=literature_source, 
                external_references=external_references)
        
class ResonanceFrequency(SiteIndicator):
    """
    Site resonance-frequency indicator.
    """

    value = _wrapped_property("value", ValueWithUncertainty)
    methods = _enum_list_property("methods", ResonanceFrequencyMethod)

    def __init__(self, value, quality_index=None, methods=None, 
                 literature_source=None, external_references=None):
        """
        :type value:
            :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, required
        :param value: Resonance Frequency (f0). Expecting float values.
        :type quality_index: float, optional
        :param quality_index: Quality index of the site indicator. Takes
            values between 0 and 1. Calculated according to the guidelines of
            the SERA D7.2 Deliverable.
        :type methods: List of Enum type
            :data:`~obspy.io.sitexml.util.ResonanceFrequencyMethod`,
            optional
        :param methods: Methods used for the estimation of ResonanceFrequency
        :type literature_source:
            :class:`~obspy.io.sitexml.core.LiteratureSource`, optional
        :param literature_source: Literature source related to the provided
            site indicator value.
        :type external_references: list of
            :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param external_references: External URIs and descriptions for this
            indicator.
        """
        super(ResonanceFrequency, self).__init__(
            name="resonanceFrequency", value=value, methods=methods, 
            quality_index=quality_index, 
            literature_source=literature_source, 
            external_references=external_references)
        
class VelocityS30(SiteIndicator):
    """
    Time-averaged shear-wave velocity over the upper 30 meters.
    """

    value = _wrapped_property("value", ValueWithUncertainty)
    methods = _enum_list_property("methods", VelocityS30Method)
    method_combined_qindex = _enum_property("velocityS30MethodCombIndex", Vs30MethodCombined)
    manual_qindex = _enum_property("velocityS30ManualIndex", Vs30ManualIndex)
    
    def __init__(self, value, quality_index=None, methods=None,
                 method_combined_qindex=None, manual_qindex=None, 
                 literature_source=None, external_references=None):
        """
        :type value:
            :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, required
        :param value: Velocity S30. Expecting float values.
        :type quality_index: float, optional
        :param quality_index: Quality index of the site indicator. Takes
            values between 0 and 1. Calculated according to the guidelines of
            the SERA D7.2 Deliverable.
        :type methods: List of Enum type
            :data:`~obspy.io.sitexml.util.VelocityS30Method`, optional
        :param methods: Methods used for the estimation of Velocity S30
        :type method_combined_qindex: Enum of type
            :data:`~obspy.io.sitexml.util.Vs30MethodCombined`,
            optional
        :param method_combined_qindex: Whether a combination of two or more
            methods has been applied to estimate the Vs30 value.
        :type manual_qindex: Enum of type
            :data:`~obspy.io.sitexml.util.Vs30ManualIndex`, optional
        :param manual_qindex: Overall qualitative factor on the knowledge of
            the maximum depth of Vs measurements.
        :type literature_source:
            :class:`~obspy.io.sitexml.core.LiteratureSource`, optional
        :param literature_source: Literature source related to the provided
            site indicator value.
        :type external_references: list of
            :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param external_references: External URIs and descriptions for this
            indicator.
        """

        self.method_combined_qindex = method_combined_qindex
        self.manual_qindex = manual_qindex
        super(VelocityS30, self).__init__(
            name="velocityS30", 
            value=value, 
            quality_index=quality_index, 
            methods=methods, 
            literature_source=literature_source, 
            external_references=external_references)

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
                 literature_source=None, external_references=None):
        """
        :type velocity_profiles: list of
            :class:`~obspy.io.sitexml.core.VelocityProfile`, optional
        :param velocity_profiles: List of Velocity Profiles.
        :type quality_index: float, optional
        :param quality_index: Quality index of the site indicator. Takes
            values between 0 and 1. Calculated according to the guidelines of
            the SERA D7.2 Deliverable.
        :type literature_source:
            :class:`~obspy.io.sitexml.core.LiteratureSource`, optional
        :param literature_source: Literature source related to the provided
            site indicator value.
        :type external_references: list of
            :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param external_references: External URIs and descriptions for this
            indicator.
        """
        self.velocity_profiles = velocity_profiles  # triggers setter/validation
        super(VelocityProfileSurvey, self).__init__(
            name="velocityProfile", 
            value=self.velocity_profiles,
            quality_index=quality_index,
            literature_source=literature_source,
            external_references=external_references)

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

    resource_id = _resource_id_property(
        "resource_id", allow_none=False, allow_empty=False)

    def __init__(self, resource_id, velocity_profile_data, layer_count=None):
        """
        :type resource_id: str or
            :class:`~obspy.core.event.resourceid.ResourceIdentifier`, required
        :param resource_id: Unique Velocity Profile Resource ID.
        :type velocity_profile_data:
            :class:`~obspy.io.sitexml.core.VelocityProfileData`, required
        :param velocity_profile_data: An array of velocity profile data for all
            layers. Must contain at least one layer.
        :type layer_count: int, optional
        :param layer_count: Non-negative int. Number of layers in velocity profile. 
            If omitted, it is derived from ``velocity_profile_data``.
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

    top_depth = _wrapped_property("top_depth", ValueWithUncertainty,
                                 allow_none=False)
    bottom_depth = _wrapped_property("bottom_depth", ValueWithUncertainty)
    density = _wrapped_property("density", ValueWithUncertainty)
    velocityP = _wrapped_property("velocityP", ValueWithUncertainty)
    velocityS = _wrapped_property("velocityS", ValueWithUncertainty)

    # Need to decide which if these arguments besides top_depth will be
    # mandatory
    def __init__(self, top_depth, bottom_depth=None, density=None, 
                velocityP=None, velocityS=None):
        """
        :type top_depth:
            :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, required
        :param top_depth: Layer top depth.
        :type bottom_depth:
            :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, optional
        :param bottom_depth: Layer bottom depth.
        :type density:
            :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, optional
        :param density: Layer density.
        :type velocityP:
            :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, optional
        :param velocityP: Layer velocityP value.
        :type velocityS:
            :class:`~obspy.io.sitexml.core.ValueWithUncertainty`, optional
        :param velocityS: Layer velocityS value.
        """
        self.top_depth = top_depth
        self.bottom_depth = bottom_depth
        self.density = density 
        self.velocityP = velocityP 
        self.velocityS = velocityS 
        
class SERASiteOwner(BaseNode):
    """
    Site owner and required contact-person metadata.

    ObsPy's :class:`~obspy.core.inventory.util.Person` and
    :class:`~obspy.core.inventory.util.Operator` classes are close, but not
    identical, representations. SiteXML stores one required owner and one
    required contact person split into first name, last name, and email. ObsPy
    stores person names, agencies, and emails as lists, and operators can hold
    multiple contacts. The conversion helpers below keep those policy choices
    explicit.
    """

    owner_codename = _scalar_property(
        "owner_codename", allow_none=False, allow_empty=False)
    owner_fullname = _scalar_property(
        "owner_fullname", allow_none=False, allow_empty=False)
    person_firstname = _scalar_property(
        "person_firstname", allow_none=False, allow_empty=False)
    person_lastname = _scalar_property(
        "person_lastname", allow_none=False, allow_empty=False)
    person_mbox = _scalar_property(
        "person_mbox", allow_none=False, allow_empty=False)
    ownerID = _resource_id_property("ownerID")
    personID = _resource_id_property("personID")
    institutionID = _resource_id_property("institutionID")

    def __init__(self, owner_codename, owner_fullname,
                 person_firstname, person_lastname, person_mbox, ownerID=None,
                 person_homepage=None, personID=None,
                 institution_name=None, institution_mbox=None, 
                 institution_phone=None, institution_homepage=None, institutionID=None,
                 address_street=None, address_locality=None, address_postal_code=None, 
                 address_country=None, address_country_code=None,
                 affiliation_department=None, affiliation_function=None):
        """
        :type owner_codename: str, required
        :param owner_codename: Short code name for the site owner.
        :type owner_fullname: str, required
        :param owner_fullname: Full name of the site owner.
        :type person_firstname: str, required
        :param person_firstname: First name of the contact person.
        :type person_lastname: str, required
        :param person_lastname: Last name of the contact person.
        :type person_mbox: str, required
        :param person_mbox: Email address of the contact person.
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
        :param affiliation_function: Function or position of the contact
            person.
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

    @staticmethod
    def _split_obspy_person_name(name):
        """
        Split an ObsPy full-name string into SiteXML first/last names.
        """
        parts = str(name).strip().split(None, 1)
        if len(parts) != 2:
            raise SiteXMLValidationError(
                "Cannot derive SiteXML person_firstname and person_lastname "
                "from an ObsPy Person name with fewer than two words"
            )
        return parts

    @classmethod
    def from_person(cls, person, owner_codename, owner_fullname, ownerID=None,
                    person_firstname=None, person_lastname=None,
                    person_mbox=None, person_homepage=None, personID=None,
                    institution_name=None, institution_mbox=None,
                    institution_phone=None, institution_homepage=None,
                    institutionID=None, address_street=None,
                    address_locality=None, address_postal_code=None,
                    address_country=None, address_country_code=None,
                    affiliation_department=None, affiliation_function=None):
        """
        Convert an ObsPy :class:`~obspy.core.inventory.util.Person` 
        to a SiteXML owner contact.

        ``owner_codename`` and ``owner_fullname`` are required because ObsPy
        :class:`~obspy.core.inventory.util.Person` only represents the contact person, 
        not the SiteXML owner identity. If ``person_firstname`` and ``person_lastname`` 
        are omitted, they are derived from the first ObsPy person name. If
        ``person_mbox`` is omitted, the first ObsPy email address is used. If
        ``institution_name`` is omitted, the first ObsPy agency is used.

        :rtype: :class:`~obspy.io.sitexml.core.SERASiteOwner`
        
        .. rubric:: Example

        >>> from obspy.core.inventory.util import Person
        >>> person = Person(
        ...     names=["Name Surname"],
        ...     agencies=["INSTITUTION_ABBR"],
        ...     emails=["someemail@domain.ab"])
        >>> site_owner = SERASiteOwner.from_person(
        ...     person,
        ...     owner_codename="SITEOWNER",
        ...     owner_fullname="Site Owner Full Name")
        >>> site_owner.person_firstname
        'Name'
        >>> site_owner.person_lastname
        'Surname'
        >>> site_owner.institution_name
        'INSTITUTION_ABBR'
        """
        names = list(person.names)
        emails = list(person.emails)
        agencies = list(person.agencies)

        if person_firstname is None or person_lastname is None:
            if not names:
                raise SiteXMLValidationError(
                    "Cannot derive SiteXML contact person from an ObsPy "
                    "Person without names"
                )
            derived_firstname, derived_lastname = (
                cls._split_obspy_person_name(names[0]))
            person_firstname = person_firstname or derived_firstname
            person_lastname = person_lastname or derived_lastname

        if person_mbox is None:
            if not emails:
                raise SiteXMLValidationError(
                    "Cannot derive SiteXML person_mbox from an ObsPy Person "
                    "without emails"
                )
            person_mbox = emails[0]

        if institution_name is None and agencies:
            institution_name = agencies[0]

        return cls(
            owner_codename=owner_codename,
            owner_fullname=owner_fullname,
            ownerID=ownerID,
            person_firstname=person_firstname,
            person_lastname=person_lastname,
            person_mbox=person_mbox,
            person_homepage=person_homepage,
            personID=personID,
            institution_name=institution_name,
            institution_mbox=institution_mbox,
            institution_phone=institution_phone,
            institution_homepage=institution_homepage,
            institutionID=institutionID,
            address_street=address_street,
            address_locality=address_locality,
            address_postal_code=address_postal_code,
            address_country=address_country,
            address_country_code=address_country_code,
            affiliation_department=affiliation_department,
            affiliation_function=affiliation_function)

    @classmethod
    def from_operator(cls, operator, owner_codename=None, owner_fullname=None,
                      contact_index=None, **kwargs):
        """
        Convert an ObsPy :class:`~obspy.core.inventory.util.Operator` 
        to a SiteXML owner contact.

        ``operator.agency`` is used as ``owner_fullname`` when no explicit
        value is provided. ``owner_codename`` defaults to ``operator.agency``
        when omitted, though callers should pass a short code when they have
        one. Operators with multiple contacts are rejected unless
        ``contact_index`` selects which contact to convert, because SiteXML has
        a single contact person in :class:`~obspy.io.sitexml.core.SERASiteOwner`. 
        Extra keyword arguments are forwarded to :meth:`from_person`.

        :rtype: :class:`~obspy.io.sitexml.core.SERASiteOwner`

        .. rubric:: Example

        >>> from obspy.core.inventory.util import Operator, Person
        >>> person = Person(
        ...     names=["Name Surname"],
        ...     emails=["someemail@domain.ab"])
        >>> operator = Operator(
        ...     agency="Site Owner Full Name",
        ...     contacts=[person],
        ...     website="https://www.domain.ab")
        >>> site_owner = SERASiteOwner.from_operator(
        ...     operator,
        ...     owner_codename="SITEOWNER")
        >>> site_owner.owner_fullname
        'Site Owner Full Name'
        >>> site_owner.institution_homepage
        'https://www.domain.ab'
        """
        contacts = list(operator.contacts)
        if not contacts:
            raise SiteXMLValidationError(
                "Cannot derive SiteXML owner contact from an ObsPy Operator "
                "without contacts"
            )
        if contact_index is None:
            if len(contacts) != 1:
                raise SiteXMLValidationError(
                    "Cannot derive one SiteXML owner contact from an ObsPy "
                    "Operator with multiple contacts; pass contact_index"
                )
            contact_index = 0

        if owner_fullname is None:
            owner_fullname = operator.agency
        if owner_codename is None:
            owner_codename = operator.agency
        kwargs.setdefault("institution_homepage", operator.website)

        return cls.from_person(
            contacts[contact_index],
            owner_codename=owner_codename,
            owner_fullname=owner_fullname,
            **kwargs)

    def to_person(self):
        """
        Convert this SiteXML owner contact to ObsPy's 
        :class:`~obspy.core.inventory.util.Person` type.

        The SiteXML first and last names are joined into one ObsPy name. The
        SiteXML institution name is mapped to the first ObsPy agency, when
        present, and ``person_mbox`` is mapped to the first ObsPy email.
        SiteXML person homepage and public IDs are not represented by ObsPy
        :class:`~obspy.core.inventory.util.Person`.

        :rtype: :class:`~obspy.core.inventory.util.Person`

        .. rubric:: Example

        >>> site_owner = SERASiteOwner(
        ...     owner_codename="SITEOWNER",
        ...     owner_fullname="Site Owner Full Name",
        ...     person_firstname="Name",
        ...     person_lastname="Surname",
        ...     person_mbox="someemail@domain.ab",
        ...     institution_name="INSTITUTION_ABBR")
        >>> person = site_owner.to_person()
        >>> person.names
        ['Name Surname']
        >>> person.agencies
        ['INSTITUTION_ABBR']
        >>> person.emails
        ['someemail@domain.ab']
        """
        from obspy.core.inventory.util import Person

        agencies = (
            [self.institution_name] if self.institution_name is not None
            else [])
        return Person(
            names=[f"{self.person_firstname} {self.person_lastname}"],
            agencies=agencies,
            emails=[self.person_mbox])

    def to_operator(self):
        """
        Convert this SiteXML owner contact to ObsPy's 
        :class:`~obspy.core.inventory.util.Operator` type.

        ``owner_fullname`` is mapped to ``operator.agency``, the converted
        contact person becomes the only operator contact, and
        ``institution_homepage`` is mapped to ``operator.website``.

        :rtype: :class:`~obspy.core.inventory.util.Operator`

        .. rubric:: Example

        >>> site_owner = SERASiteOwner(
        ...     owner_codename="SITEOWNER",
        ...     owner_fullname="Site Owner Full Name",
        ...     person_firstname="Name",
        ...     person_lastname="Surname",
        ...     person_mbox="someemail@domain.ab",
        ...     institution_homepage="https://www.domain.ab")
        >>> operator = site_owner.to_operator()
        >>> operator.agency
        'Site Owner Full Name'
        >>> operator.website
        'https://www.domain.ab'
        >>> operator.contacts[0].names
        ['Name Surname']
        """
        from obspy.core.inventory.util import Operator

        return Operator(
            agency=self.owner_fullname,
            contacts=[self.to_person()],
            website=self.institution_homepage)

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

    resource_id = _resource_id_property(
        "resource_id", allow_none=False, allow_empty=False)
    preferred_site_analysisID = _resource_id_property(
        "preferred_site_analysisID")
    preferred_velocity_profileID = _resource_id_property(
        "preferred_velocity_profileID")
    latitude = _wrapped_property("latitude", Latitude, allow_none=False)
    longitude = _wrapped_property("longitude", Longitude, allow_none=False)
    altitude = _wrapped_property("altitude", Distance)
    min_distance_from_station = _wrapped_property("min_distance_from_station", Distance)
    max_distance_from_station = _wrapped_property("max_distance_from_station", Distance)
    bedrock_depth = _wrapped_property("bedrock_depth", BedrockDepth)
    h800 = _wrapped_property("h800", H800)
    ec8 = _wrapped_property("ec8", EC8)
    geological_unit = _wrapped_property("geological_unit", GeologicalUnit)
    topographyA = _enum_property("topographyA", TopographySchemaA)
    topographyB = _enum_property("topographyB", TopographySchemaB)

    def __init__(self, resource_id, latitude, longitude, altitude=None,
                 min_distance_from_station=None, max_distance_from_station=None, 
                 station_code=None, 
                 ec8=None, bedrock_depth=None, h800=None, geological_unit=None, 
                 morphology=None, topographyA=None, topographyB=None, 
                 preferred_site_analysisID=None, preferred_velocity_profileID=None,
                 overall_quality_index=None):
        """
        :type resource_id: str or
            :class:`~obspy.core.event.resourceid.ResourceIdentifier`, required
        :param resource_id: Unique Site Description Resource ID
        :type latitude: :class:`~obspy.core.inventory.util.Latitude`, required
        :param latitude: The latitude of the site.
        :type longitude:
            :class:`~obspy.core.inventory.util.Longitude`, required
        :param longitude: The longitude of the site. 
        :type altitude: :class:`~obspy.core.inventory.util.Distance`, optional
        :param altitude: Elevation of ground with respect to sea level (m).
        :type min_distance_from_station:
            :class:`~obspy.core.inventory.util.Distance`, optional
        :param min_distance_from_station: Minimum distance between the
            permanent seismological station and site characterization
            measurement. Should be used only when representative latitude and
            longitude of site characterization measurements cannot be provided.
        :type max_distance_from_station:
            :class:`~obspy.core.inventory.util.Distance`, optional
        :param max_distance_from_station: Maximum distance between the
            permanent seismological station and site characterization
            measurement. Should be used only when representative latitude and
            longitude of site characterization measurements cannot be provided.
        :type station_code: str, optional
        :param station_code: FDSN network and station code in
            network.station notation (if any).
        :type ec8: :class:`~obspy.io.sitexml.core.EC8`, optional
        :param ec8: Ground type according to Eurocode 8, based on the
            velocity S30 value and geotechnical description.
        :type h800: :class:`~obspy.io.sitexml.core.H800`, optional
        :param h800: Engineering depth. Depth beyond which the shear-wave
            velocity Vs exceeds 800 m/s.
        :type bedrock_depth:
            :class:`~obspy.io.sitexml.core.BedrockDepth`, optional
        :param bedrock_depth: Seismological bedrock depth. 
        :type geological_unit:
            :class:`~obspy.io.sitexml.core.GeologicalUnit`, optional
        :param geological_unit: Brief description of the surface geology
            (free text).
        :type morphology: str, optional
        :param morphology: Qualitative description of the shape of the
            earth's surface (free text).
        :type topographyA: Enum of type :data:`~obspy.io.sitexml.util.TopographySchemaA`, optional
        :param topographyA: Quantitative description of the surface according
            to the Italian Code (detailed description of the scheme in SERA
            Deliverable D7.1 - Appendix I).
        :type topographyB: Enum of type :data:`~obspy.io.sitexml.util.TopographySchemaB`, optional
        :param topographyB: Quantitative description of the shape of the
            earth's surface according to Burjanek et al, 2014 (detailed
            description of the scheme in SERA Deliverable D7.1 - Appendix I).
        :type preferred_site_analysisID: str or
            :class:`~obspy.core.event.resourceid.ResourceIdentifier`, optional
        :param preferred_site_analysisID: Preferred Site Analysis ID. 
                If you provide one or more analysis for this site 
                you should use this field to designate the prefered analysis.
        :type preferred_velocity_profileID: str or
            :class:`~obspy.core.event.resourceid.ResourceIdentifier`, optional
        :param preferred_velocity_profileID: Preferred Velocity Profile ID. 
                If you provide one or more velocity profiles for this site 
                you should use this field to designate the prefered VP. If
                ``preferred_site_analysisID`` is also provided, the preferred
                VP must belong to the preferred analysis. The overall quality
                index calculation uses the Velocity Profile Survey quality
                index associated with the preferred analysis.
        :type overall_quality_index: float, optional
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

    @property
    def station_code(self):
        return self._station_code

    @station_code.setter
    def station_code(self, value):
        if value is None:
            self._station_code = None
            return
        _split_station_code(value)
        self._station_code = value


class Analysis(BaseNode):
    """
    Site-characterization analysis and related indicator metadata.
    """

    resource_id = _resource_id_property(
        "resource_id", allow_none=False, allow_empty=False)
    site_descriptionID = _resource_id_property(
        "site_descriptionID", allow_none=False, allow_empty=False)
    resonance_frequency = _wrapped_property("resonance_frequency", ResonanceFrequency)
    velocity_s30 = _wrapped_property("velocity_s30", VelocityS30)
    velocity_profile_survey = _wrapped_property("velocity_profile_survey", VelocityProfileSurvey)

    def __init__(self, resource_id, site_descriptionID, creation_date=None,
                 resonance_frequency=None, velocity_s30=None, 
                 velocity_profile_survey=None, spt_logs_count=None,
                 cpt_logs_count=None, borehole_logs_count=None):
        """
        :type resource_id: str or
            :class:`~obspy.core.event.resourceid.ResourceIdentifier`, required
        :param resource_id: Analysis resource ID. 
        :type site_descriptionID: str or
            :class:`~obspy.core.event.resourceid.ResourceIdentifier`, required
        :param site_descriptionID: The Site Description object this analysis
            refers to.
        :type creation_date: datetime, optional
        :param creation_date: Date that this analysis was published.
        :type resonance_frequency:
            :class:`~obspy.io.sitexml.core.ResonanceFrequency`, optional
        :param resonance_frequency: The Resonance frequency of the soil
            column.
        :type velocity_s30:
            :class:`~obspy.io.sitexml.core.VelocityS30`, optional
        :param velocity_s30: Average shear-wave velocity between 0 and 30
            meters depth.
        :type velocity_profile_survey:
            :class:`~obspy.io.sitexml.core.VelocityProfileSurvey`, optional
        :param velocity_profile_survey: Velocity Profile Survey.
            Parent object for Velocity Profiles. 
        :type spt_logs_count: int, optional
        :param spt_logs_count: Non-negative. Number of available SPT profile(s). 
        :type cpt_logs_count: int, optional
        :param cpt_logs_count: Non-negative. Number of available CPT profile(s). 
        :type borehole_logs_count: int, optional
        :param borehole_logs_count: Non-negative. Number of available borehole log
            profile(s).
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
    resource_id = _resource_id_property(
        "resource_id", allow_none=False, allow_empty=False)
    site_owner = _wrapped_property("site_owner", SERASiteOwner)
    site_description = _wrapped_property("site_description", SiteDescription)
    external_references = _wrapped_list_property("external_references", ExternalReference)
    analysis = _wrapped_list_property("analysis", Analysis)
    created = _wrapped_property("created", obspy.UTCDateTime)
    
    def __init__(self, resource_id, site_owner, site_description, 
                 analysis=None, created=None, external_references=None):
        """
        :type resource_id: str or
            :class:`~obspy.core.event.resourceid.ResourceIdentifier`, required
        :param resource_id: SERA SiteXML Unique Identifier (siteID).
        :type site_owner:
            :class:`~obspy.io.sitexml.core.SERASiteOwner`, required
        :param site_owner: The site owner metadata. 
        :type site_description:
            :class:`~obspy.io.sitexml.core.SiteDescription`, required
        :param site_description: The site description parameters (H800,
            Bedrock depth, EC8 class, geological unit, morphology,
            topography).
        :type analysis: list of
            :class:`~obspy.io.sitexml.core.Analysis`, optional
        :param analysis: The site characterization parameters 
                            (VS30, resonance frequency, velocity profiles).
        :type created: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param created: Root-level SiteXML document creation time. This
            value is serialization metadata for the XML document itself, not
            the creation time of the underlying site metadata. When
            :func:`~obspy.io.sitexml.sitexml.write_sitexml` serializes a
            ``SERASite`` object, it replaces this value with the current write
            time and writes that timestamp to the root ``<creationTime>``
            element.
        :type external_references: List of
            :class:`~obspy.core.inventory.util.ExternalReference`, optional
        :param external_references: Additional resources with site
            characterization metadata.
        """
        self.resource_id = resource_id
        self.site_owner = site_owner
        self.site_description = site_description
        self.analysis = analysis
        self.created = created
        self.external_references = external_references

    def get_analysis(self, resource_id):
        """
        Return the attached analysis with matching resource ID, if present.

        :type resource_id: str or
            :class:`~obspy.core.event.resourceid.ResourceIdentifier`, required
        :param resource_id: Analysis resource ID to look up.
        :rtype: :class:`~obspy.io.sitexml.core.Analysis` or None
        """
        if isinstance(resource_id, ResourceIdentifier):
            resource_id = resource_id.id

        for analysis in self.analysis or []:
            if analysis.resource_id == resource_id:
                return analysis
        return None

    def get_preferred_analysis(self):
        """
        Return the preferred analysis, falling back to the first analysis.

        If ``site_description.preferred_site_analysisID`` is set, the matching
        attached analysis is returned. If no preferred analysis is declared,
        the first attached analysis is returned. If there are no analyses, or
        the preferred ID cannot be found, ``None`` is returned.

        :rtype: :class:`~obspy.io.sitexml.core.Analysis` or None
        """
        if not self.analysis:
            return None

        preferred_id = self.site_description.preferred_site_analysisID
        if preferred_id is not None:
            return self.get_analysis(preferred_id)

        return self.analysis[0]

    def get_velocity_profile(self, resource_id, analysis=None):
        """
        Return the attached velocity profile with matching resource ID.

        If ``analysis`` is provided, only that analysis is searched.
        Otherwise all attached analyses are searched.

        :type resource_id: str or
            :class:`~obspy.core.event.resourceid.ResourceIdentifier`, required
        :param resource_id: Velocity profile resource ID to look up.
        :type analysis: :class:`~obspy.io.sitexml.core.Analysis`, optional
        :param analysis: Analysis whose velocity-profile survey should be
            searched.
        :rtype: :class:`~obspy.io.sitexml.core.VelocityProfile` or None
        """
        if isinstance(resource_id, ResourceIdentifier):
            resource_id = resource_id.id

        analyses = [analysis] if analysis is not None else self.analysis or []
        for item in analyses:
            survey = item.velocity_profile_survey
            if survey is None or not survey.velocity_profiles:
                continue
            for velocity_profile in survey.velocity_profiles:
                if velocity_profile.resource_id == resource_id:
                    return velocity_profile
        return None

    def get_indicator_object(self, name):
        """
        Return a site indicator object by SiteXML indicator name.

        Site-description indicators are read from ``site_description``.
        Analysis indicators are read from the preferred analysis, falling back
        to the first attached analysis when no preferred analysis is declared.

        Supported names are ``siteClassEC8``, ``bedrockDepth``, ``h800``,
        ``geologicalUnit``, ``resonanceFrequency``, ``velocityS30``, and
        ``velocityProfile``.

        :type name: str
        :param name: SiteXML indicator name.
        :rtype: :class:`~obspy.io.sitexml.core.SiteIndicator` or None
        """
        site_description_indicators = {
            "siteClassEC8": self.site_description.ec8,
            "bedrockDepth": self.site_description.bedrock_depth,
            "h800": self.site_description.h800,
            "geologicalUnit": self.site_description.geological_unit,
        }
        if name in site_description_indicators:
            return site_description_indicators[name]

        analysis = self.get_preferred_analysis()
        if analysis is None:
            return None

        analysis_indicators = {
            "resonanceFrequency": analysis.resonance_frequency,
            "velocityS30": analysis.velocity_s30,
            "velocityProfile": analysis.velocity_profile_survey,
        }
        if name in analysis_indicators:
            return analysis_indicators[name]

        raise SiteXMLValidationError(f"Unknown site indicator name: {name}")

    def calculate_quality_index2(self):
        """
        Calculate Q_Index2 for this site.

        The calculation uses Q_Index1 values already stored on this site's
        indicators. Missing indicator quality indexes contribute zero.

        See :func:`obspy.io.sitexml.quality_index.quality_index2` for the formula
        and indicator weights.

        :rtype: float
        """
        from .quality_index import quality_index2

        return quality_index2(self)

    def calculate_quality_index3(
            self, f0_vs30=None, f0_bedrock_depth=None, f0_h800=None,
            vs30_h800=None, vs30_geology=None):
        """
        Calculate Q_Index3 from externally assessed consistency values.

        These consistency inputs are not stored in SiteXML. The denominator is
        the number of provided consistency pairs. Each consistency value is
        binary: ``0`` for no consistency and ``1`` for consistency. ``None``
        means unavailable or not evaluated.

        See :func:`obspy.io.sitexml.quality_index.quality_index3` for the formula
        and consistency-pair definitions.

        :rtype: float or None
        """
        from .quality_index import quality_index3

        return quality_index3(
            f0_vs30=f0_vs30,
            f0_bedrock_depth=f0_bedrock_depth,
            f0_h800=f0_h800,
            vs30_h800=vs30_h800,
            vs30_geology=vs30_geology)

    def calculate_overall_quality_index(
            self, f0_vs30=None, f0_bedrock_depth=None, f0_h800=None,
            vs30_h800=None, vs30_geology=None, assign=True):
        """
        Calculate the overall quality index for this site.

        Q_Index2 is calculated from the site's stored indicator quality
        indexes. Q_Index3 is calculated from the provided consistency values.
        If no Q_Index3 consistency values are provided, Q_Index3 is treated as
        zero for the overall quality-index formula.

        If ``assign`` is true, store the final value in
        ``self.site_description.overall_quality_index``.

        See :func:`obspy.io.sitexml.quality_index.overall_quality_index` for the
        standalone formula helper.

        :rtype: float
        """
        from .quality_index import overall_quality_index

        q2 = self.calculate_quality_index2()
        q3 = self.calculate_quality_index3(
            f0_vs30=f0_vs30,
            f0_bedrock_depth=f0_bedrock_depth,
            f0_h800=f0_h800,
            vs30_h800=vs30_h800,
            vs30_geology=vs30_geology)
        value = overall_quality_index(q2, q3)

        if assign:
            self.site_description.overall_quality_index = value

        return value

    def get_sitexml_filename(self, creation_time=None):
        """
        Return the default SiteXML filename for this site.

        When ``creation_time`` is provided, return the official SiteXML
        filename containing the serialization date. Station-backed sites use
        the associated FDSN station code in ``network.station`` notation.
        Other sites use this site's resource ID. Without ``creation_time``,
        the legacy identity-only filename is returned.

        :rtype: str
        """
        station_code = self.site_description.station_code
        if station_code:
            filename = station_code
        else:
            filename = re.sub(
                r"[^A-Za-z0-9]+", "_", str(self.resource_id)
            ).strip("_")

        if creation_time is not None:
            creation_time = obspy.UTCDateTime(creation_time)
            date_text = creation_time.strftime("%d-%m-%Y")
            filename = "Site_%s_%s" % (filename, date_text)

        return filename + ".xml"

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

        if preferred_analysis_id is not None and \
                preferred_velocity_profile_id is not None:
            preferred_analysis = self.get_preferred_analysis()
            if self.get_velocity_profile(
                    preferred_velocity_profile_id,
                    analysis=preferred_analysis) is None:
                raise SiteXMLValidationError(
                    "preferred_velocity_profileID does not belong to the "
                    "preferred_site_analysisID."
                )

    def __str__(self):
        output=["\n#################\n"]
        if self.site_description.station_code:
            title = "Site Metadata (station: " + self.site_description.station_code + ")"
        else:
            title = "Site Metadata"
        output.append(title)
        output.append("\n#################\n")
       
        output.append("resource_id: " + self.resource_id + "\n")
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
