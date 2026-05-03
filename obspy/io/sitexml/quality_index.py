# -*- coding: utf-8 -*-
"""
Quality-index formula helpers and sidecar import utilities for SiteXML.

:copyright:
    ORFEUS, 2025
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import warnings

import pandas as pd

from .util import SiteXMLIOError, SiteXMLImportError


_QUALITY_INDEX2_WEIGHTS = {
    "resonanceFrequency": 1,
    "velocityProfile": 1,
    "velocityS30": 0.5,
    "bedrockDepth": 0.5,
    "h800": 0.5,
    "geologicalUnit": 0.5,
    "siteClassEC8": 0.25,
}

_QUALITY_INDEX_INDICATORS = (
    "siteClassEC8",
    "bedrockDepth",
    "h800",
    "geologicalUnit",
    "resonanceFrequency",
    "velocityS30",
    "velocityProfile",
)

_QUALITY_INDEX1_CRITERIA = (
    "method",
    "evaluation",
    "reliability",
    "report",
)

_QUALITY_INDEX3_COLUMNS = (
    "f0_vs30",
    "f0_bedrock_depth",
    "f0_h800",
    "vs30_h800",
    "vs30_geology",
)


def _quality_index_or_zero(indicator):
    if indicator is None or indicator.quality_index is None:
        return 0
    return indicator.quality_index


def quality_index1(method=None, evaluation=None, reliability=None,
                   report=None):
    """
    Standalone formula helper for Q_Index1.

    .. note::
        For object-oriented SiteXML workflows, prefer
        :meth:`~obspy.io.sitexml.core.SiteIndicator.calculate_quality_index1`,
        which can also store the result on the indicator object.

    This function calculates the Quality Index #1 according to SERA
    Deliverable 7.2.
    It varies from 0 to 1 and refers to a single mandatory indicator.

    Four criteria are used for the calculation:

    +-----------+-----------------------------------------+
    | Criterion | Meaning                                 |
    +===========+=========================================+
    | A         | Method of acquisition and analysis      |
    +-----------+-----------------------------------------+
    | B         | Estimation of the indicator             |
    +-----------+-----------------------------------------+
    | C         | Reliability of the value                |
    +-----------+-----------------------------------------+
    | D         | Report documenting the indicator value  |
    +-----------+-----------------------------------------+

    Accepted values are:

    +-------------------+------------------------------+---------+
    | Parameter         | Accepted value               | Score   |
    +===================+==============================+=========+
    | ``method``        | ``"documented"`` or ``1``    | A = 1   |
    +-------------------+------------------------------+---------+
    | ``method``        | Any other value              | A = 0   |
    +-------------------+------------------------------+---------+
    | ``evaluation``    | ``"direct"`` or ``2``        | B = 2   |
    +-------------------+------------------------------+---------+
    | ``evaluation``    | Any other value              | B = 0   |
    +-------------------+------------------------------+---------+
    | ``reliability``   | ``"yes"`` or ``1``           | C = 1   |
    +-------------------+------------------------------+---------+
    | ``reliability``   | ``"partial"`` or ``0.5``     | C = 0.5 |
    +-------------------+------------------------------+---------+
    | ``reliability``   | Any other value              | C = 0   |
    +-------------------+------------------------------+---------+
    | ``report``        | ``"yes"`` or ``1``           | D = 1   |
    +-------------------+------------------------------+---------+
    | ``report``        | ``"partial"`` or ``0.5``     | D = 0.5 |
    +-------------------+------------------------------+---------+
    | ``report``        | Any other value              | D = 0   |
    +-------------------+------------------------------+---------+

    ``None`` is treated as "any other value" for all parameters.

    The Quality Index #1 is then calculated using the following formula::

    >>> Q_Index1 = ((A + B + C) * D) / (Amax + Bmax + Cmax)

    :type method: str or float, optional
    :param method: Whether the method of acquisition and analysis is
        documented in peer-reviewed literature. Accepted documented values are
        ``"documented"`` and ``1``.
    :type evaluation: str or float, optional
    :param evaluation: Whether the target indicator is evaluated directly from
        field experiments. Accepted direct values are ``"direct"`` and ``2``.
    :type reliability: str or float, optional
    :param reliability: Confidence in the indicator value. Accepted values are
        ``"yes"``/``1`` for reliable and ``"partial"``/``0.5`` for partial
        reliability.
    :type report: str or float, optional
    :param report: Whether a report documents the field survey and data
        processing. Accepted values are ``"yes"``/``1`` for a complete report
        and ``"partial"``/``0.5`` for a partial report. Missing report
        documentation gives a zero Q_Index1 contribution because D = 0.
    :rtype: float
    """
    if method == "documented" or method == 1:
        A = 1
    else:
        A = 0

    if evaluation == "direct" or evaluation == 2:
        B = 2
    else:
        B = 0

    if reliability == "yes" or reliability == 1:
        C = 1
    elif reliability == "partial" or reliability == 0.5:
        C = 0.5
    else:
        C = 0

    if report == "yes" or report == 1:
        D = 1
    elif report == "partial" or report == 0.5:
        D = 0.5
    else:
        D = 0

    max_value = 4
    return (A + B + C) * D / max_value


def quality_index2(sera_site):
    """
    Standalone formula helper for Q_Index2.

    .. note::
        For object-oriented SiteXML workflows, prefer
        :meth:`~obspy.io.sitexml.core.SERASite.calculate_quality_index2`.

    This function calculates the Quality Index #2 for a site, according to
    SERA Deliverable 7.2.

    Quality Index #2 is a weighted sum computed on the quality index #1 of all
    site indicators evaluated at the target site and varies from 0 to 1.

    The formula used for the calculation is:

    >>> Q_Index2 = (
    ...    w1*Q_Index1_si1 + w2*Q_Index1_si2 + ... + w7*Q_Index1_si7) / 
    ...    (w1 + w2 + ... + w7)

    The weights used for this calculation for each site indicator, as proposed
    by SERA, are:

    +---------------------+--------+
    | Site indicator      | Weight |
    +=====================+========+
    | Resonance Frequency | 1      |
    +---------------------+--------+
    | Velocity Profile    | 1      |
    +---------------------+--------+
    | Velocity S30        | 0.5    |
    +---------------------+--------+
    | Bedrock Depth       | 0.5    |
    +---------------------+--------+
    | H800                | 0.5    |
    +---------------------+--------+
    | Geological Unit     | 0.5    |
    +---------------------+--------+
    | Soil Class EC8      | 0.25   |
    +---------------------+--------+

    The velocity-profile term uses the quality index of the
    ``VelocityProfileSurvey`` attached to the preferred analysis. A declared
    preferred velocity profile is expected to belong to that same preferred
    analysis.

    :type sera_site: :class:`~obspy.io.sitexml.core.SERASite`, required
    :param sera_site: The site for which to calculate quality index #2.
    :rtype: float or None
    """
    if not sera_site:
        return None

    weights_sum = sum(_QUALITY_INDEX2_WEIGHTS.values())

    Qindex1 = {}
    if sera_site.site_description:
        if sera_site.site_description.ec8:
            Qindex1["siteClassEC8"] = _quality_index_or_zero(
                sera_site.site_description.ec8)
        if sera_site.site_description.h800:
            Qindex1["h800"] = _quality_index_or_zero(
                sera_site.site_description.h800)
        if sera_site.site_description.bedrock_depth:
            Qindex1["bedrockDepth"] = _quality_index_or_zero(
                sera_site.site_description.bedrock_depth)
        if sera_site.site_description.geological_unit:
            Qindex1["geologicalUnit"] = _quality_index_or_zero(
                sera_site.site_description.geological_unit)

    analysis = sera_site.get_preferred_analysis()
    if analysis:
        if analysis.resonance_frequency:
            Qindex1["resonanceFrequency"] = _quality_index_or_zero(
                analysis.resonance_frequency)
        if analysis.velocity_s30:
            Qindex1["velocityS30"] = _quality_index_or_zero(
                analysis.velocity_s30)
        if analysis.velocity_profile_survey:
            Qindex1["velocityProfile"] = _quality_index_or_zero(
                analysis.velocity_profile_survey)

    quality_index2_sum = 0
    for key in Qindex1:
        quality_index2_sum += _QUALITY_INDEX2_WEIGHTS[key] * Qindex1[key]
    return quality_index2_sum / weights_sum


def quality_index3(f0_vs30=None, f0_bedrock_depth=None, f0_h800=None,
                   vs30_h800=None, vs30_geology=None):
    """
    Standalone formula helper for Q_Index3.

    .. note::
        For object-oriented SiteXML workflows, prefer
        :meth:`~obspy.io.sitexml.core.SERASite.calculate_quality_index3`.

    This function calculates the Quality Index #3 for a site, according to
    SERA Deliverable 7.2.

    Quality Index #3 refers to the overall consistency between the various
    indicators and varies from 0 to 1.

    The computation of Q_Index3 is given by the sum of consistency values
    divided by the number of provided consistency couples.

    Each consistency value is binary:

    - ``0``: the indicator pair is not consistent
    - ``1``: the indicator pair is consistent
    - ``None``: the indicator pair is unavailable or was not evaluated

    >>> Q_Index3 = [cons(f0, Vs30) + cons(f0, seismic_bedrock_depth) +
    ...            cons(f0, engineering_bedrock_depth) + cons(H800, Vs30) +
    ...            cons(Vs30, geology)] / n
    
    where ``n`` is the number of provided, non-``None`` consistency values.

    :type f0_vs30: float or None, optional
    :param f0_vs30: Consistency value for resonance frequency and Vs30
        (``0``, ``1``, or ``None``).
    :type f0_bedrock_depth: float or None, optional
    :param f0_bedrock_depth: Consistency value for resonance frequency and
        seismic bedrock depth (``0``, ``1``, or ``None``).
    :type f0_h800: float or None, optional
    :param f0_h800: Consistency value for resonance frequency and engineering
        bedrock depth H800 (``0``, ``1``, or ``None``).
    :type vs30_h800: float or None, optional
    :param vs30_h800: Consistency value for Vs30 and H800
        (``0``, ``1``, or ``None``).
    :type vs30_geology: float or None, optional
    :param vs30_geology: Consistency value for Vs30 and surface geology
        (``0``, ``1``, or ``None``).
    :rtype: float or None
    """
    values = [
        f0_vs30, f0_bedrock_depth, f0_h800, vs30_h800, vs30_geology]
    consistency_values = [value for value in values if value is not None]
    if not consistency_values:
        return None
    return sum(consistency_values) / len(consistency_values)


def overall_quality_index(quality_index2=0, quality_index3=0):
    """
    Standalone formula helper for the overall quality index.

    .. note::
        For object-oriented SiteXML workflows, prefer
        :meth:`~obspy.io.sitexml.core.SERASite.calculate_overall_quality_index`.

    This function calculates the overall quality index for a site, according
    to SERA Deliverable 7.2.

    The overall quality index is computed as the arithmetic mean between
    Q_Index2 and Q_Index3.

    >>> Overall_Quality_Index = (Q_Index2 + Q_Index3) / 2

    If Q_Index2 is zero, the overall quality index is zero and Q_Index3 does
    not affect the result. If Q_Index3 is ``None``, it is treated as zero.

    :type quality_index2: float, optional
    :param quality_index2: Q_Index2 value derived from the site's indicator
        quality indexes.
    :type quality_index3: float or None, optional
    :param quality_index3: Q_Index3 value derived from consistency checks
        between indicator pairs. ``None`` is treated as ``0`` according to the
        SERA overall quality-index formula.
    :rtype: float
    """
    if quality_index2 == 0:
        return 0
    if quality_index3 is None:
        quality_index3 = 0
    return (quality_index2 + quality_index3) / 2


def apply_quality_index_metadata(sera_site_dict, df_quality_index):
    """
    Apply tabular quality-index calculation inputs to imported sites.

    Q_Index1 criteria and Q_Index3 consistency values are not stored. Only the
    calculated indicator quality indexes and overall quality index are assigned
    to the SiteXML object model.

    :rtype: dict
    :return: The input ``sera_site_dict`` after applying calculated values.
    """
    for _, row in df_quality_index.iterrows():
        site_id = _read_cell(row, "siteID")
        if site_id is None:
            warnings.warn(
                "Missing siteID value. Processing of quality-index row will "
                "be skipped.",
                UserWarning)
            continue
        if site_id not in sera_site_dict:
            warnings.warn(
                f"Quality-index metadata references unknown siteID {site_id}. "
                "Processing of quality-index row will be skipped.",
                UserWarning)
            continue

        sera_site = sera_site_dict[site_id]
        has_quality_input = False

        for indicator_name in _QUALITY_INDEX_INDICATORS:
            indicator = sera_site.get_indicator_object(indicator_name)
            if indicator is None:
                continue

            criteria = {
                criterion: _read_cell(row, f"{indicator_name}_{criterion}")
                for criterion in _QUALITY_INDEX1_CRITERIA
            }
            if all(value is None for value in criteria.values()):
                continue

            indicator.calculate_quality_index1(**criteria)
            has_quality_input = True

        q3_values = {
            name: _read_quality_index_consistency(row, name)
            for name in _QUALITY_INDEX3_COLUMNS
        }
        if any(value is not None for value in q3_values.values()):
            has_quality_input = True

        if has_quality_input:
            sera_site.calculate_overall_quality_index(**q3_values)

    return sera_site_dict


def apply_quality_index_csv(sera_site_dict, quality_index_csv, delim=';'):
    """
    Apply CSV quality-index calculation inputs to existing SERASite objects.

    The sidecar values are used immediately to calculate SiteXML quality
    indexes and are not stored. The input dictionary is mutated in place and
    returned for convenience.

    :type sera_site_dict: dict of
        :class:`~obspy.io.sitexml.core.SERASite`, required
    :param sera_site_dict: Dictionary of SERASite objects keyed by site ID.
    :type quality_index_csv: str, pathlib.Path, or file-like object, required
    :param quality_index_csv: CSV file with quality-index calculation inputs.
    :type delim: str, optional
    :param delim: CSV file delimiter. Default is semicolon-delimited.
    :rtype: dict
    :return: The input ``sera_site_dict`` after applying calculated values.
    """
    try:
        df_quality_index = pd.read_csv(quality_index_csv, sep=delim)
    except OSError as e:
        raise SiteXMLIOError(
            f"Could not access quality-index CSV metadata: "
            f"{quality_index_csv}"
        ) from e
    except Exception as e:
        raise SiteXMLImportError(
            f"Could not read quality-index CSV metadata: {quality_index_csv}"
        ) from e

    return apply_quality_index_metadata(sera_site_dict, df_quality_index)


def apply_quality_index_excel(
        sera_site_dict, path_or_file_object, sheet_name="qualityIndex"):
    """
    Apply Excel quality-index calculation inputs to existing SERASite objects.

    The sidecar values are used immediately to calculate SiteXML quality
    indexes and are not stored. The input dictionary is mutated in place and
    returned for convenience.

    :type sera_site_dict: dict of
        :class:`~obspy.io.sitexml.core.SERASite`, required
    :param sera_site_dict: Dictionary of SERASite objects keyed by site ID.
    :type path_or_file_object: str, pathlib.Path, or file-like object, required
    :param path_or_file_object: Excel file containing the quality-index sheet.
    :type sheet_name: str, optional
    :param sheet_name: Sheet containing quality-index calculation inputs.
        Defaults to ``"qualityIndex"``.
    :rtype: dict
    :return: The input ``sera_site_dict`` after applying calculated values.
    """
    try:
        df_quality_index = pd.read_excel(
            path_or_file_object, sheet_name=sheet_name)
    except OSError as e:
        raise SiteXMLIOError(
            f"Could not access quality-index Excel metadata: "
            f"{path_or_file_object}"
        ) from e
    except ValueError as e:
        raise SiteXMLImportError(
            f"Could not find quality-index Excel sheet: {sheet_name}"
        ) from e
    except Exception as e:
        raise SiteXMLImportError(
            f"Could not read quality-index Excel metadata: "
            f"{path_or_file_object}"
        ) from e

    return apply_quality_index_metadata(sera_site_dict, df_quality_index)


def _read_quality_index_consistency(row, name):
    """
    Return one Q_Index3 consistency value from a tabular quality-index row.

    :rtype: int or None
    """
    value = _read_cell(row, name)
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
    try:
        value = float(value)
    except (TypeError, ValueError) as e:
        raise SiteXMLImportError(
            f"Q_Index3 consistency value {name!r} must be 0 or 1."
        ) from e
    if value not in (0, 1):
        raise SiteXMLImportError(
            f"Q_Index3 consistency value {name!r} must be 0 or 1."
        )
    return int(value)


def _read_cell(df_row, argument):
    """
    Return a non-empty cell value from a quality-index row.

    :rtype: object or None
    """
    if argument in df_row and not _empty_value(df_row[argument]):
        return df_row[argument]
    return None


def _empty_value(value):
    """
    Return whether a tabular cell should be treated as missing.

    This intentionally mirrors ``tabular._empty_value()`` locally. Keeping it
    here avoids making the lower-level ``util.py`` module depend on
    pandas/tabular import semantics solely for quality-index sidecar parsing.

    :rtype: bool
    """
    if pd.isna(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False
