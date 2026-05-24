# SiteXML Tabular Input Reference

This document is a reference for the tabular input files accepted by
`csv2sitexml` and `excel2sitexml`. 

> **Important:**  
> The input tables are organized to follow the SiteXML schema. Each table 
> corresponds to a major SiteXML object, and the columns in that table 
> correspond to the object's **elements** and **attributes**. Detailed 
> descriptions on the allowed values you can find at the 
> [SiteXML schema documentation](https://www.itsak.gr/SiteXML).

## Table Of Contents

- [CSV Specific Notes](#csv-specific-notes)
- [Common Conventions](#common-conventions)
- [Shared Indicator Columns](#shared-indicator-columns)
  - [Full Example: `siteClassEC8`](#full-example-siteclassec8)
- [Site Owner Table](#site-owner-table)
- [Site Description Table](#site-description-table)
  - [Site And Location Columns](#site-and-location-columns)
  - [Site-Description Indicator Columns](#site-description-indicator-columns)
- [Analysis Table](#analysis-table)
  - [Analysis Relationship Columns](#analysis-relationship-columns)
  - [Analysis Indicator Columns](#analysis-indicator-columns)
  - [Analysis Log Count Columns](#analysis-log-count-columns)
- [Velocity Profiles Table](#velocity-profiles-table)
- [Quality-Index Tables](#quality-index-table)
  - [Q_Index1 Criteria Columns](#q_index1-criteria-columns)
  - [Q_Index3 Consistency Columns](#q_index3-consistency-columns)
- [Notes On Preferred IDs](#notes-on-preferred-ids)
- [Notes On Quality Indexes](#notes-on-quality-indexes)

## CSV Specific Notes

The default CSV delimiter is semicolon (`;`). If your files use another
delimiter, pass it with `-s`, for example `-s ","`. 

> - **You must make sure, that the delimiter character 
> is not used inside any text values in any of the columns.**
>
> - **Column names cannot be changed and are case-sensitive.**

In the tables below, **yes** marks required inputs.

The CSV and Excel file names are not specific. When running the commands provide the actual paths to the input files.

The Excel sheet names are specific. See below for the names you must use for each table.

## Common Conventions

Resource ID examples in the test fixtures use QuakeML-like strings:

```text
quakeml:domain.ab/site/001
quakeml:domain.ab/site_description/001
quakeml:domain.ab/analysis/001
quakeml:domain.ab/velocity_profile/001
```

Empty optional cells are imported as absent metadata.

Station codes must use `network.station` notation:

```text
XX.ABCD
1A.ABC1
A.TST
```

The network code is 1-2 ASCII uppercase alpha or numeric characters. The station code is
3-5 ASCII uppercase alpha or numeric characters.

Quality-index values written directly in `*_qualityIndex` and `overallQindex`
columns must be numbers in the closed range `0` to `1`.

## Shared Indicator Columns

All site-description and analysis indicators use the same column pattern:

```text
<indicator>_value
<indicator>_qualityIndex
<indicator>_title
<indicator>_firstAuthor
<indicator>_secondaryAuthors
<indicator>_year
<indicator>_booktitle
<indicator>_language
<indicator>_doi
<indicator>_description
<indicator>_uri
```

For indicators with an uncertainty, the pattern also includes:

```text
<indicator>_uncertainty
```

The first full example below uses `siteClassEC8`. For the other indicators,
only the columns that differ from this shared pattern are listed.

### Full Example: `siteClassEC8`

| Column | Required? | Meaning | Example |
| --- | --- | --- | --- |
| `siteClassEC8_value` | no | EC8 ground type. Allowed values: `A`, `B`, `C`, `D`, `E`, `S1`, `S2`, `Undefined`. | `B` |
| `siteClassEC8_qualityIndex` | no | Calculated indicator quality index, Q_Index1. | `0.875` |
| `siteClassEC8_title` | no | Literature/source title for this indicator. If source metadata is provided, `title` and `firstAuthor` are both required. | `Site characterization report` |
| `siteClassEC8_firstAuthor` | no | First author of the source. | `Author A.` |
| `siteClassEC8_secondaryAuthors` | no | Additional authors. | `Author B., Author C.` |
| `siteClassEC8_year` | no | Four-digit publication year. | `2018` |
| `siteClassEC8_booktitle` | no | Journal, report, book, or collection title. | `Engineering Geology` |
| `siteClassEC8_language` | no | Source language code or label. | `en` |
| `siteClassEC8_doi` | no | DOI. | `10.1007/s10518-017-0135-5` |
| `siteClassEC8_description` | no | Description for the external reference. | `paper` |
| `siteClassEC8_uri` | no | External reference URI. | `https://doi.org/10.1007/s10518-017-0135-5/` |

## Site Owner Table

The site-owner table provides contact information for the site owner and it is **required**. 

It is provided as a CSV file or as an Excel sheet named `siteOwner` with just **one row**. 
This means you can use the site owner table with many sites that have the same owner.

The data read from the Site Owner table are stored in 
[SiteXML element siteOwner](https://www.itsak.gr/SiteXML/#type_SiteOwnerType).

| Column | Required? | Meaning | Example |
| --- | --- | --- | --- |
| `ownerID` | no | Resource ID for the owner object. | `quakeml:domain.ab/siteOwner/001` |
| `owner_codename` | **yes** | Short code name for the data owner. | `SITEOWNER` |
| `owner_fullname` | **yes** | Full owner name. | `Site Owner Full Name` |
| `personID` | no | Resource ID for the contact person. | `quakeml:domain.ab/person/001` |
| `person_firstname` | **yes** | Contact person's first name. | `Name` |
| `person_lastname` | **yes** | Contact person's last name. | `Surname` |
| `person_mbox` | **yes** | Contact email address. | `someemail@domain.ab` |
| `person_homepage` | no | Contact person's web page. | `https://www.domain.ab/person` |
| `institutionID` | no | Resource ID for the institution. | `quakeml:domain.ab/institution/001` |
| `institution_name` | no | Institution name or abbreviation. | `INSTITUTION_ABBR` |
| `institution_mbox` | no | Institution email address. | `info@domain.ab` |
| `institution_phone` | no | Institution phone number. | `+30 123 456789` |
| `institution_homepage` | no | Institution web page. | `https://www.domain.ab` |
| `address_street` | no | Street address. | `Some streetAddress` |
| `address_locality` | no | City/locality. | `City` |
| `address_postal_code` | no | Postal code. | `12345` |
| `address_country_code` | no | Country code. | `GR` |
| `address_country` | no | Country name. | `Greece` |
| `affiliation_department` | no | Contact person's department. | `Seismology` |
| `affiliation_function` | no | Contact person's role/function. | `Senior researcher` |

## Site Description Table

The site-description table is **required** and it provides information on site location, geology 
and values for the four site indicators: `siteClassEC8`, `bedrockDepth`, `h800`, `geologicalUnit`.

It is provided as a CSV file or as an Excel sheet named `siteDescription` with **one row per site**. 

The data read from the site-description table are stored in 
[SiteXML element siteDescription](https://www.itsak.gr/SiteXML/#type_SiteDescriptionType).


### Site And Location Columns

| Column | Required? | Meaning | Example |
| --- | --- | --- | --- |
| `siteID` | **yes** | Resource ID for the top-level SiteXML site. **This is necessary because it associates the siteDescription item with the site.** | `quakeml:domain.ab/site/001` |
| `siteDescriptionID` | **yes** | Resource ID for the site-description object. | `quakeml:domain.ab/site_description/001` |
| `station` | no | Station code in `network.station` notation. Leave empty for non-station sites. | `XX.ABCD` |
| `latitude` | **yes** | Geographic latitude in degrees. | `45.137174` |
| `longitude` | **yes** | Geographic longitude in degrees. | `5.998905` |
| `altitude` | no | Ground elevation in meters. | `239` |
| `minDistanceFromStation` | no | Minimum distance from station in meters. | `20` |
| `maxDistanceFromStation` | no | Maximum distance from station in meters. | `520.2` |
| [`morphology`](https://www.itsak.gr/SiteXML/#type_MorphologyType)  | no | Qualitative landform. Allowed values: `Plain`, `Valley - Basin`, `Slope`, `Ridge`. | `Valley - Basin` |
| [`topography_schemaA`](https://www.itsak.gr/SiteXML/#type_TopographySchemaAType)  | no | Italian Code topography class. Allowed values: `T1`, `T2`, `T3`, `T4`. | `T1` |
| [`topography_schemaB`](https://www.itsak.gr/SiteXML/#type_TopographySchemaBType)  | no | Burjanek et al. terrain class. Allowed values: `Valley`, `Lower slope`, `Flat`, `Middle slope`, `Upper slope`, `Ridge`. | `Valley` |
| `overallQindex` | no | Final overall quality index stored in SiteXML. | `0.41` |
| `preferredSiteAnalysisID` | no | Preferred analysis resource ID. Requires matching analysis metadata to be written. | `quakeml:domain.ab/analysis/001` |
| `preferredVelocityProfileID` | no | Preferred velocity-profile resource ID. Requires matching velocity-profile metadata to be written. | `quakeml:domain.ab/velocity_profile/001` |

If the analysis CSV is omitted, `preferredSiteAnalysisID` and
`preferredVelocityProfileID` are ignored with warnings. If the analysis CSV is
provided but velocity-profile metadata is omitted, `preferredVelocityProfileID`
is ignored with a warning.

### Site-Description Indicator Columns

The complete shared pattern is shown above for `siteClassEC8`.

For `bedrockDepth`, use the shared source/reference columns with these value
columns:

| Column | Meaning | Example |
| --- | --- | --- |
| `bedrockDepth_value` | Bedrock depth value. | `40` |
| `bedrockDepth_uncertainty` | Uncertainty of the bedrock depth value. | `6` |
| `bedrockDepth_qualityIndex` | Indicator Q_Index1. | `0.5` |

Complete shared metadata columns:

```text
bedrockDepth_title
bedrockDepth_firstAuthor
bedrockDepth_secondaryAuthors
bedrockDepth_year
bedrockDepth_booktitle
bedrockDepth_language
bedrockDepth_doi
bedrockDepth_description
bedrockDepth_uri
```

For `h800`, use the shared source/reference columns with these value columns:

| Column | Meaning | Example |
| --- | --- | --- |
| `h800_value` | H800 depth value. | `10` |
| `h800_uncertainty` | Uncertainty of the H800 value. | `1` |
| `h800_qualityIndex` | Indicator Q_Index1. | `0.43` |

Complete shared metadata columns:

```text
h800_title
h800_firstAuthor
h800_secondaryAuthors
h800_year
h800_booktitle
h800_language
h800_doi
h800_description
h800_uri
```

For `geologicalUnit`, use the shared source/reference columns with these value
and extra columns:

| Column | Meaning | Example |
| --- | --- | --- |
| `geologicalUnit_value` | Geological unit description. | `Holocene Deposits` |
| `geologicalUnit_qualityIndex` | Indicator Q_Index1. | `0.8` |
| `geologicalMapScale` | Scale of the source geological map. | `1:50000` |
| `geologicalUnitOGE` | Geological unit according to the OGE vocabulary, if available. | `Alluvial deposits` |

Complete shared metadata columns:

```text
geologicalUnit_title
geologicalUnit_firstAuthor
geologicalUnit_secondaryAuthors
geologicalUnit_year
geologicalUnit_booktitle
geologicalUnit_language
geologicalUnit_doi
geologicalUnit_description
geologicalUnit_uri
```

## Analysis Table

The analysis table is **optional** and it provides values for three site indicators: 
`resonanceFrequency`, `velocityS30`, `velocityProfileSet`.

It is provided as a CSV file or as an Excel sheet named `analysis` with **many rows per site**. 
This means you can provide many sets of analysis site-indicators per site.

Each row must **absolutely** identify three things:
- the site it belongs to: SiteID
- the site description it is associated with: siteDescriptionID
- the unique analysisID

> **Note:** Missing values in any of the three above resource identifiers will result in data import error.

The data read from the analysis table are stored in 
[SiteXML element analysis](https://www.itsak.gr/SiteXML/#type_AnalysisType).


### Analysis Relationship Columns

| Column | Required? | Meaning | Example |
| --- | --- | --- | --- |
| `siteID` | **yes** | Resource ID of the parent site. | `quakeml:domain.ab/site/001` |
| `siteDescriptionID` | **yes** | Resource ID of the parent site description. Must match the site row. | `quakeml:domain.ab/site_description/001` |
| `analysisID` | **yes** | Resource ID for this analysis. | `quakeml:domain.ab/analysis/001` |

### Analysis Indicator Columns

For `resonanceFrequency`, use the shared source/reference columns with these
value and method columns:

| Column | Meaning | Example |
| --- | --- | --- |
| `resonanceFrequency_value` | Resonance frequency value. | `0.7` |
| `resonanceFrequency_uncertainty` | Uncertainty of the resonance frequency value. | `0.05` |
| `resonanceFrequency_qualityIndex` | Indicator Q_Index1. | `0.8` |
| [`resonanceFrequency_method1`](https://www.itsak.gr/SiteXML/#type_ResonanceFrequencyMethodType) | Primary method. Allowed examples: `HVSR NOISE`, `SSR NOISE`, `HVSR EARTHQUAKE RECORDS`, `SSR EARTHQUAKE RECORDS`, `INFERRED`. | `HVSR NOISE` |
| [`resonanceFrequency_method2`](https://www.itsak.gr/SiteXML/#type_VelocityS30MethodType) | Secondary method. Same allowed values as method 1. | `SSR NOISE` |

Complete shared metadata columns:

```text
resonanceFrequency_title
resonanceFrequency_firstAuthor
resonanceFrequency_secondaryAuthors
resonanceFrequency_year
resonanceFrequency_booktitle
resonanceFrequency_language
resonanceFrequency_doi
resonanceFrequency_description
resonanceFrequency_uri
```

For `velocityS30`, use the shared source/reference columns with these value and
method columns:

| Column | Meaning | Example |
| --- | --- | --- |
| `velocityS30_value` | Vs30 value in m/s. | `620` |
| `velocityS30_uncertainty` | Uncertainty of the Vs30 value. | `18` |
| `velocityS30_qualityIndex` | Indicator Q_Index1. | `0.5` |
| [`velocityS30_method1`](https://www.itsak.gr/SiteXML/#type_VelocityS30MethodType) | Primary Vs30 method. Examples: `MASW`, `SPAC/F-K`, `S-REFL`, `Downhole`, `Geology`. | `MASW` |
| [`velocityS30_method2`](https://www.itsak.gr/SiteXML/#type_VelocityS30MethodType) | Secondary Vs30 method. Same allowed values as method 1. | `SPAC/F-K` |
| [`velocityS30_methodCombIndex`](https://www.itsak.gr/SiteXML/#type_VelocityS30MethodCombIndexType) | Whether methods were combined. Allowed values: `1.0`, `1.2`. | `1.2` |
| [`velocityS30_manualIndex`](https://www.itsak.gr/SiteXML/#type_VelocityS30ManualIndexType) | Qualitative factor for maximum Vs measurement depth. Allowed values: `0.2`, `0.4`, `0.8`, `1.0`. | `1.0` |

Complete shared metadata columns:

```text
velocityS30_title
velocityS30_firstAuthor
velocityS30_secondaryAuthors
velocityS30_year
velocityS30_booktitle
velocityS30_language
velocityS30_doi
velocityS30_description
velocityS30_uri
```

For `velocityProfileSet`, there is no `velocityProfileSet_value` column. 
The velocity profile data, if available, is provided as seperate table(s). Use
the shared source/reference columns with:

| Column | Meaning | Example |
| --- | --- | --- |
| `velocityProfileSet_qualityIndex` | Indicator Q_Index1 for the velocity-profile set/survey. | `1` |

Complete shared metadata columns:

```text
velocityProfileSet_title
velocityProfileSet_firstAuthor
velocityProfileSet_secondaryAuthors
velocityProfileSet_year
velocityProfileSet_booktitle
velocityProfileSet_language
velocityProfileSet_doi
velocityProfileSet_description
velocityProfileSet_uri
```

### Analysis Log Count Columns

| Column | Required? | Meaning | Example |
| --- | --- | --- | --- |
| `sptLogsCount` | no | Number of SPT logs. | `0` |
| `cptLogsCount` | no | Number of CPT logs. | `0` |
| `boreholeLogsCount` | no | Number of borehole logs. | `0` |

## Velocity Profiles Table

Velocity-profile input is optional. It may be one CSV file or a folder of CSV
files. Each row describes one layer in one velocity profile.

| Column | Required? | Meaning | Example |
| --- | --- | --- | --- |
| `siteID` | **yes** | Resource ID of the parent site. | `quakeml:domain.ab/site/001` |
| `analysisID` | **yes** | Resource ID of the parent analysis. | `quakeml:domain.ab/analysis/001` |
| `velocityProfileID` | **yes** | Resource ID of the velocity profile. Repeated rows with the same ID are layers of the same profile. | `quakeml:domain.ab/velocity_profile/001` |
| `layerCount` | no | Optional layer counter/order marker. | `1` |
| `density_value` | no | Density value. | `1800` |
| `density_uncertainty` | no | Uncertainty of the density value. | `50` |
| `velocityP_value` | no | P-wave velocity value. | `900` |
| `velocityP_uncertainty` | no | Uncertainty of the P-wave velocity value. | `20` |
| `velocityS_value` | **yes** | S-wave velocity value. | `118.08` |
| `velocityS_uncertainty` | no | Uncertainty of the S-wave velocity value. | `2` |
| `layerTopDepth_value` | **yes** | Top depth of the layer. | `0` |
| `layerTopDepth_uncertainty` | no | Uncertainty of the top depth. | `0.1` |
| `layerBottomDepth_value` | no | Bottom depth of the layer. Leave empty for an open-ended final layer. | `0.19` |
| `layerBottomDepth_uncertainty` | no | Uncertainty of the bottom depth. | `0.1` |

## Quality-Index Table

The indicator-level quality indexes and the final overall quality index 
stored in SiteXML can be calculated during tabular import. The import 
helpers read the required calculation parameters from an **optional** CSV or 
Excel table, apply them immediately, and **store only the schema-supported 
calculated results on the SiteXML objects**.

Quality-Index table is provided as a CSV file or as an Excel sheet named 
`qualityIndex` with **one row per site**. 

Each row must identify the site it belongs to, with the ``SiteID`` column. 
It also includes [criteria values for the calculation of 
Q_Index1](#q_index1-criteria-columns) for all site indicators and [criteria 
values for the calculation of Q_Index3](#q_index3-consistency-columns). All
these criteria values are **optional**.

> **Note:** 
> - A missing ``SiteID`` column would result in data import error.
> - Rows with missing or unknown ``SiteID`` values are skipped with a warning.

> **See also:** For more information on the SiteXML quality indexes, please refer
> - to the [quality indexes guide](quality-indexes-guide.md) distributed with the standalone executables
> - to the guidelines of [SERA deliverable D7.2](https://www.itsak.gr/SiteXML/SERA_D7.2_Best-practice_for_site_characterization.pdf).

### Q_Index1 Criteria Columns

For the calculation of Q_Index1, for each site indicator, you can provide values
for four criteria:

| Criterion suffix | Meaning | Example values |
| --- | --- | --- |
| `_method` | Whether acquisition/analysis method is documented. | `documented`, empty |
| `_evaluation` | Whether the indicator is evaluated directly from field experiments. | `direct`, empty |
| `_reliability` | Confidence in the indicator value. | `yes`, `partial`, empty |
| `_report` | Whether a report documents the field survey and processing. | `yes`, `partial`, empty |

The names of the columns are formed using the criterion name, preffixed by 
the name of the site indicator.

```text
<indicator>_method
<indicator>_evaluation
<indicator>_reliability
<indicator>_report
```

Indicator prefixes are:

```text
siteClassEC8
bedrockDepth
h800
geologicalUnit
resonanceFrequency
velocityS30
velocityProfileSet
```

Example columns for the EC8 site indicator:

| Column | Example |
| --- | --- |
| `siteClassEC8_method` | `documented` |
| `siteClassEC8_evaluation` | `direct` |
| `siteClassEC8_reliability` | `partial` |
| `siteClassEC8_report` | `yes` |

For the other six indicators, use the same suffixes with the corresponding
prefix. For example, use `velocityS30_method` and
`velocityProfileSet_report`.

Complete Q_Index1 columns for the other indicators:

```text
bedrockDepth_method
bedrockDepth_evaluation
bedrockDepth_reliability
bedrockDepth_report
h800_method
h800_evaluation
h800_reliability
h800_report
geologicalUnit_method
geologicalUnit_evaluation
geologicalUnit_reliability
geologicalUnit_report
resonanceFrequency_method
resonanceFrequency_evaluation
resonanceFrequency_reliability
resonanceFrequency_report
velocityS30_method
velocityS30_evaluation
velocityS30_reliability
velocityS30_report
velocityProfileSet_method
velocityProfileSet_evaluation
velocityProfileSet_reliability
velocityProfileSet_report
```

### Q_Index3 Consistency Columns

Q_Index3 is calculated overall for the site, using **consistency** values
for pairs of site indicators. You can provide consistency values for the 
following pairs:

| Column | Required? | Meaning | Example |
| --- | --- | --- | --- |
| `f0_vs30` | no | Consistency between resonance frequency and Vs30. | `1` |
| `f0_bedrock_depth` | no | Consistency between resonance frequency and seismic bedrock depth. | `0` |
| `f0_h800` | no | Consistency between resonance frequency and H800. | `1` |
| `vs30_h800` | no | Consistency between Vs30 and H800. | `1` |
| `vs30_geology` | no | Consistency between Vs30 and surface geology. | `1` |

Allowed consistency values are:

- `1`: consistent;
- `0`: not consistent;
- empty: unavailable or not evaluated.

### Notes On Quality Indexes

SiteXML stores calculated indicator-level quality indexes and the final overall
quality index. It does not store the detailed Q_Index1 criteria or Q_Index3
consistency inputs.

If you provide both **direct** (already calculated) `*_qualityIndex` values 
(from the site-description or the analysis table) and a quality-index 
calculation table:

- direct `*_qualityIndex` values are imported first;
- Q_Index1 criteria values for an existing indicator **recalculate and replace**
  that indicator's direct `*_qualityIndex` value;
- Missing Q_Index1 criteria values for an indicator, leave that indicator's 
  direct `*_qualityIndex` value unchanged;
- Quality-index rows for sites without indicator objects are skipped.

The import tools do not automatically populate the `overallQindex` value in
the SiteXML document from direct `*_qualityIndex` columns alone. To write 
`overallQindex`, use one of these explicit workflows:

- provide `overallQindex` in the site-description table;
- provide the quality-index table so the tool can calculate Q_Index1 and
  Q_Index3-derived results during import.

If no usable indicator data exists for a site, providing a quality-index
table does not write a fake `<overallQindex>0</overallQindex>`.

## Notes On Preferred IDs

The importer does not guess preferred IDs.

If you provide `preferredSiteAnalysisID`, the analysis CSV must contain a
matching `analysisID`. If the analysis CSV is omitted, the preferred analysis
ID is ignored with a warning and omitted from generated SiteXML.

If you provide `preferredVelocityProfileID`, the velocity-profile CSV must
contain a matching `velocityProfileID`. If velocity-profile metadata is
omitted, the preferred velocity-profile ID is ignored with a warning and
omitted from generated SiteXML.

> **If both preferred IDs are provided, the preferred velocity profile must belong
> to the preferred analysis.**