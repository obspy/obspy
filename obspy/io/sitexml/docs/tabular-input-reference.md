# SiteXML Tabular Input Reference

This document is a reference for the tabular input files accepted by
`csv2sitexml` and `excel2sitexml`. 

> **Important:**  
> The input tables are organized to follow the SiteXML schema. Each table 
> corresponds to a major SiteXML object, and the columns in that table 
> correspond to the object's **elements** and **attributes**. At the 
> [**SiteXML schema documentation**](https://www.itsak.gr/SiteXML) you
> can find descriptions, examples and allowed values for each column.

## Table Of Contents

- [General Formatting And Naming Rules](#general-formatting-and-naming-rules)
- [Resource Identifiers And Preferred IDs](#resource-identifiers-and-preferred-ids)
  - [Notes On Preferred IDs](#notes-on-preferred-ids)
- [Common Site Indicator Metadata](#common-site-indicator-metadata)
- [Site Owner Table](#site-owner-table)
- [Site Description Table](#site-description-table)
  - [Site And Location Columns](#site-and-location-columns)
  - [Site Description Indicator Columns](#site-description-indicator-columns)
- [Analysis Table](#analysis-table)
  - [Analysis Relationship and Log Count Columns](#analysis-relationship-and-log-count-columns)
  - [Analysis Indicator Columns](#analysis-indicator-columns)
- [Velocity Profiles Table](#velocity-profiles-table)
- [Quality-Index Tables](#quality-index-table)
  - [Q_Index1 Criteria Columns](#q_index1-criteria-columns)
  - [Q_Index3 Consistency Columns](#q_index3-consistency-columns)

## General Formatting And Naming Rules

> **Important:**  
> - Excel **sheet names** are fixed and must be exactly as shown in the tables
>   descriptions below.
> - **Column names** are fixed and are case-sensitive.
> - The same column names are used by both CSV and Excel input.
> - The default CSV delimiter is semicolon (`;`). If your files use another
    delimiter, pass it with `-s`, for example `-s ","`. 
> - The delimiter character must not be used inside text values in any column.

In the tables below, **yes** marks required inputs.

## Resource Identifiers And Preferred IDs

SiteXML uses resource identifiers to connect the top-level site object with its
nested site description, analyses, velocity profiles, owner, and contact
metadata.

The current schema **keeps the identifier pattern intentionally relaxed**, but
project data should still use **stable, unique, URI-like identifiers**. The bundled
examples use a QuakeML-style convention:

```text
quakeml:domain.ab/site/001
quakeml:domain.ab/site_description/001
quakeml:domain.ab/analysis/001
quakeml:domain.ab/velocity_profile/001
quakeml:domain.ab/siteOwner/001
```

The part after the final slash usually identifies the object within its
collection. The path segment before it, names the object type. Keeping this
shape consistent makes the XML and tabular files easier to audit, but the
important requirement is that every relationship column points to an existing
object ID.

The most important relationship and preferred-ID columns are:

- `siteID` identifies the top-level SiteXML site and is used as the key in
  dictionaries returned by CSV and Excel import.
- `siteDescriptionID` identifies the site-description object. Analysis rows
  must repeat the corresponding `siteDescriptionID` so the analysis can be
  checked against the site description it describes.
- `analysisID` identifies one analysis. A site can have multiple analyses.
- `velocityProfileID` identifies one velocity profile under a specific
  analysis. An analysis can have multiple velocity profiles.
- `preferredSiteAnalysisID` selects the analysis to use when a site has more
  than one analysis. Quality-index calculations use this preferred analysis
  when it is present.
- `preferredVelocityProfileID` selects the preferred velocity profile.

Tabular import **validates** these relationships, but it **does not repair them** by
guessing what the user meant. When a site has several analyses or velocity
profiles, a missing or inconsistent ID is ambiguous. Automatically choosing the
first available object could create valid-looking XML that no longer represents
the intended data.

Explicit relationship IDs **are preserved** as provided by the user. The import 
tools raise a validation error when an explicit relationship points to an object 
that does not exist or belongs somewhere else.

### Notes On Preferred IDs

The importer does not guess preferred IDs. Missing preferred IDs are preserved 
as missing.

If you provide preferred IDs:
- `preferredSiteAnalysisID`: the analysis CSV must contain a
matching `analysisID`. If the analysis CSV is omitted, the preferred analysis
ID is ignored with a warning and omitted from generated SiteXML.

- `preferredVelocityProfileID`: the velocity-profile CSV must
contain a matching `velocityProfileID`. If velocity-profile metadata is
omitted, the preferred velocity-profile ID is ignored with a warning and
omitted from generated SiteXML.

> **Important:**  
> If both preferred IDs are present, the preferred velocity profile must belong
> to the preferred analysis.


## Common Site Indicator Metadata

SiteXML supports seven (7) site indicators: 
[`siteClassEC8`](https://www.itsak.gr/SiteXML/#type_EC8IndicatorType), 
[`bedrockDepth`](https://www.itsak.gr/SiteXML/#type_BedrockDepthIndicatorType), 
[`h800`](https://www.itsak.gr/SiteXML/#type_H800IndicatorType), 
[`geologicalUnit`](https://www.itsak.gr/SiteXML/#type_GeologicalUnitIndicatorType), 
[`resonanceFrequency`](https://www.itsak.gr/SiteXML/#type_ResonanceFrequencyIndicatorType), 
[`velocityS30`](https://www.itsak.gr/SiteXML/#type_VelocityS30IndicatorType) and 
[`velocityProfileSet`](https://www.itsak.gr/SiteXML/#type_VelocityProfileSetIndicatorType). 

All site indicators include the same set of metadata: 
- Value metadata
  ```text
  value
  uncertainty (where applicable)
  qualityIndex
  ```
- [`literatureSource`](https://www.itsak.gr/SiteXML/#type_LiteratureSourceType) metadata:
  ```text
  title
  firstAuthor
  secondaryAuthors
  year
  booktitle
  language
  doi
  ```
- [`externalReference`](https://www.itsak.gr/SiteXML/#type_ExternalReferenceType) metadata:
  ```text
  description
  uri
  ```

Some, have extra metadata; for example `geologicalUnit` includes the `geologicalMapScale`.

Some common rules apply if you want to provide metadata for a site indicator:
- you must provide a `value`
- `uncertainty` and `qualityIndex` are optional
- if you have a [`literatureSource`](https://www.itsak.gr/SiteXML/#type_LiteratureSourceType)
  you must provide at least `title` and `firstAuthor`
- If you have an [`externalReference`](https://www.itsak.gr/SiteXML/#type_ExternalReferenceType)
  you must provide both `description` and `uri`
- All other metadata is optional.

For user convenience, site indicator metadata columns in the input tables, 
use the same naming pattern:

```text
<indicator>_value
<indicator>_uncertainty
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

In the following table, you can find the descriptions and examples for
metadata that is **common** among all site indicators.

For metadata (like `value`) that are defined differently for each site 
indicator, or for the extra metadata please refer to the sections 
[Site Description Indicator Columns](#site-description-indicator-columns) 
and [Analysis Indicator Columns](#analysis-indicator-columns). 

In this example, the soil class site indicator `siteClassEC8` is used. 

| Column | Required? | Meaning | Example |
| --- | --- | --- | --- |
| `siteClassEC8_qualityIndex` | no | User provided quality index for a site indicator value (Q_Index1). Values must be in the range [0..1] | `0.875` |
| `siteClassEC8_title` | no | Literature source title for this indicator. If literature metadata is provided, `title` and `firstAuthor` are both required. | `Site characterization report` |
| `siteClassEC8_firstAuthor` | no | First author of the source. | `Author A.` |
| `siteClassEC8_secondaryAuthors` | no | Additional authors. | `Author B., Author C.` |
| `siteClassEC8_year` | no | Four-digit publication year. | `2018` |
| `siteClassEC8_booktitle` | no | Journal, report, book, or collection title. | `Engineering Geology` |
| `siteClassEC8_language` | no | Source language code or label. | `en` |
| `siteClassEC8_doi` | no | DOI. | `10.1007/s10518-017-0135-5` |
| `siteClassEC8_description` | no | External reference description. | `paper` |
| `siteClassEC8_uri` | no | External reference URI. | `https://doi.org/10.1007/s10518-017-0135-5/` |


## Site Owner Table

The site-owner table provides contact information for the site owner and it is **required**. 

It is provided as a CSV file or as an Excel sheet named `siteOwner` with just **one row**. 
This means you can use the site owner table with many sites that have the same owner.

The data read from the Site Owner table is stored in 
[**SiteXML element siteOwner**](https://www.itsak.gr/SiteXML/#type_SiteOwnerType).

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
and values for the four site indicators: 
[`siteClassEC8`](https://www.itsak.gr/SiteXML/#type_EC8IndicatorType), 
[`bedrockDepth`](https://www.itsak.gr/SiteXML/#type_BedrockDepthIndicatorType), 
[`h800`](https://www.itsak.gr/SiteXML/#type_H800IndicatorType), 
[`geologicalUnit`](https://www.itsak.gr/SiteXML/#type_GeologicalUnitIndicatorType).

It is provided as a CSV file or as an Excel sheet named `siteDescription` with **one row per site**. 

The data read from the site-description table is stored in 
[**SiteXML element siteDescription**](https://www.itsak.gr/SiteXML/#type_SiteDescriptionType).


### Site And Location Columns

| Column | Required? | Meaning | Example |
| --- | --- | --- | --- |
| `siteID` | **yes** | Resource ID for the top-level SiteXML site. **This is necessary because it associates the siteDescription object with the site.** | `quakeml:domain.ab/site/001` |
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
| `overallQindex` | no | Overall quality index of the site stored in SiteXML. Values must be in the range [0..1] | `0.41` |
| `preferredSiteAnalysisID` | no | Preferred analysis resource ID. Requires matching analysis metadata to be written. | `quakeml:domain.ab/analysis/001` |
| `preferredVelocityProfileID` | no | Preferred velocity-profile resource ID. Requires matching velocity-profile metadata to be written. | `quakeml:domain.ab/velocity_profile/001` |

> **Please Note:**   
> The `overallQindex` can be either provided manually by the user using the
> site-description table or calculated by the tools if a quality-index input table
> is provided. If neither is true, the produced SiteXML document won't have an
> `overallQindex` value.


### Site Description Indicator Columns

In addition to the common `qualityIndex`, `literatureSource` and  `externalReference` 
columns that were described [above](#common-site-indicator-metadata), use the 
following columns to provide value/uncertainty and other metadata for site 
indicators that belong to the site-description object.

| Column | Meaning | Example |
| --- | --- | --- |
| `siteClassEC8_value` | EC8 ground type. Allowed values: `A`, `B`, `C`, `D`, `E`, `S1`, `S2`, `Undefined`. | `B` |
| `bedrockDepth_value` | Bedrock depth value. | `40` |
| `bedrockDepth_uncertainty` | Uncertainty of the bedrock depth value. | `6` |
| `h800_value` | H800 depth value. | `10` |
| `h800_uncertainty` | Uncertainty of the H800 value. | `1` |
| `geologicalUnit_value` | Geological unit description. | `Holocene Deposits` |
| `geologicalMapScale` | Scale of the source geological map. | `1:50000` |
| `geologicalUnitOGE` | Geological unit according to the OGE vocabulary, if available. | `Alluvial deposits` |


## Analysis Table

The analysis table is **optional** and it provides values for three site indicators: 
[`resonanceFrequency`](https://www.itsak.gr/SiteXML/#type_ResonanceFrequencyIndicatorType), 
[`velocityS30`](https://www.itsak.gr/SiteXML/#type_VelocityS30IndicatorType) and 
[`velocityProfileSet`](https://www.itsak.gr/SiteXML/#type_VelocityProfileSetIndicatorType).

It is provided as a CSV file or as an Excel sheet named `analysis` with **many rows per site**. 
This means you can provide many sets of analysis site-indicators per site.

Each row must identify three things:
- the site it belongs to: SiteID
- the site description it is associated with: siteDescriptionID
- the unique analysisID

> **Note:** Missing values in any of the three above resource identifiers will result in data import error.

The data read from the analysis table are stored in 
[SiteXML element analysis](https://www.itsak.gr/SiteXML/#type_AnalysisType).


### Analysis Relationship and Log Count Columns

| Column | Required? | Meaning | Example |
| --- | --- | --- | --- |
| `siteID` | **yes** | Resource ID of the parent site. | `quakeml:domain.ab/site/001` |
| `siteDescriptionID` | **yes** | Resource ID of the parent site description. Must match the site row. | `quakeml:domain.ab/site_description/001` |
| `analysisID` | **yes** | Resource ID for this analysis. | `quakeml:domain.ab/analysis/001` |
| `sptLogsCount` | no | Number of SPT logs. | `0` |
| `cptLogsCount` | no | Number of CPT logs. | `0` |
| `boreholeLogsCount` | no | Number of borehole logs. | `0` |


### Analysis Indicator Columns

In addition to the common `qualityIndex`, `literatureSource` and  `externalReference` 
columns that were described [above](#common-site-indicator-metadata), use the 
following columns to provide value/uncertainty and other metadata for site 
indicators that belong to the analysis object.

| Column | Meaning | Example |
| --- | --- | --- |
| `resonanceFrequency_value` | Resonance frequency value. | `0.7` |
| `resonanceFrequency_uncertainty` | Uncertainty of the resonance frequency value. | `0.05` |
| [`resonanceFrequency_method1`](https://www.itsak.gr/SiteXML/#type_ResonanceFrequencyMethodType) | Primary method. Allowed examples: `HVSR NOISE`, `SSR NOISE`, `HVSR EARTHQUAKE RECORDS`, `SSR EARTHQUAKE RECORDS`, `INFERRED`. | `HVSR NOISE` |
| [`resonanceFrequency_method2`](https://www.itsak.gr/SiteXML/#type_VelocityS30MethodType) | Secondary method. Same allowed values as method 1. | `SSR NOISE` |
| `velocityS30_value` | Vs30 value in m/s. | `620` |
| `velocityS30_uncertainty` | Uncertainty of the Vs30 value. | `18` |
| [`velocityS30_method1`](https://www.itsak.gr/SiteXML/#type_VelocityS30MethodType) | Primary Vs30 method. Examples: `MASW`, `SPAC/F-K`, `S-REFL`, `Downhole`, `Geology`. | `MASW` |
| [`velocityS30_method2`](https://www.itsak.gr/SiteXML/#type_VelocityS30MethodType) | Secondary Vs30 method. Same allowed values as method 1. | `SPAC/F-K` |
| [`velocityS30_methodCombIndex`](https://www.itsak.gr/SiteXML/#type_VelocityS30MethodCombIndexType) | Whether methods were combined. Allowed values: `1.0`, `1.2`. | `1.2` |
| [`velocityS30_manualIndex`](https://www.itsak.gr/SiteXML/#type_VelocityS30ManualIndexType) | Qualitative factor for maximum Vs measurement depth. Allowed values: `0.2`, `0.4`, `0.8`, `1.0`. | `1.0` |

> **Please note,** that there is not `velocityProfileSet_value` column.   
> The velocity profile data, if available, is provided as seperate table(s)
> ([see below](#velocity-profiles-table)).   
> However, you can still provide `qualityIndex`, `literatureSource` and 
> `externalReference` metadata for the `velocityProfileSet` site indicator.


## Velocity Profiles Table

Velocity-profile input is **optional**. It may be one CSV/Excel file or a folder of CSV/Excel
files. 

Each row in the table describes **one layer in one velocity profile** and must 
identify three things:
- the site it belongs to: SiteID
- the analysis it belongs to: analysisID
- the unique velocityProfileID

> **For example**, a velocity profile with 8 layers, will occupy 8 rows in the table, 
> with the same SiteID, analysisID and velocityProfileID

> **Note:** Missing values in any of the three above resource identifiers will result in data import error.

The data read from the velocity-profile  table are stored in multiple
[SiteXML VelocityProfile elements](https://www.itsak.gr/SiteXML/#type_VelocityProfile).

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

Here is an example of a velocity-profile table, containing 
- two velocity profiles for the same analysis and site and 
- one velocity profile for a different analysis and site.

> **Please note** that, the layer data of one velocity profile, share the same velocityProfileID.

| SiteID | analysisID | velocityProfileID | layerCount | velocityS | topDepth |
| --- | --- | --- | --- | --- | --- |
| quakeml:domain.ab/site/001 | quakeml:domain.ab/analysis/001	| quakeml:domain.ab/velocity_profile/001 | 1 | 118.08	| 0
| quakeml:domain.ab/site/001 | quakeml:domain.ab/analysis/001	| quakeml:domain.ab/velocity_profile/001 | 2 | 139.94	| 0.19
| quakeml:domain.ab/site/001 | quakeml:domain.ab/analysis/001	| quakeml:domain.ab/velocity_profile/001 | 3 | 180.3 | 0.57
| quakeml:domain.ab/site/001 | quakeml:domain.ab/analysis/001	| quakeml:domain.ab/velocity_profile/001 | 4 | 252.54	| 1.34
| quakeml:domain.ab/site/001 | quakeml:domain.ab/analysis/001	| quakeml:domain.ab/velocity_profile/002 | 1 | 128.08	| 0
| quakeml:domain.ab/site/001 | quakeml:domain.ab/analysis/001	| quakeml:domain.ab/velocity_profile/002 | 2 | 149.94	| 0.19
| quakeml:domain.ab/site/001 | quakeml:domain.ab/analysis/001	| quakeml:domain.ab/velocity_profile/002 | 3 | 170.3 | 0.57
| quakeml:domain.ab/site/001 | quakeml:domain.ab/analysis/001	| quakeml:domain.ab/velocity_profile/002 | 4 | 232.54	| 1.34
| quakeml:domain.ab/site/002 | quakeml:domain.ab/analysis/002	| quakeml:domain.ab/velocity_profile/003 | 1 | 128.08	| 0
| quakeml:domain.ab/site/002 | quakeml:domain.ab/analysis/002	| quakeml:domain.ab/velocity_profile/003 | 2 | 149.94	| 0.19
| quakeml:domain.ab/site/002 | quakeml:domain.ab/analysis/002	| quakeml:domain.ab/velocity_profile/003 | 3 | 170.3 | 0.57
| quakeml:domain.ab/site/002 | quakeml:domain.ab/analysis/002	| quakeml:domain.ab/velocity_profile/003 | 4 | 232.54	| 1.34


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
> - to the [**quality indexes guide**](quality-indexes-guide.md) distributed with the standalone executables
> - to the guidelines of [**SERA deliverable D7.2**](https://www.itsak.gr/SiteXML/SERA_D7.2_Best-practice_for_site_characterization.pdf).


### Q_Index1 Criteria Columns

For the calculation of Q_Index1, for each site indicator, you can provide values
for four criteria:

| Criterion suffix | Meaning | Example values |
| --- | --- | --- |
| `method` | Whether acquisition/analysis method is documented. | `documented`, empty |
| `evaluation` | Whether the indicator is evaluated directly from field experiments. | `direct`, empty |
| `reliability` | Confidence in the indicator value. | `yes`, `partial`, empty |
| `report` | Whether a report documents the field survey and processing. | `yes`, `partial`, empty |

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
