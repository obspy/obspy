# SiteXML Standalone Executable Usage Guide

This guide describes how to use the standalone `csv2serasite` and
`excel2serasite` command-line tools to convert tabular site-characterization
metadata into SiteXML files.

The tools are intended for users who do not have Python or ObsPy installed.
Each release artifact is built for one operating system and architecture. Use
the bundle built for your platform, and keep the executable inside its
distributed folder.

## What The Tools Produce

Both tools write one SiteXML file per imported site into the output folder.
The generated XML is schema-validated before it is written.

Default output filenames are:

- `Site_<network.station>_<DD-MM-YYYY>.xml` for station-backed sites, for
  example `Site_XX.ABCD_13-05-2026.xml`.
- `Site_<domain>.<site-id>_<DD-MM-YYYY>.xml` for sites without a station code,
  for example `Site_domain.ab.003_13-05-2026.xml`.

Existing files with the same generated name are overwritten.

## Minimal Required Metadata

The smallest tabular input that can produce a valid SiteXML file contains:

- one site-owner table;
- one site-description table;
- no analysis table;
- no velocity-profile table;
- no quality-index sidecar table.

For CSV input, this means:

- `site_owner.csv`
- `site_description.csv`

For Excel input, this means one workbook with:

- a `siteOwner` sheet;
- a `siteDescription` sheet.

The minimal required site-owner columns are:

```text
owner_codename
owner_fullname
person_firstname
person_lastname
person_mbox
```

The minimal required site-description columns are:

```text
siteID
siteDescriptionID
latitude
longitude
```

All other columns are optional. Optional empty cells are imported as absent
metadata.

## Running `csv2serasite`

Use `csv2serasite` when your input metadata is split across CSV files.

Minimal CSV conversion:

```bash
csv2serasite \
  -o site_owner.csv \
  -d site_description.csv \
  -out sitexml_output
```

Full CSV conversion with analysis, velocity profiles, and quality-index
calculation inputs:

```bash
csv2serasite \
  -o site_owner.csv \
  -d site_description.csv \
  -a site_analysis.csv \
  -p velocity_profiles \
  -q quality_index.csv \
  -out sitexml_output
```

Options:

```text
-o, --site-owner          Required CSV file with site-owner metadata.
-d, --site-description    Required CSV file with site-description metadata.
-a, --analysis            Optional CSV file with analysis metadata.
-p, --velocity-profiles   Optional CSV file or folder with velocity-profile metadata.
-q, --quality-index       Optional CSV file with quality-index calculation inputs.
-out, --output-folder     Required output folder for generated SiteXML files.
-s, --delim               CSV delimiter. Defaults to ';'.
-V, --version             Print the tool version.
-h, --help                Print command help.
```

If your CSV files use commas instead of semicolons, pass `-s ","`.

## Running `excel2serasite`

Use `excel2serasite` when your owner, site-description, analysis, and
quality-index tables are sheets in one workbook.

Minimal Excel conversion:

```bash
excel2serasite sera_site.xlsx -out sitexml_output
```

Excel conversion with velocity profiles:

```bash
excel2serasite sera_site.xlsx \
  -p velocity_profiles.xlsx \
  -out sitexml_output
```

Options:

```text
path_or_file_object       Required Excel workbook with site metadata.
-p, --velocity-profiles   Optional Excel file or folder with velocity-profile metadata.
-out, --output-folder     Required output folder for generated SiteXML files.
-V, --version             Print the tool version.
-h, --help                Print command help.
```

The main Excel workbook uses these sheet names:

```text
siteOwner        required
siteDescription  required
analysis         optional
qualityIndex     optional
```

Velocity-profile metadata is passed separately with `-p`. It may be one Excel
file or a folder of Excel files.

## Platform Notes

On macOS and Linux, run the executable from a terminal, for example:

```bash
./csv2serasite -o site_owner.csv -d site_description.csv -out sitexml_output
./excel2serasite sera_site.xlsx -out sitexml_output
```

On Windows PowerShell, run:

```powershell
.\csv2serasite.exe -o site_owner.csv -d site_description.csv -out sitexml_output
.\excel2serasite.exe sera_site.xlsx -out sitexml_output
```

The tools are distributed as PyInstaller `--onedir` bundles. Do not move the
executable out of its folder; it needs the bundled libraries and SiteXML schema
data next to it.

Unsigned macOS executables may trigger a Gatekeeper warning. If you trust the
source of the bundle, remove the quarantine attribute from the unpacked tool
folder:

```bash
xattr -dr com.apple.quarantine csv2serasite
xattr -dr com.apple.quarantine excel2serasite
```

Unsigned Windows executables may trigger SmartScreen or antivirus warnings.
For trusted small-group distribution, verify the source of the archive and any
provided checksums before running the tool.

## Input Tables

### Site Owner

The site-owner table describes the data owner and contact person. In CSV mode,
it is provided with `-o`. In Excel mode, it is the `siteOwner` sheet.

Required columns:

```text
owner_codename
owner_fullname
person_firstname
person_lastname
person_mbox
```

Optional columns may include owner, person, institution, address, and
affiliation resource identifiers and contact metadata. Missing optional values
are omitted from SiteXML.

### Site Description

The site-description table has one row per site. In CSV mode, it is provided
with `-d`. In Excel mode, it is the `siteDescription` sheet.

Required columns:

```text
siteID
siteDescriptionID
latitude
longitude
```

Common optional columns:

```text
station
altitude
minDistanceFromStation
maxDistanceFromStation
siteTopography_schemaA
siteTopography_schemaB
siteMorphology
preferredSiteAnalysisID
preferredVelocityProfileID
overallQindex
```

Site-description indicator columns use these prefixes:

```text
siteClassEC8
bedrockDepth
h800
geologicalUnit
```

Examples:

```text
siteClassEC8_value
siteClassEC8Qindex1
siteClassEC8_title
siteClassEC8_firstAuthor
siteClassEC8_year
siteClassEC8_uri
bedrockDepth_value
bedrockDepth_uncertainty
h800_value
geologicalUnit_value
geologicalMapScale
geologicalUnitOGE
```

Station codes must use `network.station` notation, for example `XX.ABCD`.
Bare station codes are rejected because station codes are not globally unique.

### Analysis

The analysis table is optional. In CSV mode, it is provided with `-a`. In
Excel mode, it is the optional `analysis` sheet.

Required columns when the table is provided:

```text
siteID
siteDescriptionID
analysisID
```

Analysis indicator columns use these prefixes:

```text
resonanceFrequency
velocityS30
velocityProfileSet
```

Examples:

```text
resonanceFrequency_value
resonanceFrequency_uncertainty
resonanceFrequencyQindex1
resonanceFrequencyMethod1
velocityS30_value
velocityS30_uncertainty
velocityS30Qindex1
velocityS30Method1
velocityS30Method2
velocityS30MethodCombIndex
velocityS30ManualIndex
velocityProfileSetQindex1
sptLogsCount
cptLogsCount
boreholeLogsCount
```

If the analysis table is omitted, any `preferredSiteAnalysisID` and
`preferredVelocityProfileID` values from the site-description table are ignored
with warnings and are not written to generated SiteXML.

### Velocity Profiles

Velocity-profile metadata is optional. In CSV mode, pass one CSV file or a
folder of CSV files with `-p`. In Excel mode, pass one Excel file or a folder of
Excel files with `-p`.

Each row describes one velocity-profile layer.

Required columns when velocity-profile metadata is provided:

```text
siteID
analysisID
velocityProfileID
velocityS_value
layerTopDepth_value
```

Common optional columns:

```text
layerCount
layerBottomDepth_value
layerBottomDepth_uncertainty
velocityS_uncertainty
velocityP_value
velocityP_uncertainty
density_value
density_uncertainty
```

`velocityS_value` and `layerTopDepth_value` are required for every layer.
`layerBottomDepth_value` is optional. A missing bottom depth represents an
open-ended final layer.

If analysis metadata is provided but velocity-profile metadata is omitted, any
`preferredVelocityProfileID` values from the site-description table are ignored
with warnings and are not written to generated SiteXML. Valid
`preferredSiteAnalysisID` values are preserved.

### Quality-Index Sidecar

The quality-index sidecar is optional. In CSV mode, pass it with `-q`. In Excel
mode, include an optional `qualityIndex` sheet in the main workbook.

The sidecar is keyed by:

```text
siteID
```

Rows with an empty `siteID` or an unknown `siteID` are skipped with a warning
because quality-index metadata is optional enrichment.

Q_Index1 calculation criteria use `<indicator>_<criterion>` column names.
Supported indicator prefixes are:

```text
siteClassEC8
bedrockDepth
h800
geologicalUnit
resonanceFrequency
velocityS30
velocityProfileSet
```

Supported Q_Index1 criteria are:

```text
method
evaluation
reliability
report
```

Examples:

```text
siteClassEC8_method
siteClassEC8_evaluation
siteClassEC8_reliability
siteClassEC8_report
velocityS30_method
velocityS30_evaluation
velocityS30_reliability
velocityS30_report
```

Q_Index3 consistency columns are:

```text
f0_vs30
f0_bedrock_depth
f0_h800
vs30_h800
vs30_geology
```

Consistency values must be `0`, `1`, or empty:

- `1`: the indicator pair is consistent.
- `0`: the indicator pair is not consistent.
- empty: the pair is unavailable or was not evaluated.

The sidecar inputs are not stored in SiteXML. They are used immediately to
calculate schema-supported outputs:

- indicator-level `qualityIndex` values, also called Q_Index1;
- site-description `overallQindex`.

## Quality-Index Behavior

SiteXML stores calculated indicator-level quality indexes and the final overall
quality index. It does not store the detailed Q_Index1 criteria or Q_Index3
consistency inputs.

If both direct `*_qualityIndex` columns and a quality-index sidecar are provided:

- direct `*_qualityIndex` values are imported first;
- sidecar Q_Index1 criteria for an existing indicator recalculate and replace
  that indicator's direct `*_qualityIndex` value;
- sidecar blanks for an indicator leave that indicator's direct `*_qualityIndex`
  value unchanged;
- sidecar rows for sites without indicator objects are skipped.

The tools do not automatically synthesize `overallQindex` from direct
`*_qualityIndex` columns alone. To write `overallQindex`, use one of these explicit
workflows:

- provide `overallQindex` in the site-description table;
- provide the quality-index sidecar so the tool can calculate Q_Index1 and
  Q_Index3-derived results during import.

If no usable indicator data exists for a site, providing a quality-index
sidecar does not write a fake `<overallQindex>0</overallQindex>`.

## Validation Rules And Assumptions

The import process is intentionally strict about object identifiers and
relationships when the relevant metadata is provided.

The tools validate that:

- required input files or sheets are present;
- required columns are present;
- required row values are not empty;
- station codes use `network.station` notation;
- analysis rows point to the parent site description through
  `siteDescriptionID`;
- velocity-profile rows point to an existing `analysisID`;
- duplicate analysis and velocity-profile resource IDs are rejected;
- `preferredSiteAnalysisID`, when kept, points to an attached analysis;
- `preferredVelocityProfileID`, when kept, points to an attached velocity
  profile;
- when both preferred IDs are kept, the preferred velocity profile belongs to
  the preferred analysis;
- generated XML validates against the bundled SiteXML schema before it is
  written.

The tools do not guess or generate missing relationship IDs. For example, they
do not choose the first analysis as the preferred analysis and they do not
invent missing `analysisID` or `velocityProfileID` values.

One lenient rule applies to optional target tables:

- if analysis metadata is omitted, `preferredSiteAnalysisID` and
  `preferredVelocityProfileID` values from site-description input are ignored
  with warnings;
- if analysis metadata is present but velocity-profile metadata is omitted,
  `preferredVelocityProfileID` values are ignored with warnings.

Ignored preferred IDs are omitted from generated SiteXML.

## Recommended Workflows

### Minimal SiteXML

Use this when you only need owner and site-location metadata:

```bash
csv2serasite -o site_owner.csv -d site_description.csv -out sitexml_output
```

or:

```bash
excel2serasite sera_site.xlsx -out sitexml_output
```

The site-description table should not include preferred analysis or velocity
profile IDs unless you also provide the target analysis and velocity-profile
metadata.

### SiteXML With Analysis

Use this when you have analysis-level indicators such as resonance frequency or
Vs30 but no velocity-profile layers:

```bash
csv2serasite \
  -o site_owner.csv \
  -d site_description.csv \
  -a site_analysis.csv \
  -out sitexml_output
```

If `preferredVelocityProfileID` appears in the site-description table but no
velocity-profile metadata is provided, it is ignored with a warning.

### SiteXML With Velocity Profiles

Use this when you have analysis metadata and velocity-profile layers:

```bash
csv2serasite \
  -o site_owner.csv \
  -d site_description.csv \
  -a site_analysis.csv \
  -p velocity_profiles \
  -out sitexml_output
```

or:

```bash
excel2serasite sera_site.xlsx \
  -p velocity_profiles.xlsx \
  -out sitexml_output
```

Velocity-profile rows require both `siteID` and `analysisID`, so velocity
profiles are meaningful only with analysis metadata.

### SiteXML With Calculated Quality Indexes

Use this when you want the tool to calculate indicator quality indexes and
`overallQindex` from sidecar calculation inputs:

```bash
csv2serasite \
  -o site_owner.csv \
  -d site_description.csv \
  -a site_analysis.csv \
  -p velocity_profiles \
  -q quality_index.csv \
  -out sitexml_output
```

For Excel, include a `qualityIndex` sheet in the main workbook:

```bash
excel2serasite sera_site.xlsx \
  -p velocity_profiles.xlsx \
  -out sitexml_output
```

## Troubleshooting

If the command fails before writing XML, check:

- required files or sheets are present;
- required columns are spelled exactly as expected;
- CSV delimiter matches the file contents, usually `;`;
- every required row value is filled;
- every `siteID`, `siteDescriptionID`, `analysisID`, and `velocityProfileID`
  relationship points to a real object;
- `preferredSiteAnalysisID` and `preferredVelocityProfileID` point to objects
  that are actually provided in the input tables;
- velocity-profile layers include `velocityS_value` and
  `layerTopDepth_value`;
- Q_Index3 consistency values are only `0`, `1`, or empty.

Warnings usually mean optional enrichment was skipped or an unresolved optional
preferred ID was omitted from generated XML. Errors mean the input could not be
converted into a valid, schema-validated SiteXML document.
