# SiteXML Quality Indexes Guide

This document describes the SiteXML quality-index calculations implemented by
ObsPy and used by the standalone `csv2sitexml` and `excel2sitexml` import
tools.

## Table Of Contents

- [Overview](#overview)
- [Quality Index 1](#quality-index-1)
- [Quality Index 2](#quality-index-2)
- [Quality Index 3](#quality-index-3)
- [Overall Quality Index](#overall-quality-index)
- [Import From CSV Or Excel](#import-from-csv-or-excel)
- [Automatic Calculation Of Overall QI](#automatic-calculation-of-overall-qi)

## Overview

The SiteXML quality indexes follow the guidelines of the **[SERA deliverable
D7.2](https://www.itsak.gr/SiteXML/SERA_D7.2_Best-practice_for_site_characterization.pdf)** 
for describing the reliability and consistency of site-characterization metadata. 
ObsPy implements four related values:

- Q_Index1 describes the quality of one site indicator.
- Q_Index2 combines the Q_Index1 values available for one site.
- Q_Index3 describes consistency between pairs of site indicators.
- The overall quality index combines Q_Index2 and Q_Index3.

SiteXML stores only the **calculated indicator-level quality indexes
(Q_Index1) and the final overall quality index**. The extra calculation inputs
for Q_Index1 criteria and Q_Index3 consistency checks are not part of the
SiteXML object model. CSV and Excel imports can read these inputs from an
optional quality-index table and apply them immediately.

## Quality Index 1

Q_Index1 varies from 0 to 1 and refers to a single site indicator, such as EC8
class, Vs30, resonance frequency, or a velocity profile. Four criteria are used
for the calculation:

| Parameter | Meaning | Accepted scoring values |
| --- | --- | --- |
| `method` | Method of acquisition and analysis is documented in peer-reviewed literature. | `"documented"` or `1` gives A = 1. Any other value, including an empty cell, gives A = 0. |
| `evaluation` | Indicator was evaluated directly from field experiments. | `"direct"` or `2` gives B = 2. Any other value gives B = 0. |
| `reliability` | Confidence in the indicator value. | `"yes"` or `1` gives C = 1. `"partial"` or `0.5` gives C = 0.5. Any other value gives C = 0. |
| `report` | Field survey and data processing are documented in a report. | `"yes"` or `1` gives D = 1. `"partial"` or `0.5` gives D = 0.5. Any other value gives D = 0. |

Q_Index1 is calculated as:

```text
Q_Index1 = ((A + B + C) * D) / 4
```

Because the report criterion is multiplicative, a missing or zero report value
makes the Q_Index1 contribution zero.

## Quality Index 2

Q_Index2 varies from 0 to 1 and combines the Q_Index1 values of all site
indicators evaluated at the target site. It is a weighted mean:

```text
Q_Index2 = (w1 * Q_Index1_si1 + w2 * Q_Index1_si2 + ...) / (w1 + w2 + ...)
```

The weights implemented in ObsPy are:

| Site indicator | Weight |
| --- | --- |
| Resonance frequency | 1 |
| Velocity profile | 1 |
| Velocity S30 | 0.5 |
| Bedrock depth | 0.5 |
| H800 | 0.5 |
| Geological unit | 0.5 |
| Soil class EC8 | 0.25 |

When a site has multiple analyses, Q_Index2 uses the analysis selected by
`preferredSiteAnalysisID`. If no preferred analysis is set, the first analysis
in document order is used. The velocity-profile contribution uses the
`VelocityProfileSet` quality index attached to that analysis.

## Quality Index 3

Q_Index3 varies from 0 to 1 and describes consistency between available pairs of
site indicators. Each provided consistency value is binary:

- `1` means the indicator pair is consistent.
- `0` means the indicator pair is not consistent.
- An empty value means the pair is unavailable or was not evaluated.

Q_Index3 is calculated as the average of only **the provided, non-empty
consistency values**:

```text
Q_Index3 = (
    cons(f0, Vs30)
    + cons(f0, seismic_bedrock_depth)
    + cons(f0, engineering_bedrock_depth)
    + cons(H800, Vs30)
    + cons(Vs30, geology)
) / n
```

where `n` is the number of provided consistency values. If no consistency
values are provided, Q_Index3 is `None`.

| Parameter | Consistency pair |
| --- | --- |
| `f0_vs30` | Resonance frequency and Vs30. |
| `f0_bedrock_depth` | Resonance frequency and seismic bedrock depth. |
| `f0_h800` | Resonance frequency and engineering bedrock depth H800. |
| `vs30_h800` | Vs30 and H800. |
| `vs30_geology` | Vs30 and surface geology. |

## Overall Quality Index

The overall quality index is the arithmetic mean of Q_Index2 and Q_Index3:

```text
Overall_Quality_Index = (Q_Index2 + Q_Index3) / 2
```

If Q_Index2 is zero, the overall quality index is zero. If Q_Index3 is `None`,
it is treated as zero for the overall calculation.

## Import From CSV Or Excel

The indicator-level quality indexes and the final overall quality index stored
in SiteXML can be calculated during tabular import. The import helpers read the
required calculation parameters from an optional CSV or Excel table,
apply them immediately, and store only the schema-supported calculated results
on the imported SiteXML objects.

## Automatic Calculation Of Overall QI

Tabular import is conservative about the overall quality index. If a CSV file
or Excel sheet contains indicator-level `*_qualityIndex` columns but does not
provide an `overallQindex` value and does not provide the optional quality-index
table, ObsPy imports and writes the indicator-level quality indexes
only. It does **not** automatically synthesize `overallQindex` in the output
XML.

This distinction is important because Q_Index2 gives missing indicator Q_Index1
values a zero contribution, and Q_Index3 needs consistency inputs that are only
available from the quality-index table or from explicit Python arguments.
Automatically calculating an overall value from partial tabular metadata could
therefore make absent information look like an evaluated site quality.

Use one of these explicit workflows when the output XML should contain
`overallQindex`:

- provide an `overallQindex` column in the site-description table;
- provide the quality-index sidecar CSV or Excel `qualityIndex` sheet, so ObsPy
  can calculate Q_Index1/Q_Index3-derived results during import;
