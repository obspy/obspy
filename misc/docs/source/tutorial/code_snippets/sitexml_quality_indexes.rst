.. _quality-indexes:

SiteXML Quality Indexes
=======================

.. contents:: Contents
    :local:
    :depth: 2

This advanced guide covers quality-index calculations through the Python API
and CSV/Excel import tools. See :doc:`sitexml` for general SiteXML workflows
and :doc:`sitexml_tabular_input` for the shared input table specification.

Overview
--------

The SiteXML quality indexes follow the guidelines of the `SERA deliverable D7.2
<https://www.itsak.gr/SiteXML/
SERA_D7.2_Best-practice_for_site_characterization.pdf>`_ for describing the
reliability and consistency of site-characterization
metadata. ObsPy implements four related values:

* Q_Index1 describes the quality of one site indicator.
* Q_Index2 combines the Q_Index1 values available for one site.
* Q_Index3 describes consistency between pairs of site indicators.
* The overall quality index combines Q_Index2 and Q_Index3.

SiteXML stores only the **calculated indicator-level quality indexes (Q_Index1)
and the final overall quality index**. The extra calculation inputs for
Q_Index1 criteria and Q_Index3 consistency checks are not part of the SiteXML
object model. CSV and Excel imports can read these inputs from an optional
quality-index input table and apply them immediately.

Quality Index 1
---------------

Q_Index1 varies from 0 to 1 and refers to a single site indicator, such as
EC8 class, Vs30, resonance frequency, or a velocity profile. Four criteria are
used for the calculation:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Meaning
     - Accepted scoring values
   * - ``method``
     - Method of acquisition and analysis is documented in peer-reviewed
       literature.
     - ``"documented"`` or ``1`` gives A = 1. Any other value, including an
       empty cell, gives A = 0.
   * - ``evaluation``
     - Indicator was evaluated directly from field experiments.
     - ``"direct"`` or ``2`` gives B = 2. Any other value gives B = 0.
   * - ``reliability``
     - Confidence in the indicator value.
     - ``"yes"`` or ``1`` gives C = 1. ``"partial"`` or ``0.5`` gives
       C = 0.5. Any other value gives C = 0.
   * - ``report``
     - Field survey and data processing are documented in a report.
     - ``"yes"`` or ``1`` gives D = 1. ``"partial"`` or ``0.5`` gives
       D = 0.5. Any other value gives D = 0.

Q_Index1 is calculated as:

.. code-block:: text

    Q_Index1 = ((A + B + C) * D) / 4

Because the report criterion is multiplicative, a missing or zero report value
makes the Q_Index1 contribution zero.

Quality Index 2
---------------

Q_Index2 varies from 0 to 1 and combines the Q_Index1 values of all site
indicators evaluated at the target site. It is a weighted mean:

.. code-block:: text

    Q_Index2 = (w1 * Q_Index1_si1 + w2 * Q_Index1_si2 + ...) / (w1 + w2 + ...)

The denominator includes all seven indicator weights. Missing indicators and
missing Q_Index1 values contribute zero; the calculation does not reduce the
denominator to the indicators that happen to be present. Missing stored quality
indexes remain absent metadata until a calculation explicitly assigns a value.

The weights implemented in ObsPy are:

.. list-table::
   :header-rows: 1

   * - Site indicator
     - Weight
   * - Resonance frequency
     - 1
   * - Velocity profile
     - 1
   * - Velocity S30
     - 0.5
   * - Bedrock depth
     - 0.5
   * - H800
     - 0.5
   * - Geological unit
     - 0.5
   * - Soil class EC8
     - 0.25

.. important::

    When a site has multiple analyses, Q_Index2 uses the analysis selected by
    ``preferredSiteAnalysisID``. If no preferred analysis is set, the first
    analysis in document order is used. The velocity-profile contribution uses
    the
    ``VelocityProfileSet`` quality index attached to that analysis.

Quality Index 3
---------------

Q_Index3 varies from 0 to 1 and describes consistency between available pairs
of site indicators. Each provided consistency value is binary:

* ``1`` means the indicator pair is consistent.
* ``0`` means the indicator pair is not consistent.
* An empty value means the pair is unavailable or was not evaluated.

Q_Index3 is calculated as the average of only **the provided, non-empty
consistency values**:

.. code-block:: text

    Q_Index3 = (
        cons(f0, Vs30)
        + cons(f0, seismic_bedrock_depth)
        + cons(f0, engineering_bedrock_depth)
        + cons(H800, Vs30)
        + cons(Vs30, geology)
    ) / n

where ``n`` is the number of provided consistency values. If no consistency
values are provided, Q_Index3 is ``None``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Consistency pair
   * - ``f0_vs30``
     - Resonance frequency and Vs30.
   * - ``f0_bedrock_depth``
     - Resonance frequency and seismic bedrock depth.
   * - ``f0_h800``
     - Resonance frequency and engineering bedrock depth H800.
   * - ``vs30_h800``
     - Vs30 and H800.
   * - ``vs30_geology``
     - Vs30 and surface geology.

Overall Quality Index
---------------------

The overall quality index is the arithmetic mean of Q_Index2 and Q_Index3:

.. code-block:: text

    Overall_Quality_Index = (Q_Index2 + Q_Index3) / 2

If Q_Index2 is zero, the overall quality index is zero. If Q_Index3 is
``None``, it is treated as zero for the overall calculation.

Import From CSV Or Excel
------------------------

The indicator-level quality indexes and the final overall quality index stored
in SiteXML can be calculated during tabular import. The import helpers read the
required calculation parameters from an optional CSV or Excel quality-index
input table, apply them immediately, and store only the schema-supported
calculated results on the imported SiteXML objects.

Refer to :ref:`sitexml-tabular-quality-index-table` for the full column
reference of the quality-index input table.

.. important::

    When both direct (indicator-level) ``*_qualityIndex`` columns **and** a
    quality-index input table are provided, the following rules apply:

    * **Direct values are imported first.** ObsPy then applies the input table.
    * **Provided Q_Index1 criteria replace the direct value.** For an
      existing indicator, the input table criteria recalculate its stored
      quality index and replace the directly imported ``*_qualityIndex`` value.
    * **Missing Q_Index1 criteria preserve the direct value.** If the input
      table row has no Q_Index1 criteria for an indicator, its direct
      ``*_qualityIndex`` value remains unchanged.
    * **Sites without indicator objects are skipped.** Their input table rows
      have no target indicator to calculate or combine.

Automatic Calculation Of Overall QI
-----------------------------------

Tabular import is conservative about the calculation of the overall quality
index.

While importing SiteXML metadata from CSV or Excel input files:

* if the user provides indicator-level ``*_qualityIndex`` columns **but**
* does not provide an ``overallQindex`` value **and**
* does not provide the optional quality-index input table,

**then**
ObsPy imports and writes the indicator-level quality indexes only.   
It does **not** automatically synthesize ``overallQindex`` in the output XML.

This distinction is important because Q_Index2 gives missing indicator
Q_Index1 values a zero contribution, and Q_Index3 needs consistency inputs
that are only available from the quality-index input table or from explicit
Python arguments. Automatically calculating an overall value from partial
tabular metadata could therefore make absent information look like an evaluated
site quality.

Use one of these explicit workflows when the output XML should contain
``overallQindex``:

* calculate by yourself the overall quality index and provide the value in
  the ``overallQindex`` column in the site-description input table;
* provide the quality-index CSV file or Excel ``qualityIndex`` sheet, so
  ObsPy can calculate Q_Index1/Q_Index3-derived results during import;
* call :meth:`~obspy.io.sitexml.core.SERASite.calculate_overall_quality_index`
  yourself on the imported ``SERASite`` objects before writing SiteXML.

Other rules apply when calculating overall quality indexes through the
Python API:

* If no usable indicator data exists for a site, providing a quality-index
  input table does not create a synthetic ``<overallQindex>0</overallQindex>``.
* If no Q_Index3 consistency pairs are passed,
  ``calculate_overall_quality_index()`` treats Q_Index3 as zero for the final
  formula. Pass consistency values only when they have actually been evaluated.

Examples
--------

Initial Setup
~~~~~~~~~~~~~

The examples below use bundled fixtures. Run this setup first; subsequent
examples reuse these imports and paths. Examples that write files create an
output folder named ``sitexml_output`` in the current working directory.

.. code-block:: python

    from pathlib import Path

    from obspy.core.util import get_example_file
    from obspy.io.sitexml.sitexml import (
        sitedict_to_sitexml, sitexml_to_sitedict)
    from obspy.io.sitexml.tabular import csv_to_sera_site

    site_owner_csv = get_example_file("csv/site_owner.csv")
    data_dir = Path(site_owner_csv).parent
    filename = get_example_file("xml/full_sitexml.xml")
    Path("sitexml_output").mkdir(exist_ok=True)

Calculate quality index manually for a single site
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``SERASite`` object methods to calculate quality indexes directly:

.. code-block:: python

    from obspy.io.sitexml.sitexml import read_sitexml

    site = read_sitexml(filename)
    ec8_q1 = site.site_description.ec8.calculate_quality_index1(
        method="documented",
        evaluation="direct",
        reliability="yes",
        report="yes",
        assign=True)
    q2 = site.calculate_quality_index2()
    q3 = site.calculate_quality_index3(
        f0_vs30=1,
        f0_bedrock_depth=0,
        vs30_geology=1)
    overall = site.calculate_overall_quality_index(
        f0_vs30=1,
        f0_bedrock_depth=0,
        vs30_geology=1)

    print(ec8_q1, q2, q3, overall)

Calculate quality index manually for multiple sites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If site metadata for all your sites are produced using the same methodology,
you can calculate and apply manually the quality indexes:

.. code-block:: python

    from obspy.io.sitexml.quality_index import quality_index1
    from obspy.io.sitexml.sitexml import sitedict_to_sitexml

    ec8_quality_index = quality_index1(
        method="documented",
        evaluation="direct",
        reliability="yes",
        report="yes")
    h800_quality_index = quality_index1(
        method="documented",
        evaluation="direct",
        reliability="partial",
        report="yes")
    vs30_quality_index = quality_index1(
        method="documented",
        evaluation="direct",
        reliability="yes",
        report="partial")

    # sites is a dictionary of SERASite objects
    for site in sites.values():
        if site.site_description.ec8 is not None:
            site.site_description.ec8.quality_index = ec8_quality_index
        if site.site_description.h800 is not None:
            site.site_description.h800.quality_index = h800_quality_index

        analysis = site.get_preferred_analysis()
        if analysis is not None and analysis.velocity_s30 is not None:
            analysis.velocity_s30.quality_index = vs30_quality_index

        site.calculate_overall_quality_index()

    sitedict_to_sitexml(sites, "./sitexml_output")

Calculate the Overall Quality Index Explicitly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After importing SiteXML metadata from CSV files with some indicator-level
``*_qualityIndex`` values, calculate and store the overall quality index
explicitly before producing XML output:

.. code-block:: python

    from obspy.io.sitexml.sitexml import sitedict_to_sitexml

    sites = csv_to_sera_site(
        site_owner_csv=site_owner_csv,
        site_description_csv=data_dir / "site_description.csv",
        analysis_csv=data_dir / "site_analysis.csv",
        velocity_profiles_csv=data_dir / "velocity_profiles",
        delim=";")

    for site in sites.values():
        site.calculate_overall_quality_index(
            f0_vs30=1,
            f0_bedrock_depth=0,
            vs30_geology=1)

    sitedict_to_sitexml(sites, "./sitexml_output")

Apply quality-index inputs during CSV import
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    sites = csv_to_sera_site(
        site_owner_csv=site_owner_csv,
        site_description_csv=data_dir / "site_description.csv",
        analysis_csv=data_dir / "site_analysis.csv",
        velocity_profiles_csv=data_dir / "velocity_profiles",
        quality_index_csv=data_dir / "quality_index.csv",
        delim=";")

    site = sites["quakeml:domain.ab/site/001"]
    print(site.site_description.ec8.quality_index)
    print(site.site_description.overall_quality_index)

Apply quality-index inputs to existing SiteXML files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quality-index inputs can be applied to existing SiteXML files,
after reading them using Python API into a dictionary of SiteXML objects.

.. code-block:: python

    from obspy.io.sitexml.quality_index import apply_quality_index_csv

    sites = sitexml_to_sitedict(filename)
    apply_quality_index_csv(sites, data_dir / "quality_index.csv", delim=";")

    site = sites["quakeml:domain.ab/site/001"]
    print(site.site_description.overall_quality_index)

