====================
Working With SiteXML
====================

SiteXML stores site-characterization metadata for seismic stations and
locations. ObsPy reads SiteXML files into
:class:`~obspy.io.sitexml.core.SERASite` objects, writes those objects back to
schema-validated XML, and can build the same objects from CSV or Excel input
tables.

The examples below use the small fixtures bundled with ObsPy. They are
convenient for learning the file layout before using project-specific files.

Reading And Validating SiteXML
------------------------------

Use :func:`~obspy.io.sitexml.sitexml.read_sitexml` to read one SiteXML file.
The returned object contains the owner, the site description, and optional
analysis objects.

.. code-block:: python

    from obspy.core.util import get_example_file
    from obspy.io.sitexml.sitexml import read_sitexml

    filename = get_example_file("full_sitexml.xml")
    site = read_sitexml(filename)

    print(site.resource_id)
    print(site.site_owner.owner_codename)
    print(site.site_description.station_code)
    print(len(site.analysis))

SiteXML validation is available separately with
:func:`~obspy.io.sitexml.sitexml.validate_sitexml`. It returns a boolean and a
list of schema-validation errors.

.. code-block:: python

    from obspy.io.sitexml.sitexml import validate_sitexml

    valid, errors = validate_sitexml(filename)
    if not valid:
        for error in errors:
            print(error)

Writing SiteXML
---------------

Use :func:`~obspy.io.sitexml.sitexml.write_sitexml` to write one
``SERASite``. Validation is enabled by default, so schema problems are reported
before the file is accepted.

.. code-block:: python

    from pathlib import Path
    from tempfile import TemporaryDirectory

    from obspy.io.sitexml.sitexml import write_sitexml

    with TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "site.xml"
        write_sitexml(site, output_file, validate=True)

Reading Or Writing Site Dictionaries
------------------------------------

A directory of SiteXML files can be read as a dictionary keyed by
``SERASite.resource_id``. The reverse helper writes each site to its default
SiteXML filename.

.. code-block:: python

    from pathlib import Path
    from tempfile import TemporaryDirectory

    from obspy.io.sitexml.sitexml import (
        sitedict_to_sitexml, sitexml_to_sitedict)

    sites = sitexml_to_sitedict(filename)

    with TemporaryDirectory() as tmpdir:
        sitedict_to_sitexml(sites, tmpdir)
        print(sorted(Path(tmpdir).glob("*.xml")))

CSV Input Files
---------------

CSV import uses one required owner table, one required site-description table,
and optional analysis, velocity-profile, and quality-index sidecar tables. The
default delimiter is ``;``.

The owner table describes the data owner and contact person. The minimum
required columns are:

.. code-block:: text

    owner_codename;owner_fullname;person_firstname;person_lastname;person_mbox

The site-description table has one row per site. The minimum required columns
are:

.. code-block:: text

    siteID;siteDescriptionID;latitude;longitude

Common optional columns include station association and site indicators:

.. code-block:: text

    station;altitude;siteClassEC8_value;bedrockDepth_value;h800_value
    geologicalUnit_value;preferredSiteAnalysisID;preferredVelocityProfileID

Station codes use ``network.station`` notation, for example ``XX.ABCD``.
Bare station codes are rejected because station codes are not globally unique.

The analysis table is optional. When present, each row must identify the site,
site description, and analysis:

.. code-block:: text

    siteID;siteDescriptionID;analysisID

Analysis rows may also contain indicator columns such as:

.. code-block:: text

    resonanceFrequency_value;resonanceFrequencyMethod1
    velocityS30_value;velocityS30_uncertainty;velocityS30Method1
    sptLogsCount;cptLogsCount;boreholeLogsCount

Velocity-profile input can be one CSV file or a directory of CSV files. Each
row describes one layer in one velocity profile:

.. code-block:: text

    siteID;analysisID;velocityProfileID;layerCount
    density_value;density_uncertainty
    velocityP_value;velocityP_uncertainty
    velocityS_value;velocityS_uncertainty
    layerTopDepth_value;layerTopDepth_uncertainty
    layerBottomDepth_value;layerBottomDepth_uncertainty

Indicator reference metadata uses the indicator name as a prefix. For example,
``velocityS30_title`` and ``velocityS30_firstAuthor`` create a literature
source for the Vs30 indicator, while ``velocityS30_uri`` and
``velocityS30_description`` create an external reference.

Importing CSV
-------------

Use :func:`~obspy.io.sitexml.read_csv.csv_to_sera_site` to build a dictionary
of ``SERASite`` objects from CSV files.

.. code-block:: python

    from pathlib import Path

    from obspy.core.util import get_example_file
    from obspy.io.sitexml.read_csv import csv_to_sera_site

    site_owner_csv = get_example_file("site_owner.csv")
    data_dir = Path(site_owner_csv).parent

    sites = csv_to_sera_site(
        site_owner_csv=site_owner_csv,
        site_description_csv=data_dir / "site_description.csv",
        analysis_csv=data_dir / "site_analysis.csv",
        velocity_profiles_csv=data_dir / "velocity_profiles",
        delim=";")

    site = sites["quakeml:domain.ab/site/001"]
    analysis = site.analysis[0]
    profile = analysis.velocity_profile_survey.velocity_profiles[0]

    print(site.site_description.latitude)
    print(analysis.velocity_s30.value.value)
    print(profile.layer_count)

Importing Excel
---------------

Excel import uses the same logical tables. The main workbook should contain:

* ``siteOwner``: required owner and contact metadata.
* ``siteDescription``: required site rows.
* ``analysis``: optional analysis rows.
* ``qualityIndex``: optional quality-index calculation inputs.

Velocity profiles can be supplied in a separate workbook or a directory of
workbooks. Each non-empty velocity-profile sheet must contain a ``siteID``
column and the same layer columns used by CSV import.

.. code-block:: python

    from pathlib import Path

    from obspy.core.util import get_example_file
    from obspy.io.sitexml.read_csv import excel_to_sera_site

    excel_file = get_example_file("sera_site_all.xlsx")
    data_dir = Path(excel_file).parent

    sites = excel_to_sera_site(
        excel_file,
        velocity_profiles=data_dir / "velocity_profiles.xlsx")

    site = sites["quakeml:domain.ab/site/001"]
    print(site.site_description.station_code)

Quality Indexes
---------------

SiteXML stores calculated indicator-level quality indexes and the final
overall quality index. The extra calculation inputs for Q_Index1 criteria and
Q_Index3 consistency checks are not part of the SiteXML object model, so CSV
and Excel imports can read them from an optional sidecar table and apply them
immediately.

The quality-index sidecar table is keyed by ``siteID``. Q_Index1 criteria use
``<indicator>_<criterion>`` column names:

.. code-block:: text

    siteClassEC8_method;siteClassEC8_evaluation
    siteClassEC8_reliability;siteClassEC8_report
    velocityS30_method;velocityS30_evaluation
    velocityS30_reliability;velocityS30_report

Supported indicator prefixes are ``siteClassEC8``, ``bedrockDepth``, ``h800``,
``geologicalUnit``, ``resonanceFrequency``, ``velocityS30``, and
``velocityProfile``. Q_Index3 consistency columns are:

.. code-block:: text

    f0_vs30;f0_bedrock_depth;f0_h800;vs30_h800;vs30_geology

Consistency values must be ``0`` or ``1``. Empty cells mean that the pair was
not evaluated.

Apply quality-index inputs during CSV import:

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

Quality-index inputs can also be applied later to an existing dictionary, for
example after reading SiteXML files:

.. code-block:: python

    from obspy.io.sitexml.quality_index import apply_quality_index_csv

    sites = sitexml_to_sitedict(filename)
    apply_quality_index_csv(sites, data_dir / "quality_index.csv", delim=";")

    site = sites["quakeml:domain.ab/site/001"]
    print(site.site_description.overall_quality_index)

The same formulas are available directly through convenience methods on
``SERASite``:

.. code-block:: python

    q2 = site.calculate_quality_index2()
    q3 = site.calculate_quality_index3(
        f0_vs30=1,
        f0_bedrock_depth=0,
        vs30_geology=1)
    overall = site.calculate_overall_quality_index(
        f0_vs30=1,
        f0_bedrock_depth=0,
        vs30_geology=1)

    print(q2, q3, overall)
