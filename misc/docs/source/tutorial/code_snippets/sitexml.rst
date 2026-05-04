====================
Working With SiteXML
====================

SiteXML stores site-characterization metadata for seismic stations and
locations. ObsPy reads local SiteXML files and published SiteXML URLs into
:class:`~obspy.io.sitexml.core.SERASite` objects, writes those objects back to
schema-validated XML, and can build the same objects from CSV or Excel input
tables.

The examples below use the small fixtures bundled with ObsPy. They are
convenient for learning the file layout before using project-specific files.

Reading And Validating SiteXML
------------------------------

Use :func:`~obspy.io.sitexml.sitexml.read_sitexml` to read one SiteXML file or
HTTP(S) URL. The returned object contains the owner, the site description, and
optional analysis objects.

.. code-block:: python

    from obspy.core.util import get_example_file
    from obspy.io.sitexml.sitexml import read_sitexml

    filename = get_example_file("full_sitexml.xml")
    site = read_sitexml(filename)

    print(site.resource_id)
    print(site.site_owner.owner_codename)
    print(site.site_description.station_code)
    print(len(site.analysis))

For a published SiteXML document, pass the URL directly. ObsPy retrieves the
remote XML and uses the same schema validation and parser as it does for local
files.

.. code-block:: python

    from obspy.io.sitexml.sitexml import read_sitexml

    site = read_sitexml("https://example.org/sitexml/XX.ABCD.xml")

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

When no output path or file-like object is supplied, ObsPy writes the file in
the current directory using using a default filename pattern. The filename
contains the same serialization date used for the root ``creationTime``
metadata, for example ``Site_XX.ABCD_12-01-2026.xml``.

.. code-block:: python

    from obspy.io.sitexml.sitexml import write_sitexml

    write_sitexml(site, validate=True)
    write_sitexml(site, "./site.xml", validate=True)

SiteXML can also store optional document revision history. Root
``creationTime`` records when this XML file is generated, while each
``Revision.revision_time`` records when the described document revision
occurred. These times can differ when metadata changes are reviewed, migrated,
or exported later.

.. code-block:: python

    site.add_revision(
        revision_time="2026-05-02T12:00:00Z",
        description="Updated velocity profile and quality indexes.",
        author="ORFEUS",
        version="2026-05-02",
        previous_version=(
            "https://example.org/sitexml/"
            "Site_XX.ABCD_01-05-2026.xml"))

    write_sitexml(site, validate=True)

Associating SiteXML With StationXML
-----------------------------------

SiteXML can be associated with the corresponding StationXML by adding the
current published SiteXML URL as a station-level StationXML
``ExternalReference``. The SiteXML file itself stores the station association
in ``SiteDescription.station_code`` using ``network.station`` notation, for
example ``XX.ABCD``. SiteXML filenames can include the document creation date,
so the StationXML reference is treated as the current pointer to the latest
published SiteXML document.

Use :func:`~obspy.io.sitexml.sitexml.write_stationxml_reference` to add or
update the StationXML reference that points to the SiteXML URL.

Input Options
~~~~~~~~~~~~~

The :func:`~obspy.io.sitexml.sitexml.write_stationxml_reference` supports 
three input options that are mutually exclusive:

* ``input_path`` - Read a local StationXML file 
* ``client`` - Use an already initialized FDSN Web Service client to fetch StationXML
* ``datacenter`` - Provide the datacenter of the remote FDSN Web Service from 
  which to fetch StationXML.

Output Options
~~~~~~~~~~~~~

The helper always returns an updated :class:`~obspy.core.inventory.inventory.Inventory` 
object. Optionally, it can write an output StationXML file if ``output_path`` is provided.

Local StationXML Input
~~~~~~~~~~~~~~~~~~~~~~

If the StationXML is already available locally, provide the filename using 
``input_path``. This is useful for newly created StationXML files, offline
workflows, or examples where you do not want to contact an FDSN data center:

.. code-block:: python

    from obspy import UTCDateTime
    from obspy.io.sitexml.sitexml import write_stationxml_reference

    inventory = write_stationxml_reference(
        station_code="XX.ABCD",
        sitexml_url="https://example.org/sitexml/Site_XX.ABCD_02-05-2026.xml",
        input_path="XX.ABCD.stationxml.xml",
        output_path="XX.ABCD.with_sitexml.stationxml.xml",
        added_time=UTCDateTime(2026, 5, 2))

Basic Remote Workflow
~~~~~~~~~~~~~~~~~~~~~

When ``input_path`` is omitted, StationXML is fetched from an FDSN data center.
The helper splits the ``network.station`` code into FDSN query arguments and
requests StationXML at ``level="response"``.

.. code-block:: python

    from obspy.io.sitexml.sitexml import write_stationxml_reference

    inventory = write_stationxml_reference(
        station_code="XX.ABCD",
        sitexml_url="https://example.org/sitexml/Site_XX.ABCD_02-05-2026.xml",
        datacenter="ORFEUS")

    station = inventory.select(network="XX", station="ABCD")[0][0]
    print(station.external_references[-1].description)

This returns the modified inventory but does not write an output file. To write the
updated StationXML directly, pass ``output_path``:

.. code-block:: python

    write_stationxml_reference(
        station_code="XX.ABCD",
        sitexml_url="https://example.org/sitexml/Site_XX.ABCD_02-05-2026.xml",
        output_path="XX.ABCD.stationxml.xml",
        datacenter="ORFEUS")



Keeping The Reference Current
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``sitexml_url`` should be the URL where the current SiteXML document is
published.
By default, :func:`~obspy.io.sitexml.sitexml.write_stationxml_reference`
replaces an existing SiteXML station reference instead of appending another
dated URL. It replaces references written by this helper and manually added
references whose URL basename follows the default SiteXML station filename
pattern, for example ``Site_XX.ABCD_02-05-2026.xml``. Other station external
references are left untouched.

If the StationXML file should preserve a history of SiteXML URLs, disable
replacement explicitly:

.. code-block:: python

    from obspy.io.sitexml.sitexml import write_stationxml_reference

    inventory = write_stationxml_reference(
        station_code="XX.ABCD",
        sitexml_url="https://example.org/sitexml/Site_XX.ABCD_02-05-2026.xml",
        input_path="XX.ABCD.stationxml.xml",
        replace_existing=False)

Reference Description And Date
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The StationXML ``ExternalReference.description`` records when the reference was
added, for example ``SERA SiteXML site characterization; added 2026-05-02``.
By default the added date is the current UTC date. If you need reproducible
output or want to use the SiteXML document creation date, pass
``added_time``. If the default external-reference text is not specific enough,
pass a custom ``description``; the added-date marker is appended to the custom
text:

.. code-block:: python

    inventory = write_stationxml_reference(
        station_code="XX.ABCD",
        sitexml_url="https://example.org/sitexml/Site_XX.ABCD_02-05-2026.xml",
        description="SERA SiteXML site characterization for XX.ABCD",
        added_time=UTCDateTime(2026, 5, 2))

    inventory.write("XX.ABCD.stationxml.xml", format="STATIONXML")

Bare station codes such as ``ABCD`` should not be used for this association,
because StationXML station codes are only unique together with their network
code.

Creating A SiteXML File From Scratch
------------------------------------

Creating a SiteXML file directly in Python is useful when metadata already
exists in application objects or another structured source. It is not always
shorter than preparing CSV or Excel input, but it keeps the object graph
explicit and lets ObsPy validate the result against the bundled SiteXML schema
when writing.

A minimal SiteXML document needs a top-level
:class:`~obspy.io.sitexml.core.SERASite`, a required
:class:`~obspy.io.sitexml.core.SERASiteOwner`, and a required
:class:`~obspy.io.sitexml.core.SiteDescription`. Analyses and site indicators
can then be attached as needed. Resource identifiers are ordinary strings, but
they should be stable and unique, and relationship fields such as
``site_descriptionID`` and ``preferred_site_analysisID`` must point to existing
objects.

The following illustration shows the relationships between the most basic 
SiteXML objects.

.. figure:: /_images/SERASite.png

The following example creates a small SiteXML site with owner/contact
metadata, location metadata, a few site-description indicators, one analysis,
and a Vs30 indicator:

.. code-block:: python

    from pathlib import Path
    from tempfile import TemporaryDirectory

    from obspy.io.sitexml.core import (
        Analysis, BedrockDepth, EC8, H800, SERASite, SERASiteOwner,
        SiteDescription, ValueWithUncertainty, VelocityS30)
    from obspy.io.sitexml.sitexml import validate_sitexml, write_sitexml

    site_owner = SERASiteOwner(
        owner_codename="EXAMPLE",
        owner_fullname="Example Site Owner",
        person_firstname="Ada",
        person_lastname="Lovelace",
        person_mbox="ada@example.org")

    site_description = SiteDescription(
        resource_id="quakeml:example.org/site_description/001",
        latitude=45.137174,
        longitude=5.998905,
        station_code="XX.ABCD",
        ec8=EC8("B"),
        bedrock_depth=BedrockDepth(
            ValueWithUncertainty(40.0, uncertainty=6.0)),
        h800=H800(ValueWithUncertainty(10.0, uncertainty=1.0)),
        preferred_site_analysisID="quakeml:example.org/analysis/001")

    analysis = Analysis(
        resource_id="quakeml:example.org/analysis/001",
        site_descriptionID="quakeml:example.org/site_description/001",
        velocity_s30=VelocityS30(
            ValueWithUncertainty(620.0, uncertainty=18.0),
            methods=["MASW"]))

    site = SERASite(
        resource_id="quakeml:example.org/site/001",
        site_owner=site_owner,
        site_description=site_description,
        analysis=[analysis])

    site.add_revision(
        revision_time="2026-05-02T12:00:00Z",
        description="Created initial SiteXML document.",
        author="Some Author (email@example.com)")

    with TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "site.xml"
        write_sitexml(site, output_file, validate=True)

        valid, errors = validate_sitexml(output_file)
        if not valid:
            for error in errors:
                print(error)

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

Resource Identifiers And Preferred IDs
--------------------------------------

SiteXML uses resource identifiers to connect the top-level site object with
its nested site description, analyses, velocity profiles, owner, and contact
metadata. ObsPy stores these identifiers internally as plain strings. Values
passed as ObsPy ``ResourceIdentifier`` objects are accepted as input
conveniences and normalized to their string IDs.

The current schema keeps the identifier pattern intentionally relaxed, but
project data should still use stable, unique, URI-like identifiers. The bundled
fixtures use a QuakeML-style convention:

.. code-block:: text

    quakeml:domain.ab/site/001
    quakeml:domain.ab/site_description/001
    quakeml:domain.ab/analysis/001
    quakeml:domain.ab/velocity_profile/001
    quakeml:domain.ab/siteOwner/001

The part after the final slash usually identifies the object within its
collection. The path segment before it names the object type. Keeping this
shape consistent makes the XML and tabular files easier to audit, but the
important requirement is that every relationship column points to an existing
object ID.

The most important relationship and preferred-ID columns are:

* ``siteID`` identifies the top-level ``SERASite`` and is used as the key in
  dictionaries returned by CSV and Excel import.
* ``siteDescriptionID`` identifies the ``SiteDescription`` object. Analysis
  rows must repeat the corresponding ``siteDescriptionID`` so the analysis can
  be checked against the site description it describes.
* ``analysisID`` identifies one analysis. A site can have multiple analyses.
* ``velocityProfileID`` identifies one velocity profile under a specific
  analysis. An analysis can have multiple velocity profiles.
* ``preferredSiteAnalysisID`` selects the analysis to use when a site has more
  than one analysis. Quality-index calculations use this preferred analysis
  when it is present.
* ``preferredVelocityProfileID`` selects the preferred velocity profile. **If
  both preferred IDs are present, the preferred velocity profile must belong to
  the preferred analysis.**

In Python, use ``SERASite.get_preferred_analysis()`` and
``SERASite.get_preferred_velocity_profile()`` to retrieve the selected objects.
If no preferred ID is declared, these methods return the first available
analysis or velocity profile without changing the missing preferred-ID
metadata.

The generic CSV and Excel importers **do not generate missing relationship IDs**.
For example, they will not guess ``analysisID`` or
``preferredSiteAnalysisID`` values, because that would hide user intent when a
site has several analyses or velocity profiles.

CSV Input Files
---------------

CSV import uses one required owner table, one required site-description table,
and optional analysis, velocity-profile, and quality-index sidecar tables. The
default delimiter is ``;``.

The **owner** table describes the data owner and contact person. The minimum
required columns are:

.. code-block:: text

    owner_codename;owner_fullname;person_firstname;person_lastname;person_mbox

The **site-description** table has **one row per site**. The minimum required columns
are:

.. code-block:: text

    siteID;siteDescriptionID;latitude;longitude

Common optional columns include station association, site indicators and the 
resource ids of the preferred analysis and velocity profile:

.. code-block:: text

    station;altitude;siteClassEC8_value;bedrockDepth_value;h800_value
    geologicalUnit_value;preferredSiteAnalysisID;preferredVelocityProfileID

Station codes use ``network.station`` notation, for example ``XX.ABCD``.
Bare station codes are rejected because station codes are not globally unique.

The **analysis** table is optional. When present, each row must identify the site,
site description, and analysis:

.. code-block:: text

    siteID;siteDescriptionID;analysisID

Analysis rows may also contain indicator columns such as:

.. code-block:: text

    resonanceFrequency_value;resonanceFrequencyMethod1
    velocityS30_value;velocityS30_uncertainty;velocityS30Method1
    sptLogsCount;cptLogsCount;boreholeLogsCount

**Velocity-profile** input can be one CSV file or a directory of CSV files. Each
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

Use :func:`~obspy.io.sitexml.tabular.csv_to_sera_site` to build a dictionary
of ``SERASite`` objects from CSV files.

.. code-block:: python

    from pathlib import Path

    from obspy.core.util import get_example_file
    from obspy.io.sitexml.tabular import csv_to_sera_site

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
    profile = analysis.velocity_profile_set.velocity_profiles[0]

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
    from obspy.io.sitexml.tabular import excel_to_sera_site

    excel_file = get_example_file("sera_site_all.xlsx")
    data_dir = Path(excel_file).parent

    sites = excel_to_sera_site(
        excel_file,
        velocity_profiles=data_dir / "velocity_profiles.xlsx")

    site = sites["quakeml:domain.ab/site/001"]
    print(site.site_description.station_code)

Quality Indexes
---------------

Overview
~~~~~~~~

The SiteXML quality indexes follow the guidelines of the **SERA deliverable 
D7.1** for describing the reliability and consistency of site-characterization 
metadata. ObsPy implements four related values:

* Q_Index1 describes the quality of one site indicator.
* Q_Index2 combines the Q_Index1 values available for one site.
* Q_Index3 describes consistency between pairs of site indicators.
* The overall quality index combines Q_Index2 and Q_Index3.

SiteXML stores only the **calculated indicator-level quality indexes (Q_Index1)
and the final overall quality index**. The extra calculation inputs for Q_Index1 
criteria and Q_Index3 consistency checks are not part of the SiteXML object model. 
CSV and Excel imports can read these inputs from an optional sidecar table and 
apply them immediately.

Quality Index 1
~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~

Q_Index2 varies from 0 to 1 and combines the Q_Index1 values of all site
indicators evaluated at the target site. It is a weighted mean:

.. code-block:: text

    Q_Index2 = (w1 * Q_Index1_si1 + w2 * Q_Index1_si2 + ...) / (w1 + w2 + ...)

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

When a site has multiple analyses, Q_Index2 uses the analysis selected by
``preferredSiteAnalysisID``. If no preferred analysis is set, the first
analysis in document order is used. The velocity-profile contribution uses the
``VelocityProfileSet`` quality index attached to that analysis.

Quality Index 3
~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~

The overall quality index is the arithmetic mean of Q_Index2 and Q_Index3:

.. code-block:: text

    Overall_Quality_Index = (Q_Index2 + Q_Index3) / 2

If Q_Index2 is zero, the overall quality index is zero. If Q_Index3 is
``None``, it is treated as zero for the overall calculation.

Import From CSV Or Excel
~~~~~~~~~~~~~~~~~~~~~~~~

The indicator-level quality indexes and the final overall quality index stored
in SiteXML can be calculated during tabular import. The import helpers read the
required calculation parameters from an optional CSV or Excel sidecar table,
apply them immediately, and store only the schema-supported calculated results
on the imported SiteXML objects.

The quality-index sidecar table is keyed by ``siteID``. Q_Index1 criteria use
``<indicator>_<criterion>`` column names:

.. code-block:: text

    siteClassEC8_method;siteClassEC8_evaluation
    siteClassEC8_reliability;siteClassEC8_report
    velocityS30_method;velocityS30_evaluation
    velocityS30_reliability;velocityS30_report

Supported indicator prefixes are ``siteClassEC8``, ``bedrockDepth``, ``h800``,
``geologicalUnit``, ``resonanceFrequency``, ``velocityS30``, and
``velocityProfileSet``.

Q_Index3 consistency columns are:

.. code-block:: text

    f0_vs30;f0_bedrock_depth;f0_h800;vs30_h800;vs30_geology

For CSV import, pass the sidecar file as ``quality_index_csv``. For Excel
import, include an optional ``qualityIndex`` sheet in the workbook.

Examples
~~~~~~~~

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
