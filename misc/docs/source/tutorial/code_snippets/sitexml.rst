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

    site = read_sitexml("https://example.org/sitexml/Site_XX.ABCD.xml")

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

    from obspy.io.sitexml.sitexml import write_sitexml

    write_sitexml(site, validate=True)
    write_sitexml(site, "./site.xml", validate=True)

When no output path or file-like object is supplied, ObsPy writes the file in
the current directory using the default filename pattern described in
:ref:`sitexml-filenames-and-revision-history` **dated with current date**.

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

.. _sitexml-filenames-and-revision-history:

SiteXML filenames and revision history
--------------------------------------

SiteXML filenames identify the site and, for ObsPy generated files, **the
publication date**. For a station-backed site, the base filename is built
from ``Site_`` and the ``network.station`` code stored in
``SiteDescription.station_code``:

.. code-block:: text

    Site_XX.ABCD.xml                    # Default filename - no date
    Site_XX.ABCD_05-05-2026.xml         # Default filename - publication date

For sites without an associated station code, the base filename is derived from
the ``SERASite.resource_id`` using the compact SiteXML site identifier:

.. code-block:: text

    quakeml:example.org/site/001
    Site_example.org.001.xml
    Site_example.org.001_05-05-2026.xml

Use
:meth:`~obspy.io.sitexml.core.SERASite.get_sitexml_filename` when you need to
preview or choose the default filename yourself. If you provide a date, this will
be appended in the base filename.

.. code-block:: python

    output_file = site.get_sitexml_filename()
    # Output filename is Site_XX.ABCD.xml 

    output_file = site.get_sitexml_filename("2026-05-02")
    # Output filename is Site_XX.ABCD_02-05-2026.xml 
    
    write_sitexml(site, output_file, validate=True)

.. note::

    The date appended to the filename is the document's publication date. It
    is not necessarily the date when the site was measured, or
    characterized.

SiteXML can also store optional document revision history. 
``SERASite.created`` records when this XML file is generated, while each
``Revision.revision_time`` records when the described document revision
occurred. These times can differ when metadata changes are reviewed, migrated,
or exported later. Revision entries can include the update time, a short
description, the responsible author or institution, a version label, and the
URL of the previous published version.

.. code-block:: python

    site.add_revision(
        revision_time="2026-05-02T12:00:00Z",
        description="Updated velocity profile and quality indexes.",
        author="Author Name",
        version="2026-05-02",
        previous_version=(
            "https://example.org/sitexml/Site_XX.ABCD_01-05-2026.xml"))


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
* ``preferredVelocityProfileID`` selects the preferred velocity profile. 

.. important::

    If both preferred IDs are present, the preferred velocity profile must 
    belong to the preferred analysis.

In Python, use ``SERASite.get_preferred_analysis()`` and
``SERASite.get_preferred_velocity_profile()`` to retrieve the selected objects.
If no preferred ID is declared, these methods return the first available
analysis or velocity profile without changing the missing preferred-ID
metadata.

Use the object-level helpers when editing these relationships in Python.
``SERASite.add_analysis(...)`` creates an analysis already tied to the site's
``site_description.resource_id``. ``SERASite.set_preferred_analysis(...)`` and
``SERASite.set_preferred_velocity_profile(...)`` validate that the referenced
objects are attached to the site before updating the preferred IDs.

.. code-block:: python

    from obspy.io.sitexml.core import (
        ValueWithUncertainty, VelocityProfile, VelocityProfileData)

    analysis = site.add_analysis(
        resource_id="quakeml:domain.ab/analysis/002",
        set_preferred=True)

    profile = VelocityProfile(
        resource_id="quakeml:domain.ab/velocity_profile/003",
        velocity_profile_data=[
            VelocityProfileData(
                velocityS=ValueWithUncertainty(400.0),
                top_depth=ValueWithUncertainty(0.0))])
    site.add_velocity_profiles(
        [profile],
        analysisID=analysis.resource_id)

    site.set_preferred_analysis(analysis.resource_id)
    site.set_preferred_velocity_profile(
        profile.resource_id,
        analysisID=analysis.resource_id)

Reference Validation And Preserving User Intent
-----------------------------------------------

SiteXML contains internal references between objects: a top-level site points
to one site description, each analysis points back to that site description,
velocity-profile rows belong to a specific analysis, and preferred IDs select 
one analysis or velocity profile. **ObsPy validates these relationships, but
it does not repair them by guessing what the user meant.**

This is a deliberate design choice. When a site has several analyses or
velocity profiles, a missing or inconsistent ID is ambiguous. Automatically
choosing as preferred the first available or a random object could create 
valid-looking XML that no longer represents the user's intended data. Instead, 
ObsPy preserves missing optional metadata as missing, preserves explicit IDs 
as written, and raises a clear validation error when an explicit relationship 
points to an object that does not exist or belongs somewhere else.

Use :meth:`~obspy.io.sitexml.core.SERASite.validate_references` to check the
internal object graph before writing or after programmatic edits:

.. code-block:: python

    site.validate_references()

:func:`~obspy.io.sitexml.sitexml.write_sitexml` calls this validation
automatically before XML serialization, so broken internal references fail
before a file is emitted.

Reference validation checks that:

* every ``Analysis.site_descriptionID`` matches the parent
  ``SiteDescription.resource_id``;
* analysis resource IDs are unique within the site;
* velocity-profile resource IDs are unique within the site;
* ``preferredSiteAnalysisID`` points to an attached analysis when it is set;
* ``preferredVelocityProfileID`` points to an attached velocity profile when it
  is set;
* if both preferred IDs are set, the preferred velocity profile belongs to the
  preferred analysis.

The tabular importers follow the same principle. They require explicit
relationship columns such as ``siteDescriptionID``, ``analysisID``, and
``velocityProfileID`` where those relationships are needed. They do not
generate missing relationship IDs, and they do not infer
``preferredSiteAnalysisID`` or ``preferredVelocityProfileID`` from the first
available row.

If you want ObsPy to set preferred relationships for you, use the explicit
object helpers. These helpers validate the selected objects before changing
the preferred-ID metadata:

.. code-block:: python

    analysis = site.add_analysis(
        resource_id="quakeml:domain.ab/analysis/002",
        set_preferred=True)

    site.set_preferred_analysis(analysis.resource_id)
    site.set_preferred_velocity_profile(
        "quakeml:domain.ab/velocity_profile/003",
        analysisID=analysis.resource_id)

Some convenience methods do use **fallback behavior** for lookup or calculation.
For example, ``get_preferred_analysis()`` returns the first attached analysis
when no preferred analysis has been declared, and quality-index calculations
use that same lookup behavior. 

.. important::

  These fallbacks are read-time conveniences: they do not write guessed preferred 
  IDs back into ``SiteDescription`` and do not change the XML metadata unless you 
  explicitly assign those fields.

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

The ``add_revision`` call records the initial document version. The 
revision fields are described in :ref:`sitexml-filenames-and-revision-history`.

When creating or editing a ``SERASite`` object in Python, use
``SERASite.iter_site_indicators()`` to apply shared metadata to a selected set
of existing indicators. The iterator yields SiteXML indicator names such as
``siteClassEC8``, ``bedrockDepth``, and ``h800``.

.. code-block:: python

    from obspy.core.inventory.util import ExternalReference
    from obspy.io.sitexml.core import LiteratureSource

    literature_source = LiteratureSource(
        title="Example site characterization report",
        first_author="Author A.",
        year="2026")
    external_references = [
        ExternalReference(
            uri="https://example.org/reports/site-characterization.pdf",
            description="Site characterization report")]

    shared_reference_indicators = {
        "siteClassEC8", "bedrockDepth", "h800"}
    for name, indicator in site.iter_site_indicators():
        if name in shared_reference_indicators:
            indicator.literature_source = literature_source
            indicator.external_references = external_references

CSV Input Files
---------------

CSV import uses one required owner CSV file, one required site-description CSV
file, plus optional analysis, velocity-profile, and quality-index sidecar
files. The default delimiter is ``;``.

The importer distinguishes between missing structure and missing optional
metadata:

.. list-table::
    :header-rows: 1

    * - Input
      - Required columns
      - Row handling
    * - Owner
      - ``owner_codename``, ``owner_fullname``, ``person_firstname``,
        ``person_lastname``, ``person_mbox``
      - Strict. Missing required columns or values stop import.
    * - Site description
      - ``siteID``, ``siteDescriptionID``, ``latitude``, ``longitude``
      - Strict. These rows create the ``SERASite`` objects.
    * - Analysis
      - ``siteID``, ``siteDescriptionID``, ``analysisID``
      - Strict in the public CSV importer when the file is provided.
    * - Velocity profile
      - ``siteID``, ``analysisID``, ``velocityProfileID``,
        ``velocityS_value``, ``layerTopDepth_value``
      - Strict. A malformed layer invalidates the velocity profile.
    * - Quality index
      - ``siteID``
      - Lenient by row. Empty or unknown ``siteID`` values are skipped with a
        warning.

Optional columns may be present with empty cells or omitted entirely. Missing
optional values are imported as absent metadata.

Site Owner
~~~~~~~~~

The owner CSV describes the data owner and contact person:

.. list-table::
    :header-rows: 1

    * - Type
      - Column names
    * - Required
      - ``owner_codename``, ``owner_fullname``, ``person_firstname``,
        ``person_lastname``, ``person_mbox``
    * - Optional
      - All other columns are optional, including the resource identifiers.

Site Description
~~~~~~~~~~~~~~~

The site-description CSV has one row per site. The site this description
refers to is designated by the ``siteID``. The unique resource identifier
``siteDescriptionID`` must also be present.

.. list-table::
    :header-rows: 1

    * - Type
      - Column names
      - Description
    * - Required
      - ``siteID``, ``siteDescriptionID``, ``latitude``, ``longitude``
      - Required for every site description row.
    * - Location
      - ``station``, ``altitude``, ``minDistanceFromStation``, ``maxDistanceFromStation``,
      - Optional. Use ``station`` for station-backed sites.
    * - Site indicators
      - ``siteClassEC8_*``, ``bedrockDepth_*``, ``h800_*``, ``geologicalUnit_*``
      - Optional. Use these columns for site-level characterization values.
    * - References
      - ``preferredSiteAnalysisID``, ``preferredVelocityProfileID``
      - Optional but necessary when more than one analysis or velocity
        profile is present.

Indicator metadata columns use the indicator name as a prefix.

.. list-table::
    :header-rows: 1

    * - Indicator
      - Column name
    * - EC8
      - ``siteClassEC8_value``, ``siteClassEC8Qindex1``, ``siteClassEC8_title``, etc.
    * - Bedrock depth
      - ``bedrockDepth_value``, ``bedrockDepthQindex1``, ``bedrockDepth_title``, etc.
    * - H800
      - ``h800_value``, ``h800Qindex1``, ``h800_title``, etc.
    * - Geological unit
      - ``geologicalUnit_value``, ``geologicalUnitQindex1``, ``geologicalUnit_title``, etc.

Station codes use ``network.station`` notation, for example ``XX.ABCD``.
Bare station codes are rejected because station codes are not globally unique.

Analysis
~~~~~~~~

The analysis CSV is optional. When provided, each row identifies the site with ``siteID``,
site description with ``siteDescriptionID``, and analysis with ``analysisID``.

.. list-table::
    :header-rows: 1

    * - Type
      - Column names
      - Description
    * - Required
      - ``siteID``, ``siteDescriptionID``, ``analysisID``
      - Required for every analysis row.
    * - Site indicators
      - ``resonanceFrequency_*``, ``velocityS30_*``, ``velocityProfileSet_*``
      - Optional. Use these columns for site-level characterization values.
    * - Logs counts
      - ``sptLogsCount``, ``cptLogsCount``, ``boreholeLogsCount``
      - Optional count of available logs.

Indicator metadata columns use the indicator name as a prefix.

.. list-table::
    :header-rows: 1

    * - Indicator
      - Column name
    * - Resonance Frequency
      - ``resonanceFrequency_value``, ``resonanceFrequencyQindex1``, ``resonanceFrequency_title``, etc.
    * - Velocity S30
      - ``velocityS30_value``, ``velocityS30Qindex1``, ``velocityS30_title``, etc.
    * - Velocity Profile Set
      - ``velocityProfileSet_value``, ``velocityProfileSetQindex1``, ``velocityProfileSet_title``, etc.

Velocity Profiles
~~~~~~~~~~~~~~~~~

Velocity-profile input can be one CSV file or a directory of CSV files. Each row
describes one layer in one velocity profile:

.. list-table::
    :header-rows: 1

    * - Type
      - Column names
      - Description
    * - Required references
      - ``siteID``, ``analysisID``, ``velocityProfileID``
      - Required for every velocity profile row.
    * - Required values
      - ``layerTopDepth_value`` and ``velocityS_value``
      - Required for every velocity profile row.
    * - Optional values
      - ``layerBottomDepth_value``, ``velocityP_value``, ``density_value``
      - All other velocity profile values are optional
    * - Optional uncertainties
      - ``layerTopDepth_uncertainty``, ``velocityS_uncertainty``, etc.
      - All uncertainty columns are optional
    * - Layer counter
      - ``layerCount``
      - Optional counter of the velocity profile layer
    
A missing layer bottom depth value represents an open-ended final layer.

Quality Index Sidecar
~~~~~~~~~~~~~~~~~~~~~

The optional quality-index CSV is keyed by ``siteID``. All Quality index #1 
criteria and Quality index #3 consistency columns are optional. Rows with an 
empty ``siteID`` value or an unknown site ID are skipped with a warning because 
quality-index inputs are optional enrichment.

All columns use the indicator name as a prefix followed by the name of the 
parameter used to calculate the quality index. For example, the column name 
``siteClassEC8_method`` refers to the method used for the estimation of the
``siteClassEC8`` value. 
The following table summarizes the parameter names.

.. list-table::
    :header-rows: 1

    * - Quality index
      - Parameter names
    * - Quality index #1
      - ``method``, ``evaluation``, ``reliability``, ``report``
    * - Quality index #3
      - ``f0_vs30``, ``f0_bedrock_depth``, ``f0_h800``, ``vs30_h800``, ``vs30_geology``

For more information on the parameters, their meaning and the calculation of
quality indexes refer to :ref:`quality-indexes`

.. important::

    The generic CSV and Excel importers **do not generate missing relationship IDs**.
    For example, they will not guess ``preferredSiteAnalysisID`` or
    ``preferredVelocityProfileID`` values, because that would hide user intent when a
    site has several analyses or velocity profiles.

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

Associating SiteXML With StationXML
-----------------------------------

SiteXML can be associated with the corresponding StationXML by adding the
current published SiteXML URL as a station-level StationXML
``ExternalReference``. The SiteXML file itself stores the station association
in ``SiteDescription.station_code`` using ``network.station`` notation, for
example ``XX.ABCD``. Because SiteXML filenames can include a publication date
(:ref:`sitexml-filenames-and-revision-history`), the StationXML reference is
treated as the current pointer to the latest published SiteXML document.

Use :func:`~obspy.io.sitexml.sitexml.add_sitexml_reference` to add or
update the StationXML reference that points to the SiteXML URL.

Inventory Input
~~~~~~~~~~~~~~~

The helper operates on an existing
:class:`~obspy.core.inventory.inventory.Inventory` object and returns the same
updated inventory. Read StationXML before calling the helper, and write the
updated StationXML afterwards if you need a file on disk.

Local StationXML Input
~~~~~~~~~~~~~~~~~~~~~~

If the StationXML is already available locally, read it with
:func:`~obspy.core.inventory.inventory.read_inventory` first. This is useful
for newly created StationXML files, offline workflows, or examples where you
do not want to contact an FDSN data center:

.. code-block:: python

    from obspy import UTCDateTime, read_inventory
    from obspy.io.sitexml.sitexml import add_sitexml_reference

    inventory = read_inventory("XX.ABCD.stationxml.xml")
    inventory = add_sitexml_reference(
        inventory,
        station_code="XX.ABCD",
        sitexml_url="https://example.org/sitexml/Site_XX.ABCD_02-05-2026.xml",
        added_time=UTCDateTime(2026, 5, 2))
    inventory.write("XX.ABCD.with_sitexml.stationxml.xml",
                    format="STATIONXML")

Basic Remote Workflow
~~~~~~~~~~~~~~~~~~~~~

To update StationXML from an FDSN data center, fetch the inventory first and
then pass it to the helper:

.. code-block:: python

    from obspy.clients.fdsn import Client
    from obspy.io.sitexml.sitexml import add_sitexml_reference

    client = Client("ORFEUS")
    inventory = client.get_stations(
        network="XX", station="ABCD", level="response")
    inventory = add_sitexml_reference(
        inventory,
        station_code="XX.ABCD",
        sitexml_url="https://example.org/sitexml/Site_XX.ABCD_02-05-2026.xml")

    station = inventory.select(network="XX", station="ABCD")[0][0]
    print(station.external_references[-1].description)

The helper does not perform file I/O. Use ``inventory.write(...)`` when you
want to export the updated StationXML.

Keeping The Reference Current
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``sitexml_url`` should be the URL where the current SiteXML document is
published.
By default, :func:`~obspy.io.sitexml.sitexml.add_sitexml_reference`
replaces an existing SiteXML station reference instead of appending another
dated URL. It replaces references written by this helper and manually added
references whose URL basename follows the default SiteXML station filename
pattern, for example ``Site_XX.ABCD_02-05-2026.xml``. Other station external
references are left untouched.

If the StationXML file should preserve a history of SiteXML URLs, disable
replacement explicitly:

.. code-block:: python

    from obspy import read_inventory
    from obspy.io.sitexml.sitexml import add_sitexml_reference

    inventory = read_inventory("XX.ABCD.stationxml.xml")
    inventory = add_sitexml_reference(
        inventory,
        station_code="XX.ABCD",
        sitexml_url="https://example.org/sitexml/Site_XX.ABCD_02-05-2026.xml",
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

    inventory = add_sitexml_reference(
        inventory,
        station_code="XX.ABCD",
        sitexml_url="https://example.org/sitexml/Site_XX.ABCD_02-05-2026.xml",
        description="SERA SiteXML site characterization for XX.ABCD",
        added_time=UTCDateTime(2026, 5, 2))

    inventory.write("XX.ABCD.stationxml.xml", format="STATIONXML")

Bare station codes such as ``ABCD`` should not be used for this association,
because StationXML station codes are only unique together with their network
code.

Updating a SiteXML file
-----------------------

Existing SiteXML files can be read from a local path or from a published
HTTP(S) URL, modified in memory, and written back to a new XML file. This is
useful when one indicator needs to be corrected or when new site
characterization metadata become available.

Use :meth:`~obspy.io.sitexml.core.SERASite.add_site_indicator` to add or
replace site indicators. The method routes each indicator to the correct
SiteXML object location by its indicator name. For example, ``EC8`` is written
to ``site.site_description.ec8`` and ``BedrockDepth`` is written to
``site.site_description.bedrock_depth``.

.. code-block:: python

    from obspy import UTCDateTime
    from obspy.core.inventory.util import ExternalReference
    from obspy.io.sitexml.core import (
        BedrockDepth, EC8, LiteratureSource, ValueWithUncertainty)
    from obspy.io.sitexml.sitexml import read_sitexml, write_sitexml

    # Use either a local file path ...
    site = read_sitexml("Site_XX.ABCD_01-05-2026.xml")

    # ... or a published SiteXML URL.
    # site = read_sitexml(
    #     "https://example.org/sitexml/Site_XX.ABCD_01-05-2026.xml")

    updated_ec8 = EC8(
        "C",
        quality_index=0.75,
        literature_source=LiteratureSource(
            title="Updated site classification report",
            first_author="Example",
            year=2026),
        external_references=[ExternalReference(
            uri="https://example.org/reports/XX.ABCD-ec8-2026.pdf",
            description="Updated EC8 classification report")])

    new_bedrock_depth = BedrockDepth(
        ValueWithUncertainty(42.0, uncertainty=5.0),
        quality_index=0.8,
        literature_source=LiteratureSource(
            title="Bedrock depth reassessment",
            first_author="Example",
            year=2026),
        external_references=[ExternalReference(
            uri="https://example.org/reports/XX.ABCD-bedrock-2026.pdf",
            description="Bedrock depth reassessment")])

    site.add_site_indicator([updated_ec8, new_bedrock_depth])

    revision_time = UTCDateTime("2026-05-05T10:30:00Z")
    site.add_revision(
        revision_time=revision_time,
        description="Updated EC8 class and added bedrock depth.",
        author="ORFEUS",
        version="2026-05-05",
        previous_version=(
            "https://example.org/sitexml/Site_XX.ABCD_01-05-2026.xml")))

    output_file = site.get_sitexml_filename(revision_time)
    write_sitexml(site, output_file, validate=True)

Adding an indicator to a slot that is already populated replaces the old
indicator object. For example, ``site.add_site_indicator([EC8("C")])`` creates
a new EC8 indicator and the previous EC8 indicator's quality-assessment
metadata is no longer attached to the site. That includes any
``literature_source``, ``external_references``, and ``quality_index`` stored on
the previous indicator. Add fresh assessment metadata to the replacement
indicator, or edit the existing indicator directly if the original assessment
metadata still applies.

For details on ``revisionHistory``, ``creationTime``, and dated output names
such as ``Site_XX.ABCD_05-05-2026.xml``, see
:ref:`sitexml-filenames-and-revision-history`.

Adding Velocity Profiles To Existing SiteXML
--------------------------------------------

Sometimes a SiteXML document already exists, but the velocity profiles are
available later as a CSV or Excel sidecar table. Use
:func:`~obspy.io.sitexml.tabular.add_velocity_profiles` to merge those
profiles into the existing object tree without rebuilding the whole site from
the owner, site-description, and analysis tables.

The sidecar table uses the same velocity-profile columns described above.
Rows are matched by ``siteID`` and ``analysisID``, so profiles are attached to
the correct analysis even when a site has multiple analyses. The helper accepts
either one ``SERASite`` object or a dictionary of sites keyed by ``siteID`` and
detects CSV or Excel input from the file extension.

There are two public helpers, depending on where the velocity profiles already
live:

* Use ``SERASite.add_velocity_profiles(...)`` when you already have
  :class:`~obspy.io.sitexml.core.VelocityProfile` objects and want to add them
  to one analysis on one site.
* Use :func:`~obspy.io.sitexml.tabular.add_velocity_profiles` when the
  profiles are still in CSV or Excel form, or when one sidecar table may update
  multiple sites and analyses.

For object-level updates, pass the target ``analysisID`` directly:

.. code-block:: python

    site.add_velocity_profiles(
        [velocity_profile],
        analysisID="quakeml:domain.ab/analysis/001")

For tabular updates, pass the existing site or site dictionary and the sidecar
file:

.. code-block:: python

    from pathlib import Path

    from obspy.core.util import get_example_file
    from obspy.io.sitexml.sitexml import read_sitexml, write_sitexml
    from obspy.io.sitexml.tabular import add_velocity_profiles

    filename = get_example_file("full_sitexml.xml")
    data_dir = Path(filename).parent

    site = read_sitexml(filename)

    # The same function also accepts Excel workbooks such as
    # data_dir / "velocity_profiles.xlsx".
    add_velocity_profiles(
        site,
        data_dir / "velocity_profiles.csv",
        replace_existing=True)

    write_sitexml(site, "site_with_velocity_profiles.xml")

By default, new profiles are appended and duplicate ``velocityProfileID``
values are rejected. Pass ``replace_existing=True`` when the sidecar table
should replace the profile list on the matching analysis. If the target
analysis has no ``velocityProfileSet`` yet, the helper creates one.

.. _quality-indexes:

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

If both schema-shaped ``...Qindex1`` columns and a quality-index sidecar are
provided, ObsPy imports the direct ``...Qindex1`` values first and then applies
the sidecar. Sidecar Q_Index1 criteria for an existing indicator recalculate
that indicator's stored quality index and replace the direct ``...Qindex1``
value. If the sidecar row has no Q_Index1 criteria for an indicator, the direct
``...Qindex1`` value is preserved. Sidecar rows for sites without any indicator
objects are skipped because there is no target indicator to calculate or
combine.

Automatic Calculation Of Overall QI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tabular import is conservative about the overall quality index. If a CSV file
or Excel sheet contains indicator-level ``...Qindex1`` columns but does not
provide an ``overallQindex`` value and does not provide the optional
quality-index sidecar table, ObsPy imports and writes the indicator-level
quality indexes only. It does **not** automatically synthesize
``overallQindex`` in the output XML.

This distinction is important because Q_Index2 gives missing indicator
Q_Index1 values a zero contribution, and Q_Index3 needs consistency inputs
that are only available from the sidecar table or from explicit Python
arguments. Automatically calculating an overall value from partial tabular
metadata could therefore make absent information look like an evaluated site
quality.

Use one of these explicit workflows when the output XML should contain
``overallQindex``:

* provide an ``overallQindex`` column in the site-description table;
* provide the quality-index sidecar CSV or Excel ``qualityIndex`` sheet, so
  ObsPy can calculate Q_Index1/Q_Index3-derived results during import;
* call :meth:`~obspy.io.sitexml.core.SERASite.calculate_overall_quality_index`
  yourself on the imported ``SERASite`` objects before writing SiteXML.

For example, after importing tabular files with some indicator-level
``...Qindex1`` values, calculate and store the overall quality index
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

If no Q_Index3 consistency pairs are passed,
``calculate_overall_quality_index()`` treats Q_Index3 as zero for the final
formula. Pass consistency values only when they have actually been evaluated.

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

    # site is an existing SERASite object
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
