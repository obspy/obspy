========================
Working With SERASiteXML
========================

SERASiteXML stores site-characterization metadata for seismic stations and
locations. ObsPy reads local SERASiteXML files and published SERASiteXML URLs
into :class:`~obspy.io.sitexml.core.SERASite` objects, writes those objects
back to schema-validated XML, and can build the same objects from CSV or Excel
input tables.

The examples below use the small fixtures bundled with ObsPy. They are
convenient for learning the file layout before using project-specific files.

Reading And Validating SERASiteXML
----------------------------------

Use :func:`~obspy.io.sitexml.sitexml.read_sitexml` to read one SERASiteXML file
or HTTP(S) URL. The returned object contains the owner, the site description,
and optional analysis objects.

.. code-block:: python

    from obspy.core.util import get_example_file
    from obspy.io.sitexml.sitexml import read_sitexml

    filename = get_example_file("xml/full_sitexml.xml")
    site = read_sitexml(filename)

    print(site.resource_id)
    print(site.site_owner.owner_codename)
    print(site.site_description.station_code)
    print(len(site.analysis))

For a published SERASiteXML document, pass the URL directly. ObsPy retrieves
the remote XML and uses the same schema validation and parser as it does for
local files.

.. code-block:: python

    from obspy.io.sitexml.sitexml import read_sitexml

    site = read_sitexml("https://example.org/sitexml/Site_XX.ABCD.xml")

SERASiteXML validation is available separately with
:func:`~obspy.io.sitexml.sitexml.validate_sitexml`. It returns a boolean and a
list of schema-validation errors.

.. code-block:: python

    from obspy.io.sitexml.sitexml import validate_sitexml

    valid, errors = validate_sitexml(filename)
    if not valid:
        for error in errors:
            print(error)

Writing SERASiteXML
-------------------

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

A directory of SERASiteXML files can be read as a dictionary keyed by
``SERASite.resource_id``. The reverse helper writes each site to its default
SERASiteXML filename.

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

SERASiteXML filenames and revision history
------------------------------------------

SERASiteXML filenames identify the site and, for ObsPy generated files, **the
publication date**. For a station-backed site, the base filename is built
from ``Site_`` and the ``network.station`` code stored in
``SiteDescription.station_code``:

.. code-block:: text

    Site_XX.ABCD.xml                    # Default filename - no date
    Site_XX.ABCD_05-05-2026.xml         # Default filename - publication date

For sites without an associated station code, the base filename is derived from
the ``SERASite.resource_id`` using the compact SERASiteXML site identifier:

.. code-block:: text

    quakeml:example.org/site/001
    Site_example.org.001.xml
    Site_example.org.001_05-05-2026.xml

Use
:meth:`~obspy.io.sitexml.core.SERASite.get_sitexml_filename` when you need to
preview or choose the default filename yourself. If you provide a date, this
will be appended in the base filename.

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

SERASiteXML can also store optional document revision history.
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

SERASiteXML uses resource identifiers to connect the top-level site object with
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

SERASiteXML contains internal references between objects: a top-level site
points to one site description, each analysis points back to that site
description, velocity-profile rows belong to a specific analysis, and preferred
IDs select one analysis or velocity profile. **ObsPy validates these
relationships, but it does not repair them by guessing what the user meant.**

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

There is one lenient case for optional tabular inputs. If the analysis CSV or
Excel sheet is omitted, any ``preferredSiteAnalysisID`` and
``preferredVelocityProfileID`` values in the site-description table are ignored
with a warning, because there is no analysis or velocity-profile object to
reference. If analysis metadata is present but velocity-profile metadata is
omitted, ``preferredSiteAnalysisID`` is preserved when valid, while
``preferredVelocityProfileID`` is ignored with a warning. The ignored preferred
values are not written to generated SERASiteXML.

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

  These fallbacks are read-time conveniences: they do not write guessed
  preferred IDs back into ``SiteDescription`` and do not change the XML
  metadata unless you explicitly assign those fields.

Creating A SERASiteXML File From Scratch
----------------------------------------

Creating a SERASiteXML file directly in Python is useful when metadata already
exists in application objects or another structured source. It is not always
shorter than preparing CSV or Excel input, but it keeps the object graph
explicit and lets ObsPy validate the result against the bundled SERASiteXML
schema when writing.

A minimal SERASiteXML document needs a top-level
:class:`~obspy.io.sitexml.core.SERASite`, a required
:class:`~obspy.io.sitexml.core.SERASiteOwner`, and a required
:class:`~obspy.io.sitexml.core.SiteDescription`. Analyses and site indicators
can then be attached as needed. Resource identifiers are ordinary strings, but
they should be stable and unique, and relationship fields such as
``site_descriptionID`` and ``preferred_site_analysisID`` must point to existing
objects.

The following illustration shows the relationships between the most basic
SERASiteXML objects.

.. figure:: /_images/SERASite.png

The following example creates a small SERASiteXML site with owner/contact
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
        description="Created initial SERASiteXML document.",
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
of existing indicators. The iterator yields SERASiteXML indicator names such as
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

CSV and Excel Input Format
--------------------------

CSV and Excel import share the same table structure, column names, and
validation rules. See :doc:`sitexml_tabular_input` for detailed information
on how to format the input tables either in CSV or Excel format.
The examples below demonstrate how to import these tables with the Python API.

Importing CSV
-------------

Use :func:`~obspy.io.sitexml.tabular.csv_to_sera_site` to build a dictionary
of ``SERASite`` objects from CSV files.

.. code-block:: python

    from pathlib import Path

    from obspy.core.util import get_example_file
    from obspy.io.sitexml.tabular import csv_to_sera_site

    site_owner_csv = get_example_file("csv/site_owner.csv")
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

    excel_file = get_example_file("xlsx/full_site.xlsx")
    data_dir = Path(excel_file).parent

    sites = excel_to_sera_site(
        excel_file,
        velocity_profiles=data_dir / "velocity_profiles.xlsx")

    site = sites["quakeml:domain.ab/site/001"]
    print(site.site_description.station_code)

Associating SERASiteXML With StationXML
---------------------------------------

SERASiteXML can be associated with the corresponding StationXML by adding the
current published SERASiteXML URL as a station-level StationXML
``ExternalReference``. The SERASiteXML file itself stores the station
association in ``SiteDescription.station_code`` using ``network.station``
notation, for example ``XX.ABCD``. Because SERASiteXML filenames can include
a publication date (:ref:`sitexml-filenames-and-revision-history`), the
StationXML reference is treated as the current pointer to the latest published
SERASiteXML document.

Use :func:`~obspy.io.sitexml.sitexml.add_sitexml_reference` to add or
update the StationXML reference that points to the SERASiteXML URL.

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

The ``sitexml_url`` should be the URL where the current SERASiteXML document
is published.
By default, :func:`~obspy.io.sitexml.sitexml.add_sitexml_reference`
replaces an existing SERASiteXML station reference instead of appending another
dated URL. It replaces references written by this helper and manually added
references whose URL basename follows the default SERASiteXML station filename
pattern, for example ``Site_XX.ABCD_02-05-2026.xml``. Other station external
references are left untouched.

If the StationXML file should preserve a history of SERASiteXML URLs, disable
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
added in the form ``added YYYY-MM-DD``.
By default the added date is the current UTC date. If you need reproducible
output or want to use the SERASiteXML document creation date, pass
``added_time``. If the default external-reference text is not specific enough,
pass a custom ``description``; the added-date marker is appended to the custom
text:

.. code-block:: python

    inventory = add_sitexml_reference(
        inventory,
        station_code="XX.ABCD",
        sitexml_url="https://example.org/sitexml/Site_XX.ABCD_02-05-2026.xml",
        description="SERASiteXML site characterization for XX.ABCD",
        added_time=UTCDateTime(2026, 5, 2))

    inventory.write("XX.ABCD.stationxml.xml", format="STATIONXML")

Bare station codes such as ``ABCD`` should not be used for this association,
because StationXML station codes are only unique together with their network
code.

Updating a SERASiteXML file
---------------------------

Existing SERASiteXML files can be read from a local path or from a published
HTTP(S) URL, modified in memory, and written back to a new XML file. This is
useful when one indicator needs to be corrected or when new site
characterization metadata become available.

Use :meth:`~obspy.io.sitexml.core.SERASite.add_site_indicator` to add or
replace site indicators. The method routes each indicator to the correct
SERASiteXML object location by its indicator name. For example, ``EC8`` is
written to ``site.site_description.ec8`` and ``BedrockDepth`` is written to
``site.site_description.bedrock_depth``.

.. code-block:: python

    from obspy import UTCDateTime
    from obspy.core.inventory.util import ExternalReference
    from obspy.io.sitexml.core import (
        BedrockDepth, EC8, LiteratureSource, ValueWithUncertainty)
    from obspy.io.sitexml.sitexml import read_sitexml, write_sitexml

    # Use either a local file path ...
    site = read_sitexml("Site_XX.ABCD_01-05-2026.xml")

    # ... or a published SERASiteXML URL.
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

Adding Velocity Profiles To Existing SERASiteXML
------------------------------------------------

Sometimes a SERASiteXML document already exists, but the velocity profiles are
available later as a CSV or Excel input table. Use
:func:`~obspy.io.sitexml.tabular.add_velocity_profiles` to merge those
profiles into the existing object tree without rebuilding the whole site from
the owner, site-description, and analysis tables.

The input table uses the same velocity-profile columns described above.
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
  profiles are still in CSV or Excel form, or when one input table may update
  multiple sites and analyses.

For object-level updates, pass the target ``analysisID`` directly:

.. code-block:: python

    site.add_velocity_profiles(
        [velocity_profile],
        analysisID="quakeml:domain.ab/analysis/001")

For tabular updates, pass the existing site or site dictionary and the
velocity-profile file:

.. code-block:: python

    from pathlib import Path

    from obspy.core.util import get_example_file
    from obspy.io.sitexml.sitexml import read_sitexml, write_sitexml
    from obspy.io.sitexml.tabular import add_velocity_profiles

    filename = get_example_file("xml/full_sitexml.xml")
    velocity_profiles_csv = get_example_file("csv/velocity_profiles.csv")

    site = read_sitexml(filename)

    # The same function also accepts Excel workbooks such as
    # get_example_file("xlsx/velocity_profiles.xlsx").
    add_velocity_profiles(
        site,
        velocity_profiles_csv,
        replace_existing=True)

    write_sitexml(site, "site_with_velocity_profiles.xml")

By default, new profiles are appended and duplicate ``velocityProfileID``
values are rejected. Pass ``replace_existing=True`` when the input table
should replace the profile list on the matching analysis. If the target
analysis has no ``velocityProfileSet`` yet, the helper creates one.

Quality Indexes
---------------

SERASiteXML can store optional indicator-level quality indexes and an overall
quality index. These describe the reliability and consistency of site
characterization metadata and can be provided directly or calculated through
the Python API and optional CSV/Excel quality-index inputs.

See :doc:`sitexml_quality_indexes` for the formulas, scoring criteria,
missing-data behavior, and calculation examples.
