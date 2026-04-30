.. currentmodule:: obspy.io.sitexml
.. automodule:: obspy.io.sitexml
    
    .. comment to end block
    
    Overview
    --------
    
    ``obspy.io.sitexml`` provides tools for working with seismological station
    metadata stored in SiteXML files and related tabular inputs. The module reads
    and writes SiteXML, validates XML files against the bundled schema, and imports
    metadata from CSV and Excel sources into the internal :class:`~obspy.io.sitexml.core.SERASite`
    object model.
    
    The top-level metadata object is :class:`~obspy.io.sitexml.core.SERASite`.
    It contains required :class:`~obspy.io.sitexml.core.SERASiteOwner` and
    :class:`~obspy.io.sitexml.core.SiteDescription` objects, plus optional
    :class:`~obspy.io.sitexml.core.Analysis` objects with site-characterization
    indicators such as Vs30, resonance frequency, and velocity profiles.
    
    The following illustration shows the relationships between the most basic 
    SiteXML objects.

    .. figure:: /_images/SERASite.png
    
    Common Workflows
    ----------------
    
    Read a SiteXML file into a ``SERASite`` object:
    
    .. code-block:: python
    
        from obspy.io.sitexml.sitexml import read_sitexml
    
        sera_site = read_sitexml("site.xml")
        print(sera_site.resource_id)
        print(sera_site.site_description.latitude)
    
    Validate a SiteXML file against the bundled schema:
    
    .. code-block:: python
    
        from obspy.io.sitexml.sitexml import validate_sitexml
    
        valid, errors = validate_sitexml("site.xml")
        if not valid:
            for err in errors:
                print(err)
    
    Write a ``SERASite`` object back to XML:
    
    .. code-block:: python
    
        from obspy.io.sitexml.sitexml import write_sitexml
    
        write_sitexml(sera_site, "site_out.xml", validate=True)
    
    Import site metadata from CSV files. The imported metadata is stored in a
    dictionary of ``SERASite`` objects keyed by the ``siteID``:
    
    .. code-block:: python
    
        from obspy.io.sitexml.read_csv import csv_to_sera_site
    
        sites = csv_to_sera_site(
            "site_owner.csv",
            "site_description.csv",
            analysis_csv="site_analysis.csv",
            velocity_profiles_csv="velocity_profiles",
            delim=";")
    
        sera_site = sites["quakeml:domain.ab/site/001"]
    
    Import site metadata from Excel files. The imported metadata is stored in a
    dictionary of ``SERASite`` objects keyed by the ``siteID``:
    
    .. code-block:: python
    
        from obspy.io.sitexml.read_csv import excel_to_sera_site
    
        sites = excel_to_sera_site(
            "sera_site_all.xlsx",
            velocity_profiles="velocity_profiles.xlsx")
    
    Tutorial With Bundled Test Fixtures
    -----------------------------------
    
    The ``sitexml`` package ships with XML, CSV, and Excel fixtures under
    ``obspy/io/sitexml/tests/data``. These files are useful for interactive
    exploration.

    Read a bundled SiteXML example:
    
    .. code-block:: python
    
        from obspy.core.util import get_example_file
        from obspy.io.sitexml.sitexml import read_sitexml
    
        filename = get_example_file("full_sitexml.xml")
        sera_site = read_sitexml(filename)
    
        print(sera_site.resource_id)
        print(len(sera_site.analysis))
    
    Import bundled CSV fixtures and inspect one imported site:
    
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
    
        sera_site = sites["quakeml:domain.ab/site/001"]
        analysis = sera_site.analysis[0]
        first_profile = analysis.velocity_profile_survey.velocity_profiles[0]
    
        print(sera_site.site_owner.owner_codename)
        print(analysis.velocity_s30.value.value)
        print(first_profile.layer_count)
    
    Import bundled Excel fixtures:
    
    .. code-block:: python
    
        from pathlib import Path
    
        from obspy.core.util import get_example_file
        from obspy.io.sitexml.read_csv import excel_to_sera_site
    
        excel_file = get_example_file("sera_site_all.xlsx")
        data_dir = Path(excel_file).parent
    
        sites = excel_to_sera_site(
            excel_file,
            velocity_profiles=data_dir / "velocity_profiles.xlsx")
    
        sera_site = sites["quakeml:domain.ab/site/001"]
        print(sera_site.site_description.station_code)
    
    Apply optional quality-index sidecar inputs during CSV import. The sidecar
    stores Q_Index1 criteria and Q_Index3 consistency checks used for the
    calculation; only schema-supported calculated outputs are stored on the
    imported ``SERASite`` objects:
    
    .. code-block:: python
    
        sites = csv_to_sera_site(
            site_owner_csv,
            site_description_csv,
            analysis_csv=analysis_csv,
            velocity_profiles_csv=velocity_profiles_dir,
            quality_index_csv="quality_index.csv",
            delim=";")
    
        sera_site = sites["quakeml:domain.ab/site/001"]
        print(sera_site.site_description.overall_quality_index)
    
    Apply quality-index sidecar inputs later to an existing dictionary, for
    example after reading SiteXML files:
    
    .. code-block:: python
    
        from obspy.io.sitexml.quality_index import apply_quality_index_csv
        from obspy.io.sitexml.sitexml import sitexml_to_sitedict
    
        sites = sitexml_to_sitedict("./sitexml_files")
        apply_quality_index_csv(sites, "quality_index.csv", delim=";")
    
    Write a site dictionary created from a CSV or Excel import back to
    schema-validated SiteXML:
    
    .. code-block:: python
    
        from obspy.io.sitexml.sitexml import sitedict_to_sitexml
    
        sitedict_to_sitexml(sites, "./output_folder")
    
    Write one imported site back to schema-validated SiteXML:
    
    .. code-block:: python
    
        from obspy.io.sitexml.sitexml import write_sitexml
    
        write_sitexml(sera_site, "site.xml", validate=True)

    Uncertainty Values
    ------------------
    
    SiteXML value/uncertainty pairs are represented by
    :class:`~obspy.io.sitexml.core.ValueWithUncertainty`. The class intentionally
    keeps the SiteXML shape: one numeric ``value`` and one optional symmetric
    ``uncertainty``. ObsPy also provides
    :class:`~obspy.core.util.obspy_types.FloatWithUncertainties`, but that type
    stores separate lower and upper uncertainties, so using it directly would
    require a policy for asymmetric values that SiteXML cannot represent.
    
    Use the conversion helpers when interoperability with ObsPy's float subtype is
    needed:
    
    .. code-block:: python
    
        from obspy.core.util.obspy_types import FloatWithUncertainties
        from obspy.io.sitexml.core import ValueWithUncertainty
    
        site_value = ValueWithUncertainty(18.2, uncertainty=0.5)
        obspy_value = site_value.to_float_with_uncertainties()
    
        symmetric = FloatWithUncertainties(
            18.2, lower_uncertainty=0.5, upper_uncertainty=0.5)
        site_value = ValueWithUncertainty.from_float_with_uncertainties(symmetric)
    
    Asymmetric ObsPy uncertainties are rejected during conversion to avoid silent
    data loss. The ObsPy ``measurement_method`` metadata is not represented in the
    SiteXML value/uncertainty pair.
    
    Owner Contact Conversion
    ------------------------
    
    SiteXML owner metadata is represented by
    :class:`~obspy.io.sitexml.core.SERASiteOwner`. The closest ObsPy inventory
    types are :class:`~obspy.core.inventory.util.Person` and
    :class:`~obspy.core.inventory.util.Operator`, but the models are not identical:
    SiteXML has one required owner and one required contact person, while ObsPy
    persons can have multiple names, agencies, emails, and operators can have
    multiple contacts.
    
    Use the conversion helpers when exchanging contact metadata with ObsPy
    inventory objects:
    
    .. code-block:: python
    
        from obspy.core.inventory.util import Operator, Person
        from obspy.io.sitexml.core import SERASiteOwner
    
        person = Person(
            names=["Name Surname"],
            agencies=["INSTITUTION_ABBR"],
            emails=["someemail@domain.ab"])
        site_owner = SERASiteOwner.from_person(
            person,
            owner_codename="SITEOWNER",
            owner_fullname="Site Owner Full Name")
    
        operator = site_owner.to_operator()
    
    When converting from an ObsPy ``Operator`` with multiple contacts, pass
    ``contact_index`` to select the contact to use. Without an explicit selection,
    the conversion raises a SiteXML validation error to avoid silently discarding
    contacts. SiteXML public IDs, contact homepage metadata, and institution
    address fields are not represented by ObsPy ``Person`` and ``Operator`` in a
    fully round-trippable way.
    
    Import Requirements
    -------------------
    
    The CSV and Excel import helpers expect required owner/contact and
    site-description metadata to be present. In particular, owner metadata must
    include owner code/name and contact person first name, last name, and email
    address. Site descriptions require a site ID, site-description ID, latitude,
    and longitude. Optional analysis and velocity-profile inputs may be omitted;
    when they are explicitly provided but malformed, the import raises a
    SiteXML-specific exception instead of silently ignoring the broken metadata.
    
    Reference metadata follows the current SiteXML terminology: literature
    sources use required ``title`` and ``firstAuthor`` fields, and external
    resources use ``externalReference`` metadata with ``uri`` and
    ``description`` fields. Resource identifiers are stored internally as plain
    strings; ObsPy ``ResourceIdentifier`` values may be passed as input
    conveniences and are normalized to their string IDs.
    
    Notes
    -----
    
    - XML handling should follow the schema files bundled under
      ``obspy/io/sitexml/data``.
    - CSV and Excel import paths build the same internal ``SERASite``-based object
      graph used by the XML reader and writer.
    - The active writer entry point is
      :func:`~obspy.io.sitexml.sitexml.write_sitexml`.
    
    Enums
    -----

    .. autosummary::
       :toctree: autogen
       :nosignatures:

       ~util.TopographySchemaA
       ~util.TopographySchemaB
       ~util.MorphologyType
       ~util.EC8Class
       ~util.ResonanceFrequencyMethod
       ~util.VelocityS30Method
       ~util.Vs30MethodCombined
       ~util.Vs30ManualIndex

    .. comment to end block

    Functions
    ---------
    
    .. autosummary::
       :toctree: autogen
       :nosignatures:
    
       ~sitexml._is_sitexml
       ~sitexml.validate_sitexml
       ~sitexml.read_sitexml
       ~sitexml.write_sitexml
       ~sitexml.sitedict_to_sitexml
       ~sitexml.sitexml_to_sitedict
       ~sitexml.write_stationxml_reference
       ~read_csv.csv_to_sera_site
       ~read_csv.excel_to_sera_site
       ~quality_index.quality_index1
       ~quality_index.quality_index2
       ~quality_index.quality_index3
       ~quality_index.overall_quality_index
       ~quality_index.apply_quality_index_csv
       ~quality_index.apply_quality_index_excel
       
    .. comment to end block

    Classes
    -------
    
    .. autosummary::
       :toctree: autogen
       :nosignatures:
    
       ~core.SERASite
       ~core.SERASiteOwner
       ~core.SiteDescription
       ~core.Analysis
       ~core.VelocityProfile
       ~core.VelocityProfileData

    .. comment to end block

    Modules
    -------
    
    .. autosummary::
       :toctree: autogen
       :nosignatures:
    
       core
       sitexml
       read_csv
       quality_index
       util

    .. comment to end block
