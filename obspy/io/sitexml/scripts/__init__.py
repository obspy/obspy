# -*- coding: utf-8 -*-
r"""
SiteXML Command-Line Tools
==========================

.. contents:: Contents
    :local:
    :depth: 2


This guide describes how to use the ``csv2sitexml`` and
``excel2sitexml`` command-line tools to convert tabular site-characterization
metadata into SiteXML files.

Availability
------------


Installing ObsPy provides the ``csv2sitexml`` and ``excel2sitexml`` commands.
Run ``csv2sitexml -h`` or ``excel2sitexml -h`` for help.

What The Tools Produce
----------------------


Both tools write **one SiteXML file per imported site** into an output folder.
The generated XML is **schema-validated** before it is written.

Default output filenames are:

- ``Site_<network.station>_<DD-MM-YYYY>.xml`` for station-backed sites,
  for example ``Site_XX.ABCD_13-05-2026.xml``.
- ``Site_<domain>.<site-id>_<DD-MM-YYYY>.xml`` for sites without a station code,
  for example ``Site_domain.ab.003_13-05-2026.xml``.

Existing files with the same generated name are overwritten.

Input Table Summary
-------------------


.. important::

    The input tables are organized to follow the
    `SiteXML schema <https://www.itsak.gr/SiteXML>`_. Each table corresponds to
    a
    major SiteXML object, and the columns in that table correspond to the
    object's
    elements and attributes.


CSV and Excel imports use the same logical tables:

.. list-table::
    :header-rows: 1
    :widths: auto

    * - Table
      - Major Schema Object
      - Required/optional
      - Description
      - Excel sheet name
    * - Site owner
      - `siteOwner <https://www.itsak.gr/SiteXML/#type_SiteOwnerType>`_
      - **Required**
      - Metadata owner and contact information.
      - ``siteOwner``
    * - Site description
      - `siteDescription
        <https://www.itsak.gr/SiteXML/#type_SiteDescriptionType>`_
      - **Required**
      - One row per site; creates the top-level SiteXML site objects.
      - ``siteDescription``
    * - Analysis
      - `analysis <https://www.itsak.gr/SiteXML/#type_AnalysisType>`_
      - Optional
      - Analysis-level indicators such as resonance frequency, Vs30,
        velocity-profile-set metadata, and log counts.
      - ``analysis``
    * - Velocity profiles
      - `velocityProfile <https://www.itsak.gr/SiteXML/#type_VelocityProfile>`_
      - Optional
      - Velocity-profile layer rows. Both for CSV and Excel input, this may be
        one file or a folder.
      - Separate file or folder, not a main-workbook sheet
    * - Quality-index
      - ``qualityIndex``
      - Optional
      - Q_Index1 criteria and Q_Index3 consistency inputs used to calculate
        quality indexes during import.
      - ``qualityIndex``


For CSV input, each table corresponds to a separate CSV file or folder
selected by command-line options.
For Excel input, ``siteOwner``, ``siteDescription``, ``analysis``, and
``qualityIndex``
tables are sheets in the **main workbook**, while velocity profiles are provided
in separate file(s).

.. important::

    - For Excel input the sheet names must be **exactly as shown** in the table
      above.
    - For CSV input the filenames **are not fixed**.
    - The same column names are used **by both** CSV and Excel files.
    - Column names are case sensitive


For additional input-format and quality-index examples, see the
:doc:`SiteXML tabular input reference
</tutorial/code_snippets/sitexml_tabular_input>`.


Minimal Required Metadata
-------------------------


The smallest tabular input that can produce a valid SiteXML file contains:

- one site-owner table;
- one site-description table;
- no analysis table;
- no velocity-profile table;
- no quality-index input table.

For CSV input, this means two files. For example:

- ``site_owner.csv``
- ``site_description.csv``

For Excel input, this means one workbook with:

- a ``siteOwner`` sheet;
- a ``siteDescription`` sheet.

The minimal required ``site-owner`` columns are:

.. code-block:: text

    owner_codename
    owner_fullname
    person_firstname
    person_lastname
    person_mbox


The minimal required ``site-description`` columns are:

.. code-block:: text

    siteID
    siteDescriptionID
    latitude
    longitude


.. note::

    All other columns are **optional.**
    Optional empty cells are imported as absent metadata.
    Required empty cells abort the input process.


Running ``csv2sitexml``
-----------------------


Use ``csv2sitexml`` when your input metadata is split across CSV files.
In the table below you can find a summary of the supported input options.

Use ``csv2sitexml -h``, to get a full list of supported options.

.. list-table::
    :header-rows: 1
    :widths: auto

    * - Input
      - Command option
      - Required?
      - Purpose
    * - Output folder
      - ``-out``
      - **yes**
      - Folder where the generated SiteXML files will be written.
    * - Site owner CSV
      - ``-o`` or ``--site-owner``
      - **yes**
      - One table describing the metadata owner and contact information.
    * - Site description CSV
      - ``-d`` or ``--site-description``
      - **yes**
      - One row per site. Creates the top-level SiteXML site objects.
    * - Analysis CSV
      - ``-a`` or ``--analysis``
      - no
      - One row per analysis. Adds resonance frequency, Vs30,
        velocity-profile-set metadata, and log counts.
    * - Velocity profiles CSV
      - ``-p`` or ``--velocity-profiles``
      - no
      - One CSV file or folder of CSV files.
        One row per velocity-profile layer.
    * - Quality-index CSV
      - ``-q`` or ``--quality-index``
      - no
      - Q_Index1 criteria and Q_Index3 consistency inputs.


The minimal input that can produce SiteXML is ``site_owner.csv`` plus
``site_description.csv`` with their required columns.

.. note::

    In the examples below, the file names are just an example.
    They **must** be replaced by your actual file names or full path names to
    the input files.
    See the example files below to get started.


Examples
~~~~~~~~


Minimal CSV conversion:

.. code-block:: bash

    csv2sitexml \
      -o site_owner.csv \
      -d site_description.csv \
      -out sitexml_output


Full CSV conversion with analysis, velocity profiles, and quality-index
calculation inputs:

.. code-block:: bash

    csv2sitexml \
      -o site_owner.csv \
      -d site_description.csv \
      -a site_analysis.csv \
      -p velocity_profiles \
      -q quality_index.csv \
      -out sitexml_output


.. important::

    - The default CSV delimiter is semicolon ``';'``. If your CSV files use
      another delimiter, pass it with option ``-s``, for example ``-s ","``.
    - You must make sure, that **the delimiter character
      is not used inside any text values in any of the columns.**


Running ``excel2sitexml``
-------------------------


Use ``excel2sitexml`` when your owner, site-description, analysis, and
quality-index tables are sheets in one workbook.

Use ``excel2sitexml -h``, to get a full list of supported options.

.. list-table::
    :header-rows: 1
    :widths: auto

    * - Input
      - Command option
      - Required?
      - Purpose
    * - Output folder
      - ``-out``
      - **yes**
      - Folder where the generated SiteXML files will be written.
    * - Main Workbook
      - ``file path``
      - **yes**
      - A main Workbook with at least ``siteOwner`` and ``siteDescription``
        sheets.
    * - Velocity profiles
      - ``-p`` or ``--velocity-profiles``
      - no
      - One Excel file or folder of Excel files. One row per velocity-profile
        layer.


.. note::

    In the examples below, the file names are just an example.
    They **must** be replaced by your actual file names or full path names to
    the input files.
    See the example files below to get started.


Examples
~~~~~~~~


Minimal Excel conversion:

.. code-block:: bash

    excel2sitexml sera_site.xlsx -out sitexml_output


Excel conversion with velocity profiles:

.. code-block:: bash

    excel2sitexml sera_site.xlsx \
      -p velocity_profiles.xlsx \
      -out sitexml_output


Example Files
-------------


The ObsPy source tree includes small XML, CSV, and Excel files under
``obspy/io/sitexml/tests/data``, grouped by format. Use them for testing
the tools before preparing your own metadata.

Useful starter files include:

- ``obspy/io/sitexml/tests/data/csv/minimal_site_owner.csv``
- ``obspy/io/sitexml/tests/data/csv/minimal_site_description.csv``
- ``obspy/io/sitexml/tests/data/csv/site_owner.csv``
- ``obspy/io/sitexml/tests/data/csv/site_description.csv``
- ``obspy/io/sitexml/tests/data/csv/site_analysis.csv``
- ``obspy/io/sitexml/tests/data/csv/velocity_profiles.csv``
- ``obspy/io/sitexml/tests/data/csv/velocity_profiles/``
- ``obspy/io/sitexml/tests/data/csv/quality_index.csv``
- ``obspy/io/sitexml/tests/data/xlsx/minimal_site.xlsx``
- ``obspy/io/sitexml/tests/data/xlsx/full_site.xlsx``
- ``obspy/io/sitexml/tests/data/xlsx/velocity_profiles.xlsx``
- ``obspy/io/sitexml/tests/data/xml/minimal_sitexml.xml``
- ``obspy/io/sitexml/tests/data/xml/full_sitexml.xml``

From the root of an ObsPy source checkout, for example:

.. code-block:: bash

    csv2sitexml \
      -o obspy/io/sitexml/tests/data/csv/site_owner.csv \
      -d obspy/io/sitexml/tests/data/csv/site_description.csv \
      -a obspy/io/sitexml/tests/data/csv/site_analysis.csv \
      -p obspy/io/sitexml/tests/data/csv/velocity_profiles.csv \
      -q obspy/io/sitexml/tests/data/csv/quality_index.csv \
      -out sitexml_output


or:

.. code-block:: bash

    excel2sitexml obspy/io/sitexml/tests/data/xlsx/full_site.xlsx \
      -p obspy/io/sitexml/tests/data/xlsx/velocity_profiles.xlsx \
      -out sitexml_output


Recommended Workflows
---------------------


SiteXML with only Site Description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Use this when you only need owner and site-location metadata:

.. code-block:: bash

    csv2sitexml -o site_owner.csv -d site_description.csv -out sitexml_output


or:

.. code-block:: bash

    excel2sitexml sera_site.xlsx -out sitexml_output


SiteXML With Analysis
~~~~~~~~~~~~~~~~~~~~~


Use this when you have analysis-level indicators such as resonance frequency or
Vs30 but no velocity-profile layers:

.. code-block:: bash

    csv2sitexml \
      -o site_owner.csv \
      -d site_description.csv \
      -a site_analysis.csv \
      -out sitexml_output


SiteXML With Velocity Profiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Use this when you have analysis metadata and velocity-profile layers:

.. code-block:: bash

    csv2sitexml \
      -o site_owner.csv \
      -d site_description.csv \
      -a site_analysis.csv \
      -p velocity_profiles \
      -out sitexml_output


or:

.. code-block:: bash

    excel2sitexml sera_site.xlsx \
      -p velocity_profiles.xlsx \
      -out sitexml_output


Velocity-profile rows require both ``siteID`` and ``analysisID``, so velocity
profiles are meaningful only with analysis metadata.

SiteXML With Calculated Quality Indexes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Use this when you want the tool to calculate indicator quality indexes and
``overallQindex`` from provided calculation criteria values in
``quality_index.csv``:

.. code-block:: bash

    csv2sitexml \
      -o site_owner.csv \
      -d site_description.csv \
      -a site_analysis.csv \
      -p velocity_profiles \
      -q quality_index.csv \
      -out sitexml_output


For Excel, include a ``qualityIndex`` sheet in the main workbook:

.. code-block:: bash

    excel2sitexml sera_site.xlsx \
      -p velocity_profiles.xlsx \
      -out sitexml_output


Validation Rules And Assumptions
--------------------------------


The import process is intentionally strict about object identifiers and
relationships when the relevant metadata is provided.

The tools validate that:

- required input files or sheets are present;
- required columns are present;
- required row values are not empty;
- all provided values validate against the rules imposed by the schema;
- analysis rows point to existing site description objects through
  ``siteDescriptionID``;
- velocity-profile rows point to existing analysis objects through
  ``analysisID``;
- no duplicate analysis and velocity-profile resource IDs are present;
- ``preferredSiteAnalysisID``, when provided, points to an attached analysis;
- ``preferredVelocityProfileID``, when provided, points to an attached velocity
  profile;
- when both preferred IDs are provided, the preferred velocity profile belongs
  to
  the preferred analysis;
- generated XML validates against the bundled SiteXML schema before it is
  written.

The tools do not guess or generate missing relationship IDs. For example, they
do not choose the first analysis as the preferred analysis and they do not
invent missing ``analysisID`` or ``velocityProfileID`` values.

Preferred IDs are optional. If the optional analysis table is omitted,
preferred analysis and velocity-profile IDs are ignored with warnings. If
analysis is provided but velocity-profile metadata is omitted, preferred
velocity-profile IDs are ignored with warnings. Invalid explicit IDs remain
errors when the corresponding target metadata is provided.

Troubleshooting
---------------


If the command fails before writing XML, check:

- required files or sheets are present;
- required columns are present;
- Excel sheet names are spelled exactly as expected;
- column names are spelled exactly as expected;
- CSV delimiter matches the file contents, usually ``;``;
- CSV delimiter is not used inside any text values in any of the columns;
- every required row value is filled;
- every provided value conforms to the requirements set by the SiteXML schema;
- every resource identifier ID is unique;
- every ``siteID``, ``siteDescriptionID``, ``analysisID``, and
  ``velocityProfileID``
  that represents a relationship points to a real object;
- ``preferredSiteAnalysisID`` and ``preferredVelocityProfileID`` point to
  objects
  that are actually provided in the input tables;

Warnings usually mean optional enrichment was skipped or an unresolved optional
preferred ID was omitted from generated XML. Errors mean the input could not be
converted into a valid, schema-validated SiteXML document.

See the :doc:`SiteXML tabular input reference
</tutorial/code_snippets/sitexml_tabular_input>` for more details on tabular
import and the
`SiteXML schema documentation <https://www.itsak.gr/SiteXML>`_ for the
accepted values in each column.
"""
