# -*- coding: utf-8 -*-
"""
SCML (formerly SC3ML) events read and write support.

:author:
    EOST (École et Observatoire des Sciences de la Terre)
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import io
from pathlib import Path
import re

from lxml import etree

from obspy.io.quakeml.core import Pickler, Unpickler, _xml_doc_from_anything
from obspy.io.seiscomp.core import validate as validate_scml

# note that for schema 0.9 and below the following is NOT FIXED
# but for legacy reason remains as-is.
#
#  * 26.07.2024:
#    - Fix origin/confidenceEllipsoid conversion. The unit for
#      'semiMajorAxisLength', 'semiMinorAxisLength' and
#      'semiIntermediateAxisLength' is already meter and does not need a
#      conversion.

SCHEMA_VERSION = ['0.7', '0.8', '0.9', '0.10',
                  '0.11', '0.12', '0.13', '0.14']
# from version 0.14 onwards "sc3ml" is dropped
NEW_SCHEMA_VERSION = ['0.14']


def _read_scml(filename, id_prefix='smi:org.gfz-potsdam.de/geofon/'):
    """
    Read a SeisComp XML file and returns a :class:`~obspy.core.event.Catalog`.

    An XSLT file is used to convert the SCML file to a QuakeML file. The
    catalog is then generated using the QuakeML module.

    .. warning::

        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.event.catalog.Catalog.write` method of an
        ObsPy :class:`~obspy.core.event.catalog.Catalog` object, call this
        instead.

    :type filename: str
    :param filename: SCML file to be read.
    :type id_prefix: str
    :param id_prefix: ID prefix. SCML does not enforce any particular ID
        restriction, this ID prefix allows to convert the IDs to a well
        formatted QuakeML ID. You can modify the default ID prefix with the
        reverse DNS name of your institute.
    :rtype: :class:`~obspy.core.event.Catalog`
    :return: An ObsPy Catalog object.

    .. rubric:: Example

    >>> from obspy import read_events
    >>> cat = read_events('/path/to/iris_events.sc3ml')
    >>> print(cat)
    2 Event(s) in Catalog:
    2011-03-11T05:46:24.120000Z | +38.297, +142.373
    2006-09-10T04:26:33.610000Z |  +9.614, +121.961
    """
    scml_doc = _xml_doc_from_anything(filename)

    match = re.match(
        r'{http://geofon\.gfz(?:-potsdam)?\.de/ns/seiscomp3?-schema/([-+]?'
        r'[0-9]*\.?[0-9]+)}', scml_doc.tag)

    try:
        version = match.group(1)
    except AttributeError:
        raise ValueError("Not a SCML compatible file or string.")
    else:
        if version not in SCHEMA_VERSION:
            message = ("Can't read SCML version %s, ObsPy can handle "
                       "versions [%s].") % (
                version, ', '.join(SCHEMA_VERSION))
            raise ValueError(message)

    xslt_filename = Path(__file__).parent / 'data'
    if version in NEW_SCHEMA_VERSION:
        xslt_filename = xslt_filename / f'scml_{version}__quakeml_1.2.xsl'
        # this doesn't need to be set necessarily, but may as well stay in tune
        id_prefix = 'smi:org.gfz.de/geofon/'
    else:
        xslt_filename = xslt_filename / f'sc3ml_{version}__quakeml_1.2.xsl'

    transform = etree.XSLT(etree.parse(str(xslt_filename)))
    quakeml_doc = transform(scml_doc,
                            ID_PREFIX=etree.XSLT.strparam(id_prefix))

    return Unpickler().load(io.BytesIO(quakeml_doc))


def _write_scml(catalog, filename, validate=False, verbose=False,
                event_removal=False, version='0.12',
                **kwargs):  # @UnusedVariable
    """
    Write a SCML event file at desired version. Since a XSLT file is used to
    write the SCML file from a QuakeML file, the catalog is first
    converted in QuakeML.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.event.catalog.Catalog.write` method of an
        ObsPy :class:`~obspy.core.event.catalog.Catalog` object, call this
        instead.

    :type catalog: :class:`~obspy.core.event.catalog.Catalog`
    :param catalog: The ObsPy Catalog object to write.
    :type filename: str or file
    :param filename: Filename to write or open file-like object
    :type validate: bool
    :param validate: If True, the final SCML file will be validated against
        the SCML schema file. Raises an AssertionError if the validation
        fails.
    :type verbose: bool
    :param verbose: Print validation error log if True.
    :type event_deletion: bool
    :param event_removal: If True, the event elements will be removed. This can
        be useful to associate origins with scevent when injecting SCML file
        into seiscomp.
    :type version: str
    :param version: SCML version to output (default is 0.12 ~ SC 5.0)
    """
    if version not in SCHEMA_VERSION:
        raise ValueError('%s is not a supported version. Use one of these '
                         'versions: [%s].'
                         % (version, ', '.join(SCHEMA_VERSION)))
    nsmap_ = getattr(catalog, "nsmap", {})
    quakeml_doc = Pickler(nsmap=nsmap_).dumps(catalog)
    xslt_filename = Path(__file__).parent / 'data'
    if version in NEW_SCHEMA_VERSION:
        xslt_filename = xslt_filename / f'quakeml_1.2__scml_{version}.xsl'
    else:
        xslt_filename = xslt_filename / f'quakeml_1.2__sc3ml_{version}.xsl'
    transform = etree.XSLT(etree.parse(str(xslt_filename)))
    scml_doc = transform(etree.parse(io.BytesIO(quakeml_doc)))

    # Remove events
    if event_removal:
        for event in scml_doc.xpath("//*[local-name()='event']"):
            event.getparent().remove(event)

    if validate and not validate_scml(io.BytesIO(scml_doc), verbose=verbose):
        raise AssertionError("The final SCML file did not pass validation.")

    # Open filehandler or use an existing file like object
    try:
        with open(filename, 'wb') as fh:
            fh.write(scml_doc)
    except TypeError:
        filename.write(scml_doc)
