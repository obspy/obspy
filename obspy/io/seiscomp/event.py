# -*- coding: utf-8 -*-
"""
scxml/sc3ml events read and write support.

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
from obspy.io.seiscomp.core import validate as validate_scxml


SCHEMA_VERSION = ['0.6', '0.7', '0.8', '0.9', '0.10', '0.11', '0.12']


def _read_sc3ml(filename, id_prefix='smi:org.gfz-potsdam.de/geofon/'):
    """
    Read a SeisComp XML file and returns a :class:`~obspy.core.event.Catalog`.

    An XSLT file is used to convert the SCXML file to a QuakeML file. The
    catalog is then generated using the QuakeML module.

    .. warning::

        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.event.catalog.Catalog.write` method of an
        ObsPy :class:`~obspy.core.event.catalog.Catalog` object, call this
        instead.

    :type filename: str
    :param filename: SCXML file to be read.
    :type id_prefix: str
    :param id_prefix: ID prefix. SCXML does not enforce any particular ID
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
    scxml_doc = _xml_doc_from_anything(filename)

    match = re.match(
        r'{http://geofon\.gfz-potsdam\.de/ns/seiscomp3-schema/([-+]?'
        r'[0-9]*\.?[0-9]+)}', scxml_doc.tag)

    try:
        version = match.group(1)
    except AttributeError:
        raise ValueError("Not a SCXML compatible file or string.")
    else:
        if version not in SCHEMA_VERSION:
            message = ("Can't read SCXML version %s, ObsPy can deal with "
                       "versions [%s].") % (
                version, ', '.join(SCHEMA_VERSION))
            raise ValueError(message)

    xslt_filename = Path(__file__).parent / 'data'
    xslt_filename = xslt_filename / ('sc3ml_%s__quakeml_1.2.xsl' % version)

    transform = etree.XSLT(etree.parse(str(xslt_filename)))
    quakeml_doc = transform(scxml_doc,
                            ID_PREFIX=etree.XSLT.strparam(id_prefix))

    return Unpickler().load(io.BytesIO(quakeml_doc))


def _write_sc3ml(catalog, filename, validate=False, verbose=False,
                 event_removal=False, **kwargs):  # @UnusedVariable
    """
    Write a SCXML 0.12 event file. Since a XSLT file is used to write the
    SCXML file from a QuakeML file, the catalog is first converted in QuakeML.

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
    :param validate: If True, the final SCXML file will be validated against
        the SCXML schema file. Raises an AssertionError if the validation
        fails.
    :type verbose: bool
    :param verbose: Print validation error log if True.
    :type event_deletion: bool
    :param event_removal: If True, the event elements will be removed. This can
        be useful to associate origins with scevent when injecting SCXML file
        into seiscomp.
    """
    nsmap_ = getattr(catalog, "nsmap", {})
    quakeml_doc = Pickler(nsmap=nsmap_).dumps(catalog)
    xslt_filename = Path(__file__).parent / 'data'
    xslt_filename = xslt_filename / 'quakeml_1.2__sc3ml_0.12.xsl'
    transform = etree.XSLT(etree.parse(str(xslt_filename)))
    scxml_doc = transform(etree.parse(io.BytesIO(quakeml_doc)))

    # Remove events
    if event_removal:
        for event in scxml_doc.xpath("//*[local-name()='event']"):
            event.getparent().remove(event)

    if validate and not validate_scxml(io.BytesIO(scxml_doc), verbose=verbose):
        raise AssertionError("The final SCXML file did not pass validation.")

    # Open filehandler or use an existing file like object
    try:
        with open(filename, 'wb') as fh:
            fh.write(scxml_doc)
    except TypeError:
        filename.write(scxml_doc)
