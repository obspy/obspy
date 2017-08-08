#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sc3ml events read and write support.

:author:
    EOST (École et Observatoire des Sciences de la Terre)
:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import io
import os

from lxml import etree

from obspy.io.quakeml.core import Pickler, Unpickler, _xml_doc_from_anything


def _validate_sc3ml(path_or_object, verbose=False):
    """
    Validates a SC3ML file against the SC3ML 0.9 schema. Returns either True or
    False.

    :param path_or_object: File name or file like object. Can also be an etree
        element.
    :type verbose: bool
    :param verbose: Print error log if True.
    """
    # Get the schema location.
    schema_location = os.path.join(os.path.dirname(__file__), 'data',
                                   'sc3ml_0.9.xsd')
    xmlschema = etree.XMLSchema(etree.parse(schema_location))

    if hasattr(path_or_object, "tell") and hasattr(path_or_object, "seek"):
        current_position = path_or_object.tell()

    if isinstance(path_or_object, etree._Element):
        xmldoc = path_or_object
    else:
        try:
            xmldoc = _xml_doc_from_anything(path_or_object)
        except ValueError:
            return False
        finally:
            # Make sure to reset file pointer position.
            try:
                path_or_object.seek(current_position, 0)
            except Exception:
                pass

    valid = xmlschema.validate(xmldoc)

    # Pretty error printing if the validation fails.
    if verbose and valid is not True:
        print("Error validating SC3ML file:")
        for entry in xmlschema.error_log:
            print("\t%s" % entry)

    return valid


def _read_sc3ml(filename, id_prefix='smi:org.gfz-potsdam.de/geofon/'):
    """
    Read a 0.9 SC3ML file and returns a :class:`~obspy.core.event.Catalog`.

    An XSLT file is used to convert the SC3ML file to a QuakeML file. The
    catalog is then generated using the QuakeML module.

    .. warning::
    This function should NOT be called directly, it registers via the
    the :meth:`~obspy.core.event.catalog.Catalog.write` method of an
    ObsPy :class:`~obspy.core.event.catalog.Catalog` object, call this
    instead.

    :type filename: str
    :param filename: SC3ML file to be read.
    :type id_prefix: str
    :param id_prefix: ID prefix. SC3ML does not enforce any particular ID
        restriction, this ID prefix allow to convert the IDs to a well
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
    xslt_filename = os.path.join(os.path.dirname(__file__), 'data',
                                 'sc3ml_0.9__quakeml_1.2.xsl')
    transform = etree.XSLT(etree.parse(xslt_filename))
    sc3ml_doc = _xml_doc_from_anything(filename)
    quakeml_doc = transform(sc3ml_doc,
                            ID_PREFIX=etree.XSLT.strparam(id_prefix))
    return Unpickler().load(io.BytesIO(quakeml_doc))


def _write_sc3ml(catalog, filename, validate=False, verbose=False,
                 event_removal=False, **kwargs):  # @UnusedVariable
    """
    Write a SC3ML file. Since a XSLT file is used to write the SC3ML file from
    a QuakeML file, the catalog is first converted in QuakeML.

    .. warning::
        This function should NOT be called directly, it registers via the
        the :meth:`~obspy.core.event.catalog.Catalog.write` method of an
        ObsPy :class:`~obspy.core.event.catalog.Catalog` object, call this
        instead.

    :type catalog: :class:`~obspy.core.event.catalog.Catalog`
    :param catalog: The ObsPy Catalog object to write.
    :type filename: str or file
    :param filename: Filename to write or open file-like object.
    :type validate: bool
    :param validate: If True, the final SC3ML file will be validated against
        the SC3ML schema file. Raises an AssertionError if the validation
        fails.
    :type verbose: bool
    :param verbose: Print validation error log if True.
    :type event_deletion: bool
    :param event_removal: If True, the event elements will be removed. This can
        be useful to associate origins with scevent when injecting SC3ML file
        into seiscomp.
    """
    nsmap_ = getattr(catalog, "nsmap", {})
    quakeml_doc = Pickler(nsmap=nsmap_).dumps(catalog)
    xslt_filename = os.path.join(os.path.dirname(__file__), 'data',
                                 'quakeml_1.2__sc3ml_0.9.xsl')
    transform = etree.XSLT(etree.parse(xslt_filename))
    sc3ml_doc = transform(etree.parse(io.BytesIO(quakeml_doc)))

    # Remove events
    if event_removal:
        for event in sc3ml_doc.xpath("//*[local-name()='event']"):
            event.getparent().remove(event)

    if validate and not _validate_sc3ml(io.BytesIO(sc3ml_doc), verbose):
        raise AssertionError("The final SC3ML file did not pass validation.")

    # Open filehandler or use an existing file like object.
    if not hasattr(filename, "write"):
        file_opened = True
        fh = open(filename, "wb")
    else:
        file_opened = False
        fh = filename

    try:
        fh.write(sc3ml_doc)
    finally:
        # Close if a file has been opened by this function.
        if file_opened is True:
            fh.close()
