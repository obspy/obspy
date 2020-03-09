# -*- coding: utf-8 -*-
import json

from .default import Default


def get_dump_kwargs(minify=True, no_nulls=True, **kwargs):
    """
    Return dict of kwargs for :py:func:`json.dump` or
    :py:func:`json.dumps`.

    :param bool minify: Use no spaces between separators (True)
    :param bool no_nulls: Omit null values and empty sequences/mappings (True)

    """
    if minify:
        kwargs["separators"] = (',', ':')
    kwargs["default"] = Default(omit_nulls=no_nulls)
    return kwargs


def _write_json(obj, filename, omit_nulls=False, pretty_print=True,
                **kwargs):
    """
    Write object to a file in JSON format

    :type obj: :mod:`~obspy.core.event` class object
    :param obj: The ObsPy Event-type object to write.
    :type filename: str or file
    :param filename: Filename to write or open file-like object.
    :type omit_nulls: bool
    :param omit_nulls: Don't include empty-valued attributes
    :type pretty_print: bool
    :param pretty_print: Indent for readability

    .. note::
        This function is registered via the
        :meth:`~obspy.core.event.Catalog.write` method of an ObsPy
        :class:`~obspy.core.event.Catalog` object, but is also valid for any
        obspy event-type object (or any serializable python object that
        contains an obspy event-type object)

    """
    try:
        # Open filehandler or use an existing file like object.
        if not hasattr(filename, "write"):
            file_opened = True
            fh = open(filename, "wt")
        else:
            file_opened = False
            fh = filename

        default = Default(omit_nulls=omit_nulls)
        if pretty_print:
            kwargs.setdefault('indent', 2)
        json_string = str(json.dumps(obj, default=default, **kwargs))
        fh.write(json_string)
    finally:
        # Close if a file has been opened by this function.
        if file_opened is True:
            fh.close()
