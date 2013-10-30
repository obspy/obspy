# -*- coding: utf-8 -*-
import json
from obspy.core.json import Default


def writeJSON(catalog, filename, omit_nulls=False, pretty_print=True,
              **kwargs):
    """
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
        json_string = json.dumps(catalog, default=default, **kwargs)
        fh.write(json_string)
    finally:
        # Close if a file has been opened by this function.
        if file_opened is True:
            fh.close()
