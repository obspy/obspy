# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .blockette import Blockette
from ..fields import Integer, VariableString


class Blockette051(Blockette):
    """
    Blockette 051: Station Comment Blockette.

    Sample:
    05100351992,001~1992,002~0740000000
    """

    id = 51
    name = "Station Comment"
    fields = [
        VariableString(3, "Beginning effective time", 1, 22, 'T'),
        VariableString(4, "End effective time", 0, 22, 'T', optional=True),
        Integer(5, "Comment code key", 4, xpath=31),
        Integer(6, "Comment level", 6, ignore=True)
    ]
