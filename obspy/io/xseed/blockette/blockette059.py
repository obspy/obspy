# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from .blockette import Blockette
from ..fields import Integer, VariableString


class Blockette059(Blockette):
    """
    Blockette 059: Channel Comment Blockette.

    Sample:
    05900351989,001~1989,004~4410000000
    """

    id = 59
    name = "Channel Comment"
    fields = [
        VariableString(3, "Beginning of effective time", 1, 22, 'T'),
        VariableString(4, "End effective time", 0, 22, 'T', optional=True),
        Integer(5, "Comment code key", 4, xpath=31),
        Integer(6, "Comment level", 6, ignore=True)
    ]
