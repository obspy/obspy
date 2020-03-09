# -*- coding: utf-8 -*-
from .blockette import Blockette
from ..fields import Integer, Loop, VariableString


class Blockette012(Blockette):
    """
    Blockette 012: Volume Time Span Index Blockette.

    This blockette forms an index to the time spans that encompass the actual
    data. One index entry exists for each time span recorded later in the
    volume. Time spans are not used for field station type volumes. There
    should be one entry in this index for each time span control header.
    (For more information, see the notes for blockettes [70], [73], and [74].)

    Sample:
    012006300011992,001,00:00:00.0000~1992,002,00:00:00.0000~000014
    """

    id = 12
    name = "Volume Timespan Index"
    fields = [
        Integer(3, "Number of spans in table", 4),
        # REPEAT fields 4 — 6 for the Number of spans in table:
        Loop("Timespan", "Number of spans in table", [
            VariableString(4, "Beginning of span", 1, 22, 'T'),
            VariableString(5, "End of span", 1, 22, 'T'),
            Integer(6, "Sequence number of time span header", 6, ignore=True)
        ], optional=True),
    ]
