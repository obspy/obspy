# -*- coding: utf-8 -*-
HEADER_STARTS = {
    'origins': ['date', 'time', 'err', 'rms'],
    'bibliography': ['year', 'volume', 'page1', 'page2'],
    'magnitudes': ['magnitude', 'err', 'nsta', 'author'],
    'phases': ['sta', 'dist', 'evaz', 'phase'],
    }


def _block_header(line):
    """
    Return name of block type as string or False
    """
    first_parts = [x.lower() for x in line.split()[:4]]
    if first_parts[0] == 'event':
        return 'event'
    for block_type, header_start in HEADER_STARTS.items():
        if first_parts == header_start:
            return block_type
    return False


def evaluation_mode_and_status(my_string):
    """
    Return QuakeML standard evaluation mode and status based on the single
    field ISF "analysis type" field.
    """
    my_string = my_string.lower()
    # TODO check this matching
    if my_string == 'a':
        mode = 'automatic'
        status = None
    elif my_string == 'm':
        mode = 'manual'
        status = 'reviewed'
    elif my_string == 'g':
        mode = 'manual'
        status = 'preliminary'
    elif not my_string.strip():
        return None, None
    else:
        raise ValueError()
    return mode, status


def type_or_none(my_string, type_, multiplier=None):
    my_string = my_string.strip() or None
    my_string = my_string and type_(my_string)
    if my_string is not None and multiplier is not None:
        my_string = my_string * multiplier
    return my_string and type_(my_string)


def float_or_none(my_string, **kwargs):
    return type_or_none(my_string, float)


def int_or_none(my_string, **kwargs):
    return type_or_none(my_string, int)


def fixed_flag(my_char):
    if len(my_char) != 1:
        raise ValueError()
    return my_char.lower() == 'f'
