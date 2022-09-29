# -*- coding: utf-8 -*-
"""
ALSEP module helper functions and data.
"""
from obspy import UTCDateTime


def interp(x1, x2, x3, x4):
    """Interpolate by Lagrange polynomial with two points before and after"""
    return int(((x2 + x3) * 4 - (x1 + x4) + 3) / 6)


def get_utc(year, msec_of_year):
    """Calculate date and time in UTC from year and millisecond of year"""
    return UTCDateTime(year, month=1, day=1) + msec_of_year / 1000.0


def doy2utc(year, doy):
    """Calculate date and time in UTC from year and day of year"""
    return UTCDateTime(year, julday=doy)


def check_date(apollo_station, utc):
    """Check observation date"""
    if apollo_station == 11:
        # 1969/202 - 1969/214, 1969/231 - 1969/237
        if doy2utc(1969, 202) <= utc < doy2utc(1969, 215):
            return True
        if doy2utc(1969, 231) <= utc < doy2utc(1969, 238):
            return True
    elif apollo_station == 12:
        # 1969/323-1977/273
        if doy2utc(1969, 323) <= utc < doy2utc(1977, 274):
            return True
    elif apollo_station == 14:
        # 1971/036-1977/273
        if doy2utc(1971, 36) <= utc < doy2utc(1977, 274):
            return True
    elif apollo_station == 15:
        # 1971/212-1977/273
        if doy2utc(1971, 212) <= utc < doy2utc(1977, 274):
            return True
    elif apollo_station == 16:
        # 1972/112-1977/273
        if doy2utc(1972, 112) <= utc < doy2utc(1977, 274):
            return True
    elif apollo_station == 17:
        # LSPE: 1976/228-1977/115
        # LSG: 1976/061-1977/273
        if doy2utc(1976, 61) <= utc < doy2utc(1977, 273):
            return True
    return False


def check_sync_code(sync_code):
    """Check sync pattern barker code"""
    return True if sync_code == 0b11100010010 else False
