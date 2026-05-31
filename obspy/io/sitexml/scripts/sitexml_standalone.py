#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dispatcher for the standalone SiteXML command-line executables.
"""
import sys
from pathlib import Path

from obspy.io.sitexml.scripts.csv2sitexml import main as csv_main
from obspy.io.sitexml.scripts.excel2sitexml import main as excel_main


def main(argv=None):
    program_name = Path(sys.argv[0]).stem.lower()
    if program_name == "csv2sitexml":
        return csv_main(argv)
    if program_name == "excel2sitexml":
        return excel_main(argv)
    raise SystemExit(
        "This executable must be named csv2sitexml or excel2sitexml.")


if __name__ == "__main__":
    main()
