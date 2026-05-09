#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A command-line program that converts SiteXML Excel metadata into SiteXML files.
"""
from argparse import ArgumentParser
from pathlib import Path

from obspy import __version__
from obspy.io.sitexml.sitexml import sitedict_to_sitexml
from obspy.io.sitexml.tabular import excel_to_sera_site


def excel2serasite(options):
    """
    Import Excel metadata and write one SiteXML file per site.
    """
    sera_site_dict = excel_to_sera_site(
        path_or_file_object=options.path_or_file_object,
        velocity_profiles=options.velocity_profiles)
    output_folder = Path(options.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    sitedict_to_sitexml(sera_site_dict, output_folder=output_folder)
    print("Wrote %d SiteXML file(s) to %s." % (
        len(sera_site_dict), output_folder))


def main(argv=None):
    parser = ArgumentParser(prog='excel2serasite',
                            description=__doc__.strip())
    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s ' + __version__)
    parser.add_argument(
        '-out', '--output-folder', required=True,
        help='folder where the generated SiteXML files will be written. '
            'Existing files with the same name will be overwritten.')
    parser.add_argument(
        '-p', '--velocity-profiles',
        help='optional Excel file or folder with velocity-profile metadata')
    parser.add_argument(
        'path_or_file_object',
        help='Excel file with site metadata')
    args = parser.parse_args(argv)

    excel2serasite(args)
    return 0


if __name__ == "__main__":
    main()
