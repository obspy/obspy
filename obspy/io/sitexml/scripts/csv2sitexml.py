#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A command-line program that converts SiteXML CSV metadata into SiteXML files.
"""
from argparse import ArgumentParser
from pathlib import Path

from obspy import __version__
from obspy.io.sitexml.sitexml import sitedict_to_sitexml
from obspy.io.sitexml.tabular import csv_to_sera_site


def csv2sitexml(options):
    """
    Import CSV metadata and write one SiteXML file per site.
    """
    sera_site_dict = csv_to_sera_site(
        site_owner_csv=options.site_owner,
        site_description_csv=options.site_description,
        analysis_csv=options.analysis,
        velocity_profiles_csv=options.velocity_profiles,
        quality_index_csv=options.quality_index,
        delim=options.delim)
    output_folder = Path(options.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    sitedict_to_sitexml(sera_site_dict, output_folder=output_folder)
    print("Wrote %d SiteXML file(s) to %s." % (
        len(sera_site_dict), output_folder))


def main(argv=None):
    parser = ArgumentParser(prog='csv2sitexml',
                            description=__doc__.strip())
    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s ' + __version__)
    parser.add_argument(
        '-out', '--output-folder', required=True,
        help='folder where the generated SiteXML files will be written. '
        'Existing files with the same name will be overwritten.')
    parser.add_argument(
        '-o', '--site-owner', required=True,
        help='CSV file with site owner metadata')
    parser.add_argument(
        '-d', '--site-description', required=True,
        help='CSV file with site description metadata')
    parser.add_argument(
        '-a', '--analysis',
        help='optional CSV file with analysis metadata')
    parser.add_argument(
        '-p', '--velocity-profiles',
        help='optional CSV file or folder with velocity-profile metadata')
    parser.add_argument(
        '-q', '--quality-index',
        help='optional CSV file with quality-index calculation inputs')
    parser.add_argument(
        '-s', '--delim', default=';',
        help="CSV delimiter, defaults to ';'")
    args = parser.parse_args(argv)

    csv2sitexml(args)
    return 0


if __name__ == "__main__":
    main()
