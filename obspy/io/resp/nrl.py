#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
nrl.py for navigating, and picking responses from the
Nominal response library

:copyright:
    Lloyd Carothers IRIS/PASSCAL, 2016
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
'''
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
# Urllib for py 2 and 3
from future.standard_library import install_aliases
install_aliases()
from urllib.parse import urlparse, urlencode, urljoin
from urllib.request import urlopen, Request
from urllib.error import HTTPError

from os.path import join, abspath

class NRL:
    """
    Object representing the Nominal Response library.
    Can be Accessed online from the DMC or with a local copy
    Decision tree:
    Sensor/DL
    Make
    Model
    Gain & sample rate

    """
    URLroot = 'http://ds.iris.edu/NRL/'
    URLroot_index = urljoin(URLroot, 'index.txt')
    def __init__(self, local_path=None, online=False):
        """
        :type local_path: str or os.path
        :param local_path: Root path of a local copy of NRL.
        :type online: Bool
        :param online: Use the online NRL
        """
        if online:
            self.source = self.URLroot
            #test connection
            try:
                with urlopen(self.URLroot_index) as response:
                    indexdata = response.read()
                    print(indexdata)
                    self.parse_index_txt(indexdata)
                    self.get_index = self.get_index_online
            except HTTPError as e:
                print('Could not connect to {}.\n{}'.format(self.URLroot,e))
        if local_path:
            self.source = os.path.abspath(local_path)

    @staticmethod
    def parse_index_txt(data):
        """
        :type data: bytes
        :param data: Contents of index.txt files in NRL returned by read()
        """
        import re
        m = re.findall(r'\[(.*)\]\n+((.*?)\s=\s"(.*?)"\n+)*', data)
        for x in m:
            print(x)
        return m

    def get_parsed_index(self, relativeURL):
        return self.parse_index_txt(relativeURL)

    def get_index_online(self, relative_path):
        with urlopen(urljoin(self.URLroot, relative_path)) as response:
            indexdata = response.read()
            print(indexdata)
            return self.parse_index_txt(indexdata)

    def get_index_local(self, relative_path):
        pass


if __name__ == '__main__':
    nrl = NRL(online=True)
    nrl.get_index('dataloggers/quanterra/index.txt')
