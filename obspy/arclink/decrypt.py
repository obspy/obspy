# -*- coding: utf-8 -*-
"""
Decryption class of ArcLink/WebDC client for ObsPy.

.. seealso:: http://www.seiscomp3.org/wiki/doc/applications/arclink-encryption

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

try:
    from M2Crypto import EVP
    hasM2Crypto = True
except ImportError:
    hasM2Crypto = False


class SSLWrapper:
    """
    """
    def __init__(self, password):
        if not hasM2Crypto:
            raise Exception("Module M2Crypto was not found on this system.")
        self._cypher = None
        self._password = None
        if password is None:
            raise Exception('Password should not be Empty')
        else:
            self._password = password

    def update(self, chunk):
        if self._cypher is None:
            if len(chunk) < 16:
                raise Exception('Invalid first chunk (Size < 16).')
            if chunk[0:8] != "Salted__":
                raise Exception('Invalid first chunk (expected: Salted__')
            [key, iv] = self._getKeyIv(self._password, chunk[8:16])
            self._cypher = EVP.Cipher('des_cbc', key, iv, 0)
            chunk = chunk[16:]
        if len(chunk) > 0:
            return self._cypher.update(chunk)
        else:
            return ''

    def final(self):
        if self._cypher is None:
            raise Exception('Wrapper has not started yet.')
        return self._cypher.final()

    def _getKeyIv(self, password, salt=None, size=8):
        chunk = None
        key = ""
        iv = ""
        while True:
            hash = EVP.MessageDigest('md5')
            if (chunk is not None):
                hash.update(chunk)
            hash.update(password)
            if (salt is not None):
                hash.update(salt)
            chunk = hash.final()
            i = 0
            if len(key) < size:
                i = min(size - len(key), len(chunk))
                key += chunk[0:i]
            if len(iv) < size and i < len(chunk):
                j = min(size - len(iv), len(chunk) - i)
                iv += chunk[i:i + j]
            if (len(key) == size and len(iv) == size):
                break
        return [key, iv]
