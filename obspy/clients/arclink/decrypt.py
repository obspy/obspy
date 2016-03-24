# -*- coding: utf-8 -*-
"""
Decryption class of ArcLink/WebDC client for ObsPy.

.. seealso:: https://www.seiscomp3.org/wiki/doc/applications/arclink-encryption

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import hashlib

try:
    from M2Crypto import EVP
    hasM2Crypto = True
except ImportError:
    hasM2Crypto = False

try:
    from Crypto.Cipher import DES
    hasPyCrypto = True
except ImportError:
    hasPyCrypto = False


class SSLWrapper:
    """
    """
    def __init__(self, password):
        if not (hasM2Crypto or hasPyCrypto):
            raise ImportError("Module M2Crypto or PyCrypto is needed.")
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
            if chunk[0:8] != b"Salted__":
                raise Exception('Invalid first chunk (expected: Salted__')
            [key, iv] = self._get_key_iv(self._password, chunk[8:16])
            if hasM2Crypto:
                self._cypher = EVP.Cipher('des_cbc', key, iv, 0)
            else:
                self._cypher = DES.new(key, DES.MODE_CBC, iv)
            chunk = chunk[16:]
        if len(chunk) > 0:
            if hasM2Crypto:
                return self._cypher.update(chunk)
            else:
                return self._cypher.decrypt(chunk)
        else:
            return b""

    def final(self):
        if self._cypher is None:
            raise Exception('Wrapper has not started yet.')
        if hasM2Crypto:
            return self._cypher.final()
        else:
            return b""

    def _get_key_iv(self, password, salt=None, size=8):
        # make sure password is a string and not unicode
        password = password.encode('utf-8')
        chunk = None
        key = b""
        iv = b""
        while True:
            hash = hashlib.md5()
            if (chunk is not None):
                hash.update(chunk)
            hash.update(password)
            if (salt is not None):
                hash.update(salt)
            chunk = hash.digest()
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
