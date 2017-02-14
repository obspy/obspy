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


# M2Crypto - default choice - raises early if wrong password
try:
    from M2Crypto import EVP
    HAS_M2CRYPTO = True
except ImportError:
    HAS_M2CRYPTO = False

# cryptography - a bit slower due to 3DES
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, \
        modes
    from cryptography.hazmat.backends import default_backend
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

# PyCrypto - last fallback
try:
    from Crypto.Cipher import DES
    HAS_PYCRYPTO = True
except ImportError:
    HAS_PYCRYPTO = False


HAS_CRYPTOLIB = HAS_M2CRYPTO or HAS_PYCRYPTO or HAS_CRYPTOGRAPHY


class SSLWrapper:
    """
    """
    def __init__(self, password):
        if not HAS_CRYPTOLIB:
            raise ImportError(
                'M2Crypto, PyCrypto or cryptography is not installed')
        self._cipher = None
        self._password = None
        if password is None:
            raise Exception('Password should not be Empty')
        else:
            self._password = password

    def update(self, chunk):
        if self._cipher is None:
            if len(chunk) < 16:
                raise Exception('Invalid first chunk (Size < 16).')
            if chunk[0:8] != b"Salted__":
                raise Exception('Invalid first chunk (expected: Salted__')
            [key, iv] = self._get_key_iv(self._password, chunk[8:16])
            if HAS_M2CRYPTO:
                self._cipher = EVP.Cipher('des_cbc', key, iv, 0)
            elif HAS_CRYPTOGRAPHY:
                self._cipher = Cipher(algorithms.TripleDES(key), modes.CBC(iv),
                                      backend=default_backend()).decryptor()
            else:
                self._cipher = DES.new(key, DES.MODE_CBC, iv)
            chunk = chunk[16:]
        if len(chunk) > 0:
            if HAS_M2CRYPTO:
                return self._cipher.update(chunk)
            elif HAS_CRYPTOGRAPHY:
                return self._cipher.update(chunk)
            else:
                return self._cipher.decrypt(chunk)
        else:
            return b""

    def final(self):
        if self._cipher is None:
            raise Exception('Wrapper has not started yet.')
        if HAS_M2CRYPTO:
            return self._cipher.final()
        elif HAS_CRYPTOGRAPHY:
            return self._cipher.finalize()
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
