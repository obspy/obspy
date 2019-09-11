# -*- coding: utf-8 -*-
"""
Establish a socket connection through an HTTP proxy.
Author: Fredrik Østrem <frx.apps@gmail.com>
License:
  This code can be used, modified and distributed freely, as long as it is this
  note containing the original author, the source and this license, is put
  along with the source code.

J. MacCarthy, modified from https://gist.github.com/frxstrem/4487802

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA
from future import standard_library
from future.utils import native_str

from base64 import b64encode
import socket
with standard_library.hooks():
    from urllib.request import getproxies
    from urllib.parse import urlparse


def get_proxy_tuple():
    """
    Return system http proxy as a urlparse tuple or () if unset.
    """
    proxydict = getproxies()
    proxystr = proxydict.get('http') or proxydict.get('https') or ''
    if proxystr:
        proxy = urlparse(proxystr)
    else:
        proxy = ()

    return proxy


def valid_address(addr):
    """ Verify that an IP/port tuple is valid """
    is_valid = (isinstance(addr, (list, tuple)) and
                len(addr) == 2 and
                isinstance(addr[0], (str, native_str)) and
                isinstance(addr[1], int))
    return is_valid


def http_proxy_connect(address, proxy, auth=None, timeout=None):
    """
    Establish a socket connection through an HTTP proxy.

    Arguments:
      address (required)     = The address of the target
      proxy   (required)     = The address of the proxy server
      auth  (def: None)      = A tuple of the username and password used for
                               authentication

    Returns:
      A 3-tuple of the format:
        (socket, status_code, headers)
      Where `socket' is the socket object, `status_code` is the HTTP status
      code that the server returned and `headers` is a dict of headers that the
      server returned.

    """
    if not valid_address(address):
        raise ValueError('Invalid target address')

    if not valid_address(proxy):
        raise ValueError('Invalid proxy address')

    headers = {'host': address[0]}

    if auth is not None:
        if isinstance(auth, str):
            headers['proxy-authorization'] = auth
        elif auth and isinstance(auth, (tuple, list)) and len(auth) == 2:
            auth_b64 = b64encode(bytes(('%s:%s' % auth).encode()))
            proxy_authorization = 'Basic ' + auth_b64.decode()
            headers['proxy-authorization'] = proxy_authorization
        else:
            raise ValueError('Invalid authentication specification')

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if timeout is not None:
        s.settimeout(timeout)
    s.connect(proxy)
    fp = s.makefile('rw')

    fp.write('CONNECT %s:%d HTTP/1.0\r\n' % address)
    fp.write('\r\n'.join('%s: %s' % (k, v) for (k, v) in headers.items()) +
             '\r\n\r\n')
    fp.flush()

    statusline = fp.readline().rstrip('\r\n')

    if statusline.count(' ') < 2:
        fp.close()
        s.close()
        raise IOError('Bad response. statusline: {}'.format(statusline))
    version, status, statusmsg = statusline.split(' ', 2)
    if version not in ('HTTP/1.0', 'HTTP/1.1'):
        fp.close()
        s.close()
        raise IOError('Unsupported HTTP version: {}'.format(version))
    try:
        status = int(status)
    except ValueError:
        fp.close()
        s.close()
        raise IOError('Bad response. status: {}'.format(status))

    response_headers = {}

    while True:
        line = fp.readline().rstrip('\r\n')
        if line == '':
            break
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        response_headers[k.strip().lower()] = v.strip()

    fp.close()
    return (s, status, response_headers)
