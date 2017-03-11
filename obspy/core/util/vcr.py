# -*- coding: utf-8 -*-
"""
VCR decorator for capturing and simulating network communication

Any Python socket communication in unittests (decorated with the @vcr function)
and/or doctests (containing a # doctest: +VCR) will be recorded on the first
run and saved into a special 'vcrtapes' directory as single pickled file for
each test case. Future test runs will reuse those recorded network session
allowing for faster tests without any network connection. In order to create
a new recording one just needs to remove/rename the pickled session file(s).

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA @UnusedWildImport
from future.utils import PY2

import copy
import io
import os
import pickle
import select
import socket
import sys
import tempfile
import time
import warnings


VCR_RECORD = 0
VCR_PLAYBACK = 1

vcr_debug = False
vcr_playlist = []
vcr_status = VCR_RECORD


orig_socket = socket.socket
orig_select = select.select
orig_getaddrinfo = socket.getaddrinfo
orig_gethostbyname = socket.gethostbyname
orig_gethostname = socket.gethostname
orig_sleep = time.sleep


def vcr_getaddrinfo(*args, **kwargs):
    global vcr_status, vcr_playlist
    if vcr_status == VCR_RECORD:
        # record mode
        value = orig_getaddrinfo(*args, **kwargs)
        vcr_playlist.append(
            ('getaddrinfo', args, kwargs, copy.copy(value)))
        return value
    else:
        # playback mode
        data = vcr_playlist.pop(0)
        return data[3]


def vcr_gethostbyname(host):
    global vcr_status
    if vcr_status == VCR_RECORD:
        return orig_gethostbyname(host)
    else:
        return '127.0.0.1'


def vcr_gethostname():
    global vcr_status
    if vcr_status == VCR_RECORD:
        return orig_getaddrinfo()
    else:
        return 'localhost'


def vcr_select(r, w, x, timeout=None):
    global vcr_status
    # Windows only
    if sys.platform == 'win32' and vcr_status == VCR_PLAYBACK:
        return list(r), list(w), []
    return orig_select(r, w, x, timeout)


def vcr_sleep(*args, **kwargs):
    global vcr_status
    if vcr_status == VCR_PLAYBACK:
        return
    return orig_sleep(*args, **kwargs)


class VCRSockFile(object):
    def __init__(self):
        self.file = tempfile.TemporaryFile()
        self._fileno = self.file.fileno()

    def getvalue(self):
        if hasattr(self.file, 'getvalue'):
            return self.file.getvalue()
        else:
            return self.file.read()

    def close(self):
        self.socket.close()
        self.file.close()

    def fileno(self):
        return self._fileno

    def __getattr__(self, name):
        return getattr(self.file, name)


class VCRSocket(object):
    _skip_methods = ['setsockopt', 'settimeout', 'setblocking', 'close']

    def __init__(self, *args, **kwargs):
        global vcr_debug, vcr_status
        if vcr_debug:
            print('  __init__', args, kwargs)
        self.recording = vcr_status == VCR_RECORD
        self._orig_socket = orig_socket(*args, **kwargs)
        self.fd = VCRSockFile()
        self.fd.socket = self._orig_socket

    def __del__(self):
        self.fd.close()

    def _generic_method(self, name, *args, **kwargs):
        global vcr_debug, vcr_playlist
        global vcr_recv_timeout, vcr_recv_endmarker, vcr_recv_size

        if self.recording:
            # record mode
            value = getattr(self._orig_socket, name)(*args, **kwargs)
            if vcr_debug:
                print('  ', name, args, kwargs, value)
            # handle special objects which are not pickleable
            if isinstance(value, io.BufferedIOBase) and \
               not isinstance(value, io.BytesIO):
                temp = io.BytesIO()
                self._orig_socket.setblocking(0)
                self._orig_socket.settimeout(vcr_recv_timeout)
                begin = time.time()
                # recording is slightly slower than disabled vcr due to
                # waiting for socket timeouts
                while True:
                    if temp.tell() and time.time() - begin > vcr_recv_timeout:
                        # if you got some data, then break after wait sec
                        break
                    elif time.time() - begin > vcr_recv_timeout * 2:
                        # if you got no data at all, wait a little longer
                        break

                    try:
                        if vcr_recv_size:
                            data = value.read(len(vcr_recv_size))
                        else:
                            peeked_bytes = value.peek()
                            data = value.read(len(peeked_bytes))
                        if data:
                            temp.write(data)
                            begin = time.time()
                        else:
                            time.sleep(0.1 * vcr_recv_timeout)
                        # speed up closing socket by checking for end marker
                        # (e.g. arclink uses e.g. "END\r\n" as identifier) or
                        # by a given recv length
                        if vcr_recv_endmarker and \
                           data.endswith(vcr_recv_endmarker):
                            break
                        elif vcr_recv_size:
                            break
                    except socket.error:
                        break
                temp.seek(0)
                vcr_playlist.append((name, args, kwargs, temp))
                # return new copy of BytesIO as it may get closed
                return copy.copy(temp)
            if vcr_debug:
                # test if value is pickleable - will raise exception
                pickle.dumps(value)
            # skip setters
            if name not in self._skip_methods:
                # all other methods will be recorded
                vcr_playlist.append(
                    (name, args, kwargs, copy.copy(value)))
            return value
        else:
            # playback mode
            if name in self._skip_methods:
                # skip setters which do not return anything
                return
            # always work on first element in playlist list
            data = vcr_playlist.pop(0)
            # XXX: arclink again - no idea why but works with this hack
            if name == 'recv' and data[0] == 'fileno':
                data = vcr_playlist.pop(0)
            # XXX: py < 3.5 has sometimes two sendall calls ???
            if sys.version_info < (3, 5) and name == 'makefile' and \
               data[0] == 'sendall':
                data = vcr_playlist.pop(0)
            value = data[3]
            if name in ['fileno']:
                value = self.fd.fileno()
            if vcr_debug:
                print('  ', name, args, kwargs, '|', data, '->', value)
            return value

    def __nonzero__(self):
        return bool(self.__dict__.get('_orig_socket', True))

    def send(self, *args, **kwargs):
        return self._generic_method('send', *args, **kwargs)

    def sendall(self, *args, **kwargs):
        return self._generic_method('sendall', *args, **kwargs)

    def fileno(self, *args, **kwargs):
        return self._generic_method('fileno', *args, **kwargs)

    def makefile(self, *args, **kwargs):
        return self._generic_method('makefile', *args, **kwargs)

    def getsockopt(self, *args, **kwargs):
        return self._generic_method('getsockopt', *args, **kwargs)

    def setsockopt(self, *args, **kwargs):
        return self._generic_method('setsockopt', *args, **kwargs)

    def recv(self, *args, **kwargs):
        return self._generic_method('recv', *args, **kwargs)

    def close(self, *args, **kwargs):
        return self._generic_method('close', *args, **kwargs)

    def gettimeout(self, *args, **kwargs):
        return self._generic_method('gettimeout', *args, **kwargs)

    def settimeout(self, *args, **kwargs):
        return self._generic_method('settimeout', *args, **kwargs)

    def setblocking(self, *args, **kwargs):
        return self._generic_method('setblocking', *args, **kwargs)

    def connect(self, *args, **kwargs):
        return self._generic_method('connect', *args, **kwargs)

    def detach(self, *args, **kwargs):
        return self._generic_method('detach', *args, **kwargs)

    @property
    def family(self):
        if self.recording:
            return self._orig_socket.family

    @property
    def type(self):
        if self.recording:
            return self._orig_socket.type

    @property
    def proto(self):
        if self.recording:
            return self._orig_socket.proto


class VCR(object):
    """
    """
    @classmethod
    def reset(cls, debug=False):
        global vcr_playlist, vcr_status, vcr_debug
        vcr_playlist = []
        vcr_status = VCR_RECORD
        vcr_debug = debug
        vcr_recv_timeout = 2
        vcr_recv_endmarker = None
        vcr_recv_size = None

    @classmethod
    def start(cls, debug=False):
        # reset
        cls.reset(debug)
        # monkey patching
        socket.socket = VCRSocket
        socket.getaddrinfo = vcr_getaddrinfo
        socket.gethostbyname = vcr_gethostbyname
        socket.gethostname = vcr_gethostname
        select.select = vcr_select
        time.sleep = vcr_sleep  # skips arclink sleep calls during playback

    @classmethod
    def stop(cls):
        # revert monkey patches
        socket.socket = orig_socket
        socket.getaddrinfo = orig_getaddrinfo
        socket.gethostbyname = orig_gethostbyname
        socket.gethostname = orig_gethostname
        select.select = orig_select
        time.sleep = orig_sleep
        # reset
        cls.reset()


def vcr(decorated_func=None, overwrite=False, debug=False, disabled=False,
        recv_timeout=2, recv_endmarker=None, recv_size=None):
    """
    Wrapper around _vcr_inner allowing additional arguments on decorator
    """
    global vcr_recv_timeout, vcr_recv_endmarker, vcr_recv_size

    vcr_recv_timeout = recv_timeout
    vcr_recv_endmarker = recv_endmarker
    vcr_recv_size = recv_size

    def _vcr_outer(func):
        def _vcr_inner(*args, **kwargs):
            """
            The actual decorator doing a lot of monkey patching and auto magic
            """
            global vcr_playlist, vcr_status, vcr_debug

            if disabled:
                # execute decorated function without VCR
                return func(*args, **kwargs)

            # enable VCR
            VCR.start(debug)

            # prepare VCR tape
            if func.__module__ == 'doctest':
                source_filename = func.__self__._dt_test.filename
                file_name = os.path.splitext(
                    os.path.basename(source_filename))[0]
                path = os.path.join(os.path.dirname(source_filename), 'tests',
                                    'vcrtapes')
                func_name = func.__self__._dt_test.name.split('.')[-1]
            else:
                source_filename = func.__code__.co_filename
                file_name = os.path.splitext(
                    os.path.basename(source_filename))[0]
                path = os.path.join(
                    os.path.dirname(source_filename), 'vcrtapes')
                func_name = func.__name__

            # make sure 'vcrtapes' directory exists
            if not os.path.isdir(path):
                os.makedirs(path)
            tape = os.path.join(path, '%s.%s.vcr' % (file_name, func_name))

            # check for arclink module
            if 'arclink' in source_filename:
                vcr_recv_endmarker = 'END\r\n'
            else:
                vcr_recv_endmarker = ''

            # check for tape file and determine mode
            if os.path.isfile(tape) is False or overwrite is True:
                # record mode
                if PY2:
                    msg = 'VCR records only in PY3 to be backward ' + \
                          'compatible with PY2 - skipping VCR mechanics for %s'
                    warnings.warn(msg % (func.__name__))
                    # disable VCR
                    VCR.stop()
                    # execute decorated function without VCR
                    return func(*args, **kwargs)
                if debug:
                    print('\nVCR RECORDING (%s) ...' % (func_name))
                vcr_status = VCR_RECORD
                # execute decorated function
                value = func(*args, **kwargs)
                # write to file
                if len(vcr_playlist) == 0:
                    msg = 'no socket activity - @vcr decorator unneeded for %s'
                    warnings.warn(msg % (func.__name__))
                else:
                    with open(tape, 'wb') as fh:
                        pickle.dump(vcr_playlist, fh, protocol=2)
            else:
                # playback mode
                if debug:
                    print('\nVCR PLAYBACK (%s) ...' % (func_name))
                vcr_status = VCR_PLAYBACK
                # load playlist
                with open(tape, 'rb') as fh:
                    vcr_playlist = pickle.load(fh)
                # execute decorated function
                try:
                    value = func(*args, **kwargs)
                except Exception:
                    if vcr_debug:
                        raise
                    value = None
                    msg = '@vcr playback raised an exception for %s'
                    warnings.warn(msg % (func_name))

            # disable VCR
            VCR.stop()

            return value

        return _vcr_inner

    if decorated_func is None:
        # without arguments
        return _vcr_outer
    else:
        # with arguments
        return _vcr_outer(decorated_func)
