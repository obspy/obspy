# -*- coding: utf-8 -*-
"""
VCR - decorator for capturing and simulating network communication

Any Python socket communication in unittests (decorated with the @vcr function)
and/or doctests (containing a # doctest: +VCR) will be recorded on the first
run and saved into a special 'vcrtapes' directory as single pickled file for
each test case. Future test runs will reuse those recorded network session
allowing for faster tests without any network connection. In order to create
a new recording one just needs to remove/rename the pickled session file(s).

Inspired by:
 * https://docs.python.org/3.6/howto/sockets.html
 * https://github.com/gabrielfalcao/HTTPretty
 * http://code.activestate.com/recipes/408859/

:copyright:
    Robert Barsch (barsch@egu.eu)
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
import ssl
import sys
import tempfile
import time
import warnings

from .utils import classproperty


VCR_RECORD = 0
VCR_PLAYBACK = 1


orig_socket = socket.socket
orig_sslsocket = ssl.SSLSocket
orig_select = select.select
orig_getaddrinfo = socket.getaddrinfo


class VCRSystem(object):
    """
    Use this class to overwrite default settings on global scale

    >>> from vcr import VCRSystem
    >>> VCRSystem.debug = True
    >>> run_my_tests()
    >>> VCRSystem.reset()

    ``debug`` : bool
        Enables debug mode.
    ``overwrite`` : bool
        Will run vcr in recording mode - overwrites any existing vcrtapes.
    ``disabled`` : bool
        Completely disables vcr - same effect as removing the decorator.
    ``recv_timeout`` : int
        Timeout in seconds used to break socket recv calls (default is 3).
    ``recv_endmarkers`` : list of bytes
        List of end markers which is used to check if a socket recv call
        is finished, e.g. [b'\r\n', b'END\r\n']. Will be ignored if its an
        empty list (default).
    ``recv_size`` : int
        Will request given number of bytes in socket recv calls. Option is
        ignored if not set (default).
    """
    debug = False
    disabled = False
    overwrite = False
    recv_timeout = 3
    recv_endmarkers = []
    recv_size = None

    @classmethod
    def reset(cls):
        """
        Reset to default settings
        """
        cls.debug = False
        cls.disabled = False
        cls.overwrite = False
        cls.recv_timeout = 3
        cls.recv_endmarkers = []
        cls.recv_size = None

    @classmethod
    def start(cls):
        # reset
        cls.playlist = []
        cls.status = VCR_RECORD
        # apply monkey patches
        socket.socket = VCRSocket
        ssl.SSLSocket = VCRSSLSocket
        socket.getaddrinfo = vcr_getaddrinfo
        select.select = vcr_select
        # extras
        cls.start_extras()

    @classmethod
    def start_extras(cls):
        pass

    @classmethod
    def stop(cls):
        # revert monkey patches
        socket.socket = orig_socket
        ssl.SSLSocket = orig_sslsocket
        socket.getaddrinfo = orig_getaddrinfo
        select.select = orig_select
        # reset
        cls.playlist = []
        cls.status = VCR_RECORD
        # extras
        cls.stop_extras()

    @classmethod
    def stop_extras(cls):
        pass

    @classproperty
    def is_recording(cls):  # @NoSelf
        return cls.status == VCR_RECORD

    @classproperty
    def is_playing(cls):  # @NoSelf
        return cls.status == VCR_PLAYBACK


def vcr_getaddrinfo(*args, **kwargs):
    if VCRSystem.status == VCR_RECORD:
        # record mode
        value = orig_getaddrinfo(*args, **kwargs)
        VCRSystem.playlist.append(
            ('getaddrinfo', args, kwargs, copy.copy(value)))
        if VCRSystem.debug:
            print('  ', 'vcr_getaddrinfo', args, kwargs, value)
        return value
    else:
        # playback mode
        data = VCRSystem.playlist.pop(0)
        value = data[3]
        if VCRSystem.debug:
            print('  ', 'getaddrinfo', args, kwargs, ' | ', data[0:3],
                  '->', value)
        return value


def vcr_select(r, w, x, timeout=None):
    # Windows only
    if sys.platform == 'win32' and VCRSystem.status == VCR_PLAYBACK:
        return list(r), list(w), []
    return orig_select(r, w, x, timeout)


class VCRSocket(object):
    """
    """
    def __init__(self, family=socket.AF_INET, type=socket.SOCK_STREAM,
                 proto=0, fileno=None, _sock=None):
        if VCRSystem.debug:
            print('  ', '__init__', family, type, proto, fileno)
        self._recording = VCRSystem.is_recording
        self._orig_socket = orig_socket(family, type, proto, fileno)
        # a working file descriptor is needed for telnetlib.Telnet.read_until
        if not self._recording:
            self.fd = tempfile.TemporaryFile()

    def __del__(self):
        if hasattr(self, 'fd'):
            self.fd.close()

    def _exec(self, name, *args, **kwargs):
        if self._recording:
            # record mode
            value = getattr(self._orig_socket, name)(*args, **kwargs)
            if VCRSystem.debug:
                print('  ', name, args, kwargs, value)
            # handle special objects which are not pickleable
            if isinstance(value, io.BufferedIOBase) and \
               not isinstance(value, io.BytesIO):
                temp = io.BytesIO()
                self._orig_socket.setblocking(0)
                self._orig_socket.settimeout(VCRSystem.recv_timeout)
                begin = time.time()
                # recording is slightly slower than running without vcr
                # decorator as we don't know which concept is used to listen
                # on the socket (size, end marker) - we have to wait for a
                # socket timeout - on default its already quite low - but still
                # it introduces a few extra seconds per recv request
                #
                # Note: sometimes recording fails due to the small timeout
                # usually a retry helps - otherwise set the timeout higher for
                # this test case using the recv_timeout parameter
                while True:
                    # endless loop - breaks by checking against recv_timeout
                    if temp.tell() and \
                       time.time() - begin > VCRSystem.recv_timeout:
                        # got some data -> break after recv_timeout
                        break
                    elif time.time() - begin > VCRSystem.recv_timeout * 2:
                        # no data yet -> break after 2 * recv_timeout
                        break

                    try:
                        if VCRSystem.recv_size:
                            data = value.read(len(VCRSystem.recv_size))
                        else:
                            peeked_bytes = value.peek()
                            data = value.read(len(peeked_bytes))
                        if data:
                            temp.write(data)
                            begin = time.time()
                        else:
                            time.sleep(0.1 * VCRSystem.recv_timeout)
                        # speed up closing socket by checking for end markers
                        # by a given recv length
                        if VCRSystem.recv_size:
                            break
                        elif VCRSystem.recv_endmarkers:
                            for marker in VCRSystem.recv_endmarkers:
                                if data.endswith(marker):
                                    break
                    except socket.error:
                        break
                temp.seek(0)
                VCRSystem.playlist.append((name, args, kwargs, temp))
                # return new copy of BytesIO as it may get closed
                return copy.copy(temp)
            if VCRSystem.debug:
                # test if value is pickleable - will raise exception
                pickle.dumps(value)
            # add to playlist
            VCRSystem.playlist.append((name, args, kwargs, copy.copy(value)))
            return value
        else:
            # playback mode
            # get first element in playlist
            data = VCRSystem.playlist.pop(0)
            # XXX: py < 3.5 has sometimes two sendall calls ???
            if sys.version_info < (3, 5) and name == 'makefile' and \
               data[0] == 'sendall':
                data = VCRSystem.playlist.pop(0)
            value = data[3]
            if VCRSystem.debug:
                print('  ', name, args, kwargs, ' | ', data[0:3], '->', value)
            return value

    def __nonzero__(self):
        return bool(self.__dict__.get('_orig_socket', True))

    def send(self, *args, **kwargs):
        return self._exec('send', *args, **kwargs)

    def sendall(self, *args, **kwargs):
        return self._exec('sendall', *args, **kwargs)

    def fileno(self, *args, **kwargs):
        if self._recording:
            value = self._orig_socket.fileno(*args, **kwargs)
        else:
            value = self.fd.fileno()
        if VCRSystem.debug:
            print('  ', 'fileno', args, kwargs, '->', value)
        return value

    def makefile(self, *args, **kwargs):
        return self._exec('makefile', *args, **kwargs)

    def getsockopt(self, *args, **kwargs):
        return self._exec('getsockopt', *args, **kwargs)

    def setsockopt(self, *args, **kwargs):
        if VCRSystem.debug:
            print('  ', 'setsockopt', args, kwargs)
        if self._recording:
            return self._orig_socket.setsockopt(*args, **kwargs)

    def recv(self, *args, **kwargs):
        return self._exec('recv', *args, **kwargs)

    def close(self, *args, **kwargs):
        return self._orig_socket.close(*args, **kwargs)

    def gettimeout(self, *args, **kwargs):
        return self._exec('gettimeout', *args, **kwargs)

    def settimeout(self, *args, **kwargs):
        if VCRSystem.debug:
            print('  ', 'settimeout', args, kwargs)
        if self._recording:
            return self._orig_socket.settimeout(*args, **kwargs)

    def setblocking(self, *args, **kwargs):
        if VCRSystem.debug:
            print('  ', 'setblocking', args, kwargs)
        if self._recording:
            return self._orig_socket.setblocking(*args, **kwargs)

    def connect(self, *args, **kwargs):
        return self._exec('connect', *args, **kwargs)

    def detach(self, *args, **kwargs):
        return self._exec('detach', *args, **kwargs)

    @property
    def family(self):
        return self._orig_socket.family

    @property
    def type(self):
        return self._orig_socket.type

    @property
    def proto(self):
        return self._orig_socket.proto


class VCRSSLSocket(VCRSocket):
    def __init__(self, sock=None, *args, **kwargs):
        if VCRSystem.debug:
            print('  ', '__init__', args, kwargs)
        self._recording = VCRSystem.is_recording
        self._orig_socket = orig_sslsocket(sock=sock._orig_socket,
                                           *args, **kwargs)
        # a working file descriptor is needed for telnetlib.Telnet.read_until
        if not self._recording:
            self.fd = tempfile.TemporaryFile()

    def getpeercert(self, *args, **kwargs):
        return self._exec('getpeercert', *args, **kwargs)


def vcr(decorated_func=None, debug=False, overwrite=False, disabled=False,
        tape_name=None):
    """
    Decorator for capturing and simulating network communication

    ``debug`` : bool, optional
        Enables debug mode.
    ``overwrite`` : bool, optional
        Will run vcr in recording mode - overwrites any existing vcrtapes.
    ``disabled`` : bool, optional
        Completely disables vcr - same effect as removing the decorator.
    ``tape_name`` : str, optional
        Use given custom file name instead of an auto-generated name for the
        tape file.
    """
    def _vcr_outer(func):
        """
        Wrapper around _vcr_inner allowing optional arguments on decorator
        """
        def _vcr_inner(*args, **kwargs):
            """
            The actual decorator doing a lot of monkey patching and auto magic
            """
            if disabled or VCRSystem.disabled:
                # execute decorated function without VCR
                return func(*args, **kwargs)

            # enable VCR
            if debug:
                system_debug = VCRSystem.debug
                VCRSystem.debug = True
            VCRSystem.start()

            # prepare VCR tape
            if func.__module__ == 'doctest':
                source_filename = func.__self__._dt_test.filename
                file_name = os.path.splitext(
                    os.path.basename(source_filename))[0]
                # check if a tests directory exists
                path = os.path.join(os.path.dirname(source_filename),
                                    'tests')
                if os.path.exists(path):
                    # ./test/vcrtapes/tape_name.vcr
                    path = os.path.join(os.path.dirname(source_filename),
                                        'tests', 'vcrtapes')
                else:
                    # ./vcrtapes/tape_name.vcr
                    path = os.path.join(os.path.dirname(source_filename),
                                        'vcrtapes')
                func_name = func.__self__._dt_test.name.split('.')[-1]
            else:
                source_filename = func.__code__.co_filename
                file_name = os.path.splitext(
                    os.path.basename(source_filename))[0]
                path = os.path.join(
                    os.path.dirname(source_filename), 'vcrtapes')
                func_name = func.__name__

            if tape_name:
                # tape file name is given - either full path is given or use
                # 'vcrtapes' directory
                if os.sep in tape_name:
                    temp = os.path.abspath(tape_name)
                    path = os.path.dirname(temp)
                    if not os.path.isdir(path):
                        os.makedirs(path)
                tape = os.path.join(path, '%s' % (tape_name))
            else:
                # make sure 'vcrtapes' directory exists
                if not os.path.isdir(path):
                    os.makedirs(path)
                # auto-generated file name
                tape = os.path.join(path, '%s.%s.vcr' % (file_name, func_name))

            # check for tape file and determine mode
            if not os.path.isfile(tape) or overwrite or VCRSystem.overwrite:
                # remove existing tape
                try:
                    os.remove(tape)
                except OSError:
                    pass
                # record mode
                if PY2:
                    msg = 'VCR records only in PY3 to be backward ' + \
                          'compatible with PY2 - skipping VCR mechanics for %s'
                    warnings.warn(msg % (func.__name__))
                    # disable VCR
                    VCRSystem.stop()
                    # execute decorated function without VCR
                    return func(*args, **kwargs)
                if VCRSystem.debug:
                    print('\nVCR RECORDING (%s) ...' % (func_name))
                VCRSystem.status = VCR_RECORD
                # execute decorated function
                value = func(*args, **kwargs)
                # write to file
                if len(VCRSystem.playlist) == 0:
                    msg = 'no socket activity - @vcr decorator unneeded for %s'
                    warnings.warn(msg % (func.__name__))
                else:
                    with open(tape, 'wb') as fh:
                        pickle.dump(VCRSystem.playlist, fh, protocol=2)
            else:
                # playback mode
                if VCRSystem.debug:
                    print('\nVCR PLAYBACK (%s) ...' % (func_name))
                VCRSystem.status = VCR_PLAYBACK
                # load playlist
                with open(tape, 'rb') as fh:
                    VCRSystem.playlist = pickle.load(fh)
                # execute decorated function
                try:
                    value = func(*args, **kwargs)
                except Exception:
                    VCRSystem.stop()
                    raise

            # disable VCR
            if debug:
                VCRSystem.debug = system_debug
            VCRSystem.stop()

            return value

        return _vcr_inner

    if decorated_func is None:
        # without arguments
        return _vcr_outer
    else:
        # with arguments
        return _vcr_outer(decorated_func)
