# -*- coding: utf-8 -*-
"""
VCR decorator capturing network socket communication

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""

import copy
import functools
import io
import os
import pickle
import select
import selectors
import socket
import sys
import time
import warnings

from future.utils import PY2


VCR_RECORD = 0
VCR_PLAYBACK = 1


def _vcr_wrapper(func, overwrite=False, debug=False, force_check=False,
                 disabled=False):
    """
    Wrapper around _vcr_inner allowing additional arguments on decorator
    """

    def _vcr_inner(*args, **kwargs):
        """
        The actual decorator doing a lot of monkey patching and auto magic
        """
        if disabled:
            # execute decorated function without VCR
            value = func(*args, **kwargs)

        vcr_playlist = []
        vcr_status = VCR_PLAYBACK
        vcr_arclink_hack = False

        # monkey patch socket.socket
        _orig_socket = socket.socket

        class VCRSocket:
            _skip_methods = ['setsockopt', 'settimeout', 'setblocking']

            def __init__(self, *args, **kwargs):
                if debug:
                    print('__init__', args, kwargs)
                if vcr_status == VCR_RECORD:
                    # record mode
                    self.__dict__['_socket'] = _orig_socket(*args, **kwargs)

            def _generic_method(self, name, *args, **kwargs):
                if vcr_status == VCR_RECORD:
                    # record mode
                    value = getattr(self._socket, name)(*args, **kwargs)
                    if debug:
                        print(name, args, kwargs, value)
                    # handle special objects which are not pickleable
                    if isinstance(value, io.BufferedIOBase) and \
                       not isinstance(value, io.BytesIO):
                        temp = io.BytesIO()
                        while True:
                            try:
                                peeked_bytes = value.peek()
                                bytes = value.read(len(peeked_bytes))
                                if not bytes:
                                    # EOF
                                    break
                                temp.write(bytes)
                                # ugly arclink hack to improve recording
                                # speed - otherwise it takes 20s until timeout
                                if vcr_arclink_hack and \
                                   peeked_bytes.endswith(b'END\r\n'):
                                    break
                            except OSError:
                                # timeout
                                break
                        temp.seek(0)
                        vcr_playlist.append((name, args, kwargs, temp))
                        # return new copy of BytesIO as it may get closed
                        return copy.copy(temp)
                    if debug:
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
                    try:
                        # always work on first element in playlist list
                        data = vcr_playlist.pop(0)
                    except IndexError:
                        # XXX: arclink doctests raise IndexError for some close
                        # calls - no idea yet why - but can be safely ignored
                        if vcr_arclink_hack and name == 'close':
                            return
                        raise
                    # XXX: py3 sometimes has two sendall calls ???
                    if PY2 and name == 'makefile' and data[0] == 'sendall':
                        data = vcr_playlist.pop(0)
                    if debug:
                        print(name, args, kwargs, data)
                    if name != data[0] or force_check:
                        assert (name, args, kwargs) == data[:-1], \
                            '%s != %s' % ((name, args, kwargs), data[:-1])
                    return data[3]

            def __nonzero__(self):
                return bool(self.__dict__.get('_socket', True))

            def send(self, *args, **kwargs):
                return self._generic_method('send', *args, **kwargs)

            def sendall(self, *args, **kwargs):
                return self._generic_method('sendall', *args, **kwargs)

            def fileno(self, *args, **kwargs):
                return self._generic_method('fileno', *args, **kwargs)

            def makefile(self, *args, **kwargs):
                return self._generic_method('makefile', *args, **kwargs)

            def setsockopt(self, *args, **kwargs):
                return self._generic_method('setsockopt', *args, **kwargs)

            def recv(self, *args, **kwargs):
                return self._generic_method('recv', *args, **kwargs)

            def close(self, *args, **kwargs):
                return self._generic_method('close', *args, **kwargs)

            def settimeout(self, *args, **kwargs):
                return self._generic_method('settimeout', *args, **kwargs)

            def setblocking(self, *args, **kwargs):
                return self._generic_method('setblocking', *args, **kwargs)

            def connect(self, *args, **kwargs):
                return self._generic_method('connect', *args, **kwargs)

            # raise for any method not overwritten yet
            def __getattr__(self, attr):
                raise NotImplementedError(attr)

            def __setattr__(self, attr, value):
                raise NotImplementedError(attr)

        socket.socket = VCRSocket

        # monkey patch socket.getaddrinfo
        _orig_getaddrinfo = socket.getaddrinfo

        def vcr_getaddrinfo(*args, **kwargs):
            if vcr_status == VCR_RECORD:
                # record mode
                value = _orig_getaddrinfo(*args, **kwargs)
                vcr_playlist.append(
                    ('getaddrinfo', args, kwargs, copy.copy(value)))
                return value
            else:
                # playback mode
                data = vcr_playlist.pop(0)
                return data[3]

        socket.getaddrinfo = vcr_getaddrinfo

        # monkey patch selectors.SelectSelector._select in Windows systems
        _orig_select = selectors.SelectSelector._select

        if sys.platform == 'win32':
            def vcr_select(self, r, w, _, timeout=None):
                if vcr_status == VCR_PLAYBACK:
                    return list(r), list(w), []
                r, w, x = select.select(r, w, w, timeout)
                return r, w + x, []
            selectors.SelectSelector._select = vcr_select

        # monkey patch time.sleep (prevents sleep calls during playback)
        _orig_sleep = time.sleep

        def vcr_sleep(*args, **kwargs):
            if vcr_status == VCR_PLAYBACK:
                return
            return _orig_sleep(*args, **kwargs)

        time.sleep = vcr_sleep

        # prepare VCR tape
        if func.__module__ == 'doctest':
            source_filename = func.__self__._dt_test.filename
            file_name = os.path.splitext(os.path.basename(source_filename))[0]
            path = os.path.join(os.path.dirname(source_filename), 'tests',
                                'vcrtapes')
            func_name = func.__self__._dt_test.name.split('.')[-1]
        else:
            source_filename = func.__code__.co_filename
            file_name = os.path.splitext(os.path.basename(source_filename))[0]
            path = os.path.join(os.path.dirname(source_filename), 'vcrtapes')
            func_name = func.__name__

        # make sure 'vcrtapes' directory exists
        if not os.path.isdir(path):
            os.makedirs(path)
        tape = os.path.join(path, '%s.%s.vcr' % (file_name, func_name))

        # check for arclink module
        if 'arclink' in source_filename:
            vcr_arclink_hack = True
        else:
            vcr_arclink_hack = False

        # check for tape file and determine mode
        if os.path.isfile(tape) is False or overwrite is True:
            # record mode
            if PY2:
                msg = 'VCR records only in PY3 to be backward ' + \
                    'compatible with PY2 - skipping VCR mechanics for %s'
                warnings.warn(msg % (func.__name__))
                # revert monkey patches
                socket.socket = _orig_socket
                socket.getaddrinfo = _orig_getaddrinfo
                selectors.SelectSelector._select = _orig_select
                time.sleep = _orig_sleep
                # execute decorated function without VCR
                return func(*args, **kwargs)
            if debug:
                print('VCR RECORDING ...')
            vcr_status = VCR_RECORD
            vcr_playlist = []
            # execute decorated function
            value = func(*args, **kwargs)
            # write to file
            if len(vcr_playlist) == 0:
                msg = 'no socket activity - @vcr decorator not needed for %s'
                warnings.warn(msg % (func.__name__))
            else:
                with open(tape, 'wb') as fh:
                    pickle.dump(vcr_playlist, fh, protocol=2)
        else:
            # playback mode
            if debug:
                print('VCR PLAYBACK ...')
            vcr_status = VCR_PLAYBACK
            # load playlist
            with open(tape, 'rb') as fh:
                vcr_playlist = pickle.load(fh)
            # execute decorated function
            value = func(*args, **kwargs)

        # revert monkey patches
        socket.socket = _orig_socket
        socket.getaddrinfo = _orig_getaddrinfo
        selectors.SelectSelector._select = _orig_select
        time.sleep = _orig_sleep
        return value

    return _vcr_inner


vcr = functools.partial(_vcr_wrapper, overwrite=False)
