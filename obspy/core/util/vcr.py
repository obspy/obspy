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
import socket
import warnings

from future.utils import PY2


VCR_RECORD = 0
VCR_PLAYBACK = 1


def _vcr_wrapper(func, overwrite=False, debug=False, force_check=False):
    """
    Wrapper around _vcr_inner allowing additional arguments on decorator
    """

    def _vcr_inner(*args, **kwargs):
        """
        The actual decorator doing a lot of monkey patching and auto magic
        """
        # monkey patches
        _orig_socket = socket.socket

        class VCRSocket:
            _skip_methods = ['setsockopt', 'settimeout', 'setblocking']

            playlist = []
            status = VCR_PLAYBACK

            def __init__(self, *args, **kwargs):
                if self.status == VCR_RECORD:
                    self.__dict__['_socket'] = _orig_socket(*args, **kwargs)

            def _generic_method(self, name, *args, **kwargs):
                if self.status == VCR_RECORD:
                    value = getattr(self._socket, name)(*args, **kwargs)
                    if debug:
                        print(name, args, kwargs, value)
                    # handle special objects which are not pickleable
                    if isinstance(value, io.BufferedIOBase):
                        temp = io.BytesIO(value.read())
                        self.playlist.append((name, args, kwargs, temp))
                        # return new copy of BytesIO as it may get closed
                        return copy.copy(temp)
                    if debug:
                        # test if value is pickleable - will raise exception
                        pickle.dumps(value)
                    # skip setters
                    if name not in self._skip_methods:
                        # all other methods will be recorded
                        self.playlist.append((name, args, kwargs,
                                              copy.copy(value)))
                    return value
                else:
                    # skip setters
                    if name in self._skip_methods:
                        return
                    # always work on first element in playlist list
                    data = self.playlist.pop(0)
                    # XXX: py3 sometimes has two sendall calls ???
                    if PY2 and name == 'makefile' and data[0] == 'sendall':
                        data = self.playlist.pop(0)
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

        # make sure vcrtapes directory exists
        if not os.path.isdir(path):
            os.makedirs(path)
        tape = os.path.join(path, '%s.%s.vcr' % (file_name, func_name))

        # check for tape file and determine mode
        if os.path.isfile(tape) is False or overwrite is True:
            if PY2:
                msg = 'VCR will record only in PY3 to be backward ' + \
                    'compatible with PY2 - skipping VCR mechanics for %s'
                warnings.warn(msg % (func.__name__))
                # restore original socket
                socket.socket = _orig_socket
                # return
                return func(*args, **kwargs)
            if debug:
                print('VCR RECORDING ...')
            # record mode
            VCRSocket.status = VCR_RECORD
            VCRSocket.playlist = []
            # execute function
            value = func(*args, **kwargs)
            # write to file
            if len(VCRSocket.playlist) == 0:
                msg = 'no socket activity - vcr decorator not needed for %s'
                warnings.warn(msg % (func.__name__))
            else:
                with open(tape, 'wb') as fh:
                    pickle.dump(VCRSocket.playlist, fh, protocol=2)
        else:
            if debug:
                print('VCR PLAYBACK ...')
            # playback mode
            VCRSocket.status = VCR_PLAYBACK
            # load playlist
            with open(tape, 'rb') as fh:
                VCRSocket.playlist = pickle.load(fh)
            # execute function
            value = func(*args, **kwargs)

        # restore original socket
        socket.socket = _orig_socket
        return value

    return _vcr_inner


vcr = functools.partial(_vcr_wrapper, overwrite=False)
