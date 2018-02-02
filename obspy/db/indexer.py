# -*- coding: utf-8 -*-
"""
A waveform indexer collecting metadata from a file based waveform archive and
storing in into a standard SQL database.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import fnmatch
import os
import sys
import time

from obspy import read
from obspy.core.preview import create_preview
from obspy.core.util.base import _get_entry_points
from obspy.db.db import (WaveformChannel, WaveformFeatures, WaveformFile,
                         WaveformGaps, WaveformPath)


class WaveformFileCrawler(object):
    """
    A waveform file crawler.

    This class scans periodically all given paths for waveform files and
    collects them into a watch list.
    """
    def _update_or_insert(self, dataset):
        """
        Add a new file into or modifies existing file in database.
        """
        if len(dataset) < 1:
            return
        session = self.session()
        data = dataset[0]
        # check for duplicates
        if self.options.check_duplicates:
            query = session.query(WaveformFile, WaveformChannel, WaveformPath)
            query = query.filter(WaveformPath.id == WaveformFile.path_id)
            query = query.filter(WaveformFile.id == WaveformChannel.file_id)
            query = query.filter(WaveformPath.path != data['path'])
            query = query.filter(WaveformFile.file == data['file'])
            query = query.filter(WaveformChannel.network == data['network'])
            query = query.filter(WaveformChannel.station == data['station'])
            query = query.filter(WaveformChannel.location == data['location'])
            query = query.filter(WaveformChannel.channel == data['channel'])
            query = query.filter(WaveformChannel.starttime ==
                                 data['starttime'])
            query = query.filter(WaveformChannel.endtime == data['endtime'])
            if query.count() > 0:
                msg = "Duplicate entry '%s' in '%s'."
                self.log.error(msg % (data['file'], data['path']))
                return
        # fetch or create path
        try:
            # search for existing path
            query = session.query(WaveformPath)
            path = query.filter_by(path=data['path']).one()
        except Exception:
            # create new path entry
            path = WaveformPath(data)
            session.add(path)
        # search and delete existing file entry
        msg = "Inserted"
        if path.id is not None:
            # search for existing file
            query = session.query(WaveformFile)
            files = query.filter_by(path_id=path.id,
                                    file=data['file']).all()
            if files:
                msg = "Updated"
            # delete existing file entry and all related information
            for file in files:
                session.delete(file)
        # create new file entry
        file = WaveformFile(data)
        path.files.append(file)
        # add channel entries
        for data in dataset:
            # create new channel entry
            channel = WaveformChannel(data)
            file.channels.append(channel)
            # add gaps
            for gap in data['gaps']:
                channel.gaps.append(WaveformGaps(gap))
            # add features
            for feature in data['features']:
                channel.features.append(WaveformFeatures(feature))
        try:
            session.commit()
        except Exception as e:
            session.rollback()
            self.log.error(str(e))
        else:
            self.log.debug("%s '%s' in '%s'" % (msg, data['file'],
                                                data['path']))
        session.close()

    def _delete(self, path, file=None):
        """
        Remove a file or all files with a given path from the database.
        """
        session = self.session()
        if file:
            query = session.query(WaveformFile)
            query = query.filter(WaveformPath.path == path)
            query = query.filter(WaveformFile.file == file)
            query = query.filter(WaveformPath.archived is False)
            for file_obj in query:
                session.delete(file_obj)
            try:
                session.commit()
            except Exception as e:
                session.rollback()
                msg = "Error deleting file '%s' in '%s': %s"
                self.log.error(msg % (file, path, e))
            else:
                self.log.debug("Deleting file '%s' in '%s'" % (file, path))
        else:
            query = session.query(WaveformPath)
            query = query.filter(WaveformPath.path == path)
            query = query.filter(WaveformPath.archived is False)
            for path_obj in query:
                session.delete(path_obj)
            try:
                session.commit()
            except Exception as e:
                session.rollback()
                self.log.error("Error deleting path '%s': %s" % (path, e))
            else:
                self.log.debug("Deleting path '%s'" % (path))
        session.close()

    def _select(self, path=None):
        """
        Fetch entry from database.
        """
        session = self.session()
        if path:
            # check database for file entries in specific path
            result = session.query("file", "mtime").from_statement("""
                SELECT file, mtime
                FROM default_waveform_paths as p, default_waveform_files as f
                WHERE p.id=f.path_id
                AND p.path=:path""").params(path=path).all()
            result = dict(result)
        else:
            # get all path entries from database
            result = session.query("path").from_statement("""
                SELECT path FROM default_waveform_paths""").all()
            result = [r[0] for r in result]
        session.close()
        return result

    def get_features(self):
        return self.paths[self._root][1]

    features = property(get_features)

    def get_patterns(self):
        return self.paths[self._root][0]

    patterns = property(get_patterns)

    def has_pattern(self, file):
        """
        Checks if the file name fits to the preferred file pattern.
        """
        for pattern in self.patterns:
            if fnmatch.fnmatch(file, pattern):
                return True
        return False

    def _process_output_queue(self):
        try:
            dataset = self.output_queue.pop(0)
        except Exception:
            pass
        else:
            self._update_or_insert(dataset)

    def _process_log_queue(self):
        try:
            msg = self.log_queue.pop(0)
        except Exception:
            pass
        else:
            if msg.startswith('['):
                self.log.error(msg)
            else:
                self.log.debug(msg)

    def _reset_walker(self):
        """
        Resets the crawler parameters.
        """
        # break if options run_once is set and a run was completed already
        if self.options.run_once and \
                getattr(self, 'first_run_complete', False):
            # before shutting down make sure all queues are empty!
            while self.output_queue or self.work_queue:
                msg = 'Crawler stopped but waiting for empty queues to exit.'
                self.log.debug(msg)
                if self.log_queue:
                    msg = 'log_queue still has %s item(s)'
                    self.log.debug(msg % len(self.log_queue))
                    # Fetch items from the log queue
                    self._process_log_queue()
                    continue
                if self.output_queue:
                    msg = 'output_queue still has %s item(s)'
                    self.log.debug(msg % len(self.output_queue))
                    # try to finalize a single processed stream object from
                    # output queue
                    self._process_output_queue()
                    continue
                if self.work_queue:
                    msg = 'work_queue still has %s items'
                    self.log.debug(msg % len(self.work_queue))
                time.sleep(10)
            self.log.debug('Crawler stopped by option run_once.')
            sys.exit()
            return
        self.log.debug('Crawler restarted.')
        # reset attributes
        self._current_path = None
        self._current_files = []
        self._db_files = {}
        # get search paths for waveform crawler
        self._roots = list(self.paths.keys())
        self._root = self._roots.pop(0)
        # create new walker
        self._walker = os.walk(self._root, topdown=True, followlinks=True)
        # clean up paths
        if self.options.cleanup:
            paths = self._select()
            for path in paths:
                if not os.path.isdir(path):
                    # no path in filesystem
                    self._delete(path)
                elif not self._select(path):
                    # empty path in database
                    self._delete(path)
        # logging
        self.log.debug("Crawling root '%s' ..." % self._root)
        self.first_run_complete = True

    def _step_walker(self):
        """
        Steps current walker object to the next directory.
        """
        # try to fetch next directory
        try:
            root, dirs, files = next(self._walker)
        except StopIteration:
            # finished cycling through all directories in current walker
            # try get next crawler search path
            try:
                self._root = self._roots.pop()
            except IndexError:
                # a whole cycle has been done
                # reset everything
                self._reset_walker()
                return
            # reset attributes
            self._current_path = None
            self._current_files = []
            self._db_files = {}
            # create new walker
            self._walker = os.walk(self._root, topdown=True, followlinks=True)
            # logging
            self.log.debug("Crawling root '%s' ..." % self._root)
            return
        # remove files or paths starting with a dot
        if self.options.skip_dots:
            for file in files:
                if file.startswith('.'):
                    files.remove(file)
            for dir in dirs:
                if dir.startswith('.'):
                    dirs.remove(dir)
        self._current_path = root
        self._current_files = files
        # logging
        self.log.debug("Scanning path '%s' ..." % self._current_path)
        # get all database entries for current path
        self._db_files = self._select(self._current_path)

    def _prepare_paths(self, paths):
        out = {}
        for path in paths:
            # strip features
            if '#' in path:
                parts = path.split('#')
                path = parts[0]
                features = parts[1:]
            else:
                features = []
            # strip patterns
            if '=' in path:
                path, patterns = path.split('=', 1)
                if ' ' in patterns:
                    patterns = patterns.split(' ')
                else:
                    patterns = [patterns.strip()]
            else:
                patterns = ['*.*']
            # normalize and absolute path name
            path = os.path.normpath(os.path.abspath(path))
            # check path
            if not os.path.isdir(path):
                self.log.warn("Skipping inaccessible path '%s' ..." % path)
                continue
            out[path] = (patterns, features)
        return out

    def iterate(self):
        """
        Handles exactly one directory.
        """
        # skip if service is not running
        # be aware that the processor pool is still active waiting for work
        if not self.running:
            return
        # skip if input queue is full
        if len(self.input_queue) > self.options.number_of_cpus:
            return
        # try to finalize a single processed stream object from output queue
        self._process_output_queue()
        # Fetch items from the log queue
        self._process_log_queue()
        # walk through directories and files
        try:
            file = self._current_files.pop(0)
        except IndexError:
            # file list is empty
            # clean up not existing files in current path
            if self.options.cleanup:
                for file in self._db_files.keys():
                    self._delete(self._current_path, file)
            # jump into next directory
            self._step_walker()
            return
        # skip file with wrong pattern
        if not self.has_pattern(file):
            return
        # process a single file
        path = self._current_path
        filepath = os.path.join(path, file)
        # get file stats
        try:
            stats = os.stat(filepath)
            mtime = int(stats.st_mtime)
        except Exception as e:
            self.log.error(str(e))
            return
        # check if recent
        if self.options.recent:
            # skip older files
            if time.time() - mtime > 60 * 60 * self.options.recent:
                try:
                    db_file_mtime = self._db_files.pop(file)
                except Exception:
                    pass
                return
        # option force-reindex set -> process file regardless if already in
        # database or recent or whatever
        if self.options.force_reindex:
            self.input_queue[filepath] = (path, file, self.features)
            return
        # compare with database entries
        if file not in self._db_files.keys():
            # file does not exists in database -> add file
            self.input_queue[filepath] = (path, file, self.features)
            return
        # file is already in database
        # -> remove from file list so it won't be deleted on database cleanup
        try:
            db_file_mtime = self._db_files.pop(file)
        except Exception:
            return
        # -> compare modification times of current file with database entry
        if mtime == db_file_mtime:
            return
        # modification time differs -> update file
        self.input_queue[filepath] = (path, file, self.features)


def worker(_i, input_queue, work_queue, output_queue, log_queue, mappings={}):
    try:
        # fetch and initialize all possible waveform feature plug-ins
        all_features = {}
        for (key, ep) in _get_entry_points('obspy.db.feature').items():
            try:
                # load plug-in
                cls = ep.load()
                # initialize class
                func = cls().process
            except Exception as e:
                msg = 'Could not initialize feature %s. (%s)'
                log_queue.append(msg % (key, str(e)))
                continue
            all_features[key] = {}
            all_features[key]['run'] = func
            try:
                all_features[key]['indexer_kwargs'] = cls['indexer_kwargs']
            except Exception:
                all_features[key]['indexer_kwargs'] = {}
        # loop through input queue
        while True:
            # fetch a unprocessed item
            try:
                filepath, (path, file, features) = input_queue.popitem()
            except Exception:
                continue
            # skip item if already in work queue
            if filepath in work_queue:
                continue
            work_queue.append(filepath)
            # get additional kwargs for read method from waveform plug-ins
            kwargs = {'verify_chksum': False}
            for feature in features:
                if feature not in all_features:
                    log_queue.append('%s: Unknown feature %s' % (filepath,
                                                                 feature))
                    continue
                kwargs.update(all_features[feature]['indexer_kwargs'])
            # read file and get file stats
            try:
                stats = os.stat(filepath)
                stream = read(filepath, **kwargs)
                # get gap and overlap information
                gap_list = stream.get_gaps()
                # merge channels and replace gaps/overlaps with 0 to prevent
                # generation of masked arrays
                stream.merge(fill_value=0)
            except Exception as e:
                msg = '[Reading stream] %s: %s'
                log_queue.append(msg % (filepath, e))
                try:
                    work_queue.remove(filepath)
                except Exception:
                    pass
                continue
            # build up dictionary of gaps and overlaps for easier lookup
            gap_dict = {}
            for gap in gap_list:
                id = '.'.join(gap[0:4])
                temp = {
                    'gap': gap[6] >= 0,
                    'starttime': gap[4].datetime,
                    'endtime': gap[5].datetime,
                    'samples': abs(gap[7])
                }
                gap_dict.setdefault(id, []).append(temp)
            # loop through traces
            dataset = []
            for trace in stream:
                result = {}
                # general file information
                result['mtime'] = int(stats.st_mtime)
                result['size'] = stats.st_size
                result['path'] = path
                result['file'] = file
                result['filepath'] = filepath
                # trace information
                result['format'] = trace.stats._format
                result['station'] = trace.stats.station
                result['location'] = trace.stats.location
                result['channel'] = trace.stats.channel
                result['network'] = trace.stats.network
                result['starttime'] = trace.stats.starttime.datetime
                result['endtime'] = trace.stats.endtime.datetime
                result['calib'] = trace.stats.calib
                result['npts'] = trace.stats.npts
                result['sampling_rate'] = trace.stats.sampling_rate
                # check for any id mappings
                if trace.id in mappings:
                    old_id = trace.id
                    for mapping in mappings[old_id]:
                        if trace.stats.starttime and \
                           trace.stats.starttime > mapping['endtime']:
                            continue
                        if trace.stats.endtime and \
                           trace.stats.endtime < mapping['starttime']:
                            continue
                        result['network'] = mapping['network']
                        result['station'] = mapping['station']
                        result['location'] = mapping['location']
                        result['channel'] = mapping['channel']
                        msg = "Mapping '%s' to '%s.%s.%s.%s'" % \
                            (old_id, mapping['network'], mapping['station'],
                             mapping['location'], mapping['channel'])
                        log_queue.append(msg)
                # gaps/overlaps for current trace
                result['gaps'] = gap_dict.get(trace.id, [])
                # apply feature functions
                result['features'] = []
                for key in features:
                    if key not in all_features:
                        continue
                    try:
                        # run plug-in and update results
                        temp = all_features[key]['run'](trace)
                        for key, value in temp.items():
                            result['features'].append({'key': key,
                                                       'value': value})
                    except Exception as e:
                        msg = '[Processing feature] %s: %s'
                        log_queue.append(msg % (filepath, e))
                        continue
                # generate preview of trace
                result['preview'] = None
                if '.LOG.L.' not in file or trace.stats.channel != 'LOG':
                    # create previews only for non-log files (see issue #400)
                    try:
                        trace = create_preview(trace, 30)
                        result['preview'] = trace.data.dumps()
                    except ValueError:
                        pass
                    except Exception as e:
                        msg = '[Creating preview] %s: %s'
                        log_queue.append(msg % (filepath, e))
                # update dataset
                dataset.append(result)
            del stream
            # return results to main loop
            try:
                output_queue.append(dataset)
            except Exception:
                pass
            try:
                work_queue.remove(filepath)
            except Exception:
                pass
    except KeyboardInterrupt:
        return
