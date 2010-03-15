# -*- coding: utf-8 -*-

from obspy.core import read
from obspy.db.db import WaveformFile, WaveformPath, WaveformChannel, \
    WaveformGaps, WaveformFeatures
from obspy.core.util import _getPlugins
from obspy.db.util import createPreview
import fnmatch
import os


class WaveformFileCrawler:
    """
    A waveform file crawler.
    
    This class scans periodically all given paths for waveform files and 
    collects them into a watch list.
    """

    def _update_or_insert(self, dataset):
        """
        Add a new file into or modifies existing file in database.
        """
        for data in dataset:
            # fetch or create path
            try:
                query = self.session.query(WaveformPath)
                path = query.filter_by(path=data['path']).one()
            except:
                path = WaveformPath(data)
                self.session.add(path)
            # fetch or create file
            try:
                file = path.files.filter(file=data['file']).one()
            except:
                file = WaveformFile(data)
                path.files.append(file)
            # fetch or create channel
            try:
                # search for existing channel
                channel = file.channels.filter(network=data['network'],
                                               station=data['station'],
                                               location=data['location'],
                                               channel=data['channel'],
                                               starttime=data['starttime'],
                                               endtime=data['endtime']).one()
            except:
                # create new channel object
                msg = 'Inserting'
                channel = WaveformChannel(data)
            else:
                # modify existing channel object
                msg = 'Updating'
                channel.update(data)
                # remove all gaps
                self.session.delete(channel.gaps)
                # remove all features
                self.session.delete(channel.features)
            # add gaps
            for gap in data['gaps']:
                channel.gaps.append(WaveformGaps(gap))
            # add features
            for feature in data['features']:
                channel.features.append(WaveformFeatures(feature))
            file.channels.append(channel)
            try:
                self.session.commit()
            except Exception, e:
                self.session.rollback()
                self.log.error(str(e))
            else:
                self.log.debug("%s '%s' in '%s'" % (msg, data['file'],
                                                    data['path']))

    def _delete(self, path, file=None):
        """
        Remove a file or all files with a given path from the database.
        """
        if file:
            query = self.session.query(WaveformFile)
            query = query.filter(WaveformPath.path == path)
            query = query.filter(WaveformFile.file == file)
            query = query.filter(WaveformPath.archived == False)
            for file_obj in query:
                self.session.delete(file_obj)
            try:
                self.session.commit()
            except Exception, e:
                self.session.rollback()
                msg = "Error deleting file '%s' in '%s': %s"
                self.log.error(msg % (file, path, e))
            else:
                self.log.debug("Deleting file '%s' in '%s'" % (file, path))
        else:
            query = self.session.query(WaveformPath)
            query = query.filter(WaveformPath.path == path)
            query = query.filter(WaveformPath.archived == False)
            for path_obj in query:
                self.session.delete(path_obj)
            try:
                self.session.commit()
            except Exception, e:
                self.session.rollback()
                self.log.error("Error deleting path '%s': %s" % (path, e))
            else:
                self.log.debug("Deleting path '%s'" % (path))

    def _select(self, path=None):
        """
        """
        if path:
            # check database for file entries in path
            query = self.session.query(WaveformPath)
            if path:
                query = query.filter(WaveformPath.path == path)
            try:
                query = query.first()
                return dict([(f.file, f.mtime) for f in query.files])
            except:
                return {}
        else:
            # check database for all path entries
            query = self.session.query(WaveformPath.path)
            try:
                return [p.path for p in query.all()]
            except:
                return []

    def getFeatures(self):
        return self.paths[self._root][1]

    features = property(getFeatures)

    def getPatterns(self):
        return self.paths[self._root][0]

    patterns = property(getPatterns)

    def hasPattern(self, file):
        """
        Checks if the file name fits to the preferred file pattern.
        """
        for pattern in self.patterns:
            if fnmatch.fnmatch(file, pattern):
                return True
        return False

    def _processOutputQueue(self):
        try:
            dataset = self.output_queue.pop(0)
        except:
            pass
        else:
            self._update_or_insert(dataset)

    def _processLogQueue(self):
        try:
            msg = self.log_queue.pop(0)
        except:
            pass
        else:
            self.log.error(msg)

    def _resetWalker(self):
        """
        Resets the crawler parameters.
        """
        self.log.debug('Crawler restarted.')
        # reset attributes
        self._current_path = None
        self._current_files = []
        self._db_files = {}
        # get search paths for waveform crawler
        self._roots = self.paths.keys()
        self._root = self._roots.pop(0)
        # create new walker
        self._walker = os.walk(self._root, topdown=True, followlinks=True)
        # clean up paths
        if self.cleanup:
            paths = self._select()
            for path in paths:
                if not os.path.isdir(path):
                    self._delete(path)
        # logging
        self.log.debug("Crawling root '%s' ..." % self._root)

    def _stepWalker(self):
        """
        Steps current walker object to the next directory.
        """
        # try to fetch next directory
        try:
            root, dirs, files = self._walker.next()
        except StopIteration:
            # finished cycling through all directories in current walker
            # try get next crawler search path
            try:
                self._root = self._roots.pop()
            except IndexError:
                # a whole cycle has been done
                # reset everything
                self._resetWalker()
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
        if self.skip_dots:
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

    def _preparePaths(self, paths):
        out = {}
        for path in paths:
            # strip features
            if ';' in path:
                path, features = path.split(';', 1)
                if ' ' in features:
                    features = features.split(' ')
                else:
                    features = [features.strip()]
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
        if len(self.input_queue) > self.number_of_cpus:
            return
        # try to finalize a single processed stream object from output queue
        self._processOutputQueue()
        # Fetch items from the log queue
        self._processLogQueue()
        # walk through directories and files
        try:
            file = self._current_files.pop(0)
        except IndexError:
            # file list is empty
            # clean up not existing files in current path
            if self.cleanup:
                for file in self._db_files.keys():
                    self._delete(self._current_path, file)
            # jump into next directory
            self._stepWalker()
            return
        # skip file with wrong pattern
        if not self.hasPattern(file):
            return
        # process a single file
        path = self._current_path
        filepath = os.path.join(path, file)
        # check if already processed
        if filepath in self.work_queue:
            return
        # get file stats
        try:
            stats = os.stat(filepath)
        except Exception, e:
            self.log.error(str(e))
            return
        # compare with database entries
        if file not in self._db_files:
            # file does not exists in database -> add file
            self.input_queue[filepath] = (path, file, self.features)
            return
        # file is already in database
        # -> remove from file list so it won't be deleted on database cleanup
        db_file_mtime = self._db_files.pop(file)
        # -> compare modification times of current file with database entry
        if int(stats.st_mtime) == db_file_mtime:
            return
        # modification time differs -> update file
        self.input_queue[filepath] = (path, file, [])
        return


def worker(i, input_queue, work_queue, output_queue, log_queue, filter=None):
    try:
        # fetch and initialize all possible waveform feature plug-ins
        all_features = {}
        for (key, ep) in _getPlugins('obspy.db.features').iteritems():
            try:
                # load plug-in
                cls = ep.load()
                # initialize class
                func = cls().process
            except Exception, e:
                msg = 'Could not initialize feature %s. (%s)'
                log_queue.append(msg % (key, str(e)))
                continue
            all_features[key] = {}
            all_features[key]['run'] = func
            try:
                all_features[key]['indexer_kwargs'] = cls['indexer_kwargs']
            except:
                all_features[key]['indexer_kwargs'] = {}
        # fetch and initialize all possible waveform filter plug-in
        if filter:
            plugins = _getPlugins('obspy.db.filter')
            try:
                # load plug-in
                ep = plugins[filter]
                cls = ep.load()
                # initialize class
                func = cls().filter
            except:
                msg = 'Could not initialize filter %s.'
                log_queue.append(msg % (filter))
                filter = None
            else:
                filter = func
        # loop through input queue
        while True:
            # fetch a unprocessed item
            try:
                filepath, (path, file, features) = input_queue.popitem()
            except:
                continue
            # skip item if already in work queue
            if filepath in work_queue:
                continue
            work_queue.append(filepath)
            # get additional kwargs for read method from waveform plug-ins
            kwargs = {}
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
                gap_list = stream.getGaps()
                # merge channels and replace gaps/overlaps with 0 to prevent
                # generation of masked arrays
                stream.merge(fill_value=0)
            except Exception, e:
                msg = '[Reading stream] %s: %s'
                log_queue.append(msg % (filepath, e))
                try:
                    work_queue.remove(filepath)
                except:
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
                result['format'] = format = trace.stats._format
                result['station'] = trace.stats.station
                result['location'] = trace.stats.location
                result['channel'] = trace.stats.channel
                result['network'] = trace.stats.network
                result['starttime'] = trace.stats.starttime.datetime
                result['endtime'] = trace.stats.endtime.datetime
                result['calib'] = trace.stats.calib
                result['npts'] = trace.stats.npts
                result['sampling_rate'] = trace.stats.sampling_rate
                # filter
                if filter:
                    try:
                        ok = filter(result, trace)
                    except Exception, e:
                        msg = '[Applying filter] %s: %s'
                        log_queue.append(msg % (filepath, e))
                        continue
                    else:
                        if ok == False:
                            continue
                        elif isinstance(ok, dict):
                            result = ok
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
                        for key, value in temp.iteritems():
                            result['features'].append({'key':key,
                                                       'value':value})
                    except Exception, e:
                        msg = '[Processing feature] %s: %s'
                        log_queue.append(msg % (filepath, e))
                        continue
                # generate preview of trace
                try:
                    trace = createPreview(trace, 60.0)
                    result['preview'] = trace.data.dumps()
                except Exception , e:
                    msg = '[Creating preview] %s: %s'
                    log_queue.append(msg % (filepath, e))
                    result['preview'] = None
                # update dataset
                dataset.append(result)
            del stream
            # return results to main loop
            try:
                output_queue.append(dataset)
            except:
                pass
            try:
                work_queue.remove(filepath)
            except:
                pass
    except KeyboardInterrupt:
        return
