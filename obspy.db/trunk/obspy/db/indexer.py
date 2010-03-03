# -*- coding: utf-8 -*-

from obspy.core import read
from obspy.db.db import WaveformFile, WaveformPath, WaveformChannel, \
    WaveformGaps
from obspy.db.util import _getInstalledWaveformFeaturesPlugins, createPreview
import copy
import fnmatch
import logging
import multiprocessing
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
                q = self.session.query(WaveformPath)
                path = q.filter_by(path=data['path']).one()
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
                channel = file.channels.filter(network=data['network'],
                                               channel=data['file']).one()
            except:
                channel = WaveformChannel(data)
                for gap in data['gaps']:
                    channel.gaps.append(WaveformGaps(gap))
                file.channels.append(channel)
            try:
                self.session.commit()
            except Exception, e:
                self.session.rollback()
                logging.error(str(e))
            else:
                logging.info('Inserting %s %s' % (data['path'], data['file']))

    def _delete(self, path, file=None):
        """
        Remove a file or all files with a given path from the database.
        """
        if file:
            q = self.session.query(WaveformFile)
            q = q.filter(WaveformPath.path == path)
            q = q.filter(WaveformFile.file == file)
            q = q.filter(WaveformPath.archived == False)
            try:
                file_obj = q.one()
                self.session.delete(file_obj)
                self.session.commit()
            except:
                self.session.rollback()
                logging.error('Error deleting file %s %s' % (path, file))
            else:
                logging.info('Deleting file %s %s' % (file_obj.file,
                                                      file_obj.path.path))
        else:
            q = self.session.query(WaveformPath)
            q = q.filter(WaveformPath.path == path)
            q = q.filter(WaveformPath.archived == False)
            try:
                path_obj = q.one()
                self.session.delete(path_obj)
            except:
                self.session.rollback()
                logging.error('Error deleting path %s' % (path))
            else:
                logging.info('Deleting path %s' % (path_obj.path))

    def _select(self, path, file=None):
        """
        """
        if file:
            # check database for entry
            q = self.session.query(WaveformFile.mtime)
            q = q.filter(WaveformFile.file == file)
            q = q.filter(WaveformPath.path == path)
            q = q.first()
        else:
            q = self.session.query(WaveformPath)
            q = q.filter(WaveformPath.path == path)
            try:
                q = q.first()
                return dict([(f.file, f.mtime) for f in q.files])
            except:
                return {}

    def _hasPattern(self, file):
        """
        Checks if the file name fits to the preferred file pattern.
        """
        for pattern in self.paths[self._root]:
            if fnmatch.fnmatch(file, pattern):
                return True
        return False

#    def _selectAllPaths(self):
#        """
#        Query for all paths inside the database.
#        """
#        sql_obj = sql.select([waveform_tab.c['path']]).distinct()
#        try:
#            result = self.db.execute(sql_obj)
#        except:
#            result = []
#        return [path[0] for path in result]

    def _processOutputQueue(self):
        try:
            dataset = self.output_queue.pop(0)
        except:
            pass
        else:
            self._update_or_insert(dataset)

    def _resetWalker(self):
        """
        Resets the crawler parameters.
        """
        logging.info('Crawler restarted.')
        # update configuration
        self.crawler_paths = [os.path.normcase(path) for path in self.paths]
        # reset attributes
        self._current_path = None
        self._current_files = []
        self._db_files = []
        self._db_paths = []
        # get search paths for waveform crawler
        self._paths = []
        self._roots = copy.copy(self.crawler_paths)
        self._root = self._roots.pop()
        # create new walker
        self._walker = os.walk(self._root, topdown=True, followlinks=True)
        # logging
        logging.info("Crawling root '%s' ..." % self._root)

    def _stepWalker(self):
        """
        Steps current walker object to the next directory.
        """
        # try to fetch next directory
        try:
            root, dirs, files = self._walker.next()
        except StopIteration:
            # finished cycling through all directories in current walker
            # remove remaining entries from database

            # try get next crawler search path
            try:
                self._root = self._roots.pop()
            except IndexError:
                # a whole cycle has been done - check paths
#                db_paths = self._selectAllPaths()
#                for path in db_paths:
#                    # skip existing paths
#                    if path in self._paths:
#                        continue
#                    # remove the others
#                    self._delete(path)
                # reset everything
                self._resetWalker()
                return
            # reset attributes
            self._current_path = None
            self._current_files = []
            self._db_files = []
            # create new walker
            self._walker = os.walk(self._root, topdown=True, followlinks=True)
            # logging
            logging.info("Crawling root '%s' ..." % self._root)
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
        logging.info("Scanning path '%s' ..." % self._current_path)
        # get all database entries for current path
        self._db_files = self._select(self._current_path)

    def iterate(self):
        """
        Handles exactly one directory.
        """
        # skip if service is not running
        # be aware that the processor pool is still active waiting for work
        if not self.running:
            return
        # try to finalize a single processed stream object from output queue
        self._processOutputQueue()
        # skip if input queue is full
        if len(self.input_queue) > 10:
            return
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
            self._stepWalker()
            return
        # skip file with wrong pattern
        if not self._hasPattern(file):
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
            logging.warning(str(e))
            return
        # compare with database entries
        if file not in self._db_files:
            # file does not exists in database -> add file
            args = (path, file, stats, [])
            self.input_queue[filepath] = args
            return
        # file is already in database
        # -> remove from file list so it won't be deleted on database cleanup
        db_file_mtime = self._db_files.pop(file)
        # -> compare modification times of current file with database entry
        if int(stats.st_mtime) == db_file_mtime:
            return
        # modification time differs -> update file
        args = (path, file, stats, [])
        self.input_queue[filepath] = args
        return


def worker(i, input_queue, work_queue, output_queue, lock, preview_dir=None):
    logger = multiprocessing.log_to_stderr(logging.DEBUG)
    logger.info("Starting Process #%d" % i)
    # fetch and initialize all possible waveform feature plug-ins
    all_features = {}
    for (key, ep) in _getInstalledWaveformFeaturesPlugins().iteritems():
        try:
            # load plug-in
            cls = ep.load()
            # initialize class
            func = cls().process
        except Exception, e:
            logger.error(str(e) + '\n')
            continue
        all_features[key] = {}
        all_features[key]['run'] = func
        try:
            all_features[key]['indexer_kwargs'] = cls['indexer_kwargs']
        except:
            all_features[key]['indexer_kwargs'] = {}
    # loop through input queue
    while True:
        # fetch a unprocessed item
        try:
            key, args = input_queue.popitem()
        except:
            continue
        if key in work_queue:
            continue
        work_queue.append(key)
        (path, file, stats, features) = args
        filepath = os.path.join(path, file)
        logger.debug('Reading File %s' % filepath)
        # get additional kwargs for read method from waveform plug-ins
        kwargs = {}
        for feature in features:
            if feature not in all_features:
                logger.error(filepath + '\n')
                logger.error('Unknown feature %s\n' % feature)
                work_queue.remove(filepath)
                continue
            kwargs.update(all_features[feature]['indexer_kwargs'])
        # read file
        try:
            stream = read(str(filepath), kwargs)
            # get gap and overlap information
            gap_list = stream.getGaps()
            # merge channels
            stream.merge()
        except Exception, e:
            logger.error(filepath + '\n')
            logger.error(str(e) + '\n')
            work_queue.remove(filepath)
            continue
        # build up dictionary of gaps and overlaps for easier lookup
        gap_dict = {}
        for gap in gap_list:
            id = '.'.join(gap[0:4])
            temp = {
                'gap': gap[6] >= 0,
                'starttime': gap[4],
                'endtime': gap[5],
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
            # gaps/overlaps
            result['gaps'] = gap_dict.get(trace.id, [])
            # apply feature functions
            for feature in features:
                if feature not in all_features:
                    continue
                try:
                    # run plug-in and update results
                    result.update(all_features[feature]['run'](trace))
                except Exception, e:
                    logger.error(filepath + '\n')
                    logger.error(str(e) + '\n')
                    continue
            dataset.append(result)
            # generate preview of trace
            if preview_dir:
                # check for network
                if not trace.stats.network:
                    network = '__'
                else:
                    network = trace.stats.network
                # 2010/BW/MANZ/EHE.01D/
                preview_path = os.path.join(preview_dir,
                                            str(trace.stats.starttime.year),
                                            network, trace.stats.station,
                                            '%s.%sD' % (trace.stats.channel,
                                                        trace.stats.location))
                if not os.path.isdir(preview_path):
                    os.makedirs(preview_path)
                start = trace.stats.starttime.timestamp
                end = trace.stats.endtime.timestamp
                preview_file = os.path.join(preview_path,
                                            "%s-%s.mseed" % (start, end))
                try:
                    trace = createPreview(trace, 60.0)
                    trace.write(preview_file, format="MSEED", reclen=1024)
                except Exception , e:
                    logger.error(filepath + '\n')
                    logger.error(str(e) + '\n')
        del stream
        # return results to main loop
        try:
            output_queue.append(dataset)
        except:
            pass
        work_queue.remove(filepath)
