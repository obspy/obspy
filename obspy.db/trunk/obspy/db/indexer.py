# -*- coding: utf-8 -*-

from Queue import Empty as QueueEmpty, Full as QueueFull
from SimpleXMLRPCServer import SimpleXMLRPCServer
from obspy.core import read
from obspy.core.util import AttribDict
from obspy.db.defaults import Base, WaveformFile, WaveformPath
from obspy.db.utils import _getInstalledWaveformFeaturesPlugins, createPreview
from sqlalchemy import create_engine
from sqlalchemy.orm.session import sessionmaker
import copy
import fnmatch
import logging
import multiprocessing
import os
import select
import sys


# create logger
logger = logging.getLogger("simple_example")
logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
# add ch to logger
logger.addHandler(ch)


ACTION_INSERT = 0
ACTION_UPDATE = 1


class WaveformFileCrawler:
    """
    A waveform file crawler.
    
    This class scans periodically all given paths for waveform files and 
    collects them into a watch list.
    """

#    def _insert(self, dataset):
#        """
#        Add a new file into the database.
#        """
#        for entry in dataset:
#            sql_obj = waveform_tab.insert().values(**entry)
#            try:
#                self.db.execute(sql_obj)
#                logger.debug('Inserting %s %s' % (entry['path'],
#                                                   entry['file']))
#            except:
#                pass
#
#    def _update(self, dataset):
#        """
#        Modify a file in the database.
#        """
#        for entry in dataset:
#            sql_obj = waveform_tab.update()
#            sql_obj = sql_obj.where(waveform_tab.c['file'] == entry['file'])
#            sql_obj = sql_obj.where(waveform_tab.c['path'] == entry['path'])
#            sql_obj = sql_obj.values(**entry)
#            try:
#                self.db.execute(sql_obj)
#                logger.debug('Updating %s %s' % (entry['path'],
#                                                  entry['file']))
#            except Exception, e:
#                logger.error(str(e))
#                pass
#
#    def _delete(self, path, file=None):
#        """
#        Remove a file or all files with a given path from the database.
#        """
#        if self.keep_files:
#            return
#        sql_obj = waveform_tab.delete()
#        if file:
#            sql_obj = sql_obj.where(sql.and_(waveform_tab.c['file'] == file,
#                                             waveform_tab.c['path'] == path))
#        else:
#            sql_obj = sql_obj.where(waveform_tab.c['path'] == path)
#        try:
#            self.db.execute(sql_obj)
#            logger.debug('Deleting %s %s' % (path, file))
#        except:
#            pass

    def _select(self, path, file=None):
        """
        """
        if file:
            # check database for entry
            q = self.db.query(WaveformFile.mtime)
            q = q.filter(WaveformFile.file == file)
            q = q.filter(WaveformPath.path == path)
            q = q.first()
        else:
            q = self.db.query(WaveformFile.mtime, WaveformFile.file)
            q = q.filter(WaveformPath.path == path)
            q = q.all()
        return dict(q)

    def _updateCurrentConfiguration(self):
        """
        """
        self.patterns = self.options.patterns
        self.crawler_paths = [os.path.normcase(path)
                              for path in self.options.paths]
        self.keep_files = self.options.keep_files

    def _hasPattern(self, file):
        """
        Checks if the file name fits to the preferred file pattern.
        """
        for pattern in self.patterns:
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

    def _processQueue(self):
        try:
            action, dataset = self.output_queue.get_nowait()
        except QueueEmpty:
            pass
        #else:
            #if action == ACTION_INSERT:
            #    self._insert(dataset)
            #else:
            #    self._update(dataset)

    def _resetWalker(self):
        """
        Resets the crawler parameters.
        """
        logger.debug('Crawler restarted.')
        # update configuration
        self._updateCurrentConfiguration()
        # reset attributes
        self._current_path = None
        self._current_files = []
        self._db_files_in_current_path = []
        # get search paths for waveform crawler
        self._paths = []
        self._roots = copy.copy(self.crawler_paths)
        self._root = self._roots.pop()
        # create new walker
        self._walker = os.walk(self._root, topdown=True, followlinks=True)
        # logging
        logger.debug("Crawling root '%s' ..." % self._root)

    def _stepWalker(self):
        """
        Steps current walker object to the next directory.
        """
        # try to fetch next directory
        try:
            self._current_path, _ , self._current_files = self._walker.next()
        except StopIteration:
            # finished cycling through all directories in current walker
            # remove remaining entries from database
            for file in self._db_files_in_current_path:
                self._delete(self._current_path, file)
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
            self._db_files_in_current_path = []
            # create new walker
            self._walker = os.walk(self._root, topdown=True, followlinks=True)
            # logging
            logger.debug("Crawling root '%s' ..." % self._root)
            return
        # logging
        logger.debug("Scanning path '%s' ..." % self._current_path)
        # get all database entries for current path
        self._db_files_in_current_path = self._select(self._current_path)

    def iterate(self):
        """
        Handles exactly one directory.
        """
        # skip if service is not running
        # be aware that the processor pool is still active waiting for work
        if not self.running:
            return
        # try to finalize a single processed stream object from output queue
        self._processQueue()
        # skip if input queue is full
        if self.input_queue.full():
            return
        # check for files which could not put into queue yet
        if self._not_yet_queued:
            try:
                self.input_queue.put_nowait(self._not_yet_queued)
            except QueueFull:
                # try later
                return
            self._not_yet_queued = None
        # walk through directories and files
        try:
            file = self._current_files.pop(0)
        except AttributeError:
            # first loop ever after initialization - call reset
            self._resetWalker()
            self._stepWalker()
            return
        except IndexError:
            # file list is empty - jump into next directory
            self._stepWalker()
            return
        # skip file with wrong pattern
        if not self._hasPattern(file):
            return
        # process a single file
        path = self._current_path
        filepath = os.path.join(path, file)
        # get file stats
        try:
            stats = os.stat(filepath)
        except Exception, e:
            logger.warning(str(e))
            return
        # compare with database entries
        if file not in self._db_files_in_current_path:
            # file does not exists in database -> add file
            args = (ACTION_INSERT, path, file, stats, [])
            try:
                self.input_queue.put_nowait(args)
            except QueueFull:
                self._not_yet_queued = args
            return
        # file is already in database
        # -> compare modification times of current file with database entry
        # -> remove from file list so it won't be deleted on database cleanup
        db_file_mtime = self._db_files_in_current_path.pop(file)
        if int(stats.st_mtime) == db_file_mtime:
            return
        # modification time differs -> update file
        args = (ACTION_UPDATE, path, file, stats, [])
        try:
            self.input_queue.put_nowait(args)
        except QueueFull:
            self._not_yet_queued = args
        return


class WaveformIndexer(SimpleXMLRPCServer, WaveformFileCrawler):
    """
    A waveform indexer server.
    """
    def __init__(self, session, input_queue, output_queue, options):
        SimpleXMLRPCServer.__init__(self, ("localhost", 8000))
        # init db + options
        self.db = session
        self.options = options
        self._updateCurrentConfiguration()
        # set queues
        self.input_queue = input_queue
        self.output_queue = output_queue
        self._not_yet_queued = None

    def serve_forever(self, poll_interval=0.5):
        self.running = True
        #self.__is_shut_down.clear()
        while self.running:
            r, _w, _e = select.select([self], [], [], poll_interval)
            if r:
                self._handle_request_noblock()
            self.iterate()
        #self.__is_shut_down.set()


def worker(i, input_queue, output_queue, preview_dir=None):
    logger.debug("Starting Process #%d" % i)
    # fetch all possible entry points for waveform features
    features_ep = _getInstalledWaveformFeaturesPlugins()
    # fetch action from input queue
    for args in iter(input_queue.get, 'STOP'):
        (action, path, file, stats, features) = args
        filepath = os.path.join(path, file)
        try:
            stream = read(str(filepath))
        except Exception, e:
            logger.error(filepath + '\n')
            logger.error(str(e) + '\n')
            continue
        try:
            stream.merge()
        except Exception , e:
            logger.error(filepath + '\n')
            logger.error(str(e) + '\n')
            continue
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
            # apply feature functions
            for feature in features:
                if feature not in features_ep:
                    logger.error(filepath + '\n')
                    logger.error('Unknown feature %s\n' % feature)
                    continue
                try:
                    # load plug-in
                    plugin = features_ep[feature].load()
                    # run plug-in and update results
                    result.update(plugin(trace))
                except Exception, e:
                    logger.error(filepath + '\n')
                    logger.error(str(e) + '\n')
                    continue
            dataset.append(result)
            # generate preview of trace
            if preview_dir:
                # 2010/BW/MANZ/EHE.01D/
                preview_path = os.path.join(preview_dir,
                                            str(trace.stats.starttime.year),
                                            trace.stats.network,
                                            trace.stats.station,
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
            output_queue.put_nowait((action, dataset))
        except:
            pass
    logger.debug("Stopping Process #%d" % i)


def run():
    logger.info("Starting Indexer ...")
    # create file queue and worker processes
    NUMBER_OF_PROCESSORS = multiprocessing.cpu_count()
    PREVIEW_DIR = 'C:\\Users\\barsch\\Desktop\\preview'
    input_queue = multiprocessing.Queue(NUMBER_OF_PROCESSORS * 2)
    output_queue = multiprocessing.Queue()
    for i in range(NUMBER_OF_PROCESSORS):
        args = (i, input_queue, output_queue, PREVIEW_DIR)
        p = multiprocessing.Process(target=worker, args=args)
        p.daemon = True
        p.start()
    # connect to database
    uri = 'sqlite:///test.sqlite'
    engine = create_engine(uri, encoding='utf-8', convert_unicode=True)
    metadata = Base.metadata
    metadata.create_all(engine, checkfirst=True)
    Session = sessionmaker(bind=engine)
    session = Session()
    # options
    options = AttribDict()
    options.patterns = ["BW.*"]
    options.paths = ["C:\\Users\\barsch\\Workspace\\seishub\\trunk\\data"]
    options.keep_files = False
    # init crawler
    service = WaveformIndexer(session, input_queue, output_queue, options)
    try:
        service.serve_forever(0.0)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    run()
