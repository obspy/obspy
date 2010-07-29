from PyQt4 import QtCore, QtGui

from datetime import datetime
import numpy as np
from obspy.core import read
import os

from batch import create_HVSR

class batchProgress(QtCore.QThread):
    """
    Class that actually runs the batch processing and shows a progress bar.
    """
    def __init__(self, progress):
        """
        Init method.
        """
        super(batchProgress, self).__init__(None)
        self.progress = progress

    def begin(self, groups, settings, output_dir, hour_split):
        """
        Will be called to start the thread because run takes no arguments.
        """
        self.groups = groups
        self.settings = settings
        self.output_dir = output_dir
        self.hour_split = hour_split
        # Create unique group names.
        self.uniquifyGroups()
        # Set the range and the start of the progress dialog.
        self.progress.setRange(0, len(self.groups))
        self.progress.setValue(0)
        # Init the logfile.
        logfile = os.path.join(self.output_dir, 'HVSR.log')
        self.logfile = open(logfile, 'w')
        # Make sure that all names in groups are unique.
        self.start()

    def uniquifyGroups(self):
        """
        Creates unique group names.
        """
        new_groups = []
        for group in self.groups:
            name = group.keys()[0]
            if name in new_groups:
                alt_name = name + '_alt'
                new_groups.append(alt_name)
                group[alt_name] = group[name]
                del group[name]
                continue
            new_groups.append(name)

    def createHVSR(self, group):
        """
        Takes a single group and creates an HVSR spectrum.
        """
        # Seperate name and files list.
        name = group.keys()[0]
        files = group[name]
        # Read the files.
        st = read(files[0])
        for file in files[1:]:
            st += read(file)
        # Get the smallest starttime and the largest endtime.
        starttime = min([trace.stats.starttime for trace in st])
        endtime = max([trace.stats.endtime for trace in st])
        time_range = endtime - starttime
        # Get the length of one split part.
        if self.hour_split:
            span = self.hour_split * 3600
        else:
            span = time_range
        # Count how many will be there.
        split_count = int(time_range/span)
        if (time_range/float(span)) % 1 >= 0.1:
            split_count += 1
        # Loop and always add the span to the times.
        cur_count = 1
        while True:
            # Terminate loop if the cancel button was pressed in between.
            if self.progress.cancelPressed:
                break
            # End of the loop.
            if starttime >= endtime:
                break
            # Calculate endtime for this loop.
            cur_endtime = starttime + span
            # Only go until endtime.
            if cur_endtime > endtime:
                cur_endtime = endtime
            # Use only reasonably large time spans. Otherwise end the loop.
            if cur_endtime - starttime < 0.1*span:
                break
            # The minus 1 is somewhat hacky but sometimes necessary because
            # some traces have an offset of less than 1 sample spacing. If the
            # Trace has previously been cut it might result in different sample
            # counts and therefore lost data if the 1 second is not substracted
            # from both times.
            current_stream = st.slice(starttime+1, cur_endtime-1)
            # Later used for naming the files.
            middle = starttime + 0.5 * (cur_endtime - starttime)
            output_name = '%s_%s.hvsr' % (name, str(middle))
            # Increment the count and the starttime.
            cur_count += 1
            starttime += span

            # Some checks to ensure that the stream object is likely to work
            # witht the HVSR calculation routines.

            # Only works for exactly three traces with different ids.
            ids = set([tr.stats.channel for tr in current_stream])
            # Only use for valid slices.
            if len(current_stream) != 3:
                msg = 'The current stream object does not contain exactly ' +\
                      '3 traces:\n%s' % str(current_stream)
                self.addToLogFile(msg)
                continue
            if len(ids) != 3:
                msg = 'The current stream object does not contain exactly ' +\
                      'three different channels:\n%s' % str(current_stream)
                self.addToLogFile(msg)
                continue
            # The sample count needs to be the same.
            sample_count = set([tr.stats.npts for tr in current_stream])
            if len(sample_count) != 1:
                msg = 'The traces of the stream object are not of equal ' +\
                      'length:\n%s' % str(current_stream)
                self.addToLogFile(msg)
                continue
            # The start- and endtimes also need to be identical.
            starttimes = list(set([tr.stats.starttime for tr in\
                                   current_stream]))
            endtimes = list(set([tr.stats.endtime for tr in current_stream]))
            # Or at least within one delta.
            if len(starttimes) != 1:
                delta = current_stream[0].stats.delta
                first = starttimes[0]
                times_ok = True
                for time in starttimes[1:]:
                    if abs(first - time) < delta:
                        continue
                    times_ok = False
                if not times_ok:
                    msg = 'The times of the Traces are not the same:' +\
                          '\n%s' % str(current_stream)
                    self.addToLogFile(msg)
                    continue
            if len(endtimes) != 1:
                delta = current_stream[0].stats.delta
                first = endtimes[0]
                times_ok = True
                for time in endtimes[1:]:
                    if abs(first - time) < delta:
                        continue
                    times_ok = False
                if not times_ok:
                    msg = 'The times of the Traces are not the same:' +\
                          '\n%s' % str(current_stream)
                    self.addToLogFile(msg)
                    continue
            # Set the status of the progress dialog.
            msg = 'Processing split part %i of %i' % (cur_count-1, split_count)
            self.emit(QtCore.SIGNAL('minorStatusChanged(QString, int, int)'),
                      msg, cur_count, split_count)
            # Calculate the HVSR frequency.
            retval = create_HVSR(current_stream,
                                message_function=self.changeLabel,
                                **self.settings)
            if retval is False:
                msg = 'No suitable window could be found for:' +\
                      '\n%s' % str(current_stream)
                self.addToLogFile(msg)
                continue
            master_curve, hvsr_freq, error = retval
            # Save the output.
            output_file = os.path.join(self.output_dir, output_name)
            np.savez(output_file, hvsr_curve=master_curve,
                     frequencies=hvsr_freq, error=error)
            msg = 'Saved HVSR in %s' % output_name
            self.addToLogFile(msg)

    def addToLogFile(self, msg):
        """
        Add msg to the logfile. A Timestamp will always be included.
        """

        log = '[%s]\n%s\n\n' % (str(datetime.now()), msg)
        self.logfile.write(log)

    def changeLabel(self, label):
        """
        Fires a signal that the label of the progress dialog should get
        changed.
        """
        self.emit(QtCore.SIGNAL('labelChanged(QString)'), label)

    def run(self):
        """
        Actual thread.
        """
        count = len(self.groups)
        # Reset the cancel status.
        self.progress.cancelPressed = False
        for _i, group in enumerate(self.groups):
            # Will be true if the cancel button was pressed.
            if self.progress.cancelPressed:
                break
            name = group.keys()[0]
            msg = 'Processing %s [%i of %i]' % (name, _i+1, count)
            # Emit a signal to the progress dialog. The first value is the step
            # and the second the label.
            self.emit(QtCore.SIGNAL('statusChanged(int, QString)'), _i, msg)
            self.createHVSR(group)
        self.logfile.close()
