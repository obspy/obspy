# -*- coding: utf-8 -*-
#-------------------------------------------------------------------
# Filename: htov_toolbox.py
#  Purpose: GUI for calculating HVSR.
#   Author: Lion Krischer
#    Email: krischer@geophysik.uni-muenchen.de
#  License: GPLv2
#
# Copyright (C) 2010 Lion Krischer
#---------------------------------------------------------------------

from PyQt4 import QtCore, QtGui

from copy import deepcopy
import datetime
import os
import sys
import time

import matplotlib.cm
from matplotlib.patches import Rectangle, Polygon
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib.dates import MinuteLocator, epoch2num, DateFormatter, date2num
import numpy as np
from obspy.core import read

from batch_processing import Ui_BatchDialog as BatchProcessingDialog
from batch_progress_dialog import BatchProgressDialog
from htov import *
from htov_design import Ui_MainWindow as HtoVWindow
from runBatchProgress import batchProgress
from save_plot_dialog_design import Ui_SavePlotDialog as SavePlotDialog
from overlay import Overlay
from utils import *


class HtoV(HtoVWindow):
    """
    Main Window with the design loaded from the Qt Designer.
    """
    def __init__(self, MainWindow):
        """
        Standard init.
        """
        HtoVWindow.__init__(self)
        # This is how the init is called in the file created py pyuic4.
        self.setupUi(MainWindow)
        self.main_window = MainWindow
        # Add overlay.
        self.overlay = Overlay(self.centralwidget, self)
        self.overlay.hide()
        # The central widget is handling the resizing of the overlay.
        self.centralwidget.resizeEvent = self.resizeCentralWidget
        # Connect the signals to methods.
        self.connectSignalsAndSlots()
        # Set to the multitaper spectral method.
        self.spectraMethodStack.setCurrentIndex(0)

    def resizeCentralWidget(self, event):
        """
        Extra resizing that may be necessary.
        """
        self.overlay.resize(event.size())
        event.accept()
        # Resize the raw data table. Use try except because it is not filled if
        # called before any file is opened.
        try:
            self.resizeTableCells(self.rawDataTable)
        except:
            pass

    def connectSignalsAndSlots(self):
        """
        Connect signals and slots.
        """
        # Connect the load file button.
        QtCore.QObject.connect(self.loadFileButton,
                               QtCore.SIGNAL('clicked()'),
                               self.loadFile)
        # Connect the apply edit button.
        QtCore.QObject.connect(self.editDataButton,
                               QtCore.SIGNAL('clicked()'),
                               self.applyEdit)
        # Connect the save edited file button.
        QtCore.QObject.connect(self.saveEditedDataButton,
                               QtCore.SIGNAL('clicked()'),
                               self.saveEditedFile)
        # Connect the find noise button.
        QtCore.QObject.connect(self.findNoiseButton,
                               QtCore.SIGNAL('clicked()'),
                               self.autodetectNoise)
        # Connect the button to start calculating the HVSR.
        QtCore.QObject.connect(self.calculateHVSR,
                               QtCore.SIGNAL('clicked()'),
                               self.calculateHVSpectralRatio)
        # Connect the button to update the plot.
        QtCore.QObject.connect(self.applyChangesToPlotButton,
                               QtCore.SIGNAL('clicked()'),
                               self.applyPlotChanges)
        # Connect the button to save the plot.
        QtCore.QObject.connect(self.savePlotButton,
                               QtCore.SIGNAL('clicked()'),
                               self.savePlot)
        # Connect the button the button for the batch processing.
        QtCore.QObject.connect(self.batchProcessingButton,
                               QtCore.SIGNAL('clicked()'),
                               self.startBatch)
        # Connect the filter sliders.
        QtCore.QObject.connect(self.lowpassSlider,
                               QtCore.SIGNAL('sliderMoved(int)'),
                               self.lowpassSliderMoved)
        QtCore.QObject.connect(self.highpassSlider,
                               QtCore.SIGNAL('sliderMoved(int)'),
                               self.highpassSliderMoved)
        # Connect the filter spin boxes.
        QtCore.QObject.connect(self.lowpassSpinBox,
                               QtCore.SIGNAL('valueChanged(double)'),
                               self.lowpassSpinBoxChanged)
        QtCore.QObject.connect(self.highpassSpinBox,
                               QtCore.SIGNAL('valueChanged(double)'),
                               self.highpassSpinBoxChanged)
        # Connect the time selectors.
        QtCore.QObject.connect(self.starttimeEdit,
                               QtCore.SIGNAL('dateTimeChanged(QDateTime)'),
                               self.starttimeChanged)
        QtCore.QObject.connect(self.endtimeEdit,
                               QtCore.SIGNAL('dateTimeChanged(QDateTime)'),
                               self.endtimeChanged)
        # Connect the resample spin box.
        QtCore.QObject.connect(self.resampleSpinBox,
                               QtCore.SIGNAL('valueChanged(double)'),
                               self.resampleSpinBoxChanged)
        # The value of the spectra calculation method combo box has changed.
        QtCore.QObject.connect(self.spectraMethodComboBox,
                               QtCore.SIGNAL('currentIndexChanged(int)'),
                               self.spectraMethodStackChanged)
        # The value of the master method calculation combo box has changed.
        QtCore.QObject.connect(self.masterCurveComboBox,
                               QtCore.SIGNAL('currentIndexChanged(int)'),
                               self.masterCurveComboBoxChanged)
        # Connect the check box of the multitaper padding.
        QtCore.QObject.connect(self.multitaperPadCheckBox,
                               QtCore.SIGNAL('stateChanged(int)'),
                               self.multitaperPadCheckStateChanged)
        # Connect the smoothing check box.
        QtCore.QObject.connect(self.smoothingCheckBox,
                               QtCore.SIGNAL('stateChanged(int)'),
                               self.toggleSmoothingControls)
        # Connect the matplotlib HVSR canvas to a mouse click event.
        cid = self.hvPlot.fig.canvas.mpl_connect('button_press_event',
                                                 self.hvsrCanvasClick)
        # Connect the matplotlib HVSR canvas to a resize event because some
        # selection items need to be updated.
        cid = self.hvPlot.fig.canvas.mpl_connect('resize_event',
                                                 self.hvsrCanvasResize)

    def hvsrCanvasClick(self, event):
        """
        Handles the picking of the HVSR peak.
        """
        fig = self.hvPlot.fig
        ax = fig.hv_ax
        min, max = ax.get_xlim()
        xdata = event.xdata
        # If no figure, return.
        if xdata is None:
            return
        # If out of bounds, return.
        if xdata <= min or xdata >= max:
            return
        if event.button == 1:
            # Remove any old line and also the error bars when plotting a new
            # center bar.
            try:
                ax.hvsr_line.remove()
                del ax.hvsr_line
            except: pass
            try:
                ax.hvsr_left_error.remove()
                del ax.hvsr_left_error
            except: pass
            try:
                ax.hvsr_right_error.remove()
                del ax.hvsr_right_error
            except: pass
            try:
                ax.hvsr_error_box.remove()
                del ax.hvsr_error_box
            except:
                pass
            # Snap to the highest point within 3% of the plot.
            range = max-min
            # Use log scale.
            if ax.get_xscale() == 'log':
                log_range = np.log10(max - min)
                span = 0.025*log_range
                peak_min = 10**(np.log10(xdata) - span)
                peak_max = 10**(np.log10(xdata) + span)
            # Use a linear scale.
            else:
                span = 0.025*range
                peak_min = xdata - span
                peak_max = xdata + span
            if peak_min < min:
                peak_min = min
            if peak_max > max:
                peak_max = max
            min_index = np.where(self.hvsr_freq >= peak_min)[0].min()
            max_index = np.where(self.hvsr_freq <= peak_max)[0].max()
            xdata = self.master_curve[min_index: max_index].max()
            index = np.where(self.master_curve == xdata)
            self.hvsr_frequency = self.hvsr_freq[index][0]
            ax.hvsr_line = ax.axvline(self.hvsr_frequency, color='black',
                                      linewidth=3)
            self.writeToStatusBar('Fundamental frequency: %.3f' %
                                  self.hvsr_frequency)
            # Add an indicator.
            self.addFrequencyIndicator(ax)
        # Set the error bars with the right mouse button.
        elif event.button == 3:
            # Do nothing if no real pick has been set.
            if not hasattr(ax, 'hvsr_line'):
                return
            # Remove and draw a new left one.
            if xdata < self.hvsr_frequency:
                self.hvsr_left_error = xdata
                # Remove any old line.
                try:
                    ax.hvsr_left_error.remove()
                    del ax.hvsr_left_error
                except: pass
                ax.hvsr_left_error = ax.axvline(self.hvsr_left_error, ls='--',
                                                color='0.3', linewidth=2)
            # Remove and draw a new right one.
            elif xdata > self.hvsr_frequency:
                self.hvsr_right_error = xdata
                # Remove any old line.
                # Remove any old line.
                try:
                    ax.hvsr_right_error.remove()
                    del ax.hvsr_right_error
                except: pass
                ax.hvsr_right_error = ax.axvline(self.hvsr_right_error,
                                         ls='--', color='0.3', linewidth=2)
            # Draw an error rectangle.
            if hasattr(ax, 'hvsr_left_error'):
                left = self.hvsr_left_error
            else:
                left = self.hvsr_frequency
            if hasattr(ax, 'hvsr_right_error'):
                right = self.hvsr_right_error
            else:
                right = self.hvsr_frequency
            # Update the rectangle if it already exists.
            if hasattr(ax, 'hvsr_error_box'):
                ax.hvsr_error_box.set_x(left)
                ax.hvsr_error_box.set_width(right - left)
            else:
                bottom, top = ax.get_ylim()
                ax.hvsr_error_box = Rectangle((left, bottom), right - left, top - bottom,
                                  color='grey', alpha=0.5)
                ax.add_patch(ax.hvsr_error_box)
        try:
            ax.freq_text.remove()
            del ax.freq_text
        except: pass
        msg = 'Fundmental Frequency = %.3f Hz' % \
              self.hvsr_frequency
        if hasattr(ax, 'hvsr_left_error'):
            left = self.hvsr_left_error
        else:
            left = self.hvsr_frequency
        if hasattr(ax, 'hvsr_right_error'):
            right = self.hvsr_right_error
        else:
            right = self.hvsr_frequency
        if hasattr(ax, 'hvsr_left_error') or hasattr(ax, 'hvsr_right_error'):
            msg = '%s\n (%.3f - %.3f Hz)' % (msg, left, right)
        ax.freq_text = ax.text(0.01, 0.97, msg, horizontalalignment='left',
                       verticalalignment='top', backgroundcolor='white',
                       transform=ax.transAxes, zorder='100000')
        # Redraw the canvas.
        fig.canvas.draw()

    def hvsrCanvasResize(self, event):
        """
        Handles the resizing of the canvas.
        """
        fig = self.hvPlot.fig
        ax = fig.hv_ax
        self.addFrequencyIndicator(ax)
        # Redraw the error Rectangle because updating does not work for some
        # resaon.
        if not hasattr(ax, 'hvsr_error_box'):
            return
        try:
            ax.hvsr_error_box.remove()
            del ax.hvsr_error_box
        except: pass
        if hasattr(ax, 'hvsr_left_error'):
            left = self.hvsr_left_error
        else:
            left = self.hvsr_frequency
        if hasattr(ax, 'hvsr_right_error'):
            right = self.hvsr_right_error
        else:
            right = self.hvsr_frequency
        # Update the rectangle if it already exists.
        if hasattr(ax, 'hvsr_error_box'):
            ax.hvsr_error_box.set_x(left)
            ax.hvsr_error_box.set_width(right - left)
        else:
            bottom, top = ax.get_ylim()
            ax.hvsr_error_box = Rectangle((left, bottom), right - left, top - bottom,
                              color='grey', alpha=0.5)
            ax.add_patch(ax.hvsr_error_box)
        # Redraw the canvas.
        fig.canvas.draw()

    def addFrequencyIndicator(self, ax):
        """
        Adds an indicator to fig at self.hvsr_frequency. Does not redraw the
        figure.
        """
        # Width and height of the triangle in percent.
        width = 0.01
        height = 0.03
        if not hasattr(self, 'hvsr_frequency'):
            return
        # Remove any old indicator.
        try:
            ax.hvsr_frequency_indicator.remove()
            del ax.hvsr_frequency_indicator
        except: pass
        # Get the limits.
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xrange = xmax-xmin
        yrange = ymax-ymin
        # It is necessary to differentiate between linear and log scales.
        if ax.get_xscale() == 'log':
            log_range = np.log10(xmax) - np.log10(xmin)
            log_value = np.log10(self.hvsr_frequency)
            left = 10**(log_value - 0.5*width*log_range)
            right = 10**(log_value + 0.5*width*log_range)
        else:
            left = self.hvsr_frequency - 0.5*width*xrange
            right = self.hvsr_frequency + 0.5*width*xrange
        if ax.get_yscale() == 'log':
            log_range = np.log10(ymax) - np.log10(ymin)
            bottom = 10**(np.log10(ymin) + (1-height)*log_range)
        else:
            bottom = ymax - height*yrange
        # Add the polygon.
        ax.hvsr_frequency_indicator =\
            Polygon(np.array([[self.hvsr_frequency, bottom], [right, ymax],
                    [left, ymax]]), color='red', zorder=1000)
        ax.add_patch(ax.hvsr_frequency_indicator)

    def resizeTableCells(self, table):
        """
        Resizes all table cell proportionally to adjust to the available space.
        Event is the resize event and table the corresponding table.
        """
        new_width = table.viewport().size().width()
        column_count = table.model().columnCount()
        # Get the old widths and resize proportionally.
        widths = [table.columnWidth(_i) for _i in xrange(column_count)]
        old_width = sum(widths)
        widths = [float(width) / old_width * new_width for width in widths]
        # Finally set the new size.
        for col in xrange(column_count):
            table.setColumnWidth(col, widths[col])

    def starttimeChanged(self, new_starttime):
        """
        Fired when the starttime is changed. Used to set the new minimum
        endtime.
        """
        self.endtimeEdit.setMinimumDateTime(new_starttime)

    def endtimeChanged(self, new_endtime):
        """
        Fired when the endtime is changed. Used to set the new maximum
        endme.
        """
        self.starttimeEdit.setMaximumDateTime(new_endtime)

    def applyEdit(self):
        """
        Apply the editing of the trace. The data will first be cut, then
        filtered and resampled and then the noisy areas are searched.
        """
        self.overlay.show()
        # Create a deepcopy of the original stream.
        self.edited_stream = deepcopy(self.original_stream)
        # Read the variables from the GUI.
        resampling_rate = self.resampleSpinBox.value()
        # Get the low- and highpass values.
        lowpass_value = highpass_value = False
        # Determine whether low- or highpass filters are being used.
        if self.lowpassSpinBox.value() < self.nyquist_freq - 0.1:
            lowpass_value = self.lowpassSpinBox.value()
        if self.highpassSpinBox.value() > 0.0:
            highpass_value = self.highpassSpinBox.value()
        if self.zerophaseCheckbox.checkState() == 2:
            zerophase = True
            corners = int(self.cornersSpinBox.value() / 2)
            if corners == 0:
                corners = 1
        else:
            zerophase = False
            corners = int(self.cornersSpinBox.value())
        # Get the times for trimming.
        starttime = fromQDateTime(self.starttimeEdit.dateTime())
        endtime = fromQDateTime(self.endtimeEdit.dateTime())
        # Resample, filter and cut the trace.
        resampleFilterAndCutTraces(self.edited_stream, resampling_rate,
                                   lowpass_value, highpass_value, zerophase,
                                   corners, starttime, endtime,
                                   message_function=self.writeToStatusBar)
        # Plot the data.
        self.writeToStatusBar('Plotting edited file...')
        self.plotData(self.editDataPlot.fig, self.edited_stream)
        # Change the upper tab so it shows the edited trace.
        self.plotTabWidget.setCurrentIndex(1)
        self.writeToStatusBar('')
        self.overlay.hide()
        # Enable the save button.
        self.saveEditedDataButton.setEnabled(True)

    def lowpassSpinBoxChanged(self, value):
        """
        Fired every time the value of the lowpass spin box is changed.
        """
        # The slider value is the same times 1000.
        slider_value = float(value * 1000.0)
        self.lowpassSlider.setValue(slider_value)
        self.highpassSlider.setMaximum(slider_value)
        self.highpassSpinBox.setMaximum(value)

    def toggleSmoothingControls(self, value):
        """
        Toggles the smoothing controls when the smoothing check box is checked
        or unchecked.
        """
        if value == 2:
            self.smoothingComboBox.setEnabled(True)
            self.smoothingSpinBox.setEnabled(True)
            self.smoothingSpinBox2.setEnabled(True)
            self.smoothingLabel.setEnabled(True)
            self.smoothingLabel2.setEnabled(True)
        else:
            self.smoothingComboBox.setEnabled(False)
            self.smoothingSpinBox.setEnabled(False)
            self.smoothingSpinBox2.setEnabled(False)
            self.smoothingLabel.setEnabled(False)
            self.smoothingLabel2.setEnabled(False)

    def highpassSpinBoxChanged(self, value):
        """
        Fired every time the value of the lowpass spin box is changed.
        """
        # The slider value is the same times 1000.
        slider_value = float(value * 1000.0)
        self.highpassSlider.setValue(slider_value)
        self.lowpassSlider.setMinimum(slider_value)
        self.lowpassSpinBox.setMinimum(value)

    def resampleSpinBoxChanged(self, value):
        """
        Fired every time the value of the resample spin box is changed.
        """
        # New nyquist frequency.
        self.nyquist_freq = value / 2.0
        self.lowpassSpinBox.setMaximum(self.nyquist_freq)
        self.lowpassSlider.setMaximum(1000 * self.nyquist_freq)

    def lowpassSliderMoved(self, new_position):
        """
        Fired every time the lowpass slider is moved.
        """
        # The sliders only accept integer values. Therefore steps of 1000 are
        # used.
        real_value = float(new_position) / 1000.0
        self.lowpassSpinBox.setValue(real_value)
        self.highpassSlider.setMaximum(new_position)
        self.highpassSpinBox.setMaximum(real_value)

    def highpassSliderMoved(self, new_position):
        """
        Fired every time the lowpass slider is moved.
        """
        # The sliders only accept integer values. Therefore steps of 1000 are
        # used.
        real_value = float(new_position) / 1000.0
        self.highpassSpinBox.setValue(real_value)
        self.lowpassSlider.setMinimum(new_position)
        self.lowpassSpinBox.setMinimum(real_value)

    def saveEditedFile(self):
        """
        Load the file.
        """
        # File dialog.
        caption = 'Save Edited File As'
        filename = QtGui.QFileDialog.getSaveFileName(caption=caption)
        # If not filename is given, return.
        if not filename:
            return
        # Write the stream as a Mini-SEED file.
        self.edited_stream.write(filename, format='MSEED')

    def clearAllFigures(self):
        """
        Clears and redraws all figures.
        """
        figs = [self.hvPlot.fig, self.rawDataPlot.fig, self.editDataPlot.fig]
        for fig in figs:
            fig.clear()
            fig.canvas.draw()

    def deleteAttributes(self):
        """
        Tries to delete all attributes that are not necessary after loading
        another file.
        """
        try:
            del self.hvsr_frequency
        except: pass

    def loadFile(self):
        """
        Load the file.
        """
        # File dialog.
        filename = QtGui.QFileDialog.getOpenFileName()
        # If not filename is given, return.
        if not filename:
            return
        self.filename = str(filename)
        self.overlay.show()
        self.writeToStatusBar('Opening file...')
        # Gracefully exit if obspy cannot read the file.
        # XXX: Add proper error message.
        self.original_stream = read(self.filename)
        # Clear all figures.
        self.clearAllFigures()
        # Delete some variables
        self.deleteAttributes()
        # Set the filename label.
        self.filenameLabel.setText('Loaded file: %s [%.2f MB]' % (self.filename,
                                   os.path.getsize(self.filename) / 1024.0 ** 2))
        # Add information to the table for each trace.
        self.rawDataTable.setRowCount(len(self.original_stream))
        items = ['network', 'station', 'location', 'channel', 'sampling_rate',
                 'npts', 'starttime', 'endtime']
        pretty_items = ['Network', 'Station', 'Location', 'Channel',
                        'Sampling Rate [Hz]', 'Sample Count',
                        'Start Time [UTC]', 'End Time[UTC]',
                        'Trace Orientation']
        # Try to autodetermine the orientation of each channel. Will be written
        # ti stats attribute in orientation
        detectTraceOrientation(self.original_stream)
        self.rawDataTable.setColumnCount(len(items) + 1)
        self.rawDataTable.setHorizontalHeaderLabels(pretty_items)
        self.orientations = []
        for _i, trace in enumerate(self.original_stream):
            for _j, item in enumerate(items):
                table_item = QtGui.QTableWidgetItem(str(trace.stats[item]))
                self.rawDataTable.setItem(_i, _j, table_item)
            orientation_chooser = QtGui.QComboBox()
            # Add attribute to be able to identify it later on.
            orientation_chooser.numberInTable = _i
            orientation_chooser.addItem('horizontal')
            orientation_chooser.addItem('vertical')
            orientation_chooser.last_changed = time.time()
            # Set the style sheet to make it look better.
            orientation_chooser.setStyleSheet('QComboBox{border-radius:0px;}')
            self.rawDataTable.setCellWidget(_i, len(items),
                                            orientation_chooser)
            # Append to list to have access later on.
            self.orientations.append(orientation_chooser)
            if trace.stats.orientation == 'vertical':
                orientation_chooser.setCurrentIndex(1)
            else:
                orientation_chooser.setCurrentIndex(0)
            # Connect the load file button.
            QtCore.QObject.connect(orientation_chooser,
                                   QtCore.SIGNAL('currentIndexChanged(int)'),
                                   self.adjustOrientationComboBoxes)
        self.rawDataTable.resizeColumnsToContents()
        # Auto resize to stretch.
        self.resizeTableCells(self.rawDataTable)
        # Plot the file to the rawDataPlot.
        self.writeToStatusBar('Plotting raw data...')
        self.plotData(self.rawDataPlot.fig, self.original_stream)
        self.overlay.hide()
        # Setup everything to edit the traces.
        self.setupEdits()
        # Change the upper tab so it shows the raw data.
        self.plotTabWidget.setCurrentIndex(0)
        self.writeToStatusBar('')

    def adjustOrientationComboBoxes(self, index):
        """
        Makes sure that one component will always be vertical and one will be
        horizontal.
        """
        cur_time = time.time()
        last_time = max([ori.last_changed for ori in self.orientations])
        if cur_time - last_time < 0.1:
            return
        # Get the sender of the signal.
        sender = self.qApp.sender()
        id = sender.numberInTable
        # The current combo box has been set to vertical.
        # Set the default last_changed value of all but the last.
        for orientation in self.orientations:
            orientation.last_changed = cur_time
        if index == 1:
            # Set all other horizontal.
            for _i, orientation in enumerate(self.orientations):
                if _i == id:
                    orientation.setCurrentIndex(1)
                else:
                    orientation.setCurrentIndex(0)
        # Set all to horizontal and the first or second one to vertical.
        else:
            if id == 0:
                id = 1
            else:
                id = 0
            for _i, orientation in enumerate(self.orientations):
                if _i == id:
                    orientation.setCurrentIndex(1)
                else:
                    orientation.setCurrentIndex(0)
        # Set the traces attributes accordingly.
        for _i, trace in enumerate(self.original_stream):
            # Vertical.
            if self.orientations[_i].currentIndex() == 1:
                trace.stats.orientation = 'vertical'
            # Horizontal
            else:
                trace.stats.orientation = 'horizontal'

    def setupEdits(self):
        """
        Sets the minimum and maximum frequencies for the filter GUI elements.
        """
        self.max_freq = self.original_stream[0].stats.sampling_rate
        self.nyquist_freq = self.max_freq / 2.0
        # Set the sliders.
        self.lowpassSlider.setRange(0, int(self.nyquist_freq * 1000))
        self.highpassSlider.setRange(0, int(self.nyquist_freq * 1000))
        self.lowpassSlider.setSliderPosition(int(self.nyquist_freq * 1000))
        self.highpassSlider.setSliderPosition(0)
        # Set the spin boxes.
        self.lowpassSpinBox.setRange(0.0, self.nyquist_freq)
        self.highpassSpinBox.setRange(0.0, self.nyquist_freq)
        self.lowpassSpinBox.setValue(self.nyquist_freq)
        self.highpassSpinBox.setValue(0.0)
        # Set the times.
        self.starttime = self.original_stream[0].stats.starttime
        self.endtime = self.original_stream[0].stats.endtime
        starttime = toQDateTime(self.starttime)
        endtime = toQDateTime(self.endtime)
        self.starttimeEdit.setDateTimeRange(starttime, endtime)
        self.endtimeEdit.setDateTimeRange(starttime, endtime)
        self.starttimeEdit.setDateTime(starttime)
        self.endtimeEdit.setDateTime(endtime)
        # Set the resampling frequency.
        self.resampleSpinBox.setValue(self.max_freq)

    def writeToStatusBar(self, msg):
        """
        Write msg to the status bar and processes all qApp events to
        immediately show the changes.
        """
        self.statusbar.showMessage(msg)
        self.qApp.processEvents()

    def autodetectNoise(self):
        """
        Detects the noise and saves the characteristic functions.
        """
        self.overlay.show()
        self.writeToStatusBar('Deleting old plot...')
        # Delete everything that might be on the plot already.
        for _i in xrange(3):
            if hasattr(self.editDataPlot.fig, 'ax%i' % _i):
                # Get the axis.
                ax = getattr(self.editDataPlot.fig, 'ax%i' % _i)
                # Remove the characteristic function and the treshold.
                if hasattr(ax, 'twin_xaxis'):
                    ax.twin_xaxis.extra_plot.remove()
                    ax.twin_xaxis.threshold.remove()
                # Remove the quiet areas.
                if hasattr(ax, 'quiet_areas'):
                    for patch in ax.quiet_areas:
                        patch.remove()
                # Remove the common areas.
                if hasattr(ax, 'common_areas'):
                    for patch in ax.common_areas:
                        patch.remove()
                # Remove the intervals.
                if hasattr(ax, 'intervals'):
                    for patch in ax.intervals:
                        patch.remove()
        self.window_length = self.windowLengthSpinBox.value() *\
                        self.edited_stream[0].stats.sampling_rate

        # Read the paramters from the GUI.
        threshold = self.thresholdSpinBox.value() / 100.0
        z_detector_window_length = self.zDetectorSpinBox.value()
        # Get the characteristic function and the thresholds.
        self.charNoiseFunctions, thresholds = \
                calculateCharacteristicNoiseFunction(self.edited_stream,
                threshold, z_detector_window_length,
                message_function=self.writeToStatusBar)
        # Plot the characteristic functions on top of the data.
        self.plotSomethingOverData(self.editDataPlot.fig,
                                   self.charNoiseFunctions,
                                   thresholds=thresholds)
        self.writeToStatusBar('Find quiet areas...')

        # Read from the GUI again.
        npts = self.edited_stream[0].stats.npts
        window_length = self.window_length
        self.intervals, self.quiet_areas, self.common_quiet_areas = \
            getQuietIntervals(self.charNoiseFunctions, thresholds,
                              window_length, npts)

        # Currently the quiet areas are in samples. Convert to a dateformat
        # that matplotlib understands.
        self.quiet_areas_times = self.convertSamplesToMatplotlibDate(\
                                    self.edited_stream, self.quiet_areas)
        self.common_quiet_areas_times = self.convertSamplesToMatplotlibDate(\
                            self.edited_stream, [self.common_quiet_areas])[0]
        self.intervals_times = self.convertSamplesToMatplotlibDate(\
                            self.edited_stream, [self.intervals])[0]
        self.writeToStatusBar('Plot the results...')
        # Plot the quiet areas to each trace.
        self.plotAreasToDatePlot(self.editDataPlot.fig, self.quiet_areas_times)
        # Plot the common area to every trace.
        self.plotAreaToEveryPlot(self.editDataPlot.fig,
                                 self.common_quiet_areas_times,
                                 tag='common_areas')
        # Plot the intervals.
        self.plotAreaToEveryPlot(self.editDataPlot.fig,
                                 self.intervals_times, colormap='hsv',
                                 alpha=0.6, linewidth=2,
                                 tag='intervals')
        # Set the minimum value of the padding box for the spectraCalculations.
        self.nfftSpinBox.setMinimum(self.window_length)
        self.nfftSpinBox.setValue(self.window_length)
        # Change the upper tab so it shows the edited trace.
        self.plotTabWidget.setCurrentIndex(1)
        self.writeToStatusBar('')
        self.overlay.hide()

    def calculateHVSpectralRatio(self):
        """
        Finally calculates the HVSR.
        """
        self.overlay.show()
        self.writeToStatusBar('Calculating HVSR...')
        # Get the figure and delete any content on it.
        fig = self.hvPlot.fig
        fig.clear()

        # Determine the method to calculate the spectra.
        method = str(self.spectraMethodComboBox.currentText()).lower()
        if method == 'multitaper':
            # Get the variables for the multitaper spectra calculation from the
            # GUI.
            time_bandwidth = self.timeBandwidthSpinBox.value()
            number_of_tapers = \
                    str(self.numberOfTapersComboBox.currentText()).lower()
            if number_of_tapers == 'auto':
                number_of_tapers = None
            else:
                number_of_tapers = int(number_of_tapers)
            quadratic = self.quadraticCheckBox.checkState()
            if quadratic == 2:
                quadratic = True
            else:
                quadratic = False
            adaptive = self.adaptiveCheckBox.checkState()
            if adaptive == 2:
                adaptive = True
            else:
                adaptive = False
            if self.multitaperPadCheckBox.checkState() == 2:
                nfft = self.nfftSpinBox.value()
                if nfft <= self.window_length:
                    nfft = None
            else:
                nfft = None
            spec_options = {'time_bandwidth': time_bandwidth,
              'number_of_tapers': number_of_tapers, 'quadratic': quadratic,
              'adaptive': adaptive, 'nfft': nfft}
        elif method == 'sine multitaper':
            # Read the parameters from the GUI.
            number_of_tapers = self.sinePSDTapersSpinBox.value()
            if number_of_tapers == 0:
                number_of_tapers = None
            number_of_iterations = self.sinePSDIterationsSpinBox.value()
            degree_of_smoothing = self.sinePSDSmoothingSpinBox.value()
            spec_options = {'number_of_tapers': number_of_tapers,
                       'degree_of_smoothing': degree_of_smoothing,
                       'number_of_iterations': number_of_iterations}
        elif method == 'single taper':
            # Read the taper from the GUI.
            taper = str(self.singleTaperComboBox.currentText()).lower()
            spec_options = {'taper': taper}
        # Calculate the master method.
        master_method = str(self.masterCurveComboBox.currentText()).lower()
        # Geometric average.
        if 'geometric' in master_method:
            master_method = 'geometric average'
        # Mean.
        elif 'mean' in master_method:
            master_method = 'mean'
        # Median.
        elif 'median' in master_method:
            master_method = 'median'
        # Discard the top and bottom percentile.
        cutoff_value = self.cutoffSpinBox.value() / 100.0
        # Determine whether or not to smooth.
        if self.smoothingCheckBox.checkState() == 2:
            smoothing = str(self.smoothingComboBox.currentText()).lower()
            smoothing_count = self.smoothingSpinBox2.value()
            smoothing_constant = self.smoothingSpinBox.value()
        else:
            smoothing = None
            smoothing_count = 1
            smoothing_constant = 40
        # Call the HVSR Method.
        hvsr_matrix, self.hvsr_freq, length, self.master_curve, self.error = \
                calculateHVSR(self.edited_stream, self.intervals,
                self.window_length, method, spec_options, master_method,
                cutoff_value, smoothing=smoothing,
                smoothing_count=smoothing_count,
                smoothing_constant=smoothing_constant,
                message_function=self.writeToStatusBar)
        self.writeToStatusBar('Plotting results...')
        # Plot the spectra.
        ax = fig.add_subplot(111)
        # Append axis to the plot.
        fig.hv_ax = ax
        for _i in xrange(len(self.intervals)):
            ax.plot(self.hvsr_freq, hvsr_matrix[_i, :], color=self.colors[_i],
                    alpha=0.5)
        # Plot master curve and errors.
        ax.plot(self.hvsr_freq, self.master_curve, color='black', linewidth=3)
        ax.plot(self.hvsr_freq, self.error[:, 0], '--', color='black',
                linewidth=3)
        ax.plot(self.hvsr_freq, self.error[:, 1], '--', color='black',
                linewidth=3)
        # Read the values for the plot from the GUI.
        x_min = self.xAxisMinSpinBox.value()
        x_max = self.xAxisMaxSpinBox.value()
        y_min = self.yAxisMinSpinBox.value()
        y_max = self.yAxisMaxSpinBox.value()
        if self.xAxisLogCheckBox.checkState() == 2:
            x_scale = 'log'
        else:
            x_scale = 'linear'
        if self.yAxisLogCheckBox.checkState() == 2:
            y_scale = 'log'
        else:
            y_scale = 'linear'
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_locator(MaxNLocator(10))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.grid()
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('H/V Spectral Ratio using %i Windows' % \
                      len(self.intervals))
        es = self.edited_stream[0].stats
        ax.set_title('%s.%s.%s ||  %s - %s [%s]' % (es.network, es.station,
                es.location, str(es.starttime)[:19], str(es.endtime)[:19],
                os.path.basename(self.filename)), fontsize='small')
        # Change the upper tab so it shows the HVSR.
        self.plotTabWidget.setCurrentIndex(2)
        # Force the redraw of the figure canvas.
        fig.canvas.draw()
        self.writeToStatusBar('')
        self.overlay.hide()

    def savePlot(self):
        """
        Opens a dialog with configuration options to save the plot.
        """
        dialog = QtGui.QDialog()
        # This writes the dialog interface to the dialog. The figure also needs
        # to be given to the class so that is can be totally self contained.
        SaveHVSRPlotDialog(dialog, self.hvPlot.fig)
        dialog.exec_()

    def startBatch(self):
        """
        Opens a dialog with configuration options for the batch processing.
        """
        dialog = QtGui.QDialog()
        # This writes the dialog interface to the dialog. The figure also needs
        # to be given to the class so that is can be totally self contained.
        BatchHVSRProcessingDialog(dialog, self)
        dialog.exec_()

    def applyPlotChanges(self):
        """
        Reads everything necessary from the GUI and updates the plot.
        """
        self.overlay.show()
        self.writeToStatusBar('Applying plot changes...')
        # Read the values.
        x_min = self.xAxisMinSpinBox.value()
        x_max = self.xAxisMaxSpinBox.value()
        y_min = self.yAxisMinSpinBox.value()
        y_max = self.yAxisMaxSpinBox.value()
        if self.xAxisLogCheckBox.checkState() == 2:
            x_scale = 'log'
        else:
            x_scale = 'linear'
        if self.yAxisLogCheckBox.checkState() == 2:
            y_scale = 'log'
        else:
            y_scale = 'linear'
        # Apply to the plot.
        fig = self.hvPlot.fig
        ax = fig.hv_ax
        # Set the values.
        ax.set_xscale(x_scale)
        ax.set_yscale(y_scale)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.xaxis.set_major_locator(MaxNLocator(10))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        # Change the upper tab so it shows the HVSR.
        self.plotTabWidget.setCurrentIndex(2)
        self.hvsrCanvasResize(ax)
        # Force the redraw of the figure canvas.
        fig.canvas.draw()
        self.writeToStatusBar('')
        # Write the fundamental frequency to the status bar.
        if hasattr(self, 'hvsr_frequency'):
            self.writeToStatusBar('Fundamental frequency: %.3f' %
                                  self.hvsr_frequency)
        self.overlay.hide()

    def convertSamplesToMatplotlibDate(self, stream, areas):
        """
        areas contains tuples with that are the start and the end of an area in
        samples. This method converts these to a dateformat that matplotlib
        understands.
        The stream is needed to get the date information.
        """
        new_areas = []
        starttime = stream[0].stats.starttime
        endtime = stream[0].stats.endtime
        length = float(stream[0].stats.npts)
        # Convert to matplotlib epoch times.
        starttime, endtime = date2num((starttime, endtime))
        range = endtime - starttime
        # Loop over every trace.
        for data in areas:
            data = starttime + ((data / length) * range)
            new_areas.append(data)
        return new_areas

    def plotAreasToDatePlot(self, fig, quiet_areas):
        """
        Plots the quiet areas in samples to the fig.
        """
        for _i, areas in enumerate(quiet_areas):
            # Get the axis.
            ax = getattr(fig, 'ax%i' % _i)
            ax.quiet_areas = []
            bottom, top = ax.get_ylim()
            for left, right in areas:
                patch = Rectangle((left, bottom), right - left, top - bottom,
                                  color='green', alpha=0.1)
                ax.add_patch(patch)
                ax.quiet_areas.append(patch)
        # Force the redraw of the figure canvas.
        fig.canvas.draw()

    def plotAreaToEveryPlot(self, fig, quiet_area, colormap=None, alpha=0.1,
                            linewidth=1, tag=None):
        """
        Plots the quiet areas in samples to the fig.
        """
        if colormap:
            cm = getattr(matplotlib.cm, colormap)
            self.colors = []
        for _i in xrange(len(self.edited_stream)):
            # Get the axis.
            ax = getattr(fig, 'ax%i' % _i)
            bottom, top = ax.get_ylim()
            if not tag:
                tag = 'temp'
            # Save the patches somewhere.
            setattr(ax, tag, [])
            patch_list = getattr(ax, tag)
            for _j, area in enumerate(quiet_area):
                left, right = area
                if colormap:
                    color = cm(int(float(_j) / len(quiet_area) * cm.N))
                    # Add the color to the intervals so they can be retrieved
                    # later on.
                    self.colors.append(color)
                else:
                    color = 'green'
                patch = Rectangle((left, bottom), right - left, top - bottom,
                                  color=color, alpha=0.2)
                ax.add_patch(patch)
                patch_list.append(patch)
        msg = 'Found %i windows' % len(self.intervals)
        # Add the number of windows to the plot.
        ax.freq_text = fig.text(0.01, 0.97, msg, horizontalalignment='left',
                       verticalalignment='top', backgroundcolor='0.5')
        # Force the redraw of the figure canvas.
        fig.canvas.draw()

    def plotSomethingOverData(self, fig, data_list, thresholds=None):
        """
        Plots arrays in data_list to fig. The fig should have as many subplots
        as the data_list has arrays.
        """
        for _i, data in enumerate(data_list):
            # Get the axis.
            ax = getattr(fig, 'ax%i' % _i)
            twinx = ax.twinx()
            ax.twin_xaxis = twinx
            ax = ax.twin_xaxis
            # Plot whatever data there is. Name it to be able to access it
            # later on.
            ax.extra_plot, = ax.plot_date(fig.epoch_dates, data, color='red',
                                         linestyle='-', marker='None')
            # Plot the threshold if there is one. The threshold is always given
            # as a percentile.
            if thresholds:
                ax.threshold, = ax.plot_date([fig.epoch_dates[0],
                               fig.epoch_dates[-1]], [thresholds[_i],
                               thresholds[_i]], color='blue',
                               linestyle='-', marker='None')
            ax.set_yticks([])
            # The next stuff just duplicates the formatting of the original
            # plot so it will not change the appearance of the plot.
            # Set the xaxis locators.
            ax.xaxis.set_major_locator(MaxNLocator(6))
            # Only the last trace should contain labels on the xaxis.
            if _i != 2:
                ax.xaxis.set_major_formatter(DateFormatter(''))
            else:
                ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            ax.xaxis.set_minor_locator(MinuteLocator())
            ax.grid()
        # Force the redraw of the figure canvas.
        fig.canvas.draw()

    def plotData(self, fig, stream):
        """
        Plots any stream to fig.
        """
        # Ensure the same scale for all traces.
        self.max_distance = 0
        for trace in stream:
            trace.stats.mean = trace.data.mean()
            max_distance = max([trace.data.max() - trace.stats.mean,
                                trace.stats.mean - trace.data.min()])
            if max_distance > self.max_distance:
                self.max_distance = max_distance
        # Add 10 percent for nicer looks.
        self.max_distance *= 1.1
        length = len(stream[0].data)
        starttime = stream[0].stats.starttime
        endtime = stream[0].stats.endtime
        dates = np.linspace(starttime.timestamp, endtime.timestamp, length)
        # Convert to gregorian epochs.
        dates = epoch2num(dates)
        # Add them to the fig object for later access.
        fig.epoch_dates = dates
        # Plot the data.
        fig.clear()
        for _i, trace in enumerate(stream):
            ax = fig.add_subplot(len(stream), 1, _i + 1)
            # Add the subplots as attributes to the figures.
            setattr(fig, 'ax%i' % _i, ax)
            # Add the data plot to the axis to be able to change it later on.
            ax.data_plot = ax.plot_date(dates, trace.data, color='black',
                                        linestyle='-', marker='None')
            # Same scale for all traces.
            ax.set_ylim(trace.stats.mean - self.max_distance,
                        trace.stats.mean + self.max_distance)
            # Set the title for each trace.
            ax.set_title('%s (%.2f Hz - %i samples)' % (trace.id,
                         trace.stats.sampling_rate, trace.stats.npts),
                         size='small')
            # Set the xaxis locators.
            ax.xaxis.set_major_locator(MaxNLocator(6))
            # Only the last trace should contain labels on the xaxis.
            if _i != 2:
                ax.xaxis.set_major_formatter(DateFormatter(''))
            else:
                ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            ax.xaxis.set_minor_locator(MinuteLocator())
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.grid()
        # Force the redraw of the figure canvas.
        fig.canvas.draw()

    def spectraMethodStackChanged(self, new_index):
        """
        Gets fired every time the user changes the value of the spectra method
        selection combo box.
        """
        self.spectraMethodStack.setCurrentIndex(new_index)

    def multitaperPadCheckStateChanged(self, state):
        """
        Enables/disables the nfft value settings based on the multitaper pad
        check box check state.
        """
        # Box is checked.
        if state == 2:
            self.nfftSpinBox.setEnabled(True)
            self.nfftLabel.setEnabled(True)
        # Box is uncheckd.
        else:
            self.nfftSpinBox.setEnabled(False)
            self.nfftLabel.setEnabled(False)

    def masterCurveComboBoxChanged(self, index):
        """
        Enables disables the box for the cutoff value.
        """
        # Get the sender of the signal.
        sender = self.qApp.sender()
        # Disable it for the median.
        if 'median' in str(sender.currentText()).lower():
            self.cutoffLabel1.setEnabled(False)
            self.cutoffLabel2.setEnabled(False)
            self.cutoffSpinBox.setEnabled(False)
        # Enable otherwise.
        else:
            self.cutoffLabel1.setEnabled(True)
            self.cutoffLabel2.setEnabled(True)
            self.cutoffSpinBox.setEnabled(True)


class SaveHVSRPlotDialog(SavePlotDialog):
    """
    Main Window with the design loaded from the Qt Designer.
    """
    def __init__(self, dialog, figure):
        """
        Standard init.
        """
        self.imageformats = ['png', 'pdf', 'ps', 'eps', 'svg']
        SavePlotDialog.__init__(SavePlotDialog)
        # This is how the init is called in the file created py pyuic4.
        self.setupUi(dialog)
        self.dialog = dialog
        # Set the parent of the dialog to be able to access the parents
        # attributes from the dialog.
        self.dialog.parent_widget = self
        # The figure to be saved. Create a copy of it to not alter the original
        # figure.
        self.fig = figure
        # Set the standard path for the file.
        home_dir = os.path.abspath(os.path.expanduser('~'))
        self.filenameLineEdit.setText(os.path.join(home_dir, 'HVSR.png'))
        # XXX: No idea why this workaround is necessary. But otherwise the
        # signals do not seem to work at all.
        self.dialog.getPlotFilename = self.getPlotFilename
        self.dialog.plot = self.plot
        # Connect signals and slots.
        self.__connectSignals()

    def getPlotFilename(self):
        """
        Opens a save file dialog and gets the filename.
        """
        # File dialog.
        caption = 'Save Plot to'
        # Define the possible image formats.
        png = "Portable Network Graphics (*.png)"
        pdf = "Portable Document Format (*.pdf)"
        ps = "PostScript (*.ps)"
        eps = "Encapsulated PostScript (*.eps)"
        svg = "Scalable Vector Graphics (*.svg)"
        filter = [png, pdf, ps, eps, svg]
        filter = ';;'.join(filter)
        # Get the filename.
        filename = QtGui.QFileDialog.getSaveFileName(caption=caption,
                   filter=filter,
                   directory=os.path.dirname(str(\
                                    self.filenameLineEdit.text())),
                   options=QtGui.QFileDialog.DontConfirmOverwrite)
        # If not filename is given, return.
        if not filename:
            return
        # Set the filename.
        self.filenameLineEdit.setText(filename)

    def plot(self):
        """
        Plots and closes the dialog.

        Also checks the filename for any error, not existing directories and
        already existing files.
        """
        filename = str(self.filenameLineEdit.text())
        # Check if the fileformat is correct. The first extension character
        # will always be the seperator. The extra spaces are used as an ugly
        # way to make the message box a little bit bigger.
        if os.path.splitext(filename)[1].lower()[1:] not in self.imageformats:
            msg = 20 * ' ' + 'Invalid image format.' + 20 * ' '
            info = 'Please choose a valid format.\nValid formats: %s' % \
                  ', '.join(self.imageformats)
            box = QtGui.QMessageBox()
            box.setText(msg)
            box.setInformativeText(info)
            box.exec_()
            return
        # Check if it is a valid path.
        path = os.path.dirname(filename)
        if not os.path.exists(path):
            msg = 20 * ' ' + 'Invalid path' + 20 * ' '
            info = 'The path %s does not exist.\n' + \
                   'Please choose a valid path.' % path
            box = QtGui.QMessageBox()
            box.setText(msg)
            box.setInformativeText(info)
            box.exec_()
            return
        # Check if the file already exists Offer option to override.
        if os.path.exists(filename):
            msg = 20 * ' ' + 'Overwrite file?' + 20 * ' '
            info = 'The file %s already exists.\nOverwrite it?' %\
                    filename
            box = QtGui.QMessageBox()
            box.setText(msg)
            box.setInformativeText(info)
            box.setStandardButtons(QtGui.QMessageBox.No | \
                                   QtGui.QMessageBox.Yes)
            ret_code = box.exec_()
            if ret_code == QtGui.QMessageBox.No:
                return
        # Save the old values and restore them after the plot has been saved.
        old_width = self.fig.get_figwidth()
        old_height = self.fig.get_figheight()
        old_dpi = self.fig.get_dpi()
        # Receive the new ones.
        width = self.widthSpinBox.value()
        height = self.heightSpinBox.value()
        dpi = self.dpiSpinBox.value()
        # Set the figure properties.
        self.fig.set_dpi(dpi)
        self.fig.set_figwidth(int(width / float(dpi)))
        self.fig.set_figheight(int(height / float(dpi)))
        # Save it.
        self.fig.savefig(filename)
        # Restore the old properties.
        self.fig.set_dpi(old_dpi)
        self.fig.set_figwidth(old_width)
        self.fig.set_figheight(old_height)
        # Close the dialog.
        self.dialog.accept()

    def __connectSignals(self):
        """
        Connects this class' signals to methods so everything works.
        """
        # Connect the save to button.
        QtCore.QObject.connect(self.cancelButton, QtCore.SIGNAL("clicked()"),
                               self.dialog.reject)
        QtCore.QObject.connect(self.okButton, QtCore.SIGNAL("clicked()"),
                               self.dialog.plot)
        QtCore.QObject.connect(self.fileDialogButton,
                               QtCore.SIGNAL('clicked()'),
                               self.dialog.getPlotFilename)

class BatchHVSRProcessingDialog(BatchProcessingDialog):
    """
    Main Window with the design loaded from the Qt Designer.
    """
    def __init__(self, dialog, parent):
        """
        Standard init.
        """
        BatchProcessingDialog.__init__(BatchProcessingDialog)
        # This is how the init is called in the file created py pyuic4.
        self.parent = parent
        self.setupUi(dialog)
        self.dialog = dialog
        # Settings dictionary.
        self.settings = {}
        # Set the parent of the dialog to be able to access the parents
        # attributes from the dialog.
        self.dialog.parent_widget = self
        # XXX: No idea why this workaround is necessary. But otherwise the
        # signals do not seem to work at all.
        self.dialog.getFilenames = self.getFilenames
        # Read the values from the GUI.
        self.readValuesFromGUI()
        # Connect signals and slots.
        self.__connectSignals()
        # Figure out the home directory.
        self.home_dir = os.path.abspath(os.path.expanduser('~'))
        # Set the default directory of the output directory.
        self.outputDirLineEdit.setText(self.home_dir)

        self.progress = BatchProgressDialog()
        self.progress.setModal(True)

        self.worker = batchProgress(self.progress)
        QtCore.QObject.connect(self.worker,
                               QtCore.SIGNAL('statusChanged(int, QString)'),
                               self.updateProgressDialog)
        QtCore.QObject.connect(self.worker,
                               QtCore.SIGNAL('labelChanged(QString)'),
                               self.changeProgressDialogLabel)
        QtCore.QObject.connect(self.worker,
                               QtCore.SIGNAL('finished()'),
                               self.closeProgressDialog)
        QtCore.QObject.connect(self.worker,
                               QtCore.SIGNAL('terminated(int)'),
                               self.closeProgressDialog)
        QtCore.QObject.connect(self.worker,
                               QtCore.SIGNAL('canceled()'),
                               self.cancelProgressDialog)
        QtCore.QObject.connect(self.worker,
                       QtCore.SIGNAL('minorStatusChanged(QString, int, int)'),
                       self.changeMinorStatus)
        QtCore.QObject.connect(self.worker,
                       QtCore.SIGNAL('cancelButtonPressed()'),
                       self.terminateThread)

    def startBatchProcessing(self):
        """
        Starts the actual batch processing.
        """
        # Disable the start button.
        self.startButton.setEnabled(False)

        # Read the filenames and the corresponding groups.
        tree = self.filenameTreeView
        groups = []
        for _i in xrange(tree.topLevelItemCount()):
            # Get the group and create a dictionary.
            group = tree.topLevelItem(_i)
            name = str(group.text(0))
            cur_group = {name :[]}
            # Loop over each subitem.
            for _j in xrange(group.childCount()):
                child = group.child(_j)
                cur_group[name].append(str(child.text(0)))
            # Only append to the big list if it does contain children.
            if cur_group[name]:
                groups.append(cur_group)
        # groups is now a list with dictionaries that have exactly one item.
        # The key is the name of the group and the value is a list of filenames
        # that belong to this group.
        output_dir = str(self.outputDirLineEdit.text())
        # Get the split status.
        if self.splitCheckBox.checkState() == 2:
            hour_split = self.splitSpinBox.value()
        else:
            hour_split = None
        self.worker.begin(groups, self.settings, output_dir, hour_split)

    def terminateThread(self):
        """
        Terminates the worker thread.
        """
        self.worker.terminate()

    def changeMinorStatus(self, msg, frac, count):
        """
        Changes the minor status.
        """
        self.progress.setMinorLabel(msg)
        self.progress.setMinorFracValue(frac, count)

    def cancelProgressDialog(self):
        """
        Just set a cancel variable.
        """
        self.progress.cancel_status = True

    def updateProgressDialog(self, step, label):
        """
        Changes the status of the Progress Dialog.
        """
        self.progress.setMajorLabel(label)
        self.progress.setValue(step)
        self.progress.show()

    def changeProgressDialogLabel(self, label):
        """
        Changes the label of the Progress Dialog.
        """
        self.progress.setStatusLabel(label)
        self.progress.show()

    def closeProgressDialog(self):
        """
        Called when the worker thread closes or terminates. Will close the
        Progress Dialog and reenable the start button.
        """
        self.startButton.setEnabled(True)
        self.progress.setValue(self.progress.maximum())
        self.progress.reject()

    def updateFileText(self):
        """
        Updates the file text.
        """
        size = 0
        for file in self.filenameTreeView.plain_list:
            size += os.path.getsize(file)
        self.filesLabel.setText('%i files [%.2f MB]' \
            % (len(self.filenameTreeView.plain_list), 
               (size/1000000.0)))

    def chooseOutputDirectory(self):
        """
        Get the output file directory.
        """
        filename = QtGui.QFileDialog.getExistingDirectory(\
            caption='Choose output directory',
            directory=self.outputDirLineEdit.text(),
            options = QtGui.QFileDialog.ShowDirsOnly)
        if not filename:
            return
        self.outputDirLineEdit.setText(str(filename))

    def readValuesFromGUI(self):
        """
        Reads the settings for the HVSR from the GUI.
        """
        parent = self.parent
        # Edit Trace settings.
        self.lowpass_value = float(parent.lowpassSpinBox.value())
        self.settings['lowpass_value'] = self.lowpass_value
        self.highpass_value = float(parent.highpassSpinBox.value())
        self.settings['highpass_value'] = self.highpass_value
        self.new_sample_rate = float(parent.resampleSpinBox.value())
        self.settings['new_sample_rate'] = self.new_sample_rate
        self.corners = int(parent.cornersSpinBox.value())
        self.settings['corners'] = self.corners
        if parent.zerophaseCheckbox.checkState() == 2:
            self.zerophase = True
        else:
            self.zerophase = False
        self.settings['zerophase'] = self.zerophase
        # Do the time calculations.
        starttime = fromQDateTime(parent.starttimeEdit.dateTime())
        endtime = fromQDateTime(parent.endtimeEdit.dateTime())
        # If there is an original file do some checks and convert to offsets or
        # set to 0.
        if hasattr(parent, 'original_stream'):
            file_starttime = parent.original_stream[0].stats.starttime
            file_endtime = parent.original_stream[0].stats.endtime
            if starttime >= file_starttime and startttime <= file_endtime:
                self.new_starttime = int(starttime - file_starttime)
            else:
                self.new_starttime = 0
            if endtime >= file_starttime and endttime <= file_endtime:
                self.new_endtime = int(file_endtime - endtime)
            else:
                self.new_endtime = 0
        else:
            self.new_starttime = 0
            self.new_endtime = 0
        self.settings['starttime'] = self.new_starttime
        self.settings['endtime'] = self.new_endtime
        # HVSR settings.
        self.window_length = float(parent.windowLengthSpinBox.value())
        self.settings['window_length'] = self.window_length
        self.spectra_method = str(parent.spectraMethodComboBox.currentText()).lower()
        self.settings['spectra_method'] = self.spectra_method
        if self.spectra_method == 'multitaper':
            # Get the variables for the multitaper spectra calculation from the
            # GUI.
            time_bandwidth = parent.timeBandwidthSpinBox.value()
            number_of_tapers = \
                    str(parent.numberOfTapersComboBox.currentText()).lower()
            if number_of_tapers == 'auto':
                number_of_tapers = None
            else:
                number_of_tapers = int(number_of_tapers)
            quadratic = parent.quadraticCheckBox.checkState()
            if quadratic == 2:
                quadratic = True
            else:
                quadratic = False
            adaptive = parent.adaptiveCheckBox.checkState()
            if adaptive == 2:
                adaptive = True
            else:
                adaptive = False
            if parent.multitaperPadCheckBox.checkState() == 2:
                nfft = parent.nfftSpinBox.value()
                if nfft <= parent.window_length:
                    nfft = None
            else:
                nfft = None
            self.spec_options = {'time_bandwidth': time_bandwidth,
              'number_of_tapers': number_of_tapers, 'quadratic': quadratic,
              'adaptive': adaptive, 'nfft': nfft}
        elif self.spectra_method == 'sine multitaper':
            # Read the parameters from the GUI.
            number_of_tapers = parent.sinePSDTapersSpinBox.value()
            if number_of_tapers == 0:
                number_of_tapers = 'Auto'
            number_of_iterations = parent.sinePSDIterationsSpinBox.value()
            degree_of_smoothing = parent.sinePSDSmoothingSpinBox.value()
            self.spec_options = {'number_of_tapers': number_of_tapers,
                       'degree_of_smoothing': degree_of_smoothing,
                       'number_of_iterations': number_of_iterations}
        elif self.spectra_method == 'single taper':
            # Read the taper from the GUI.
            taper = str(parent.singleTaperComboBox.currentText()).lower()
            self.spec_options = {'taper': taper}
        self.settings['spectra_options'] = self.spec_options
        # Calculate the master method.
        master_method = str(parent.masterCurveComboBox.currentText()).lower()
        # Geometric average.
        if 'geometric' in master_method:
            self.master_curve_method = 'geometric average'
        # Mean.
        elif 'mean' in master_method:
            self.master_curve_method = 'mean'
        # Median.
        elif 'median' in master_method:
            self.master_curve_method = 'median'
        self.settings['master_curve_method'] = self.master_curve_method
        # Discard the top and bottom percentile.
        self.cutoff_value = parent.cutoffSpinBox.value() / 100.0
        self.settings['cutoff_value'] = self.cutoff_value
        # Z-Detector settings.
        self.z_detector_window_length = int(parent.zDetectorSpinBox.value())
        self.settings['zdetector_window_length'] = \
                self.z_detector_window_length
        self.threshold = float(parent.thresholdSpinBox.value())
        self.settings['threshold'] = self.threshold/100.0
        # Fill the TreeView.
        self.fillSettingsTree()

    def fillSettingsTree(self):
        """
        Fills the settings Tree with the previously read values.
        """
        tree = self.settingsTreeWidget
        # Add the general settings.
        item = QtGui.QTreeWidgetItem(['Edit Traces', ''])
        item.addChild(QtGui.QTreeWidgetItem(['Starttime Offset',
                                '%i s' % self.new_starttime]))
        item.addChild(QtGui.QTreeWidgetItem(['Endtime Offset',
                                '%i s' % self.new_endtime]))
        item.addChild(QtGui.QTreeWidgetItem(['Highpass Frequency',
                                '%.3f Hz' % self.highpass_value]))
        item.addChild(QtGui.QTreeWidgetItem(['Lowpass Frequency',
                                '%.3f Hz' % self.lowpass_value]))
        item.addChild(QtGui.QTreeWidgetItem(['Filter Corners',
                                '%i' % self.corners]))
        item.addChild(QtGui.QTreeWidgetItem(['Use Zerophase Filter',
                                str(self.zerophase)]))
        item.addChild(QtGui.QTreeWidgetItem(['Resample to',
                                '%.3f Hz' % self.new_sample_rate]))
        tree.addTopLevelItem(item)
        tree.expandItem(item)
        # Add the Z-detector settings.
        item = QtGui.QTreeWidgetItem(['z-Detector Settings', ''])
        item.addChild(QtGui.QTreeWidgetItem(['Window Length',
                            '%i samples' % self.z_detector_window_length]))
        item.addChild(QtGui.QTreeWidgetItem(['Threshold as Percentile',
                            '%.3f %%' % self.threshold]))
        tree.addTopLevelItem(item)
        tree.expandItem(item)
        # Add the HVSR values.
        item = QtGui.QTreeWidgetItem(['HVSR Settings', ''])
        item.addChild(QtGui.QTreeWidgetItem(['Spectrum Calculation Method',
                '%s' % ' '.join([_i.lower().capitalize() for _i in \
                                self.spectra_method.split()])]))
        keys = self.spec_options.keys()
        keys.sort()
        for key in keys:
            name = key.split('_')
            # Capitalize and remove underscores.
            split = key.split('_')
            names = []
            for _i in split:
                if _i.lower() == 'of':
                    names.append(_i.lower())
                    continue
                names.append(_i.lower().capitalize())
            name = ' '.join(names)
            value = self.spec_options[key]
            if type(value) == int:
                value = '%i' % value
            if type(value) == float:
                value = '%.3f' % value
            if type(value) == str:
                value = '%s' % value.lower().capitalize()
            else:
                value = str(value)
            item.addChild(QtGui.QTreeWidgetItem([name, value]))
        item.addChild(QtGui.QTreeWidgetItem(['Master Curve',
            ' '.join([_i.lower().capitalize() for _i in\
                      self.master_curve_method.split()])]))
        item.addChild(QtGui.QTreeWidgetItem(['Discard Bottom/Top',
            '%.1f %%' % (self.cutoff_value * 100)]))
        tree.addTopLevelItem(item)
        tree.expandItem(item)
        tree.resizeColumnToContents(0)

    def resizeView(self, *args, **kwargs):
        """
        Resize the first column.
        """
        self.filenameTreeView.resizeColumnToContents(0)

    def removeItems(self):
        """
        Removes all currently selected items.
        """
        items = list(self.filenameTreeView.selectedItems())
        self.filenameTreeView.removeItems(items)
        self.updateFileText()

    def getFilenames(self):
        """
        Opens a save file dialog and gets the filenames and send them to the
        filename model.
        """
        # File dialog.
        caption = 'Select Files'
        # Get the filename.
        filenames = list(QtGui.QFileDialog.getOpenFileNames(caption=caption))
        # If not filename is given, return.
        if not filenames:
            return
        # Convert to standart Python strings.
        filenames = [str(file) for file in filenames]
        self.filenameTreeView.addFilenames(filenames)
        self.updateFileText()

    def splitToggled(self, check_state):
        """
        Fired when the split check box is toggled.
        """
        if check_state == 2:
            self.splitLabel1.setEnabled(True)
            self.splitLabel2.setEnabled(True)
            self.splitSpinBox.setEnabled(True)
            return
        self.splitLabel1.setEnabled(False)
        self.splitLabel2.setEnabled(False)
        self.splitSpinBox.setEnabled(False)

    def __connectSignals(self):
        """
        Connects signals and slots.
        """
        QtCore.QObject.connect(self.fileDialogButton,
                               QtCore.SIGNAL('clicked()'),
                               self.dialog.getFilenames)
        QtCore.QObject.connect(self.automatchButton,
                               QtCore.SIGNAL('clicked()'),
                               self.filenameTreeView.autoMatchFiles)
        QtCore.QObject.connect(self.filenameTreeView,
                               QtCore.SIGNAL('expanded(QModelIndex)'),
                               self.resizeView)
        QtCore.QObject.connect(self.filenameTreeView,
                               QtCore.SIGNAL('collapsed(QModelIndex)'),
                               self.resizeView)
        QtCore.QObject.connect(self.removeItemsButton,
                               QtCore.SIGNAL('clicked()'),
                               self.removeItems)
        QtCore.QObject.connect(self.cancelButton,
                               QtCore.SIGNAL('clicked()'),
                               self.dialog.reject)
        QtCore.QObject.connect(self.outputDirButton,
                               QtCore.SIGNAL('clicked()'),
                               self.chooseOutputDirectory)
        QtCore.QObject.connect(self.startButton,
                               QtCore.SIGNAL('clicked()'),
                               self.startBatchProcessing)
        QtCore.QObject.connect(self.splitCheckBox,
                               QtCore.SIGNAL('stateChanged(int)'),
                               self.splitToggled)


if __name__ == "__main__":
    # Create the GUI application
    qApp = QtGui.QApplication(sys.argv)
    main_window = QtGui.QMainWindow()
    # Create and show the main window.
    w = HtoV(main_window)
    # Ensure access to qApp inside the class. It is only available after the
    # __init__ method has been called.
    w.qApp = qApp
    #qApp.setMainWidget(w)
    main_window.showMaximized()
    # start the Qt main loop execution, exiting from this script with
    # the same return code of Qt application
    sys.exit(qApp.exec_())
