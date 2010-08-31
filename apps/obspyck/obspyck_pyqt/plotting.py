import matplotlib
from matplotlib.widgets import MultiCursor as mplMultiCursor
from matplotlib.ticker import FuncFormatter
import numpy as np

from utils import formatXTicklabels

#Monkey patch (need to remember the ids of the mpl_connect-statements to remove them later)
#See source: http://matplotlib.sourcearchive.com/documentation/0.98.1/widgets_8py-source.html
class MultiCursor(mplMultiCursor):
    def __init__(self, canvas, axes, useblit=True, **lineprops):
        self.canvas = canvas
        self.axes = axes
        xmin, xmax = axes[-1].get_xlim()
        xmid = 0.5*(xmin+xmax)
        self.lines = [ax.axvline(xmid, visible=False, **lineprops) for ax in axes]
        self.visible = True
        self.useblit = useblit
        self.background = None
        self.needclear = False
        self.id1=self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.id2=self.canvas.mpl_connect('draw_event', self.clear)

class Plotting(object):
    def __init__(self, fig, picker):
        """
        Canvas is the matplotlib canvas and picker the picker object which
        serves as a glue.
        """
        self.fig = fig
        self.picker = picker

    def drawAxes(self):
        """
        Set the axes of the matplotlib figure.
        """
        st = self.picker.streams[self.picker.stPt]
        #we start all our x-axes at 0 with the starttime of the first (Z) trace
        starttime_global = st[0].stats.starttime
        self.axs = []
        self.plts = []
        self.trans = []
        self.t = []
        trNum = len(st.traces)
        for i in range(trNum):
            npts = st[i].stats.npts
            smprt = st[i].stats.sampling_rate
            #make sure that the relative times of the x-axes get mapped to our
            #global stream (absolute) starttime (starttime of first (Z) trace)
            starttime_local = st[i].stats.starttime - starttime_global
            dt = 1. / smprt
            sampletimes = np.arange(starttime_local,
                    starttime_local + (dt * npts), dt)
            # sometimes our arange is one item too long (why??), so we just cut
            # off the last item if this is the case
            if len(sampletimes) == npts + 1:
                sampletimes = sampletimes[:-1]
            self.t.append(sampletimes)
            if i == 0:
                self.axs.append(self.fig.add_subplot(trNum,1,i+1))
                self.trans.append(matplotlib.transforms.blended_transform_factory(self.axs[i].transData,
                                                                             self.axs[i].transAxes))
            else:
                self.axs.append(self.fig.add_subplot(trNum, 1, i+1, 
                        sharex=self.axs[0], sharey=self.axs[0]))
                self.trans.append(matplotlib.transforms.blended_transform_factory(self.axs[i].transData,
                                                                             self.axs[i].transAxes))
                self.axs[i].xaxis.set_ticks_position("top")
            self.axs[-1].xaxis.set_ticks_position("both")
            self.axs[i].xaxis.set_major_formatter(FuncFormatter(formatXTicklabels))
            # XXX: Reenable!
            #if self.togglebuttonSpectrogram.get_active():
            if False:
                log = self.checkbuttonSpectrogramLog.get_active()
                spectrogram(st[i].data, st[i].stats.sampling_rate, log=log,
                            cmap=self.spectrogramColormap, axis=self.axs[i],
                            zorder=-10)
            else:
                self.plts.append(self.axs[i].plot(self.t[i], st[i].data, color='k',zorder=1000)[0])
        self.supTit = self.fig.suptitle("%s.%03d -- %s.%03d" % (st[0].stats.starttime.strftime("%Y-%m-%d  %H:%M:%S"),
                                                         st[0].stats.starttime.microsecond / 1e3 + 0.5,
                                                         st[0].stats.endtime.strftime("%H:%M:%S"),
                                                         st[0].stats.endtime.microsecond / 1e3 + 0.5), ha="left", va="bottom", x=0.01, y=0.01)
        self.xMin, self.xMax=self.axs[0].get_xlim()
        self.yMin, self.yMax=self.axs[0].get_ylim()
        #self.fig.subplots_adjust(bottom=0.04, hspace=0.01, right=0.999, top=0.94, left=0.06)
        self.fig.subplots_adjust(bottom=0.001, hspace=0.000, right=0.999, top=0.999, left=0.001)
        # XXX: Reenable next three lines!
        #self.toolbar.update()
        #self.toolbar.pan(False)
        #self.toolbar.zoom(True)
    
    def drawSavedPicks(self):
        """
        Draw the saved picks to the matplotlib widget.
        """
        self.drawPLine()
        self.drawPLabel()
        self.drawPErr1Line()
        self.drawPErr2Line()
        self.drawPsynthLine()
        self.drawPsynthLabel()
        self.drawSLine()
        self.drawSLabel()
        self.drawSErr1Line()
        self.drawSErr2Line()
        self.drawSsynthLine()
        self.drawSsynthLabel()
        self.drawMagMinCross1()
        self.drawMagMaxCross1()
        self.drawMagMinCross2()
        self.drawMagMaxCross2()
    
    def drawPLine(self):
        if hasattr(self, "PLines"):
            self.delPLine()
        dict = self.dicts[self.stPt]
        if not 'P' in dict:
            return
        self.PLines = []
        for ax in self.axs:
            line = ax.axvline(dict['P'], color=self.dictPhaseColors['P'],
                              linewidth=self.axvlinewidths,
                              linestyle=self.dictPhaseLinestyles['P'])
            self.PLines.append(line)
    
    def delPLine(self):
        if not hasattr(self, "PLines"):
            return
        for i, ax in enumerate(self.axs):
            if self.PLines[i] in ax.lines:
                ax.lines.remove(self.PLines[i])
        del self.PLines
    
    def drawPsynthLine(self):
        if hasattr(self, "PsynthLines"):
            self.delPsynthLine()
        dict = self.dicts[self.stPt]
        if not 'Psynth' in dict:
            return
        self.PsynthLines = []
        for ax in self.axs:
            line = ax.axvline(dict['Psynth'], linewidth=self.axvlinewidths,
                              color=self.dictPhaseColors['Psynth'],
                              linestyle=self.dictPhaseLinestyles['Psynth'])
            self.PsynthLines.append(line)
    
    def delPsynthLine(self):
        if not hasattr(self, "PsynthLines"):
            return
        for i, ax in enumerate(self.axs):
            if self.PsynthLines[i] in ax.lines:
                ax.lines.remove(self.PsynthLines[i])
        del self.PsynthLines
    
    def drawPLabel(self):
        dict = self.dicts[self.stPt]
        if not 'P' in dict:
            return
        label = 'P:'
        if 'POnset' in dict:
            if dict['POnset'] == 'impulsive':
                label += 'I'
            elif dict['POnset'] == 'emergent':
                label += 'E'
            else:
                label += '?'
        else:
            label += '_'
        if 'PPol' in dict:
            if dict['PPol'] == 'up':
                label += 'U'
            elif dict['PPol'] == 'poorup':
                label += '+'
            elif dict['PPol'] == 'down':
                label += 'D'
            elif dict['PPol'] == 'poordown':
                label += '-'
            else:
                label += '?'
        else:
            label += '_'
        if 'PWeight' in dict:
            label += str(dict['PWeight'])
        else:
            label += '_'
        self.PLabel = self.axs[0].text(dict['P'], 1 - 0.01 * len(self.axs),
                                       '  ' + label, transform = self.trans[0],
                                       color = self.dictPhaseColors['P'],
                                       family = 'monospace', va="top")
    
    def delPLabel(self):
        if not hasattr(self, "PLabel"):
            return
        if self.PLabel in self.axs[0].texts:
            self.axs[0].texts.remove(self.PLabel)
        del self.PLabel
    
    def drawPsynthLabel(self):
        dict = self.dicts[self.stPt]
        if not 'Psynth' in dict:
            return
        label = 'Psynth: %+.3fs' % dict['Pres']
        self.PsynthLabel = self.axs[0].text(dict['Psynth'],
                1 - 0.03 * len(self.axs), '  ' + label, va="top",
                transform=self.trans[0], color=self.dictPhaseColors['Psynth'])
    
    def delPsynthLabel(self):
        if not hasattr(self, "PsynthLabel"):
            return
        if self.PsynthLabel in self.axs[0].texts:
            self.axs[0].texts.remove(self.PsynthLabel)
        del self.PsynthLabel
    
    def drawPErr1Line(self):
        if hasattr(self, "PErr1Lines"):
            self.delPErr1Line()
        dict = self.dicts[self.stPt]
        if not 'P' in dict or not 'PErr1' in dict:
            return
        self.PErr1Lines = []
        for ax in self.axs:
            line = ax.axvline(dict['PErr1'], ymin=0.25, ymax=0.75,
                              color=self.dictPhaseColors['P'],
                              linewidth=self.axvlinewidths)
            self.PErr1Lines.append(line)
    
    def delPErr1Line(self):
        if not hasattr(self, "PErr1Lines"):
            return
        for i, ax in enumerate(self.axs):
            if self.PErr1Lines[i] in ax.lines:
                ax.lines.remove(self.PErr1Lines[i])
        del self.PErr1Lines
    
    def drawPErr2Line(self):
        if hasattr(self, "PErr2Lines"):
            self.delPErr2Line()
        dict = self.dicts[self.stPt]
        if not 'P' in dict or not 'PErr2' in dict:
            return
        self.PErr2Lines = []
        for ax in self.axs:
            line = ax.axvline(dict['PErr2'], ymin=0.25, ymax=0.75,
                              color=self.dictPhaseColors['P'],
                              linewidth=self.axvlinewidths)
            self.PErr2Lines.append(line)
    
    def delPErr2Line(self):
        if not hasattr(self, "PErr2Lines"):
            return
        for i, ax in enumerate(self.axs):
            if self.PErr2Lines[i] in ax.lines:
                ax.lines.remove(self.PErr2Lines[i])
        del self.PErr2Lines

    def drawSLine(self):
        if hasattr(self, "SLines"):
            self.delSLine()
        dict = self.dicts[self.stPt]
        if not 'S' in dict:
            return
        self.SLines = []
        for ax in self.axs:
            line = ax.axvline(dict['S'], color=self.dictPhaseColors['S'],
                              linewidth=self.axvlinewidths,
                              linestyle=self.dictPhaseLinestyles['S'])
            self.SLines.append(line)
    
    def delSLine(self):
        if not hasattr(self, "SLines"):
            return
        for i, ax in enumerate(self.axs):
            if self.SLines[i] in ax.lines:
                ax.lines.remove(self.SLines[i])
        del self.SLines
    
    def drawSsynthLine(self):
        if hasattr(self, "SsynthLines"):
            self.delSsynthLine()
        dict = self.dicts[self.stPt]
        if not 'Ssynth' in dict:
            return
        self.SsynthLines = []
        for ax in self.axs:
            line = ax.axvline(dict['Ssynth'], linewidth=self.axvlinewidths,
                              color=self.dictPhaseColors['Ssynth'],
                              linestyle=self.dictPhaseLinestyles['Ssynth'])
            self.SsynthLines.append(line)
    
    def delSsynthLine(self):
        if not hasattr(self, "SsynthLines"):
            return
        for i, ax in enumerate(self.axs):
            if self.SsynthLines[i] in ax.lines:
                ax.lines.remove(self.SsynthLines[i])
        del self.SsynthLines
    
    def drawSLabel(self):
        dict = self.dicts[self.stPt]
        if not 'S' in dict:
            return
        label = 'S:'
        if 'SOnset' in dict:
            if dict['SOnset'] == 'impulsive':
                label += 'I'
            elif dict['SOnset'] == 'emergent':
                label += 'E'
            else:
                label += '?'
        else:
            label += '_'
        if 'SPol' in dict:
            if dict['SPol'] == 'up':
                label += 'U'
            elif dict['SPol'] == 'poorup':
                label += '+'
            elif dict['SPol'] == 'down':
                label += 'D'
            elif dict['SPol'] == 'poordown':
                label += '-'
            else:
                label += '?'
        else:
            label += '_'
        if 'SWeight' in dict:
            label += str(dict['SWeight'])
        else:
            label += '_'
        self.SLabel = self.axs[0].text(dict['S'], 1 - 0.01 * len(self.axs),
                                       '  ' + label, transform = self.trans[0],
                                       color = self.dictPhaseColors['S'],
                                       family = 'monospace', va="top")
    
    def delSLabel(self):
        if not hasattr(self, "SLabel"):
            return
        if self.SLabel in self.axs[0].texts:
            self.axs[0].texts.remove(self.SLabel)
        del self.SLabel
    
    def drawSsynthLabel(self):
        dict = self.dicts[self.stPt]
        if not 'Ssynth' in dict:
            return
        label = 'Ssynth: %+.3fs' % dict['Sres']
        self.SsynthLabel = self.axs[0].text(dict['Ssynth'],
                1 - 0.03 * len(self.axs), '  ' + label, va="top",
                transform=self.trans[0], color=self.dictPhaseColors['Ssynth'])
    
    def delSsynthLabel(self):
        if not hasattr(self, "SsynthLabel"):
            return
        if self.SsynthLabel in self.axs[0].texts:
            self.axs[0].texts.remove(self.SsynthLabel)
        del self.SsynthLabel
    
    def drawSErr1Line(self):
        if hasattr(self, "SErr1Lines"):
            self.delSErr1Line()
        dict = self.dicts[self.stPt]
        if not 'S' in dict or not 'SErr1' in dict:
            return
        self.SErr1Lines = []
        for ax in self.axs:
            line = ax.axvline(dict['SErr1'], ymin=0.25, ymax=0.75,
                              color=self.dictPhaseColors['S'],
                              linewidth=self.axvlinewidths)
            self.SErr1Lines.append(line)
    
    def delSErr1Line(self):
        if not hasattr(self, "SErr1Lines"):
            return
        for i, ax in enumerate(self.axs):
            if self.SErr1Lines[i] in ax.lines:
                ax.lines.remove(self.SErr1Lines[i])
        del self.SErr1Lines
    
    def drawSErr2Line(self):
        if hasattr(self, "SErr2Lines"):
            self.delSErr2Line()
        dict = self.dicts[self.stPt]
        if not 'S' in dict or not 'SErr2' in dict:
            return
        self.SErr2Lines = []
        for ax in self.axs:
            line = ax.axvline(dict['SErr2'], ymin=0.25, ymax=0.75,
                              color=self.dictPhaseColors['S'],
                              linewidth=self.axvlinewidths)
            self.SErr2Lines.append(line)
    
    def delSErr2Line(self):
        if not hasattr(self, "SErr2Lines"):
            return
        for i, ax in enumerate(self.axs):
            if self.SErr2Lines[i] in ax.lines:
                ax.lines.remove(self.SErr2Lines[i])
        del self.SErr2Lines

    def drawMagMinCross1(self):
        dict = self.dicts[self.stPt]
        if not 'MagMin1' in dict or len(self.axs) < 2:
            return
        # we have to force the graph to the old axes limits because of the
        # completely new line object creation
        xlims = list(self.axs[0].get_xlim())
        ylims = list(self.axs[0].get_ylim())
        self.MagMinCross1 = self.axs[1].plot([dict['MagMin1T']],
                [dict['MagMin1']], markersize=self.magMarkerSize,
                markeredgewidth=self.magMarkerEdgeWidth,
                color=self.dictPhaseColors['Mag'], marker=self.magMinMarker,
                zorder=2000)[0]
        self.axs[0].set_xlim(xlims)
        self.axs[0].set_ylim(ylims)
    
    def delMagMinCross1(self):
        if not hasattr(self, "MagMinCross1"):
            return
        ax = self.axs[1]
        if self.MagMinCross1 in ax.lines:
            ax.lines.remove(self.MagMinCross1)
    
    def drawMagMaxCross1(self):
        dict = self.dicts[self.stPt]
        if not 'MagMax1' in dict or len(self.axs) < 2:
            return
        # we have to force the graph to the old axes limits because of the
        # completely new line object creation
        xlims = list(self.axs[0].get_xlim())
        ylims = list(self.axs[0].get_ylim())
        self.MagMaxCross1 = self.axs[1].plot([dict['MagMax1T']],
                [dict['MagMax1']], markersize=self.magMarkerSize,
                markeredgewidth=self.magMarkerEdgeWidth,
                color=self.dictPhaseColors['Mag'], marker=self.magMinMarker,
                zorder=2000)[0]
        self.axs[0].set_xlim(xlims)
        self.axs[0].set_ylim(ylims)
    
    def delMagMaxCross1(self):
        if not hasattr(self, "MagMaxCross1"):
            return
        ax = self.axs[1]
        if self.MagMaxCross1 in ax.lines:
            ax.lines.remove(self.MagMaxCross1)
    
    def drawMagMinCross2(self):
        dict = self.dicts[self.stPt]
        if not 'MagMin2' in dict or len(self.axs) < 3:
            return
        # we have to force the graph to the old axes limits because of the
        # completely new line object creation
        xlims = list(self.axs[0].get_xlim())
        ylims = list(self.axs[0].get_ylim())
        self.MagMinCross2 = self.axs[2].plot([dict['MagMin2T']],
                [dict['MagMin2']], markersize=self.magMarkerSize,
                markeredgewidth=self.magMarkerEdgeWidth,
                color=self.dictPhaseColors['Mag'], marker=self.magMinMarker,
                zorder=2000)[0]
        self.axs[0].set_xlim(xlims)
        self.axs[0].set_ylim(ylims)
    
    def delMagMinCross2(self):
        if not hasattr(self, "MagMinCross2"):
            return
        ax = self.axs[2]
        if self.MagMinCross2 in ax.lines:
            ax.lines.remove(self.MagMinCross2)
    
    def drawMagMaxCross2(self):
        dict = self.dicts[self.stPt]
        if not 'MagMax2' in dict or len(self.axs) < 3:
            return
        # we have to force the graph to the old axes limits because of the
        # completely new line object creation
        xlims = list(self.axs[0].get_xlim())
        ylims = list(self.axs[0].get_ylim())
        self.MagMaxCross2 = self.axs[2].plot([dict['MagMax2T']],
                [dict['MagMax2']], markersize=self.magMarkerSize,
                markeredgewidth=self.magMarkerEdgeWidth,
                color=self.dictPhaseColors['Mag'], marker=self.magMinMarker,
                zorder=2000)[0]
        self.axs[0].set_xlim(xlims)
        self.axs[0].set_ylim(ylims)
    
    def delMagMaxCross2(self):
        if not hasattr(self, "MagMaxCross2"):
            return
        ax = self.axs[2]
        if self.MagMaxCross2 in ax.lines:
            ax.lines.remove(self.MagMaxCross2)
    
    def delP(self):
        dict = self.dicts[self.stPt]
        if not 'P' in dict:
            return
        del dict['P']
        msg = "P Pick deleted"
        self.textviewStdOutImproved.write(msg)
            
    def delPsynth(self):
        dict = self.dicts[self.stPt]
        if not 'Psynth' in dict:
            return
        del dict['Psynth']
        msg = "synthetic P Pick deleted"
        self.textviewStdOutImproved.write(msg)
            
    def delPWeight(self):
        dict = self.dicts[self.stPt]
        if not 'PWeight' in dict:
            return
        del dict['PWeight']
        msg = "P Pick weight deleted"
        self.textviewStdOutImproved.write(msg)
            
    def delPPol(self):
        dict = self.dicts[self.stPt]
        if not 'PPol' in dict:
            return
        del dict['PPol']
        msg = "P Pick polarity deleted"
        self.textviewStdOutImproved.write(msg)
            
    def delPOnset(self):
        dict = self.dicts[self.stPt]
        if not 'POnset' in dict:
            return
        del dict['POnset']
        msg = "P Pick onset deleted"
        self.textviewStdOutImproved.write(msg)
            
    def delPErr1(self):
        dict = self.dicts[self.stPt]
        if not 'PErr1' in dict:
            return
        del dict['PErr1']
        msg = "PErr1 Pick deleted"
        self.textviewStdOutImproved.write(msg)
            
    def delPErr2(self):
        dict = self.dicts[self.stPt]
        if not 'PErr2' in dict:
            return
        del dict['PErr2']
        msg = "PErr2 Pick deleted"
        self.textviewStdOutImproved.write(msg)
            
    def delS(self):
        dict = self.dicts[self.stPt]
        if not 'S' in dict:
            return
        del dict['S']
        if 'Saxind' in dict:
            del dict['Saxind']
        msg = "S Pick deleted"
        self.textviewStdOutImproved.write(msg)
            
    def delSsynth(self):
        dict = self.dicts[self.stPt]
        if not 'Ssynth' in dict:
            return
        del dict['Ssynth']
        msg = "synthetic S Pick deleted"
        self.textviewStdOutImproved.write(msg)
            
    def delSWeight(self):
        dict = self.dicts[self.stPt]
        if not 'SWeight' in dict:
            return
        del dict['SWeight']
        msg = "S Pick weight deleted"
        self.textviewStdOutImproved.write(msg)
            
    def delSPol(self):
        dict = self.dicts[self.stPt]
        if not 'SPol' in dict:
            return
        del dict['SPol']
        msg = "S Pick polarity deleted"
        self.textviewStdOutImproved.write(msg)
            
    def delSOnset(self):
        dict = self.dicts[self.stPt]
        if not 'SOnset' in dict:
            return
        del dict['SOnset']
        msg = "S Pick onset deleted"
        self.textviewStdOutImproved.write(msg)
            
    def delSErr1(self):
        dict = self.dicts[self.stPt]
        if not 'SErr1' in dict:
            return
        del dict['SErr1']
        msg = "SErr1 Pick deleted"
        self.textviewStdOutImproved.write(msg)
            
    def delSErr2(self):
        dict = self.dicts[self.stPt]
        if not 'SErr2' in dict:
            return
        del dict['SErr2']
        msg = "SErr2 Pick deleted"
        self.textviewStdOutImproved.write(msg)
            
    def delMagMin1(self):
        dict = self.dicts[self.stPt]
        if not 'MagMin1' in dict:
            return
        del dict['MagMin1']
        del dict['MagMin1T']
        msg = "Magnitude Minimum Estimation Pick deleted"
        self.textviewStdOutImproved.write(msg)
            
    def delMagMax1(self):
        dict = self.dicts[self.stPt]
        if not 'MagMax1' in dict:
            return
        del dict['MagMax1']
        del dict['MagMax1T']
        msg = "Magnitude Maximum Estimation Pick deleted"
        self.textviewStdOutImproved.write(msg)
            
    def delMagMin2(self):
        dict = self.dicts[self.stPt]
        if not 'MagMin2' in dict:
            return
        del dict['MagMin2']
        del dict['MagMin2T']
        msg = "Magnitude Minimum Estimation Pick deleted"
        self.textviewStdOutImproved.write(msg)
            
    def delMagMax2(self):
        dict = self.dicts[self.stPt]
        if not 'MagMax2' in dict:
            return
        del dict['MagMax2']
        del dict['MagMax2T']
        msg = "Magnitude Maximum Estimation Pick deleted"
        self.textviewStdOutImproved.write(msg)
    
    def delAxes(self):
        for ax in self.axs:
            if ax in self.fig.axes: 
                self.fig.delaxes(ax)
            del ax
        if self.supTit in self.fig.texts:
            self.fig.texts.remove(self.supTit)
    
    def redraw(self):
        for line in self.multicursor.lines:
            line.set_visible(False)
        self.canv.draw()
    
    def updatePlot(self):
        filt = []
        #filter data
        if self.togglebuttonFilter.get_active():
            zerophase = self.checkbuttonZeroPhase.get_active()
            freq_highpass = self.spinbuttonHighpass.get_value()
            freq_lowpass = self.spinbuttonLowpass.get_value()
            filter_name = self.comboboxFilterType.get_active_text()
            for tr in self.streams[self.stPt].traces:
                if filter_name == "Bandpass":
                    filt.append(bandpass(tr.data, freq_highpass, freq_lowpass,
                            df=tr.stats.sampling_rate, zerophase=zerophase))
                    msg = "%s (zerophase=%s): %.2f-%.2f Hz" % \
                            (filter_name, zerophase, freq_highpass,
                             freq_lowpass)
                elif filter_name == "Bandstop":
                    filt.append(bandstop(tr.data, freq_highpass, freq_lowpass,
                            df=tr.stats.sampling_rate, zerophase=zerophase))
                    msg = "%s (zerophase=%s): %.2f-%.2f Hz" % \
                            (filter_name, zerophase, freq_highpass,
                             freq_lowpass)
                elif filter_name == "Lowpass":
                    filt.append(lowpass(tr.data, freq_lowpass,
                            df=tr.stats.sampling_rate, zerophase=zerophase))
                    msg = "%s (zerophase=%s): %.2f Hz" % (filter_name,
                                                          zerophase,
                                                          freq_lowpass)
                elif filter_name == "Highpass":
                    filt.append(highpass(tr.data, freq_highpass,
                            df=tr.stats.sampling_rate, zerophase=zerophase))
                    msg = "%s (zerophase=%s): %.2f Hz" % (filter_name,
                                                          zerophase,
                                                          freq_highpass)
                else:
                    err = "Error: Unrecognized Filter Option. Showing " + \
                          "unfiltered data."
                    self.textviewStdErrImproved.write(err)
                    filt.append(tr.data)
            self.textviewStdOutImproved.write(msg)
            #make new plots
            for i, plot in enumerate(self.plts):
                plot.set_ydata(filt[i])
        else:
            #make new plots
            for i, plot in enumerate(self.plts):
                plot.set_ydata(self.streams[self.stPt][i].data)
            msg = "Unfiltered Traces"
            self.textviewStdOutImproved.write(msg)
        # Update all subplots
        self.redraw()

    # Define the event that handles the setting of P- and S-wave picks
    def keypress(self, event):
        if self.togglebuttonShowMap.get_active():
            return
        phase_type = self.comboboxPhaseType.get_active_text()
        keys = self.dictKeybindings
        dict = self.dicts[self.stPt]
        
        #######################################################################
        # Start of key events related to picking                              #
        #######################################################################
        # For some key events (picking events) we need information on the x/y
        # position of the cursor:
        if event.key in (keys['setPick'], keys['setPickError'],
                         keys['setMagMin'], keys['setMagMax']):
            # some keypress events only make sense inside our matplotlib axes
            if not event.inaxes in self.axs:
                return
            #We want to round from the picking location to
            #the time value of the nearest time sample:
            samp_rate = self.streams[self.stPt][0].stats.sampling_rate
            pickSample = event.xdata * samp_rate
            pickSample = round(pickSample)
            pickSample = pickSample / samp_rate
            # we need the position of the cursor location
            # in the seismogram array:
            xpos = pickSample * samp_rate

        if event.key == keys['setPick']:
            # some keypress events only make sense inside our matplotlib axes
            if not event.inaxes in self.axs:
                return
            if phase_type == 'P':
                self.delPLine()
                self.delPLabel()
                self.delPsynthLine()
                dict['P'] = pickSample
                self.drawPLine()
                self.drawPLabel()
                self.drawPsynthLine()
                self.drawPsynthLabel()
                #check if the new P pick lies outside of the Error Picks
                if 'PErr1' in dict and dict['P'] < dict['PErr1']:
                    self.delPErr1Line()
                    self.delPErr1()
                if 'PErr2' in dict and dict['P'] > dict['PErr2']:
                    self.delPErr2Line()
                    self.delPErr2()
                self.redraw()
                msg = "P Pick set at %.3f" % dict['P']
                self.textviewStdOutImproved.write(msg)
                return
            elif phase_type == 'S':
                self.delSLine()
                self.delSLabel()
                self.delSsynthLine()
                dict['S'] = pickSample
                dict['Saxind'] = self.axs.index(event.inaxes)
                self.drawSLine()
                self.drawSLabel()
                self.drawSsynthLine()
                self.drawSsynthLabel()
                #check if the new S pick lies outside of the Error Picks
                if 'SErr1' in dict and dict['S'] < dict['SErr1']:
                    self.delSErr1Line()
                    self.delSErr1()
                if 'SErr2' in dict and dict['S'] > dict['SErr2']:
                    self.delSErr2Line()
                    self.delSErr2()
                self.redraw()
                msg = "S Pick set at %.3f" % dict['S']
                self.textviewStdOutImproved.write(msg)
                return

        if event.key == keys['setWeight0']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['PWeight']=0
                self.drawPLabel()
                self.redraw()
                msg = "P Pick weight set to %i"%dict['PWeight']
                self.textviewStdOutImproved.write(msg)
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SWeight']=0
                self.drawSLabel()
                self.redraw()
                msg = "S Pick weight set to %i"%dict['SWeight']
                self.textviewStdOutImproved.write(msg)
                return

        if event.key == keys['setWeight1']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['PWeight']=1
                msg = "P Pick weight set to %i"%dict['PWeight']
                self.textviewStdOutImproved.write(msg)
                self.drawPLabel()
                self.redraw()
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SWeight']=1
                self.drawSLabel()
                self.redraw()
                msg = "S Pick weight set to %i"%dict['SWeight']
                self.textviewStdOutImproved.write(msg)
                return

        if event.key == keys['setWeight2']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['PWeight']=2
                msg = "P Pick weight set to %i"%dict['PWeight']
                self.textviewStdOutImproved.write(msg)
                self.drawPLabel()
                self.redraw()
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SWeight']=2
                self.drawSLabel()
                self.redraw()
                msg = "S Pick weight set to %i"%dict['SWeight']
                self.textviewStdOutImproved.write(msg)
                return

        if event.key == keys['setWeight3']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['PWeight']=3
                msg = "P Pick weight set to %i"%dict['PWeight']
                self.textviewStdOutImproved.write(msg)
                self.drawPLabel()
                self.redraw()
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SWeight']=3
                self.drawSLabel()
                self.redraw()
                msg = "S Pick weight set to %i"%dict['SWeight']
                self.textviewStdOutImproved.write(msg)
                return

        if event.key == keys['setPolUp']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['PPol']='up'
                self.drawPLabel()
                self.redraw()
                msg = "P Pick polarity set to %s"%dict['PPol']
                self.textviewStdOutImproved.write(msg)
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SPol']='up'
                self.drawSLabel()
                self.redraw()
                msg = "S Pick polarity set to %s"%dict['SPol']
                self.textviewStdOutImproved.write(msg)
                return

        if event.key == keys['setPolPoorUp']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['PPol']='poorup'
                self.drawPLabel()
                self.redraw()
                msg = "P Pick polarity set to %s"%dict['PPol']
                self.textviewStdOutImproved.write(msg)
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SPol']='poorup'
                self.drawSLabel()
                self.redraw()
                msg = "S Pick polarity set to %s"%dict['SPol']
                self.textviewStdOutImproved.write(msg)
                return

        if event.key == keys['setPolDown']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['PPol']='down'
                self.drawPLabel()
                self.redraw()
                msg = "P Pick polarity set to %s"%dict['PPol']
                self.textviewStdOutImproved.write(msg)
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SPol']='down'
                self.drawSLabel()
                self.redraw()
                msg = "S Pick polarity set to %s"%dict['SPol']
                self.textviewStdOutImproved.write(msg)
                return

        if event.key == keys['setPolPoorDown']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['PPol']='poordown'
                self.drawPLabel()
                self.redraw()
                msg = "P Pick polarity set to %s"%dict['PPol']
                self.textviewStdOutImproved.write(msg)
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SPol']='poordown'
                self.drawSLabel()
                self.redraw()
                msg = "S Pick polarity set to %s"%dict['SPol']
                self.textviewStdOutImproved.write(msg)
                return

        if event.key == keys['setOnsetImpulsive']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['POnset'] = 'impulsive'
                self.drawPLabel()
                self.redraw()
                msg = "P pick onset set to %s" % dict['POnset']
                self.textviewStdOutImproved.write(msg)
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SOnset'] = 'impulsive'
                self.drawSLabel()
                self.redraw()
                msg = "S pick onset set to %s" % dict['SOnset']
                self.textviewStdOutImproved.write(msg)
                return

        if event.key == keys['setOnsetEmergent']:
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                self.delPLabel()
                dict['POnset'] = 'emergent'
                self.drawPLabel()
                self.redraw()
                msg = "P pick onset set to %s" % dict['POnset']
                self.textviewStdOutImproved.write(msg)
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                self.delSLabel()
                dict['SOnset'] = 'emergent'
                self.drawSLabel()
                self.redraw()
                msg = "S pick onset set to %s" % dict['SOnset']
                self.textviewStdOutImproved.write(msg)
                return

        if event.key == keys['delPick']:
            if phase_type == 'P':
                self.delPLine()
                self.delP()
                self.delPWeight()
                self.delPPol()
                self.delPOnset()
                self.delPLabel()
                self.delPErr1Line()
                self.delPErr1()
                self.delPErr2Line()
                self.delPErr2()
                self.redraw()
                return
            elif phase_type == 'S':
                self.delSLine()
                self.delS()
                self.delSWeight()
                self.delSPol()
                self.delSOnset()
                self.delSLabel()
                self.delSErr1Line()
                self.delSErr1()
                self.delSErr2Line()
                self.delSErr2()
                self.redraw()
                return

        if event.key == keys['setPickError']:
            # some keypress events only make sense inside our matplotlib axes
            if not event.inaxes in self.axs:
                return
            if phase_type == 'P':
                if not 'P' in dict:
                    return
                # Set left Error Pick
                if pickSample < dict['P']:
                    self.delPErr1Line()
                    dict['PErr1'] = pickSample
                    self.drawPErr1Line()
                    self.redraw()
                    msg = "P Error Pick 1 set at %.3f" % dict['PErr1']
                    self.textviewStdOutImproved.write(msg)
                # Set right Error Pick
                elif pickSample > dict['P']:
                    self.delPErr2Line()
                    dict['PErr2'] = pickSample
                    self.drawPErr2Line()
                    self.redraw()
                    msg = "P Error Pick 2 set at %.3f" % dict['PErr2']
                    self.textviewStdOutImproved.write(msg)
                return
            elif phase_type == 'S':
                if not 'S' in dict:
                    return
                # Set left Error Pick
                if pickSample < dict['S']:
                    self.delSErr1Line()
                    dict['SErr1'] = pickSample
                    self.drawSErr1Line()
                    self.redraw()
                    msg = "S Error Pick 1 set at %.3f" % dict['SErr1']
                    self.textviewStdOutImproved.write(msg)
                # Set right Error Pick
                elif pickSample > dict['S']:
                    self.delSErr2Line()
                    dict['SErr2'] = pickSample
                    self.drawSErr2Line()
                    self.redraw()
                    msg = "S Error Pick 2 set at %.3f" % dict['SErr2']
                    self.textviewStdOutImproved.write(msg)
                return

        if event.key == keys['setMagMin']:
            # some keypress events only make sense inside our matplotlib axes
            if not event.inaxes in self.axs:
                return
            if phase_type == 'Mag':
                if len(self.axs) < 2:
                    err = "Error: Magnitude picking only supported with a " + \
                          "minimum of 2 axes."
                    self.textviewStdErrImproved.write(err)
                    return
                if event.inaxes is self.axs[1]:
                    self.delMagMinCross1()
                    ydata = event.inaxes.lines[0].get_ydata() #get the first line hoping that it is the seismogram!
                    cutoffSamples = xpos - self.magPickWindow #remember, how much samples there are before our small window! We have to add this number for our MagMinT estimation!
                    dict['MagMin1'] = np.min(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                    # save time of magnitude minimum in seconds
                    tmp_magtime = cutoffSamples + np.argmin(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                    tmp_magtime = tmp_magtime / samp_rate
                    dict['MagMin1T'] = tmp_magtime
                    #delete old MagMax Pick, if new MagMin Pick is higher
                    if 'MagMax1' in dict and dict['MagMin1'] > dict['MagMax1']:
                        self.delMagMaxCross1()
                        self.delMagMax1()
                    self.drawMagMinCross1()
                    self.redraw()
                    msg = "Minimum for magnitude estimation set: %s at %.3f" \
                            % (dict['MagMin1'], dict['MagMin1T'])
                    self.textviewStdOutImproved.write(msg)
                elif event.inaxes is self.axs[2]:
                    self.delMagMinCross2()
                    ydata = event.inaxes.lines[0].get_ydata() #get the first line hoping that it is the seismogram!
                    cutoffSamples = xpos - self.magPickWindow #remember, how much samples there are before our small window! We have to add this number for our MagMinT estimation!
                    dict['MagMin2'] = np.min(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                    # save time of magnitude minimum in seconds
                    tmp_magtime = cutoffSamples + np.argmin(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                    tmp_magtime = tmp_magtime / samp_rate
                    dict['MagMin2T'] = tmp_magtime
                    #delete old MagMax Pick, if new MagMin Pick is higher
                    if 'MagMax2' in dict and dict['MagMin2'] > dict['MagMax2']:
                        self.delMagMaxCross2()
                        self.delMagMax2()
                    self.drawMagMinCross2()
                    self.redraw()
                    msg = "Minimum for magnitude estimation set: %s at %.3f" \
                            % (dict['MagMin2'], dict['MagMin2T'])
                    self.textviewStdOutImproved.write(msg)
                return

        if event.key == keys['setMagMax']:
            # some keypress events only make sense inside our matplotlib axes
            if not event.inaxes in self.axs:
                return
            if phase_type == 'Mag':
                if len(self.axs) < 2:
                    err = "Error: Magnitude picking only supported with a " + \
                          "minimum of 2 axes."
                    self.textviewStdErrImproved.write(err)
                    return
                if event.inaxes is self.axs[1]:
                    self.delMagMaxCross1()
                    ydata = event.inaxes.lines[0].get_ydata() #get the first line hoping that it is the seismogram!
                    cutoffSamples = xpos - self.magPickWindow #remember, how much samples there are before our small window! We have to add this number for our MagMinT estimation!
                    dict['MagMax1'] = np.max(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                    # save time of magnitude maximum in seconds
                    tmp_magtime = cutoffSamples + np.argmax(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                    tmp_magtime = tmp_magtime / samp_rate
                    dict['MagMax1T'] = tmp_magtime
                    #delete old MagMax Pick, if new MagMax Pick is higher
                    if 'MagMin1' in dict and dict['MagMin1'] > dict['MagMax1']:
                        self.delMagMinCross1()
                        self.delMagMin1()
                    self.drawMagMaxCross1()
                    self.redraw()
                    msg = "Maximum for magnitude estimation set: %s at %.3f" \
                            % (dict['MagMax1'], dict['MagMax1T'])
                    self.textviewStdOutImproved.write(msg)
                elif event.inaxes is self.axs[2]:
                    self.delMagMaxCross2()
                    ydata = event.inaxes.lines[0].get_ydata() #get the first line hoping that it is the seismogram!
                    cutoffSamples = xpos - self.magPickWindow #remember, how much samples there are before our small window! We have to add this number for our MagMinT estimation!
                    dict['MagMax2'] = np.max(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                    # save time of magnitude maximum in seconds
                    tmp_magtime = cutoffSamples + np.argmax(ydata[xpos-self.magPickWindow:xpos+self.magPickWindow])
                    tmp_magtime = tmp_magtime / samp_rate
                    dict['MagMax2T'] = tmp_magtime
                    #delete old MagMax Pick, if new MagMax Pick is higher
                    if 'MagMin2' in dict and dict['MagMin2'] > dict['MagMax2']:
                        self.delMagMinCross2()
                        self.delMagMin2()
                    self.drawMagMaxCross2()
                    self.redraw()
                    msg = "Maximum for magnitude estimation set: %s at %.3f" \
                            % (dict['MagMax2'], dict['MagMax2T'])
                    self.textviewStdOutImproved.write(msg)
                return

        if event.key == keys['delMagMinMax']:
            if phase_type == 'Mag':
                if event.inaxes is self.axs[1]:
                    self.delMagMaxCross1()
                    self.delMagMinCross1()
                    self.delMagMin1()
                    self.delMagMax1()
                elif event.inaxes is self.axs[2]:
                    self.delMagMaxCross2()
                    self.delMagMinCross2()
                    self.delMagMin2()
                    self.delMagMax2()
                else:
                    return
                self.redraw()
                return
        #######################################################################
        # End of key events related to picking                                #
        #######################################################################
        
        if event.key == keys['switchWheelZoom']:
            self.flagWheelZoom = not self.flagWheelZoom
            if self.flagWheelZoom:
                msg = "Mouse wheel zooming activated"
                self.textviewStdOutImproved.write(msg)
            else:
                msg = "Mouse wheel zooming deactivated"
                self.textviewStdOutImproved.write(msg)
            return

        if event.key == keys['switchWheelZoomAxis']:
            self.flagWheelZoomAmplitude = True

        if event.key == keys['switchPan']:
            self.toolbar.pan()
            self.canv.widgetlock.release(self.toolbar)
            self.redraw()
            msg = "Switching pan mode"
            self.textviewStdOutImproved.write(msg)
            return
        
        # iterate the phase type combobox
        if event.key == keys['switchPhase']:
            combobox = self.comboboxPhaseType
            phase_count = len(combobox.get_model())
            phase_next = (combobox.get_active() + 1) % phase_count
            combobox.set_active(phase_next)
            msg = "Switching Phase button"
            self.textviewStdOutImproved.write(msg)
            return
            
        if event.key == keys['prevStream']:
            #would be nice if the button would show the click, but it is
            #too brief to be seen
            #self.buttonPreviousStream.set_state(True)
            self.buttonPreviousStream.clicked()
            #self.buttonPreviousStream.set_state(False)
            return

        if event.key == keys['nextStream']:
            #would be nice if the button would show the click, but it is
            #too brief to be seen
            #self.buttonNextStream.set_state(True)
            self.buttonNextStream.clicked()
            #self.buttonNextStream.set_state(False)
            return
    
    def keyrelease(self, event):
        keys = self.dictKeybindings
        if event.key == keys['switchWheelZoomAxis']:
            self.flagWheelZoomAmplitude = False

    # Define zooming for the mouse scroll wheel
    def scroll(self, event):
        if self.togglebuttonShowMap.get_active():
            return
        if not self.flagWheelZoom:
            return
        # Calculate and set new axes boundaries from old ones
        (left, right) = self.axs[0].get_xbound()
        (bottom, top) = self.axs[0].get_ybound()
        # Zoom in on scroll-up
        if event.button == 'up':
            if self.flagWheelZoomAmplitude:
                top /= 2.
                bottom /= 2.
            else:
                left += (event.xdata - left) / 2
                right -= (right - event.xdata) / 2
        # Zoom out on scroll-down
        elif event.button == 'down':
            if self.flagWheelZoomAmplitude:
                top *= 2.
                bottom *= 2.
            else:
                left -= (event.xdata - left) / 2
                right += (right - event.xdata) / 2
        if self.flagWheelZoomAmplitude:
            self.axs[0].set_ybound(lower=bottom, upper=top)
        else:
            self.axs[0].set_xbound(lower=left, upper=right)
        self.redraw()
    
    # Define zoom reset for the mouse button 2 (always scroll wheel!?)
    def buttonpress(self, event):
        if self.togglebuttonShowMap.get_active():
            return
        # set widgetlock when pressing mouse buttons and dont show cursor
        # cursor should not be plotted when making a zoom selection etc.
        if event.button == 1 or event.button == 3:
            self.multicursor.visible = False
            self.canv.widgetlock(self.toolbar)
        # show traces from start to end
        # (Use Z trace limits as boundaries)
        elif event.button == 2:
            self.axs[0].set_xbound(lower=self.xMin, upper=self.xMax)
            self.axs[0].set_ybound(lower=self.yMin, upper=self.yMax)
            # Update all subplots
            self.redraw()
            msg = "Resetting axes"
            self.textviewStdOutImproved.write(msg)
    
    def buttonrelease(self, event):
        if self.togglebuttonShowMap.get_active():
            return
        # release widgetlock when releasing mouse buttons
        if event.button == 1 or event.button == 3:
            self.multicursor.visible = True
            self.canv.widgetlock.release(self.toolbar)
