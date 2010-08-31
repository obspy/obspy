class GUI_connectors(object):
    """
    Class that connects all buttons to functions.
    """
    def __init__(self, logic):
        """
        logic is the class containing all the logic.
        """

        self.logic = logic
        self.interface = self.logic.interface
        # Connects all signals and slots!
        self._connectSignalsAndSlots()
        print 'got here'

    def _connectSignalsAndSlots(self):
        pass

    def on_windowObspyck_destroy(self, event):
        self.logic.cleanQuit()

    def on_buttonClearAll_clicked(self, event):
        self.logic.delAllItems()
        self.logic.clearDictionaries()
        self.logic.drawAllItems()
        self.logic.redraw()

    def on_buttonClearOrigMag_clicked(self, event):
        self.logic.delAllItems()
        self.logic.clearOriginMagnitudeDictionaries()
        self.logic.drawAllItems()
        self.logic.redraw()

    def on_buttonClearFocMec_clicked(self, event):
        self.logic.clearFocmecDictionary()

    def on_buttonDoHyp2000_clicked(self, event):
        self.logic.delAllItems()
        self.logic.clearOriginMagnitudeDictionaries()
        self.logic.dictOrigin['Program'] = "hyp2000"
        self.logic.doHyp2000()
        self.logic.loadHyp2000Data()
        self.logic.calculateEpiHypoDists()
        self.logic.dictMagnitude['Program'] = "obspy"
        self.logic.calculateStationMagnitudes()
        self.logic.updateNetworkMag()
        self.logic.drawAllItems()
        self.logic.redraw()
        self.logic.togglebuttonShowMap.set_active(True)

    def on_buttonDo3dloc_clicked(self, event):
        self.logic.delAllItems()
        self.logic.clearOriginMagnitudeDictionaries()
        self.logic.dictOrigin['Program'] = "3dloc"
        self.logic.do3dLoc()
        self.logic.load3dlocSyntheticPhases()
        self.logic.load3dlocData()
        self.logic.calculateEpiHypoDists()
        self.logic.dictMagnitude['Program'] = "obspy"
        self.logic.calculateStationMagnitudes()
        self.logic.updateNetworkMag()
        self.logic.drawAllItems()
        self.logic.redraw()
        self.logic.togglebuttonShowMap.set_active(True)

    def on_buttonDoNLLoc_clicked(self, event):
        self.logic.delAllItems()
        self.logic.clearOriginMagnitudeDictionaries()
        self.logic.dictOrigin['Program'] = "NLLoc"
        self.logic.doNLLoc()
        self.logic.loadNLLocOutput()
        self.logic.calculateEpiHypoDists()
        self.logic.dictMagnitude['Program'] = "obspy"
        self.logic.calculateStationMagnitudes()
        self.logic.updateNetworkMag()
        self.logic.drawAllItems()
        self.logic.redraw()
        self.logic.togglebuttonShowMap.set_active(True)

    def on_buttonCalcMag_clicked(self, event):
        self.logic.calculateEpiHypoDists()
        self.logic.dictMagnitude['Program'] = "obspy"
        self.logic.calculateStationMagnitudes()
        self.logic.updateNetworkMag()

    def on_buttonDoFocmec_clicked(self, event):
        self.logic.clearFocmecDictionary()
        self.logic.dictFocalMechanism['Program'] = "focmec"
        self.logic.doFocmec()

    def on_togglebuttonShowMap_clicked(self, event):
        buttons_deactivate = [self.logic.buttonClearAll, self.logic.buttonClearOrigMag,
                              self.logic.buttonClearFocMec, self.logic.buttonDoHyp2000,
                              self.logic.buttonDo3dloc, self.logic.buttonDoNLLoc,
                              self.logic.buttonCalcMag, self.logic.comboboxNLLocModel,
                              self.logic.buttonDoFocmec, self.logic.togglebuttonShowFocMec,
                              self.logic.buttonNextFocMec,
                              self.togglebuttonShowWadati,
                              self.buttonGetNextEvent, self.buttonSendEvent,
                              self.buttonUpdateEventList,
                              self.checkbuttonPublishEvent,
                              self.checkbuttonSysop, self.entrySysopPassword,
                              self.buttonDeleteEvent,
                              self.buttonPreviousStream, self.buttonNextStream,
                              self.togglebuttonOverview,
                              self.comboboxStreamName, self.labelStreamNumber,
                              self.comboboxPhaseType, self.togglebuttonFilter,
                              self.comboboxFilterType,
                              self.checkbuttonZeroPhase,
                              self.labelHighpass, self.labelLowpass,
                              self.spinbuttonHighpass, self.spinbuttonLowpass,
                              self.togglebuttonSpectrogram,
                              self.checkbuttonSpectrogramLog]
        state = self.logic.togglebuttonShowMap.get_active()
        for button in buttons_deactivate:
            button.set_sensitive(not state)
        if state:
            self.logic.delAxes()
            self.logic.fig.clear()
            self.logic.drawEventMap()
            self.logic.multicursor.visible = False
            self.logic.toolbar.pan()
            self.logic.toolbar.zoom()
            self.logic.toolbar.update()
            self.logic.canv.draw()
            self.logic.textviewStdOutImproved.write("http://maps.google.de/maps" + \
                    "?f=q&q=%.6f,%.6f" % (self.logic.dictOrigin['Latitude'],
                    self.logic.dictOrigin['Longitude']))
        else:
            self.logic.delEventMap()
            self.logic.fig.clear()
            self.logic.drawAxes()
            self.logic.toolbar.update()
            self.logic.drawSavedPicks()
            self.logic.multicursorReinit()
            self.logic.updatePlot()
            self.logic.updateStreamLabels()
            self.logic.canv.draw()

    def on_togglebuttonOverview_clicked(self, event):
        buttons_deactivate = [self.logic.buttonClearAll, self.logic.buttonClearOrigMag,
                              self.logic.buttonClearFocMec, self.logic.buttonDoHyp2000,
                              self.logic.buttonDo3dloc, self.logic.buttonDoNLLoc,
                              self.logic.buttonCalcMag, self.logic.comboboxNLLocModel,
                              self.logic.buttonDoFocmec, self.logic.togglebuttonShowMap,
                              self.logic.togglebuttonShowFocMec,
                              self.logic.buttonNextFocMec,
                              self.logic.togglebuttonShowWadati,
                              self.logic.buttonGetNextEvent, self.logic.buttonSendEvent,
                              self.logic.buttonUpdateEventList,
                              self.logic.checkbuttonPublishEvent,
                              self.logic.checkbuttonSysop, self.logic.entrySysopPassword,
                              self.logic.buttonDeleteEvent,
                              self.logic.buttonPreviousStream, self.logic.buttonNextStream,
                              self.logic.comboboxStreamName, self.logic.labelStreamNumber,
                              self.logic.comboboxPhaseType, self.logic.togglebuttonFilter,
                              self.logic.comboboxFilterType,
                              self.logic.checkbuttonZeroPhase,
                              self.logic.labelHighpass, self.logic.labelLowpass,
                              self.logic.spinbuttonHighpass, self.logic.spinbuttonLowpass,
                              self.logic.togglebuttonSpectrogram,
                              self.logic.checkbuttonSpectrogramLog]
        state = self.logic.togglebuttonOverview.get_active()
        for button in buttons_deactivate:
            button.set_sensitive(not state)
        if state:
            self.logic.delAxes()
            self.logic.fig.clear()
            self.logic.drawStreamOverview()
            self.logic.multicursor.visible = False
            self.logic.toolbar.pan()
            self.logic.toolbar.zoom()
            self.logic.toolbar.update()
            self.logic.canv.draw()
        else:
            self.logic.delAxes()
            self.logic.fig.clear()
            self.logic.drawAxes()
            self.logic.toolbar.update()
            self.logic.drawSavedPicks()
            self.logic.multicursorReinit()
            self.logic.updatePlot()
            self.logic.updateStreamLabels()
            self.logic.canv.draw()

    def on_togglebuttonShowFocMec_clicked(self, event):
        buttons_deactivate = [self.logic.buttonClearAll, self.logic.buttonClearOrigMag,
                              self.logic.buttonClearFocMec, self.logic.buttonDoHyp2000,
                              self.logic.buttonDo3dloc, self.logic.buttonDoNLLoc,
                              self.logic.buttonCalcMag, self.logic.comboboxNLLocModel,
                              self.logic.buttonDoFocmec, self.logic.togglebuttonShowMap,
                              self.logic.togglebuttonShowWadati,
                              self.logic.buttonGetNextEvent, self.logic.buttonSendEvent,
                              self.logic.buttonUpdateEventList,
                              self.logic.checkbuttonPublishEvent,
                              self.logic.checkbuttonSysop, self.logic.entrySysopPassword,
                              self.logic.buttonDeleteEvent,
                              self.logic.buttonPreviousStream, self.logic.buttonNextStream,
                              self.logic.togglebuttonOverview,
                              self.logic.comboboxStreamName, self.logic.labelStreamNumber,
                              self.logic.comboboxPhaseType, self.logic.togglebuttonFilter,
                              self.logic.comboboxFilterType,
                              self.logic.checkbuttonZeroPhase,
                              self.logic.labelHighpass, self.logic.labelLowpass,
                              self.logic.spinbuttonHighpass, self.logic.spinbuttonLowpass,
                              self.logic.togglebuttonSpectrogram,
                              self.logic.checkbuttonSpectrogramLog]
        state = self.logic.togglebuttonShowFocMec.get_active()
        for button in buttons_deactivate:
            button.set_sensitive(not state)
        if state:
            self.logic.delAxes()
            self.logic.fig.clear()
            self.logic.drawFocMec()
            self.logic.multicursor.visible = False
            self.logic.toolbar.pan()
            self.logic.toolbar.zoom()
            self.logic.toolbar.zoom()
            self.logic.toolbar.update()
            self.logic.canv.draw()
        else:
            self.logic.delFocMec()
            self.logic.fig.clear()
            self.logic.drawAxes()
            self.logic.toolbar.update()
            self.logic.drawSavedPicks()
            self.logic.multicursorReinit()
            self.logic.updatePlot()
            self.logic.updateStreamLabels()
            self.logic.canv.draw()

    def on_buttonNextFocMec_clicked(self, event):
        self.logic.nextFocMec()
        if self.logic.togglebuttonShowFocMec.get_active():
            self.logic.delFocMec()
            self.logic.fig.clear()
            self.logic.drawFocMec()
            self.logic.canv.draw()

    def on_togglebuttonShowWadati_clicked(self, event):
        buttons_deactivate = [self.logic.buttonClearAll, self.logic.buttonClearOrigMag,
                              self.logic.buttonClearFocMec, self.logic.buttonDoHyp2000,
                              self.logic.buttonDo3dloc, self.logic.buttonDoNLLoc,
                              self.logic.buttonCalcMag, self.logic.comboboxNLLocModel,
                              self.logic.buttonDoFocmec, self.logic.togglebuttonShowFocMec,
                              self.logic.buttonNextFocMec, self.logic.togglebuttonShowMap,
                              self.logic.buttonGetNextEvent, self.logic.buttonSendEvent,
                              self.logic.buttonUpdateEventList,
                              self.logic.checkbuttonPublishEvent,
                              self.logic.checkbuttonSysop, self.logic.entrySysopPassword,
                              self.logic.buttonDeleteEvent,
                              self.logic.buttonPreviousStream, self.logic.buttonNextStream,
                              self.logic.togglebuttonOverview,
                              self.logic.comboboxStreamName, self.logic.labelStreamNumber,
                              self.logic.comboboxPhaseType, self.logic.togglebuttonFilter,
                              self.logic.comboboxFilterType,
                              self.logic.checkbuttonZeroPhase,
                              self.logic.labelHighpass, self.logic.labelLowpass,
                              self.logic.spinbuttonHighpass, self.logic.spinbuttonLowpass,
                              self.logic.togglebuttonSpectrogram,
                              self.logic.checkbuttonSpectrogramLog]
        state = self.logic.togglebuttonShowWadati.get_active()
        for button in buttons_deactivate:
            button.set_sensitive(not state)
        if state:
            self.logic.delAxes()
            self.logic.fig.clear()
            self.logic.drawWadati()
            self.logic.multicursor.visible = False
            self.logic.toolbar.pan()
            self.logic.toolbar.update()
            self.logic.canv.draw()
        else:
            self.logic.delWadati()
            self.logic.fig.clear()
            self.logic.drawAxes()
            self.logic.toolbar.update()
            self.logic.drawSavedPicks()
            self.logic.multicursorReinit()
            self.logic.updatePlot()
            self.logic.updateStreamLabels()
            self.logic.canv.draw()

    def on_buttonGetNextEvent_clicked(self, event):
        # check if event list is empty and force an update if this is the case
        if not hasattr(self, "seishubEventList"):
            self.logic.updateEventListFromSeishub(self.logic.streams[0][0].stats.starttime,
                                            self.logic.streams[0][0].stats.endtime)
        if not self.logic.seishubEventList:
            msg = "No events available from seishub."
            self.logic.textviewStdOutImproved.write(msg)
            return
        # iterate event number to fetch
        self.logic.seishubEventCurrent = (self.logic.seishubEventCurrent + 1) % \
                                   self.logic.seishubEventCount
        event = self.logic.seishubEventList[self.logic.seishubEventCurrent]
        resource_name = event.xpath(u"resource_name")[0].text
        self.logic.delAllItems()
        self.logic.clearDictionaries()
        self.logic.getEventFromSeishub(resource_name)
        #self.logic.getNextEventFromSeishub(self.logic.streams[0][0].stats.starttime, 
        #                             self.logic.streams[0][0].stats.endtime)
        self.logic.drawAllItems()
        self.logic.redraw()
        
        #XXX 

    def on_buttonUpdateEventList_clicked(self, event):
        self.logic.updateEventListFromSeishub(self.logic.streams[0][0].stats.starttime,
                                        self.logic.streams[0][0].stats.endtime)

    def on_buttonSendEvent_clicked(self, event):
        self.logic.uploadSeishub()
        self.logic.checkForSysopEventDuplicates(self.logic.streams[0][0].stats.starttime,
                                          self.logic.streams[0][0].stats.endtime)

    def on_checkbuttonPublishEvent_toggled(self, event):
        newstate = self.logic.checkbuttonPublishEvent.get_active()
        msg = "Setting \"public\" flag of event to: %s" % newstate
        self.logic.textviewStdOutImproved.write(msg)

    def on_buttonDeleteEvent_clicked(self, event):
        event = self.logic.seishubEventList[self.logic.seishubEventCurrent]
        resource_name = event.xpath(u"resource_name")[0].text
        account = event.xpath(u"account")
        user = event.xpath(u"user")
        if account:
            account = account[0].text
        else:
            account = None
        if user:
            user = user[0].text
        else:
            user = None
        dialog = gtk.MessageDialog(self.logic.win, gtk.DIALOG_MODAL,
                                   gtk.MESSAGE_INFO, gtk.BUTTONS_YES_NO)
        msg = "Delete event from database?\n\n"
        msg += "<tt><b>%s</b> (account: %s, user: %s)</tt>" % (resource_name,
                                                               account, user)
        dialog.set_markup(msg)
        dialog.set_title("Delete?")
        response = dialog.run()
        dialog.destroy()
        if response == gtk.RESPONSE_YES:
            self.logic.deleteEventInSeishub(resource_name)
            self.logic.on_buttonUpdateEventList_clicked(event)
    
    def on_checkbuttonSysop_toggled(self, event):
        newstate = self.logic.checkbuttonSysop.get_active()
        msg = "Setting usage of \"sysop\"-account to: %s" % newstate
        self.logic.textviewStdOutImproved.write(msg)
    
    # the corresponding signal is emitted when hitting return after entering
    # the password
    def on_entrySysopPassword_activate(self, event):
        # test authentication information:
        passwd = self.logic.entrySysopPassword.get_text()
        auth = 'Basic ' + (base64.encodestring('sysop:' + passwd)).strip()
        webservice = httplib.HTTP(self.logic.server['Server'])
        webservice.putrequest("HEAD", '/xml/seismology/event/just_a_test')
        webservice.putheader('Authorization', auth)
        webservice.endheaders()
        statuscode = webservice.getreply()[0]
        # if authentication test fails empty password field and uncheck sysop
        if statuscode == 401: # 401 means "Unauthorized"
            self.logic.checkbuttonSysop.set_active(False)
            self.logic.entrySysopPassword.set_text("")
            err = "Error: Authentication as sysop failed! (Wrong password!?)"
            self.logic.textviewStdErrImproved.write(err)
        else:
            self.logic.checkbuttonSysop.set_active(True)
        self.logic.canv.grab_focus()

    def on_buttonSetFocusOnPlot_clicked(self, event):
        self.logic.setFocusToMatplotlib()

    def on_buttonDebug_clicked(self, event):
        self.logic.debug()

    def on_buttonQuit_clicked(self, event):
        self.logic.checkForSysopEventDuplicates(self.logic.streams[0][0].stats.starttime,
                                          self.logic.streams[0][0].stats.endtime)
        self.logic.cleanQuit()

    def on_buttonPreviousStream_clicked(self, event):
        self.logic.stPt = (self.logic.stPt - 1) % self.logic.stNum
        self.logic.comboboxStreamName.set_active(self.logic.stPt)

    def on_comboboxStreamName_changed(self, event):
        self.logic.stPt = self.logic.comboboxStreamName.get_active()
        xmin, xmax = self.logic.axs[0].get_xlim()
        self.logic.delAllItems()
        self.logic.delAxes()
        self.logic.fig.clear()
        self.logic.drawAxes()
        self.logic.drawSavedPicks()
        self.logic.multicursorReinit()
        self.logic.axs[0].set_xlim(xmin, xmax)
        self.logic.updatePlot()
        msg = "Going to stream: %s" % self.logic.dicts[self.logic.stPt]['Station']
        self.logic.updateStreamNumberLabel()
        self.logic.textviewStdOutImproved.write(msg)

    def on_buttonNextStream_clicked(self, event):
        self.logic.stPt = (self.logic.stPt + 1) % self.logic.stNum
        self.logic.comboboxStreamName.set_active(self.logic.stPt)

    def on_comboboxPhaseType_changed(self, event):
        self.logic.updateMulticursorColor()
        self.logic.updateButtonPhaseTypeColor()
        self.logic.redraw()

    def on_togglebuttonFilter_toggled(self, event):
        self.logic.updatePlot()

    def on_comboboxFilterType_changed(self, event):
        if self.logic.togglebuttonFilter.get_active():
            self.logic.updatePlot()

    def on_checkbuttonZeroPhase_toggled(self, event):
        # if the filter flag is not set, we don't have to update the plot
        if self.logic.togglebuttonFilter.get_active():
            self.logic.updatePlot()

    def on_spinbuttonHighpass_value_changed(self, event):
        if not self.logic.togglebuttonFilter.get_active() or \
           self.logic.comboboxFilterType.get_active_text() == "Lowpass":
            self.logic.canv.grab_focus()
            return
        # if the filter flag is not set, we don't have to update the plot
        # XXX if we have a lowpass, we dont need to update!! Not yet implemented!! XXX
        if self.logic.spinbuttonLowpass.get_value() < self.logic.spinbuttonHighpass.get_value():
            err = "Warning: Lowpass frequency below Highpass frequency!"
            self.logic.textviewStdErrImproved.write(err)
        # XXX maybe the following check could be done nicer
        # XXX check this criterion!
        minimum  = float(self.logic.streams[self.logic.stPt][0].stats.sampling_rate) / \
                self.logic.streams[self.logic.stPt][0].stats.npts
        if self.logic.spinbuttonHighpass.get_value() < minimum:
            err = "Warning: Lowpass frequency is not supported by length of trace!"
            self.logic.textviewStdErrImproved.write(err)
        self.logic.updatePlot()
        # XXX we could use this for the combobox too!
        # reset focus to matplotlib figure
        self.logic.canv.grab_focus()

    def on_spinbuttonLowpass_value_changed(self, event):
        if not self.logic.togglebuttonFilter.get_active() or \
           self.logic.comboboxFilterType.get_active_text() == "Highpass":
            self.logic.canv.grab_focus()
            return
        # if the filter flag is not set, we don't have to update the plot
        # XXX if we have a highpass, we dont need to update!! Not yet implemented!! XXX
        if self.logic.spinbuttonLowpass.get_value() < self.logic.spinbuttonHighpass.get_value():
            err = "Warning: Lowpass frequency below Highpass frequency!"
            self.logic.textviewStdErrImproved.write(err)
        # XXX maybe the following check could be done nicer
        # XXX check this criterion!
        maximum  = self.logic.streams[self.logic.stPt][0].stats.sampling_rate / 2.0
        if self.logic.spinbuttonLowpass.get_value() > maximum:
            err = "Warning: Highpass frequency is lower than Nyquist!"
            self.logic.textviewStdErrImproved.write(err)
        self.logic.updatePlot()
        # XXX we could use this for the combobox too!
        # reset focus to matplotlib figure
        self.logic.canv.grab_focus()

    def on_togglebuttonSpectrogram_toggled(self, event):
        buttons_deactivate = [self.logic.togglebuttonFilter,
                              self.logic.togglebuttonOverview,
                              self.logic.comboboxFilterType,
                              self.logic.checkbuttonZeroPhase,
                              self.logic.labelHighpass, self.logic.labelLowpass,
                              self.logic.spinbuttonHighpass, self.logic.spinbuttonLowpass]
        state = self.logic.togglebuttonSpectrogram.get_active()
        for button in buttons_deactivate:
            button.set_sensitive(not state)
        if state:
            msg = "Showing spectrograms (takes a few seconds with log-option)."
        else:
            msg = "Showing seismograms."
        xmin, xmax = self.logic.axs[0].get_xlim()
        self.logic.delAllItems()
        self.logic.delAxes()
        self.logic.fig.clear()
        self.logic.drawAxes()
        self.logic.drawSavedPicks()
        self.logic.multicursorReinit()
        self.logic.axs[0].set_xlim(xmin, xmax)
        self.logic.updatePlot()
        self.logic.textviewStdOutImproved.write(msg)

    def on_checkbuttonSpectrogramLog_toggled(self, event):
        if self.logic.togglebuttonSpectrogram.get_active():
            self.logic.on_togglebuttonSpectrogram_toggled(event)
