# -*- coding: utf-8 -*-

from __future__ import with_statement

from ConfigParser import RawConfigParser, DEFAULTSECT
from obspy.core import UTCDateTime
import os
import sys


class DBConfigParser(RawConfigParser):
    """
    Config parser for the database viewer. Inherits from
    ConfigParser.RawConfigParser and adds comments.

    It overwrites the write method so that each sections and the items in each
    sections are sorted alphabetically and it also adds support to give each
    item a commentary to ease documentation of the configuration file.
    """
    def __init__(self, env):
        """
        Init method that will also read and write the configuration file. No
        further interaction is necessary.
        """
        self.env = env
        # XXX: I do not know how to use super here because RawConfigParser does
        # not inherit from 'object'.
        RawConfigParser.__init__(self)
        # Dictionary to contain the optional comments.
        self.comments = {}
        self.config_file = self.env.config_file
        # Check if the config file exists. Otherwise give a message, create the
        # file and exits.
        if not os.path.exists(self.config_file):
            exists = False
        else:
            exists = True
        self.getConfiguration()
        if not exists:
            msg = "No config file exists. A default one has been created at %s. Please edit it and restart the application." \
                  % self.config_file
            print msg
            sys.exit()

    def write(self, fp):
        """
        Write an .ini-format representation of the configuration state.

        Override the write method to allow for comments in the configuration
        file.
        """
        if self._defaults:
            fp.write("[%s]\n" % DEFAULTSECT)
            for (key, value) in self._defaults.items():
                fp.write("%s = %s\n" % (key, str(value).replace('\n', '\n\t')))
            fp.write("\n")
        # Sort the sections.
        sections = self._sections.keys()
        sections.sort()
        for section in sections:
            fp.write("[%s]\n" % section)
            # Enable sorting after keys.
            keys = self._sections[section].keys()
            keys.sort()
            for key in keys:
                if key != "__name__":
                    value = self._sections[section][key]
                    # Write the comment if there is one.
                    if self.comments.has_key(key):
                        lines = \
                        self.formatStringToCertainLength(self.comments[key],
                                                          77)
                        for line in lines:
                            fp.write("# %s\n" % line)
                    fp.write("%s = %s\n" %
                             (key, str(value).replace('\n', '\n\t')))
            fp.write("\n")

    def getConfiguration(self):
        """
        Reads the config file and adds all values to self.env for application
        wide access.

        Will create the config file if it does not exists and add any missing
        entries automatically.
        """
        self.read(self.config_file)
        # Check if sections exist and add them if necessary.
        sections = ['Files and Directories', 'General Settings', 'Server',
                    'Picker', 'Appearance']
        for section in sections:
            if not self.has_section(section):
                self.add_section(section)
        # Read and set all values. Use a try/except construction for everything
        # to write them if they are not set.
        self.getOrSetDefault('Server', 'Seishub Server', 'seishub_server',
                             'http://teide.geophysik.uni-muenchen.de:8080',
                             comment='The address of the SeisHub server.')
        self.getOrSetDefault('Server', 'Password', 'seishub_password',
                             'dbviewer',
                             comment='The password for the SeisHub server.')
        self.getOrSetDefault('Server', 'User', 'seishub_user',
                             'dbviewer',
                             comment='The user for the SeisHub server.')
        self.getOrSetDefault('Server', 'Timeout', 'seishub_timeout',
                             10, value_type='int',
                             comment='The timeout in seconds for the SeisHub server.')
        self.getOrSetDefault('Files and Directories', 'Cache Directory', 'cache_dir',
                 os.path.join(self.env.home_dir, 'cache'),
                 comment='All cached files and databases will be stored in this directory.')
        # Default set it for a span of one week ending one week ago.
        self.getOrSetDefault('General Settings', 'Default Starttime', 'starttime',
                 '$today - 14$',
                 comment='The starttime when the database viewer is opened.  Possible values are any string that obspy.core.UTCDateTime can parse or $today$ which represents the current day. It is also possible to add or remove whole days from today, e.g. $today - 7$ would be today one week ago.')
        self.getOrSetDefault('General Settings', 'Default Endtime', 'endtime',
                 '$today - 7$',
                 comment='The endtime when the database viewer is opened. All options applicable to Default Starttime are also valid here.')
        # Parse the start- and endtime. The endtime will always be at the end
        # of the day.
        self.env.starttime = self.parseTimes(self.env.starttime)
        self.env.endtime = self.parseTimes(self.env.endtime) + 86399
        # Debug mode.
        self.getOrSetDefault('General Settings', 'Debug', 'debug', False,
                 value_type='boolean',
                 comment='Debugging messages True/False')
        # Force software rendering.
        comment = 'OpenGL is not supported on all machines and the program might crash right after the initial splash screen. Set this to True to enforce software rendering which is slower but works in any case.'
        self.getOrSetDefault('General Settings', 'Force Software Rendering',
                 'software_rendering', False, value_type='boolean',
                 comment=comment)
        # Details.
        self.getOrSetDefault('Appearance', 'Detail', 'detail',
                250, value_type='int',
                comment='The number of vertical bars drawn for each plot')
        # Double click time.
        self.getOrSetDefault('General Settings', 'Double click time',
                'double_click_time', 0.2, value_type='float',
                comment='Maximum time in seconds between to clicks to be registered as a double click.')
        # Buffer in days.
        self.getOrSetDefault('General Settings',
                 'Buffer', 'buffer', 3,
                 value_type='int',
                 comment='Buffer in days before and after the shown plots when requesting from Seishub')
        # Log scale of the plots.
        self.getOrSetDefault('Appearance', 'Log Scale', 'log_scale', False,
                 value_type='boolean',
                 comment='Plots have a log scale True/False')
        # To help determine the maximum zoom level.
        self.getOrSetDefault('General Settings', 'Preview delta',
                 'preview_delta', 30.0, value_type='float',
                 comment='The sample spacing in seconds of the preview files that are received from Seishub. This is dynamically adjusted once the first file has been requested but it is needed to set a maximum zoom level before any file has been requested.')
        # Add picker settings. 
        self.getOrSetDefault('Picker',
                 'System call command',
                 'picker_command', 'obspyck.py -t $starttime$ -d $duration$ -i $channels$',
                 comment='System call command for the picker, e.g. obspyck.py -t $starttime$ -d $duration$ -i $channels$ (everything enclosed in $ symbols will be replaced with the corresponding variable. Available are starttime, endtime, duration (in seconds), channels)')
        self.getOrSetDefault('Picker', 'Time format', 'picker_strftime',
                 '%Y-%m-%dT%H:%M:%S',
                 comment='Format for start- and endtime in strftime notation.')
        self.getOrSetDefault('Picker', 'Channels enclosure',
                 'channel_enclosure', "''",
                 comment='The channels will be enclosed in these two symbols. More then two symbols are not allowed.')
        self.getOrSetDefault('Picker', 'Channels seperator', 'channel_seperator',
                 ',', comment='The single channels will be seperated by this.')
        # Set event download range.
        self.getOrSetDefault('General Settings', 'Events Starttime',
                     'event_starttime', '$today - 100$',
                     comment='Requesting events from SeisHub takes time.  Therefore events will not be dynamically loaded during runtime (manual loading during runtime is possible) but they rather will be preloaded during the startup of the application.  This defines the beginning of the time frame for the preloaded events. The same options as in Default Starttime are valid here.')
        self.getOrSetDefault('General Settings', 'Events Endtime',
                     'event_endtime', '$today$',
                     comment='This defines the end of the time frame for the preloaded events. The same options as in Default Starttime are valid here.')
        # Plot height.
        self.getOrSetDefault('Appearance', 'Plot Height', 'plot_height', 50,
                 value_type='int',
                 comment='Height of a single plot in pixel.')
        # Parse the start- and endtime for the event time frame. The endtime
        # will always be at the end of the day.
        self.env.event_starttime = self.parseTimes(self.env.event_starttime)
        self.env.event_endtime = self.parseTimes(self.env.event_endtime) + 86399
        # Ensure that at least the time span that is plotted is in the event
        # time span.
        if self.env.starttime < self.env.event_starttime:
            self.env.event_starttime = self.env.starttime
        if self.env.endtime > self.env.event_endtime:
            self.env.event_endtime = self.env.endtime
        # Writing the configuration file.
        with open(self.config_file, 'wb') as configfile:
                self.write(configfile)

    def getOrSetDefault(self, section, name, env_name, default_value,
                        value_type=None, comment=None):
        """
        Use a try except construction to get the value of name and write it to
        self.env.

        section = The section in the config file.
        name = The name of the item in the config file.
        env_name = The corresponding name in the environment
        default_value = The default value in the file if the config file is
                        faulty or could not be found.
        value_type = Type of the value.
        comment = Optional commen of the configuration option.
        """
        # Always set the comment to ensure it will always be written.
        if comment:
            self.comments[self.optionxform(name)] = comment
        # Determine the type.
        if value_type == 'int':
            funct = self.getint
        elif value_type == 'float':
            funct = self.getfloat
        elif value_type == 'boolean':
            funct = self.getboolean
        else:
            funct = self.get
        # Actually set the attribute.
        try:
            setattr(self.env, env_name, funct(section, name))
        except:
            setattr(self.env, env_name, default_value)
            self.set(section, name, default_value)

    def formatStringToCertainLength(self, string, length):
        """
        Takes a string and formats it to a certain length. Will remove leading
        and trailing whitespaces.

        Returns a list with strings each with the maximum length.

        XXX: Does currently not work with words longer than length.
        """
        string = string.strip()
        # Do nothing if nothing is to be done.
        if len(string) <= length:
            return [string]
        items = string.split(' ')
        strings = ['']
        for item in items:
            current_length = len(strings[-1])
            item_length = len(item)
            if current_length + item_length + 1 > length:
                strings.append('')
            strings[-1] = strings[-1] + ' ' + item
        return strings

    def parseTimes(self, time):
        """
        Parses the start- and endtimes. Possible values are any string that
        obspy.core.UTCDateTime can parse or $today$ which represents the
        current day. Whole days can be added or removed from today, e.g.
        $today - 7$ represents today one week ago.
        """
        if '$' in time:
            # Remove the $ symbols.
            time = time.strip()[1:-1]
            items = time.split(' ')
            today = UTCDateTime()
            today = UTCDateTime(today.year, today.month, today.day)
            if len(items) == 1:
                return today
            if len(items) == 3:
                if items[1] == '-':
                    today = today - (int(items[2]) * 86400)
                elif items[1] == '+':
                    today = today + (int(items[2]) * 86400)
                else:
                    msg = 'Error in the times of the configuration file.'
                    raise Error(msg)
                return today
            msg = 'Error in the times of the configuration file.'
            raise Error(msg)
        return UTCDateTime(time)
