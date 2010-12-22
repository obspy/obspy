# -*- coding: utf-8 -*-

from lxml.etree import Element, SubElement, parse as xmlparse
from lxml.etree import tostring


class ChannelListParser(object):
    """
    Writes and reads the xml file with the channel lists.
    """
    def __init__(self, env):
        """
        Standard init. It will automatically read the channel_lists file and
        write its contents to the environment.
        """
        self.env = env
        # Init a dictionary which will hold all channel_lists.
        self.channel_lists = {}
        self.file = self.env.channel_lists_xml
        self.readFile()
        # Write it to the environment.
        self.env.channel_lists = self.channel_lists

    def readFile(self):
        """
        Reads a file and writes everything it finds to self.channel_lists.
        """
        # Parse the file. Return if it could not be read.
        try:
            xml = xmlparse(self.file)
        except:
            return
        # Add some error handling.
        try:
            root = xml.getroot()
        except:
            return
        # This is the last check. Otherwise it is just assumed to be a correct
        # xml file.
        if root.tag != 'channel_lists':
            return
        # Get the lists.
        lists = root.getchildren()
        # If no lists are there return.
        if len(lists) == 0:
            return
        # Loop over each channel.
        for channel in lists:
            channels = channel.getchildren()
            # If there are no channels in the list remove it.
            if len(channels) == 0:
                continue
            list_name = channel.attrib['name']
            channel_list = []
            for item in channels:
                channel_list.append(item.text)
            # Now that all information if given write it to the dictionary.
            _i = 1
            # Limit the while loop to 100 for savety reasons.
            while _i < 100:
                if _i == 1:
                    cur_name = list_name
                else:
                    cur_name = '%s_%i' % (list_name, _i)
                # Check if the name is alreadt in the dictionary, otherwise
                # increment the number.
                if cur_name in self.channel_lists:
                    _i += 1
                    continue
                self.channel_lists[cur_name] = channel_list
                break

    def writeFile(self):
        """
        Writes the contents of self.env.channel_lists to a file.
        """
        # Init the xml file.
        self.createEmptyXML()
        lists = self.env.channel_lists.keys()
        lists.sort()
        # Add every valid item to the the xml object.
        for item in lists:
            self.addList(item, self.env.channel_lists[item])
        # Write it.
        f = open(self.file, 'w')
        f.write(tostring(self.root, pretty_print=True, xml_declaration=True,
                       encoding='utf-8'))
        f.close()

    def addList(self, list_name, channel_list):
        """
        Add list with name list_name and a list with channels to the class xml
        object.
        """
        list_tag = SubElement(self.root, 'list', name=list_name)
        # For each name in the list add a channel.
        for channel in channel_list:
            sub = SubElement(list_tag, 'channel')
            sub.text = channel

    def createEmptyXML(self):
        """
        Inits XML file.
        """
        # Create the root element.
        self.root = Element('channel_lists')
