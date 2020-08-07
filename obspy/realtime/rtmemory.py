# -*- coding: utf-8 -*-
"""
Module for handling ObsPy RtMemory objects.

:copyright:
    The ObsPy Development Team (devs@obspy.org) & Anthony Lomax
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import numpy as np


class RtMemory:
    """
    Real time memory class.
    """
    def __init__(self):
        self.initialized = False

    def initialize(self, data_type, length_input, length_output,
                   input_initial_value=0, output_initial_value=0):
        """
        Create and initialize input and output arrays for this RtMemory object.

        :type data_type: numpy.dtype
        :param data_type: Desired array data-type.
        :type length_input: int
        :param length_input: length of the input memory array.
        :type length_output: int
        :param length_output: length of the output memory array.
        :type input_initial_value: float, optional
        :param input_initial_value: Initialization value for the input
            memory array (default is 1.0).
        :type output_initial_value: float, optional
        :param output_initial_value: Initialization value for the output
            memory array (default is 1.0).
        """
        self.input = np.empty(length_input, data_type)
        self.input.fill(input_initial_value)

        self.output = np.empty(length_output, data_type)
        self.output.fill(output_initial_value)

        self.initialized = True

    def _update(self, memory_array, data):
        """
        Update specified memory array using specified number of points from
        end of specified data array.

        :type memory_array: numpy.ndarray
        :param memory_array:  Memory array (input or output) in this
            RtMemory object to update.
        :type data: numpy.ndarray
        :param data:  Data array to use for update.
        :return: NumPy :class:`~numpy.ndarray` object containing updated
            memory array (input or output).
        """
        if data.size >= np.size(memory_array):
            # data length greater than or equal to memory length
            memory_array = data[np.size(data) - np.size(memory_array):]
        else:
            # data length less than memory length
            # shift memory
            memory_array = memory_array[data.size:]
            # append data
            memory_array = np.concatenate((memory_array, data))
        return memory_array

    def update_output(self, data):
        """
        Update output memory using specified number of points from end of
        specified array.

        :type data: numpy.ndarray
        :param data:  Data array to use for update.
        """
        self.output = self._update(self.output, data)

    def update_input(self, data):
        """
        Update input memory using specified number of points from end of
        specified array.

        :type data: numpy.ndarray
        :param data:  Data array to use for update.
        """
        self.input = self._update(self.input, data)
