# -*- coding: utf-8 -*-
"""
Information about files/segy useful for all tests.
"""
import numpy as np


# All the files and information about them. These files will be used in
# most tests. data_sample_enc is the encoding of the data value and
# sample_size the size in bytes of these samples.
FILES = {'00001034.sgy_first_trace': {
    'endian': '<',
    'data_sample_enc': 1, 'textual_header_enc': 'ASCII',
    'sample_count': 2001, 'sample_size': 4,
    'non_normalized_samples': [
         21, 52, 74, 89, 123,
         126, 128, 132, 136, 155, 213, 221, 222, 223,
         236, 244, 258, 266, 274, 281, 285, 286, 297, 298,
         299, 300, 301, 302, 318, 335, 340, 343, 346, 353,
         362, 382, 387, 391, 393, 396, 399, 432, 434, 465,
         466, 470, 473, 474, 481, 491, 494, 495, 507, 513,
         541, 542, 555, 556, 577, 615, 616, 622, 644, 652,
         657, 668, 699, 710, 711, 717, 728, 729, 738, 750,
         754, 768, 770, 771, 774, 775, 776, 780, 806, 830,
         853, 857, 869, 878, 885, 890, 891, 892, 917, 962,
         986, 997, 998, 1018, 1023, 1038, 1059, 1068, 1073,
         1086, 1110, 1125, 1140, 1142, 1150, 1152, 1156,
         1157, 1165, 1168, 1169, 1170, 1176, 1182, 1183,
         1191, 1192, 1208, 1221, 1243, 1250, 1309, 1318,
         1328, 1360, 1410, 1412, 1414, 1416, 1439, 1440,
         1453, 1477, 1482, 1483, 1484, 1511, 1518, 1526,
         1530, 1544, 1553, 1571, 1577, 1596, 1616, 1639,
         1681, 1687, 1698, 1701, 1718, 1734, 1739, 1745,
         1758, 1786, 1796, 1807, 1810, 1825, 1858, 1864,
         1868, 1900, 1904, 1912, 1919, 1928, 1941, 1942,
         1943, 1953, 1988]},
         '1.sgy_first_trace': {
             'endian': '>', 'data_sample_enc': 2,
             'textual_header_enc': 'ASCII', 'sample_count': 8000,
             'sample_size': 4, 'non_normalized_samples': []},
         'example.y_first_trace': {
             'endian': '>', 'data_sample_enc': 3,
             'textual_header_enc': 'EBCDIC', 'sample_count': 500,
             'sample_size': 2,
             'non_normalized_samples': []},
         'ld0042_file_00018.sgy_first_trace': {
             'endian': '>',
             'data_sample_enc': 1,
             'textual_header_enc': 'EBCDIC',
             'sample_count': 2050, 'sample_size': 4,
             'non_normalized_samples': []},
         'planes.segy_first_trace': {
             'endian': '<',
             'data_sample_enc': 1,
             'textual_header_enc': 'EBCDIC',
             'sample_count': 512, 'sample_size': 4,
             'non_normalized_samples': []},
         'one_trace_year_11.sgy': {
             'endian': '>',
             'data_sample_enc': 2, 'textual_header_enc': 'ASCII',
             'sample_count': 8000, 'sample_size': 4,
             'non_normalized_samples': []},
         'one_trace_year_99.sgy': {
             'endian': '>',
             'data_sample_enc': 2, 'textual_header_enc': 'ASCII',
             'sample_count': 8000, 'sample_size': 4,
             'non_normalized_samples': []}}
# The expected NumPy dtypes for the various sample encodings.
DTYPES = {1: np.float32,
          2: np.int32,
          3: np.int16,
          5: np.float32}
