# -*- coding: utf-8 -*-
SIZE_ALSEP_PSE_RECORD = 19456
SIZE_ALSEP_RECORD_HEADER = 16
SIZE_ALSEP_PSE_FRAME_OLD = 72
SIZE_ALSEP_PSE_FRAME_NEW = 36

FORMAT_ALSEP_PSE_OLD = 0
FORMAT_ALSEP_PSE_NEW = 1
FORMAT_ALSEP_WTN = 2
FORMAT_ALSEP_WTH = 3

DATA_RATE_530_BPS = 0
DATA_RATE_1060_BPS = 1

channels = {
    # The channel names should be determined as below:
    #
    # https://ds.iris.edu/ds/nodes/dmc/data/formats/seed-channel-naming/
    # https://www.geonet.org.nz/data/supplementary/channels
    # 'spz': {'channel': 'SHZ', 'sampling_rate': 53.0},
    # 'lpx': {'channel': 'MH1', 'sampling_rate': 6.625},
    # 'lpy': {'channel': 'MH2', 'sampling_rate': 6.625},
    # 'lpz': {'channel': 'MH3', 'sampling_rate': 6.625},
    # 'tdx': {'channel': 'LH1', 'sampling_rate': 0.828125},
    # 'tdy': {'channel': 'LH2', 'sampling_rate': 0.828125},
    # 'tdz': {'channel': 'LH3', 'sampling_rate': 0.828125},
    # 'inst_temp': {'channel': 'LK2', 'sampling_rate': 0.828125},
    # 'lsm': {'channel': 'FMF', 'sampling_rate': 9.9375},
    # 'lsm_status': {'channel': 'YLF', 'sampling_rate': 1.65625},
    # 'lsg': {'channel': 'SPZ', 'sampling_rate': 53.0},
    # ...
    #
    # Defined ALSEP specific channel names for clarity
    'spz': {'channel': 'SPZ', 'sampling_rate': 53.0},
    'lpx': {'channel': 'LPX', 'sampling_rate': 6.625},
    'lpy': {'channel': 'LPY', 'sampling_rate': 6.625},
    'lpz': {'channel': 'LPZ', 'sampling_rate': 6.625},
    'tdx': {'channel': 'TDX', 'sampling_rate': 0.828125},
    'tdy': {'channel': 'TDY', 'sampling_rate': 0.828125},
    'tdz': {'channel': 'TDZ', 'sampling_rate': 0.828125},
    'inst_temp': {'channel': 'TMP', 'sampling_rate': 0.828125},
    'lsm': {'channel': 'LSM', 'sampling_rate': 9.9375},
    'lsm_status': {'channel': 'LSS', 'sampling_rate': 1.65625},
    'lsg': {'channel': 'LSG', 'sampling_rate': 53.0},
    'lsg_tide': {'channel': 'LGT', 'sampling_rate': 1.65625},
    'lsg_free': {'channel': 'LGF', 'sampling_rate': 1.65625},
    'lsg_temp': {'channel': 'LGP', 'sampling_rate': 1.65625},
    'hk': {'channel': 'HK_', 'sampling_rate': 1.65625},
    'cv': {'channel': 'CV_', 'sampling_rate': 1.65625},
    # LUNAR SEISMIC PROFILING (LSP) EXPERIMENT
    #   3.5333 KBPS normal rate, NRZ-C PCM
    #   1.06 KBPS low rate, NRZ-C PCM
    #   1.963 frames per second normal rate
    #   0.588 frames per second low rate
    #   3 subframes per frame
    #   20 words per subframe
    #   30 bits per word
    # sampling rate is as below:
    #   20 data samples per frame in each channel.
    #   For normal rate:
    #   20 [samples/frame] x 1.963 [frames/second] = 39.26 [samples/second]
    #   For low rate:
    #   20 [samples/frame] x 0.588 [frames/second] = 11.76 [samples/second]
    'geo1': {'channel': 'GP1', 'sampling_rate': 39.26},
    'geo2': {'channel': 'GP2', 'sampling_rate': 39.26},
    'geo3': {'channel': 'GP3', 'sampling_rate': 39.26},
    'geo4': {'channel': 'GP4', 'sampling_rate': 39.26},
}

package_id_to_apollo_station = {
    1: 12,
    2: 15,
    3: 16,
    4: 14,
    5: 17,
}
