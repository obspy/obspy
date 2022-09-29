# -*- coding: utf-8 -*-
"""
Assign ALSEP words to each channel
"""
from .define import FORMAT_ALSEP_PSE_OLD, FORMAT_ALSEP_WTN
from .util import interp


def assign_alsep_words(alsep_word, apollo_station, format_type, frame_count,
                       prev_values):
    data = {}
    # spz (Apollo 11,12,14,15,16) or lsg(Apollo 17)
    if format_type == FORMAT_ALSEP_PSE_OLD and apollo_station != 17:
        spz = [None] * 32
        for i in range(32):
            spz[i] = alsep_word[(i + 1) * 2]
        if prev_values is None:
            spz[0] = spz[1]
        else:
            spz[0] = interp(prev_values[0], prev_values[1], spz[1], spz[2])
        if apollo_station == 15:
            spz[11] = interp(spz[9], spz[10], spz[12], spz[13])
        if apollo_station != 14:
            spz[22] = interp(spz[20], spz[21], spz[23], spz[24])
        spz[27] = interp(spz[25], spz[26], spz[28], spz[29])
        data['spz'] = spz
    elif apollo_station == 17:
        lsg = [None] * 32
        for i in range(32):
            lsg[i] = 511 - alsep_word[(i + 1) * 2]
        if prev_values is None:
            lsg[0] = lsg[1]
        else:
            lsg[0] = interp(prev_values[0], prev_values[1], lsg[1], lsg[2])
        data['lsg'] = lsg
        data['lsg_tide'] = [511 - alsep_word[25]]
        data['lsg_free'] = [511 - alsep_word[27]]
        data['lsg_temp'] = [511 - alsep_word[29]]
    # lpx/lpy/lpz
    lpx = [None] * 4
    lpy = [None] * 4
    lpz = [None] * 4
    for i in range(4):
        lpx[i] = alsep_word[16 * i + 9]
        lpy[i] = alsep_word[16 * i + 11]
        lpz[i] = alsep_word[16 * i + 13]
    data['lpx'] = lpx
    data['lpy'] = lpy
    data['lpz'] = lpz
    # TidalX/TidalY/TidalZ/Inst_temp
    if frame_count % 2 == 0:
        data['tdx'] = [alsep_word[35]]
        data['tdy'] = [alsep_word[37]]
    else:
        data['tdz'] = [alsep_word[35]]
        data['inst_temp'] = [alsep_word[37]]
    # LSM
    if format_type == FORMAT_ALSEP_WTN:
        if apollo_station in [12, 15, 16]:
            lsm = [None] * 6
            lsm[0] = alsep_word[17]
            lsm[1] = alsep_word[19]
            lsm[2] = alsep_word[21]
            lsm[3] = alsep_word[49]
            lsm[4] = alsep_word[51]
            lsm[5] = alsep_word[53]
            data['lsm'] = lsm
            data['lsm_status'] = [alsep_word[5]]
    # House Keeping (HK)
    data['hk'] = [alsep_word[33]]
    # Command Verification (CV)
    if apollo_station == 14:
        data['cv'] = [(alsep_word[5] >> 1)]
    else:
        data['cv'] = [(alsep_word[46] >> 1)]

    return data
