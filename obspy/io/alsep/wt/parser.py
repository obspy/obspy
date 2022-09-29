# -*- coding: utf-8 -*-


def parse_wtn_frame_to_alsep_words(frame):
    alsep_words = [None] * 65
    for i in range(21):
        j = i * 4 + 8
        alsep_words[i * 3 + 1] = (frame[j] << 2) | (frame[j + 1] >> 6)
        alsep_words[i * 3 + 2] = \
            ((frame[j + 1] & 0x1f) << 5) | ((frame[j + 2] >> 3) & 0x1f)
        alsep_words[i * 3 + 3] = ((frame[j + 2] & 0x03) << 8) | (frame[j + 3])
    alsep_words[64] = (frame[92] << 2) | (frame[93] >> 6)
    return alsep_words


def parse_wth_frame_to_geophone_data(frame):
    geophone = {1: [0] * 20, 2: [0] * 20, 3: [0] * 20, 4: [0] * 20,
                'status': [0] * 20}

    part1 = (frame[9] >> 1) & 0x1f
    part1 <<= 3
    geophone[1][0] = part1

    part2 = frame[9] & 0x01
    part2 = (part2 << 4) | ((frame[10] >> 4) & 0x0f)
    part2 <<= 3
    geophone[2][0] = part2

    part3 = frame[10] & 0x0f
    part3 = (part3 << 1) | (frame[11] >> 7)
    part3 <<= 3
    geophone[3][0] = part3

    part4 = (frame[11] >> 2) & 0x1f
    part4 <<= 3
    geophone[4][0] = part4

    geophone['status'][0] = -1

    for i in range(1, 20):
        j = (i + 2) * 4

        part1 = (frame[j] >> 1) << 1
        geophone[1][i] = part1

        part2 = frame[j] & 0x01
        part2 = (part2 << 6) | (frame[j + 1] >> 2)
        part2 <<= 1
        geophone[2][i] = part2

        part3 = frame[j + 1] & 0x03
        part3 = (part3 << 5) | (frame[j + 2] >> 3)
        part3 <<= 1
        geophone[3][i] = part3

        part4 = frame[j + 2] & 0x07
        part4 = (part4 << 4) | (frame[j + 3] >> 4)
        part4 <<= 1
        geophone[4][i] = part4

        geophone['status'][i] = (frame[j + 3] >> 2) & 0x03

    sub_frame_array = [-1, 2, 3, 1]
    geophone['sub_frame'] = sub_frame_array[geophone['status'][19]]

    return geophone
