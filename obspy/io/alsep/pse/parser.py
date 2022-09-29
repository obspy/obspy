# -*- coding: utf-8 -*-


def parse_pse_frame_to_alsep_words(frame, is_old_format):
    alsep_words = [None] * 65
    if is_old_format is True:
        word_assign = (
            1, 2, 3,
            4, 6, 8,
            9, 10, 11,
            12, 13, 14,
            16, 18, 20,
            22, 24, 25,
            26, 27, 28,
            29, 30, 32,
            33, 34, 35,
            36, 37, 38,
            40, 41, 42,
            43, 44, 45,
            46, 48, 50,
            52, 54, 57,
            58, 59, 60,
            61, 62, 64,
        )
        for i in range(16):
            j = i * 4 + 8
            alsep_words[word_assign[i * 3]] = \
                (frame[j] << 2) | (frame[j + 1] >> 6)
            alsep_words[word_assign[i * 3 + 1]] = \
                ((frame[j + 1] & 0x1f) << 5) | ((frame[j + 2] >> 3) & 0x1f)
            alsep_words[word_assign[i * 3 + 2]] = \
                ((frame[j + 2] & 0x03) << 8) | (frame[j + 3])
    else:
        word_assign = (
            1, 2, 3,
            9, 11, 13,
            25, 27, 29,
            33, 35, 37,
            41, 43, 45,
            46, 57, 59,
            61,
        )
        for i in range(6):
            j = i * 4 + 8
            alsep_words[word_assign[i * 3]] = \
                (frame[j] << 2) | (frame[j + 1] >> 6)
            alsep_words[word_assign[i * 3 + 1]] = \
                ((frame[j + 1] & 0x1f) << 5) | ((frame[j + 2] >> 3) & 0x1f)
            alsep_words[word_assign[i * 3 + 2]] = \
                ((frame[j + 2] & 0x03) << 8) | (frame[j + 3])
        alsep_words[word_assign[18]] = (frame[32] << 2) | (frame[33] >> 6)
    return alsep_words
