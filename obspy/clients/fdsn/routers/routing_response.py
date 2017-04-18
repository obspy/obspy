#!/usr/bin/env python
# -*- coding: utf-8 -*-
class RoutingResponse(object):
    '''base for all routed responses'''
    def __init__(self, code):
        self.code = code
        self.request_lines = []

    def __len__(self):
        return len(self.request_lines)

    def __str__(self):
        if len(self) != 1:
            line_or_lines = " lines"
        else:
            line_or_lines = " line"
        return self.code + ", with " + str(len(self)) + line_or_lines

    def add_request_line(self, request_line):
        '''override this'''
        self.request_lines.append(request_line)



