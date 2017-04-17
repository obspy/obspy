#!/usr/bin/env python
# -*- coding: utf-8 -*-
class RoutingResponse(object):
    '''base for all routed responses'''
    def __init__(self, code):
        self.code = code
        self.request_lines = []

    def __len__(self):
        return len(self.request_lines)

    def add_request_line(self, request_line):
        raise NotImplementedError



