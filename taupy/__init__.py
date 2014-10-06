#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect, os

# Most generic way to get the data directory.
__DATA_DIR  =os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe()))), "data")
