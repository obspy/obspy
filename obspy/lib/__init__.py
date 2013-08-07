#!/usr/bin/env python
# -*- coding: utf-8 -*-
import platform

# Import libtau in a platform specific way.
libtau = __import__('libtau_%s_%s_py%s' % (platform.system(),
    platform.architecture()[0], ''.join([str(i) for i in
    platform.python_version_tuple()[:2]])), globals(), locals())
