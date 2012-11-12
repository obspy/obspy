# -*- coding: utf-8 -*-
"""
Poles, zeros, normalization and sensitivity constants for common seismometers.

PAZ objects are created from Seismic Handler FLF files situated in the
obspy.sh.filter directory.
"""
from obspy.sh.core import readFLF


PAZ_WOOD_ANDERSON_DSP = readFLF('TF_DSP_S+WOODAND.FLF')
PAZ_WOOD_ANDERSON_DSP.name = 'Wood-Anderson (Classic) / Displacement'

PAZ_WOOD_ANDERSON_VEL = readFLF('TF_VEL_S+WOODAND.FLF')
PAZ_WOOD_ANDERSON_VEL.name = 'Wood-Anderson (Classic) / Velocity'

PAZ_WOOD_ANDERSON_DSP_IASPEI = readFLF('TF_DSP_S+WOODAND_IASPEI.FLF')
PAZ_WOOD_ANDERSON_DSP_IASPEI.name = 'Wood-Anderson (IASPEI) / Displacement'

PAZ_WOOD_ANDERSON_VEL_IASPEI = readFLF('TF_VEL_S+WOODAND_IASPEI.FLF')
PAZ_WOOD_ANDERSON_DSP_IASPEI.name = 'Wood-Anderson (IASPEI) / Velocity'
