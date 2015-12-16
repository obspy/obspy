# -*- coding: utf-8 -*-
"""
Library name handling for ObsPy.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

import importlib
import warnings
from types import ModuleType


FUNCTION_MAPS = {
    'obspy.io.ah.core': {
        'is_AH': '_is_ah',
        'read_AH': '_read_ah',
        'read_AH1': '_read_ah1',
        'read_AH2': '_read_ah2'},
    'obspy.io.cnv.core': {
        'write_CNV': '_write_cnv'},
    'obspy.core.compatibility': {
        'frombuffer': 'from_buffer'},
    'obspy.core.event': {
        'readEvents': 'read_events'},
    'obspy.io.json': {
        'writeJSON': '_write_json'},
    'obspy.io.json.core': {
        'writeJSON': '_write_json'},
    'obspy.io.quakeml.core': {
        "isQuakeML": "_is_quakeml",
        "readQuakeML": "_read_quakeml",
        "readSeisHubEventXML": "_read_seishub_event_xml",
        "writeQuakeML": "_write_quakeml"},
    'obspy.core.stream': {
        'isPickle': '_is_pickle',
        'readPickle': '_read_pickle',
        'writePickle': '_write_pickle'},
    'obspy.geodetics': {
        "calcVincentyInverse": "calc_vincenty_inverse",
        "gps2DistAzimuth": "gps2dist_azimuth"},
    # 'obspy.core.util': {
    #     'complexifyString': 'complexify_string',
    #     'createEmptyDataChunk': 'create_empty_data_chunk',
    #     'getExampleFile': 'get_example_file',
    #     'getScriptDirName': 'get_script_dir_name',
    #     'guessDelta': 'guess_delta',
    #     'scoreatpercentile': 'score_at_percentile',
    #     'uncompressFile': 'uncompress_file'},
    # 'obspy.core.util.base': {
    #     'createEmptyDataChunk': 'create_empty_data_chunk',
    #     'getBasemapVersion': 'get_basemap_version',
    #     'getExampleFile': 'get_example_file',
    #     'getMatplotlibVersion': 'get_matplotlib_version',
    #     'getSciPyVersion': 'get_scipy_version',
    #     'getScriptDirName': 'get_script_dir_name'},
    # 'obspy.core.util.decorator': {
    #     'getExampleFile': 'get_example_file',
    #     'raiseIfMasked': 'raise_if_masked',
    #     'skipIfNoData': 'skip_if_no_data',
    #     'uncompressFile': 'uncompress_file'},
    # 'obspy.core.util.misc': {
    #     'complexifyString': 'complexify_string',
    #     'flatnotmaskedContiguous': 'flat_not_masked_contiguous',
    #     'guessDelta': 'guess_delta',
    #     'scoreatpercentile': 'score_at_percentile'},
    'obspy.io.css.core': {
        'isCSS': '_is_css',
        'readCSS': '_read_css'},
    'obspy.io.css.station': {
        'writeCSS': '_write_css'},
    'obspy.io.datamark.core': {
        'isDATAMARK': '_is_datamark',
        'readDATAMARK': '_read_datamark'},
    'obspy.db.client': {
        'mergePreviews': 'merge_previews'},
    'obspy.db.indexer': {
        'createPreview': 'create_preview'},
    'obspy.db.util': {
        'parseMappingData': 'parse_mapping_data'},
    'obspy.clients.earthworm.waveserver': {
        'getMenu': 'get_menu',
        'getNumpyType': 'get_numpy_type',
        'getSockBytes': 'get_sock_bytes',
        'getSockCharLine': 'get_sock_char_line',
        'readWaveServerV': 'read_wave_server_v',
        'sendSockReq': 'send_sock_req',
        'tracebuf2': 'TraceBuf2',
        'tracebufs2obspyStream': 'trace_bufs2obspy_stream'},
    # 'obspy.io.gse2.core': {
    #     'isGSE1': 'isGSE1',
    #     'isGSE2': 'isGSE2',
    #     'readGSE1': 'readGSE1',
    #     'readGSE2': 'readGSE2',
    #     'writeGSE2': 'writeGSE2'},
    # 'obspy.io.gse2.libgse1': {
    #     'readHeader': 'readHeader',
    #     'readIntegerData': 'readIntegerData',
    #     'verifyChecksum': 'verifyChecksum'},
    # 'obspy.io.gse2.libgse2': {
    #     'isGse2': 'isGse2',
    #     'readHeader': 'readHeader',
    #     'verifyChecksum': 'verifyChecksum',
    #     'writeHeader': 'writeHeader'},
    # 'obspy.io.gse2.paz': {
    #     'readPaz': 'readPaz'},
    # 'obspy.imaging.beachball': {
    #     'MT2Axes': 'MT2Axes',
    #     'MT2Plane': 'MT2Plane',
    #     'Pol2Cart': 'Pol2Cart',
    #     'StrikeDip': 'StrikeDip',
    #     'TDL': 'TDL',
    #     'plotDC': 'plotDC',
    #     'plotMT': 'plotMT'},
    # 'obspy.io.mseed.core': {
    #     'isMSEED': 'isMSEED',
    #     'readMSEED': 'readMSEED',
    #     'writeMSEED': 'writeMSEED'},
    # 'obspy.io.ndk.core': {
    #     'is_ndk': 'is_ndk',
    #     'read_ndk': 'read_ndk'},
    # 'obspy.clients.neic.util': {
    #     'getProperty': 'getProperty'},
    # 'obspy.io.pdas': {
    #     'isPDAS': 'isPDAS',
    #     'readPDAS': 'readPDAS'},
    # 'obspy.io.pdas.core': {
    #     'isPDAS': 'isPDAS',
    #     'readPDAS': 'readPDAS'},
    # 'obspy.io.pde.mchedr': {
    #     'isMchedr': 'isMchedr',
    #     'readMchedr': 'readMchedr'},
    # 'obspy.io.sac': {
    #     'SacIO': 'SacIO'},
    # 'obspy.io.sac.core': {
    #     'SacIO': 'SacIO',
    #     'isSAC': 'isSAC',
    #     'isSACXY': 'isSACXY',
    #     'readSAC': 'readSAC',
    #     'readSACXY': 'readSACXY',
    #     'writeSAC': 'writeSAC',
    #     'writeSACXY': 'writeSACXY'},
    # 'obspy.io.sac.sacpz': {
    #     'write_SACPZ': 'write_SACPZ'},
    # 'obspy.io.seg2.seg2': {
    #     'isSEG2': 'isSEG2',
    #     'readSEG2': 'readSEG2'},
    # 'obspy.io.segy.core': {
    #     'isSEGY': 'isSEGY',
    #     'isSU': 'isSU',
    #     'readSEGY': 'readSEGY',
    #     'readSEGYrev1': 'readSEGYrev1',
    #     'readSU': 'readSU',
    #     'readSUFile': 'readSUFile',
    #     'writeSEGY': 'writeSEGY',
    #     'writeSU': 'writeSU'},
    # 'obspy.io.segy.segy': {
    #     'readSEGY': 'readSEGY',
    #     'readSU': 'readSU'},
    # 'obspy.io.seisan.core': {
    #     'isSEISAN': 'isSEISAN',
    #     'readSEISAN': 'readSEISAN'},
    # 'obspy.io.sh.core': {
    #     'fromUTCDateTime': 'fromUTCDateTime',
    #     'isASC': 'isASC',
    #     'isQ': 'isQ',
    #     'readASC': 'readASC',
    #     'readQ': 'readQ',
    #     'writeASC': 'writeASC',
    #     'writeQ': 'writeQ'},
    # 'obspy.signal': {
    #     'arPick': 'arPick',
    #     'bandpass': 'bandpass',
    #     'bandstop': 'bandstop',
    #     'bwith': 'bwith',
    #     'carlSTATrig': 'carlSTATrig',
    #     'centroid': 'centroid',
    #     'cfrequency': 'cfrequency',
    #     'classicSTALTA': 'classicSTALTA',
    #     'classicSTALTAPy': 'classicSTALTAPy',
    #     'coincidenceTrigger': 'coincidenceTrigger',
    #     'cornFreq2Paz': 'cornFreq2Paz',
    #     'cosTaper': 'cosTaper',
    #     'delayedSTALTA': 'delayedSTALTA',
    #     'domperiod': 'domperiod',
    #     'eigval': 'eigval',
    #     'envelope': 'envelope',
    #     'estimateMagnitude': 'estimateMagnitude',
    #     'highpass': 'highpass',
    #     'instBwith': 'instBwith',
    #     'instFreq': 'instFreq',
    #     'integerDecimation': 'integerDecimation',
    #     'konnoOhmachiSmoothing': 'konnoOhmachiSmoothing',
    #     'logcep': 'logcep',
    #     'lowpass': 'lowpass',
    #     'lowpassFIR': 'lowpassFIR',
    #     'normEnvelope': 'normEnvelope',
    #     'pazToFreqResp': 'pazToFreqResp',
    #     'pkBaer': 'pkBaer',
    #     'psd': 'psd',
    #     'recSTALTA': 'recSTALTA',
    #     'recSTALTAPy': 'recSTALTAPy',
    #     'remezFIR': 'remezFIR',
    #     'rotate_LQT_ZNE': 'rotate_LQT_ZNE',
    #     'rotate_NE_RT': 'rotate_NE_RT',
    #     'rotate_RT_NE': 'rotate_RT_NE',
    #     'rotate_ZNE_LQT': 'rotate_ZNE_LQT',
    #     'seisSim': 'seisSim',
    #     'sonogram': 'sonogram',
    #     'specInv': 'specInv',
    #     'triggerOnset': 'triggerOnset',
    #     'utlGeoKm': 'utlGeoKm',
    #     'utlLonLat': 'utlLonLat',
    #     'xcorr': 'xcorr',
    #     'xcorrPickCorrection': 'xcorrPickCorrection',
    #     'xcorr_3C': 'xcorr_3C',
    #     'zDetect': 'zDetect'},
    # 'obspy.signal.calibration': {
    #     'relcalstack': 'relcalstack'},
    # 'obspy.signal.cpxtrace': {
    #     'instBwith': 'instBwith',
    #     'instFreq': 'instFreq',
    #     'normEnvelope': 'normEnvelope'},
    # 'obspy.signal.cross_correlation': {
    #     'xcorrPickCorrection': 'xcorrPickCorrection'},
    # 'obspy.signal.freqattributes': {
    #     'bwith': 'bwith',
    #     'cfrequency': 'cfrequency',
    #     'cfrequency_unwindowed': 'cfrequency_unwindowed',
    #     'domperiod': 'domperiod',
    #     'logbankm': 'logbankm',
    #     'logcep': 'logcep',
    #     'mper': 'mper',
    #     'pgm': 'pgm',
    #     'seisSim': 'seisSim'},
    'obspy.signal.konnoohmachismoothing': {
        'calculateSmoothingMatrix': 'calculate_smoothing_matrix',
        'konnoOhmachiSmoothing': 'konno_ohmachi_smoothing',
        'konnoOhmachiSmoothingWindow': 'konno_ohmachi_smoothing_window'},
    # 'obspy.signal.trigger': {
    #     'arPick': 'arPick',
    #     'carlSTATrig': 'carlSTATrig',
    #     'classicSTALTA': 'classicSTALTA',
    #     'classicSTALTAPy': 'classicSTALTAPy',
    #     'coincidenceTrigger': 'coincidenceTrigger',
    #     'delayedSTALTA': 'delayedSTALTA',
    #     'pkBaer': 'pkBaer',
    #     'plotTrigger': 'plotTrigger',
    #     'recSTALTA': 'recSTALTA',
    #     'recSTALTAPy': 'recSTALTAPy',
    #     'triggerOnset': 'triggerOnset',
    #     'zDetect': 'zDetect'},
    # 'obspy.signal.util': {
    #     'nearestPow2': 'nearestPow2',
    #     'prevpow2': 'prevpow2'},
    # 'obspy.io.stationxml.core': {
    #     'is_StationXML': 'is_StationXML',
    #     'read_StationXML': 'read_StationXML',
    #     'validate_StationXML': 'validate_StationXML',
    #     'write_StationXML': 'write_StationXML'},
    # 'obspy.io.wav.core': {
    #     'isWAV': 'isWAV',
    #     'readWAV': 'readWAV',
    #     'writeWAV': 'writeWAV'},
    # 'obspy.io.xseed.utils': {
    #     'Blockette34Lookup': 'Blockette34Lookup',
    #     'DateTime2String': 'DateTime2String',
    #     'LookupCode': 'LookupCode',
    #     'compareSEED': 'compareSEED',
    #     'formatRESP': 'formatRESP',
    #     'getXPath': 'getXPath',
    #     'setXPath': 'setXPath',
    #     'toString': 'toString',
    #     'toTag': 'toTag',
    #     'uniqueList': 'uniqueList'},
    # 'obspy.io.y.core': {
    #     'isY': 'isY',
    #     'readY': 'readY'},
    # 'obspy.io.zmap.core': {
    #     'isZmap': 'isZmap',
    #     'readZmap': 'readZmap',
    #     'writeZmap': 'writeZmap'}
    }
FUNCTION_RENAME_MSG = (
    "Function '{old}' in module '{mod}' is deprecated and was renamed to "
    "'{new}'. Please use this name instead, the old name will stop working "
    "in the next major ObsPy release.")


class ObsPyDeprecationWarning(UserWarning):
    """
    Make a custom deprecation warning as deprecation warnings or hidden by
    default since Python 2.7 and 3.2 and we really want users to notice these.
    """
    pass


class DynamicAttributeImportRerouteModule(ModuleType):
    """
    Class assisting in dynamically rerouting attribute access like imports.

    This essentially makes

    >>> import obspy  # doctest: +SKIP
    >>> obspy.station.Inventory  # doctest: +SKIP

    work. Remove this once 0.11 has been released!
    """
    def __init__(self, name, doc, locs, import_map, function_map=None):
        super(DynamicAttributeImportRerouteModule, self).__init__(name=name)
        self.import_map = import_map
        self.function_map = function_map
        # Keep the metadata of the module.
        self.__dict__.update(locs)

    def __getattr__(self, name):
        # Functions, and not modules.
        if self.function_map and name in self.function_map:
            new_name = self.function_map[name].split(".")
            module = importlib.import_module(".".join(new_name[:-1]))
            warnings.warn("Function '%s' is deprecated and will stop working "
                          "with the next ObsPy version. Please use '%s' "
                          "instead." % (self.__name__ + "." + name,
                                        self.function_map[name]),
                          ObsPyDeprecationWarning)
            return getattr(module, new_name[-1])

        try:
            real_module_name = self.import_map[name]
        except:
            raise AttributeError
        warnings.warn("Module '%s' is deprecated and will stop working with "
                      "the next ObsPy version. Please import module "
                      "'%s'instead." % (self.__name__ + "." + name,
                                        self.import_map[name]),
                      ObsPyDeprecationWarning)
        return importlib.import_module(real_module_name)


def _create_deprecated_function(old_name, new_name, module_name, func):
    from obspy.core.util.decorator import deprecated
    msg = FUNCTION_RENAME_MSG.format(old=old_name, new=new_name,
                                     mod=module_name)

    @deprecated(msg)
    def deprecated_func(*args, **kwargs):
        return func(*args, **kwargs)
    return deprecated_func


def _add_deprecated_functions():
    for mod_name, mapping in FUNCTION_MAPS.items():
        mod = importlib.import_module(mod_name)
        for old_name, new_name in mapping.items():
            func = getattr(mod, new_name)
            deprecated_func = _create_deprecated_function(
                old_name, new_name, mod_name, func)
            setattr(mod, old_name, deprecated_func)
