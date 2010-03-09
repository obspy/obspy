# -*- coding: utf-8 -*-

from sqlalchemy import ForeignKey, Column, Integer, DateTime, Float, String, \
    Binary, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relation
from obspy.core import Trace, UTCDateTime
import numpy as np

##    Column('DQ_amplifier_saturation', Integer, nullable=True),
##    Column('DQ_digitizer_clipping', Integer, nullable=True),
##    Column('DQ_spikes', Integer, nullable=True),
##    Column('DQ_glitches', Integer, nullable=True),
##    Column('DQ_missing_or_padded_data', Integer, nullable=True),
##    Column('DQ_telemetry_synchronization', Integer, nullable=True),
##    Column('DQ_digital_filter_charging', Integer, nullable=True),
##    Column('DQ_questionable_time_tag', Integer, nullable=True),
##    Column('TQ_min', Numeric, nullable=True),
##    Column('TQ_avg', Numeric, nullable=True),
##    Column('TQ_max', Numeric, nullable=True),
##    Column('TQ_lq', Numeric, nullable=True),
##    Column('TQ_median', Numeric, nullable=True),
##    Column('TQ_uq', Numeric, nullable=True),



Base = declarative_base()


class WaveformPath(Base):
    __tablename__ = 'default_waveform_paths'
    id = Column(Integer, primary_key=True)
    path = Column(String, nullable=False, index=True)
    archived = Column(Boolean, default=False)

    files = relation("WaveformFile", order_by="WaveformFile.id",
                     backref="path")

    def __init__(self, data={}):
        self.path = data.get('path')

    def __repr__(self):
        return "<WaveformPath('%s')>" % self.path


class WaveformFile(Base):
    __tablename__ = 'default_waveform_files'
    id = Column(Integer, primary_key=True)
    file = Column(String, nullable=False, index=True)
    size = Column(Integer, nullable=False)
    mtime = Column(Integer, nullable=False, index=True)
    format = Column(String, nullable=False, index=True)
    path_id = Column(Integer, ForeignKey('default_waveform_paths.id'))

    channels = relation("WaveformChannel", order_by="WaveformChannel.id",
                        backref="file")

    def __init__(self, data={}):
        self.file = data.get('file')
        self.size = data.get('size')
        self.mtime = int(data.get('mtime'))
        self.format = data.get('format')

    def __repr__(self):
        return "<WaveformFile('%s')>" % self.file


class WaveformChannel(Base):
    __tablename__ = 'default_waveform_channels'
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('default_waveform_files.id'))
    network = Column(String(2), nullable=False, index=True)
    station = Column(String(5), nullable=False, index=True)
    location = Column(String(2), nullable=False, index=True)
    channel = Column(String(3), nullable=False, index=True)
    starttime = Column(DateTime, nullable=False, index=True)
    endtime = Column(DateTime, nullable=False, index=True)
    calib = Column(Float, nullable=False)
    sampling_rate = Column(Float, nullable=False)
    npts = Column(Integer, nullable=False)
    preview = Column(Binary, nullable=True)

    gaps = relation("WaveformGaps", order_by="WaveformGaps.id",
                    backref="channel")

    def __init__(self, data={}):
        self.network = data.get('network', '')
        self.station = data.get('station', '')
        self.location = data.get('location', '')
        self.channel = data.get('channel', '')
        self.starttime = data.get('starttime')
        self.endtime = data.get('endtime')
        self.calib = data.get('calib', 1.0)
        self.npts = data.get('npts', 0)
        self.sampling_rate = data.get('sampling_rate', 1.0)
        self.preview = data.get('preview', None)

    def __repr__(self):
        return "<WaveformChannel('%s')>" % self.channel

    def getPreview(self):
        tr = Trace(data=np.loads(str(self.preview)))
        tr.stats.starttime = UTCDateTime(self.starttime)
        tr.stats.delta = 60.0
        tr.stats.network = self.network
        tr.stats.station = self.station
        tr.stats.location = self.location
        tr.stats.channel = self.channel
        return tr


class WaveformGaps(Base):
    __tablename__ = 'default_waveform_gaps'
    id = Column(Integer, primary_key=True)
    channel_id = Column(Integer, ForeignKey('default_waveform_channels.id'))
    gap = Column(Boolean, nullable=False, index=True)
    starttime = Column(DateTime, nullable=False, index=True)
    endtime = Column(DateTime, nullable=False, index=True)
    samples = Column(Integer, nullable=False)

    def __init__(self, data={}):
        self.gap = data.get('gap', True)
        self.starttime = data.get('starttime')
        self.endtime = data.get('endtime')
        self.samples = data.get('samples', 0)

    def __repr__(self):
        return "<WaveformGaps('%s')>" % self.gap
