# -*- coding: utf-8 -*-

from pkg_resources import iter_entry_points
from sqlalchemy import ForeignKey, Column, Integer, Text, DateTime, Float, \
    String, create_engine, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relation, backref
from sqlalchemy.orm.session import sessionmaker


#waveform_tab = Table(WAVEFORM_TABLE, metadata,
#  
#    Column('starttime', DateTime, nullable=False, index=True),
#    Column('endtime', DateTime, nullable=False, index=True),
#    Column('calib', Float, nullable=False),
#    Column('sampling_rate', Float, nullable=False),
#    Column('npts', Integer, nullable=False),
##    Column('DQ_gaps', Integer, nullable=True),
##    Column('DQ_overlaps', Integer, nullable=True),
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
#    useexisting=True,
#)

#Index('idx_' + WAVEFORM_TABLE + '_net_sta_cha',
#      waveform_tab.c.network, waveform_tab.c.station,
#      waveform_tab.c.location, waveform_tab.c.channel)



Base = declarative_base()


class WaveformPath(Base):
    __tablename__ = 'waveform_paths'
    id = Column(Integer, primary_key=True)
    path = Column(String, nullable=False, index=True)
    archived = Column(Boolean, default=False)

    def __init__(self, data={}):
        self.path = data.get('path')

    def __repr__(self):
        return "<WaveformPath('%s')>" % self.path


class WaveformFile(Base):
    __tablename__ = 'waveform_files'
    id = Column(Integer, primary_key=True)
    file = Column(String, nullable=False, index=True)
    size = Column(Integer, nullable=False)
    mtime = Column(Integer, nullable=False)
    path_id = Column(Integer, ForeignKey('waveform_paths.id'))

    path = relation(WaveformPath, backref=backref('files', order_by=id))

    def __init__(self, data={}):
        self.file = data.get('file')
        self.size = data.get('size')
        self.mtime = int(data.get('mtime'))
        self.format = data.get('format')

    def __repr__(self):
        return "<WaveformFile('%s')>" % self.file


class WaveformChannel(Base):
    __tablename__ = 'waveform_channels'
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('waveform_files.id'))
    network = Column(String(2), nullable=False, index=True)
    station = Column(String(5), nullable=False, index=True)
    location = Column(String(2), nullable=False, index=True)
    channel = Column(String(3), nullable=False, index=True)
    starttime = Column(DateTime, nullable=False, index=True)
    endtime = Column(DateTime, nullable=False, index=True)
    calib = Column(Float, nullable=False)
    sampling_rate = Column(Float, nullable=False)
    npts = Column(Integer, nullable=False)

    file = relation(WaveformFile, backref=backref('channels', order_by=id))

    def __init__(self, data={}):
        self.network = data.get('network')
        self.station = data.get('station')
        self.location = data.get('location')
        self.channel = data.get('channel')
        self.starttime = data.get('starttime')
        self.endtime = data.get('endtime')
        self.calib = data.get('calib')
        self.npts = data.get('npts')
        self.sampling_rate = data.get('sampling_rate')

    def __repr__(self):
        return "<WaveformChannel('%s')>" % self.channel




