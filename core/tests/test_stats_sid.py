# -*- coding: utf-8 -*-
"""
Tests for the FDSN Source Identifier support on
:class:`~obspy.core.trace.Stats` and :class:`~obspy.core.trace.Trace` (Stage A:
derived ``band`` / ``source`` / ``subsource`` / ``sid`` attributes over the
canonical ``channel`` field).
"""
import pickle

import numpy as np
import pytest

from obspy import Trace, Stream
from obspy.core.trace import (
    Stats, _parse_channel_codes, _build_channel_code, _parse_sid)


class TestChannelCodeHelpers:
    """The pure decomposition/rebuild helpers."""

    @pytest.mark.parametrize("channel,expected", [
        ("EHZ", ("E", "H", "Z")),
        ("BHZ", ("B", "H", "Z")),
        ("G_SR_D", ("G", "SR", "D")),      # extended, multi-char source
        ("_H_Z", ("", "H", "Z")),          # extended, empty band
        ("B_HH_ZZ", ("B", "HH", "ZZ")),    # extended, multi-char source+sub
        ("", ("", "", "")),                # empty channel
        ("HZ", ("", "HZ", "")),            # legacy 2-char -> whole to source
        ("Z", ("", "Z", "")),              # legacy 1-char
    ])
    def test_parse_channel_codes(self, channel, expected):
        assert _parse_channel_codes(channel) == expected

    @pytest.mark.parametrize("codes,expected", [
        (("E", "H", "Z"), "EHZ"),
        (("G", "SR", "D"), "G_SR_D"),
        (("", "H", "Z"), "_H_Z"),          # empty band -> extended form
        (("B", "HH", "ZZ"), "B_HH_ZZ"),
        (("", "", ""), ""),                # all empty -> empty channel
    ])
    def test_build_channel_code(self, codes, expected):
        assert _build_channel_code(*codes) == expected

    @pytest.mark.parametrize("channel", [
        "EHZ", "BHZ", "G_SR_D", "_H_Z", "B_HH_ZZ", "",
    ])
    def test_channel_roundtrip(self, channel):
        """channel->codes->channel is identity for SEED/extended forms."""
        band, source, subsource = _parse_channel_codes(channel)
        assert _build_channel_code(band, source, subsource) == channel

    def test_parse_sid_basic(self):
        assert _parse_sid("FDSN:BW_MANZ__E_H_Z") == ("BW", "MANZ", "", "EHZ")

    def test_parse_sid_without_prefix(self):
        assert _parse_sid("IU_ANMO_00_B_H_Z") == ("IU", "ANMO", "00", "BHZ")

    def test_parse_sid_extended(self):
        assert _parse_sid("FDSN:XX_LONGSTA_00_B_HH_ZZ") == (
            "XX", "LONGSTA", "00", "B_HH_ZZ")

    @pytest.mark.parametrize("bad", ["not a sid", "FDSN:too_few_fields", ""])
    def test_parse_sid_invalid_raises(self, bad):
        with pytest.raises(ValueError):
            _parse_sid(bad)


class TestStatsSID:
    """The derived attributes on Stats/Trace."""

    def test_band_source_subsource_read(self):
        stats = Stats({"network": "BW", "station": "MANZ", "channel": "EHZ"})
        assert stats.band == "E"
        assert stats.source == "H"
        assert stats.subsource == "Z"

    def test_component_still_works(self):
        stats = Stats({"channel": "EHZ"})
        assert stats.component == "Z"
        stats.component = "N"
        assert stats.channel == "EHN"

    def test_set_band_source_subsource_rebuilds_channel(self):
        stats = Stats({"channel": "EHZ"})
        stats.source = "N"
        assert stats.channel == "ENZ"
        stats.band = "B"
        assert stats.channel == "BNZ"
        stats.subsource = "1"
        assert stats.channel == "BN1"

    def test_sid_read(self):
        stats = Stats({"network": "BW", "station": "MANZ",
                       "location": "", "channel": "EHZ"})
        assert stats.sid == "FDSN:BW_MANZ__E_H_Z"

    def test_sid_set_populates_nslc(self):
        stats = Stats()
        stats.sid = "FDSN:IU_ANMO_00_B_H_Z"
        assert stats.network == "IU"
        assert stats.station == "ANMO"
        assert stats.location == "00"
        assert stats.channel == "BHZ"

    def test_sid_set_extended_lossless(self):
        stats = Stats()
        stats.sid = "FDSN:XX_LONGSTATION_00_B_HH_ZZ"
        assert stats.station == "LONGSTATION"
        assert stats.channel == "B_HH_ZZ"
        assert stats.source == "HH"
        assert stats.sid == "FDSN:XX_LONGSTATION_00_B_HH_ZZ"

    def test_sid_empty_band_roundtrip(self):
        stats = Stats()
        stats.sid = "FDSN:XX_STA___H_Z"  # band empty
        assert stats.band == ""
        assert stats.source == "H"
        assert stats.subsource == "Z"
        assert stats.channel == "_H_Z"
        assert stats.sid == "FDSN:XX_STA___H_Z"

    def test_sid_invalid_raises(self):
        stats = Stats()
        with pytest.raises(ValueError):
            stats.sid = "garbage"

    def test_defaults_have_empty_codes(self):
        stats = Stats()
        assert stats.band == ""
        assert stats.source == ""
        assert stats.subsource == ""
        assert stats.sid == "FDSN:_____"

    def test_repr_unchanged(self):
        """Virtual attributes must not appear in the default Stats repr."""
        stats = Stats({"network": "BW", "station": "MANZ", "channel": "EHZ"})
        text = str(stats)
        assert "band" not in text
        assert "subsource" not in text
        assert "network" in text and "channel" in text

    def test_attributes_present_from_both_constructions(self):
        """The Stage A guarantee: every code readable whether built from an
        NSLC header or from a SID, with no AttributeError."""
        from_nslc = Stats({"network": "BW", "station": "MANZ",
                           "channel": "EHZ"})
        from_sid = Stats()
        from_sid.sid = "FDSN:BW_MANZ__E_H_Z"
        for stats in (from_nslc, from_sid):
            # none of these should raise
            _ = (stats.network, stats.station, stats.location, stats.channel,
                 stats.band, stats.source, stats.subsource, stats.component,
                 stats.sid)
        assert from_nslc.sid == from_sid.sid


class TestTraceSID:

    def test_trace_sid_property(self):
        tr = Trace(np.zeros(3), {"network": "BW", "station": "MANZ",
                                 "channel": "EHZ"})
        assert tr.sid == "FDSN:BW_MANZ__E_H_Z"
        assert tr.get_sid() == tr.sid

    def test_trace_id_unchanged(self):
        tr = Trace(np.zeros(3), {"network": "BW", "station": "MANZ",
                                 "channel": "EHZ"})
        assert tr.id == "BW.MANZ..EHZ"

    def test_pickle_preserves_type_and_codes(self):
        tr = Trace(np.zeros(3))
        tr.stats.sid = "FDSN:XX_LONGSTATION_00_B_HH_ZZ"
        tr2 = pickle.loads(pickle.dumps(tr))
        assert type(tr2.stats) is Stats
        assert tr2.stats.sid == tr.stats.sid
        assert tr2.stats.channel == "B_HH_ZZ"


class TestStreamSelectSID:

    def _stream(self):
        st = Stream([
            Trace(np.zeros(3), {"network": "BW", "station": "MANZ",
                                "channel": "EHZ"}),
            Trace(np.zeros(3), {"network": "BW", "station": "MANZ",
                                "channel": "EHN"}),
            Trace(np.zeros(3), {"network": "IU", "station": "ANMO",
                                "location": "00", "channel": "BHZ"}),
        ])
        tr = Trace(np.zeros(3))
        tr.stats.sid = "FDSN:XX_LONGSTA_00_B_HH_ZZ"
        st.append(tr)
        return st

    def test_select_band(self):
        assert len(self._stream().select(band="E")) == 2

    def test_select_source(self):
        assert len(self._stream().select(source="H")) == 3

    def test_select_subsource(self):
        assert len(self._stream().select(subsource="Z")) == 2

    def test_select_source_extended(self):
        assert len(self._stream().select(source="HH")) == 1

    def test_select_sid_exact(self):
        st = self._stream()
        assert len(st.select(sid="FDSN:IU_ANMO_00_B_H_Z")) == 1

    def test_select_sid_wildcard(self):
        assert len(self._stream().select(sid="FDSN:BW_*")) == 2

    def test_existing_channel_select_unaffected(self):
        st = self._stream()
        assert len(st.select(channel="EHZ")) == 1
        assert len(st.select(id="BW.MANZ..EHZ")) == 1
