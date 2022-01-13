# -*- coding: utf-8 -*-
from obspy import UTCDateTime
from obspy.core.inventory import Comment
from obspy.core.util import (ComplexWithUncertainties, Enum,
                             FloatWithUncertainties)
from obspy.core.util.obspy_types import (FloatWithUncertaintiesAndUnit)
import pytest


class TestUtilTypes:
    """
    Test suite for obspy.core.util.base
    """
    def test_enum(self):
        """
        Tests for the enum type.
        """
        items = ["m", "s", "m/s", "m/(s*s)", "m*s", "dimensionless", "other"]
        units = Enum(items)
        # existing selections
        assert units.other == "other"
        assert units.M == "m"
        assert units['m/s'] == "m/s"
        assert units.get('m/s') == "m/s"
        assert units[0] == "m"
        assert units[-1] == "other"
        # not existing selections should fail
        with pytest.raises(Exception):
            units.__getitem__('5')
        with pytest.raises(Exception):
            units.__getattr__('xx')
        with pytest.raises(Exception):
            units.get('xx', 'default')
        with pytest.raises(Exception):
            units.__getitem__(99)
        with pytest.raises(Exception):
            units.__getitem__(-99)
        # test in operator
        assert "other" in units
        assert "ot21her" not in units
        # test typical dict methods
        assert units.values() == items
        assert units.items() == list(zip(items, items))
        assert units.keys() == items
        # call will either return correct enum label or return None
        assert units('m') == 'm'
        assert units('m/(s*s)') == 'm/(s*s)'
        assert units(5) == 'dimensionless'
        assert units(99) is None
        assert units('xxx') is None

    def _check_complex_with_u(self, c, real, r_lo, r_up, imag, i_lo, i_up):
        """
        Check for six equalities for a ComplexWithUncertainties
        """
        assert isinstance(c.real, FloatWithUncertainties)
        assert isinstance(c.imag, FloatWithUncertainties)
        assert c.real == real
        assert c.imag == imag
        assert c.real.upper_uncertainty == r_up
        assert c.real.lower_uncertainty == r_lo
        assert c.imag.upper_uncertainty == i_up
        assert c.imag.lower_uncertainty == i_lo

    def test_complex(self):
        """
        Test the ComplexWithUncertainties for proper usage
        """
        f1 = float(3.5)
        f2 = float(12)
        lu1 = 1
        uu1 = 2
        lu2 = 4.1
        uu2 = 7.2
        fu1 = FloatWithUncertainties(f1, lower_uncertainty=lu1,
                                     upper_uncertainty=uu1)
        fu2 = FloatWithUncertainties(f2, lower_uncertainty=lu2,
                                     upper_uncertainty=uu2)
        c1 = ComplexWithUncertainties()
        c2 = ComplexWithUncertainties(f1, f2)
        c3 = ComplexWithUncertainties(
            f1, f2, lower_uncertainty=complex(lu1, lu2),
            upper_uncertainty=complex(uu1, uu2))
        c4 = ComplexWithUncertainties(fu1, fu2)
        # c1 should be 0+0j with uncertainties of None
        self._check_complex_with_u(c1, 0, None, None, 0, None, None)
        # c2 should return the floats
        self._check_complex_with_u(c2, f1, None, None, f2, None, None)
        # c3 and c4 should be the same.
        self._check_complex_with_u(c3, f1, lu1, uu1, f2, lu2, uu2)
        self._check_complex_with_u(c4, f1, lu1, uu1, f2, lu2, uu2)
        assert c4.real == fu1
        assert c4.imag == fu2

    def test_floating_point_types_are_indempotent(self):
        """
        Applying the constructor multiple times should not change the values.
        """
        f = FloatWithUncertainties(1.0, lower_uncertainty=0.5,
                                   upper_uncertainty=1.5)
        assert f == 1.0
        assert f.lower_uncertainty == 0.5
        assert f.upper_uncertainty == 1.5
        f = FloatWithUncertainties(f)
        assert f == 1.0
        assert f.lower_uncertainty == 0.5
        assert f.upper_uncertainty == 1.5

        f = FloatWithUncertaintiesAndUnit(1.0, lower_uncertainty=0.5,
                                          upper_uncertainty=1.5, unit="AB")
        assert f == 1.0
        assert f.lower_uncertainty == 0.5
        assert f.upper_uncertainty == 1.5
        assert f.unit == "AB"
        f = FloatWithUncertaintiesAndUnit(f)
        assert f == 1.0
        assert f.lower_uncertainty == 0.5
        assert f.upper_uncertainty == 1.5
        assert f.unit == "AB"

    def test_comment_str(self):
        """
        Tests the __str__ method of the Comment object.
        """
        c = Comment(value='test_comment', id=9,
                    begin_effective_time=UTCDateTime(1240561632),
                    end_effective_time=UTCDateTime(1584561632), authors=[])
        cs = str(c)
        assert cs == "Comment:\ttest_comment\n" \
                     "\tBegin Effective Time:\t2009-04-24T08:27:12" \
                     ".000000Z\n" \
                     "\tEnd Effective Time:\t2020-03-18T20:00:32.000000Z\n" \
                     "\tAuthors:\t\t[]\n" \
                     "\tId:\t\t\t9"
