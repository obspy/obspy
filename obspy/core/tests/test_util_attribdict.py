# -*- coding: utf-8 -*-
import pickle
import warnings

from obspy.core import AttribDict
import pytest


class DefaultTestAttribDict(AttribDict):
    defaults = {'test': 1}


class TestAttribDict:
    """
    Test suite for obspy.core.util.attribdict
    """

    def test_pop(self):
        """
        Tests pop method of AttribDict class.
        """
        ad = AttribDict()
        ad.test = 1
        ad['test2'] = 'test'
        # removing via pop
        temp = ad.pop('test')
        assert temp == 1
        assert not ('test' in ad)
        assert 'test2' in ad
        assert not ('test' in ad.__dict__)
        assert 'test2' in ad.__dict__
        assert not hasattr(ad, 'test')
        assert hasattr(ad, 'test2')
        # using pop() for not existing element raises a KeyError
        with pytest.raises(KeyError):
            ad.pop('test')

    def test_popitem(self):
        """
        Tests pop method of AttribDict class.
        """
        ad = AttribDict()
        ad['test2'] = 'test'
        # removing via popitem
        temp = ad.popitem()
        assert temp == ('test2', 'test')
        assert not ('test2' in ad)
        assert not ('test2' in ad.__dict__)
        assert not hasattr(ad, 'test2')
        # popitem for empty AttribDict raises a KeyError
        with pytest.raises(KeyError):
            ad.popitem()

    def test_delete(self):
        """
        Tests delete method of AttribDict class.
        """
        ad = AttribDict()
        ad.test = 1
        ad['test2'] = 'test'
        # deleting test using dictionary
        del ad['test']
        assert not ('test' in ad)
        assert 'test2' in ad
        assert not ('test' in ad.__dict__)
        assert 'test2' in ad.__dict__
        assert not hasattr(ad, 'test')
        assert hasattr(ad, 'test2')
        # deleting test2 using attribute
        del ad.test2
        assert not ('test2' in ad)
        assert not ('test2' in ad.__dict__)
        assert not hasattr(ad, 'test2')

    def test_init(self):
        """
        Tests initialization of AttribDict class.
        """
        ad = AttribDict({'test': 'NEW'})
        assert ad['test'] == 'NEW'
        assert ad.test == 'NEW'
        assert ad.get('test') == 'NEW'
        assert ad.__getattr__('test') == 'NEW'
        assert ad.__getitem__('test') == 'NEW'
        assert ad.__dict__['test'] == 'NEW'
        assert ad.__dict__.get('test') == 'NEW'
        assert 'test' in ad
        assert 'test' in ad.__dict__

    def test_setitem(self):
        """
        Tests __setitem__ method of AttribDict class.
        """
        # 1
        ad = AttribDict()
        ad['test'] = 'NEW'
        assert ad['test'] == 'NEW'
        assert ad.test == 'NEW'
        assert ad.get('test') == 'NEW'
        assert ad.__getattr__('test') == 'NEW'
        assert ad.__getitem__('test') == 'NEW'
        assert ad.__dict__['test'] == 'NEW'
        assert ad.__dict__.get('test') == 'NEW'
        assert 'test' in ad
        assert 'test' in ad.__dict__
        # 2
        ad = AttribDict()
        ad.__setitem__('test', 'NEW')
        assert ad['test'] == 'NEW'
        assert ad.test == 'NEW'
        assert ad.get('test') == 'NEW'
        assert ad.__getattr__('test') == 'NEW'
        assert ad.__getitem__('test') == 'NEW'
        assert ad.__dict__['test'] == 'NEW'
        assert ad.__dict__.get('test') == 'NEW'
        assert 'test' in ad
        assert 'test' in ad.__dict__

    def test_setattr(self):
        """
        Tests __setattr__ method of AttribDict class.
        """
        # 1
        ad = AttribDict()
        ad.test = 'NEW'
        assert ad['test'] == 'NEW'
        assert ad.test == 'NEW'
        assert ad.get('test') == 'NEW'
        assert ad.__getattr__('test') == 'NEW'
        assert ad.__getitem__('test') == 'NEW'
        assert ad.__dict__['test'] == 'NEW'
        assert ad.__dict__.get('test') == 'NEW'
        assert 'test' in ad
        assert 'test' in ad.__dict__
        # 2
        ad = AttribDict()
        ad.__setattr__('test', 'NEW')
        assert ad['test'] == 'NEW'
        assert ad.test == 'NEW'
        assert ad.get('test') == 'NEW'
        assert ad.__getattr__('test') == 'NEW'
        assert ad.__getitem__('test') == 'NEW'
        assert ad.__dict__['test'] == 'NEW'
        assert ad.__dict__.get('test') == 'NEW'
        assert 'test' in ad
        assert 'test' in ad.__dict__

    def test_setdefault(self):
        """
        Tests setdefault method of AttribDict class.
        """
        ad = AttribDict()
        # 1
        default = ad.setdefault('test', 'NEW')
        assert default == 'NEW'
        assert ad['test'] == 'NEW'
        assert ad.test == 'NEW'
        assert ad.get('test') == 'NEW'
        assert ad.__getattr__('test') == 'NEW'
        assert ad.__getitem__('test') == 'NEW'
        assert ad.__dict__['test'] == 'NEW'
        assert ad.__dict__.get('test') == 'NEW'
        assert 'test' in ad
        assert 'test' in ad.__dict__
        # 2 - existing key should not be overwritten
        default = ad.setdefault('test', 'SOMETHINGDIFFERENT')
        assert default == 'NEW'
        assert ad['test'] == 'NEW'
        assert ad.test == 'NEW'
        assert ad.get('test') == 'NEW'
        assert ad.__getattr__('test') == 'NEW'
        assert ad.__getitem__('test') == 'NEW'
        assert ad.__dict__['test'] == 'NEW'
        assert ad.__dict__.get('test') == 'NEW'
        assert 'test' in ad
        assert 'test' in ad.__dict__
        # 3 - default value isNone
        ad = AttribDict()
        default = ad.setdefault('test')
        assert default is None
        assert ad['test'] is None
        assert ad.test is None
        assert ad.get('test') is None
        assert ad.__getattr__('test') is None
        assert ad.__getitem__('test') is None
        assert ad.__dict__['test'] is None
        assert ad.__dict__.get('test') is None
        assert 'test' in ad
        assert 'test' in ad.__dict__

    def test_clear(self):
        """
        Tests clear method of AttribDict class.
        """
        ad = AttribDict()
        ad.test = 1
        ad['test2'] = 'test'
        # removing via pop
        ad.clear()
        assert not ('test' in ad)
        assert not ('test2' in ad)
        assert not ('test' in ad.__dict__)
        assert not ('test2' in ad.__dict__)
        assert not hasattr(ad, 'test')
        assert not hasattr(ad, 'test2')
        # class attributes should be still present
        assert hasattr(ad, 'readonly')
        assert hasattr(ad, 'defaults')

    def test_init_argument(self):
        """
        Tests initialization of AttribDict with various arguments.
        """
        # one dict works as expected
        ad = AttribDict({'test': 1})
        assert ad.test == 1
        # multiple dicts results into TypeError
        with pytest.raises(TypeError):
            AttribDict({}, {})
        with pytest.raises(TypeError):
            AttribDict({}, {}, blah=1)
        # non-dicts results into TypeError
        with pytest.raises(TypeError):
            AttribDict(1)
        with pytest.raises(TypeError):
            AttribDict(object())

    def test_defaults(self):
        """
        Tests default of __getitem__/__getattr__ methods of AttribDict class.
        """
        # 1
        ad = AttribDict()
        ad['test'] = 'NEW'
        assert ad.__getitem__('test') == 'NEW'
        assert ad.__getitem__('xxx', 'blub') == 'blub'
        assert ad.__getitem__('test', 'blub') == 'NEW'
        assert ad.__getattr__('test') == 'NEW'
        assert ad.__getattr__('xxx', 'blub') == 'blub'
        assert ad.__getattr__('test', 'blub') == 'NEW'
        # should raise KeyError without default item
        with pytest.raises(KeyError):
            ad.__getitem__('xxx')
        with pytest.raises(AttributeError):
            ad.__getattr__('xxx')
        # 2
        ad2 = AttribDict(defaults={'test2': 'NEW'})
        assert ad2.__getitem__('test2') == 'NEW'
        with pytest.raises(KeyError):
            ad2.__getitem__('xxx')

    def test_set_readonly(self):
        """
        Tests of setting readonly attributes.
        """
        class MyAttribDict(AttribDict):
            readonly = ['test']
            defaults = {'test': 1}

        ad = MyAttribDict()
        assert ad.test == 1
        with pytest.raises(AttributeError):
            ad.__setitem__('test', 1)

    def test_deepcopy_and_pickle(self):
        """
        Tests deepcopy and pickle of AttribDict.
        """
        ad = DefaultTestAttribDict()
        ad.muh = 2
        ad2 = ad.copy()
        assert ad2.test == 1
        assert ad2.muh == 2
        assert ad2 == ad
        ad3 = pickle.loads(pickle.dumps(ad, protocol=2))
        assert ad3.test == 1
        assert ad3.muh == 2
        assert ad3 == ad

    def test_compare_with_dict(self):
        """
        Checks if AttribDict is still comparable to a dict object.
        """
        adict = {'test': 1}
        ad = AttribDict(adict)
        assert ad == adict
        assert adict == ad

    def test_pretty_str(self):
        """
        Test _pretty_str method of AttribDict.
        """
        # 1
        ad = AttribDict({'test1': 1, 'test2': 2})
        out = '           test1: 1\n           test2: 2'
        assert ad._pretty_str() == out
        # 2
        ad = AttribDict({'test1': 1, 'test2': 2})
        out = '           test2: 2\n           test1: 1'
        assert ad._pretty_str(priorized_keys=['test2']) == out
        # 3
        ad = AttribDict({'test1': 1, 'test2': 2})
        out = ' test1: 1\n test2: 2'
        assert ad._pretty_str(min_label_length=6) == out

    def test_types(self):
        """
        Test that types are enforced with _types attribute
        """
        class AttrOcity(AttribDict):
            _types = {'string': str, 'number': (float, int), 'int': int,
                      'another_number': (float, int)}

        ad = AttrOcity()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('default')
            ad.string = int
            ad.number = '1'
            ad.int = 1.0
            ad.not_type_controlled = 2
            ad.another_number = 1

        assert len(w) == 3
        assert isinstance(ad.string, str)
        assert isinstance(ad.number, float)
        assert isinstance(ad.int, int)
        assert isinstance(ad.another_number, int)
