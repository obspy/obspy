.. _quakeml-extra:

=============================================================================
Handling custom defined tags in QuakeML and the ObsPy Catalog/Event framework
=============================================================================

QuakeML allows use of custom elements in addition to the 'usual' information
defined by the QuakeML standard. It allows *a)* custom namespace attributes to
QuakeML namespace tags and *b)* custom namespace subtags to QuakeML namespace
elements.
ObsPy can handle both basic custom tags in event type objects (*a*) and custom
attributes (*b*) during input/output to/from QuakeML.
The following basic example illustrates how to output a valid QuakeML file
with custom xml tags/attributes:

.. code-block:: python

    from obspy import Catalog, UTCDateTime

    extra = {'my_tag': {'value': True,
                        'namespace': 'http://some-page.de/xmlns/1.0',
                        'attrib': {'{http://some-page.de/xmlns/1.0}my_attrib1': '123.4',
                                   '{http://some-page.de/xmlns/1.0}my_attrib2': '567'}},
             'my_tag_2': {'value': u'True',
                          'namespace': 'http://some-page.de/xmlns/1.0'},
             'my_tag_3': {'value': 1,
                          'namespace': 'http://some-page.de/xmlns/1.0'},
             'my_tag_4': {'value': UTCDateTime('2013-01-02T13:12:14.600000Z'),
                          'namespace': 'http://test.org/xmlns/0.1'},
             'my_attribute': {'value': 'my_attribute_value',
                              'type': 'attribute',
                              'namespace': 'http://test.org/xmlns/0.1'}}

    cat = Catalog()
    cat.extra = extra
    cat.write('my_catalog.xml', format='QUAKEML',
              nsmap={'my_ns': 'http://test.org/xmlns/0.1'})

All custom information to be stored in the customized QuakeML has to
be stored in form of a :class:`dict` or
:class:`~obspy.core.util.attribdict.AttribDict`
object as the ``extra`` attribute of the object that should carry the
additional custom information (e.g. ``Catalog``, ``Event``, ``Pick``). The
keys are used as the name of the xml tag, the content of the xml tag is defined
in a simple dictionary: ``'value'`` defines the content of the tag (the string
representation of the object gets stored in the textual xml output).
``'namespace'`` has to specify a custom namespace for the tag.
``'type'`` can be used to specify whether the extra information should be
stored as a subelement (``'element'``, default) or as an attribute
(``'attribute'``). Attributes to custom subelements can be provided in form of
a dictionary as ``'attrib'``.
If desired for better (human-)readability, namespace abbreviations in the
output xml can be specified during output as QuakeML by providing a dictionary
of namespace abbreviation mappings as `nsmap` parameter to
:meth:`Catalog.write() <obspy.core.event.catalog.Catalog.write>`.
The xml output of the above example looks like:

.. code-block:: xml

    <?xml version='1.0' encoding='utf-8'?>
    <q:quakeml xmlns:q='http://quakeml.org/xmlns/quakeml/1.2'
               xmlns:ns0='http://some-page.de/xmlns/1.0'
               xmlns:my_ns='http://test.org/xmlns/0.1'
               xmlns='http://quakeml.org/xmlns/bed/1.2'>
      <eventParameters publicID='smi:local/b425518c-9445-40c7-8284-d1f299ed2eac'
                       my_ns:my_attribute='my_attribute_value'>
        <ns0:my_tag ns0:my_attrib1='123.4' ns0:my_attrib2='567'>true</ns0:my_tag>
        <my_ns:my_tag_4>2013-01-02T13:12:14.600000Z</my_ns:my_tag_4>
        <ns0:my_tag_2>True</ns0:my_tag_2>
        <ns0:my_tag_3>1</ns0:my_tag_3>
      </eventParameters>
    </q:quakeml>

When reading the above xml again, using
:meth:`read_events() <obspy.core.event.read_events>`, the custom tags get
parsed and attached to the respective Event type objects (in this example to
the Catalog object) as ``.extra``.
Note that all values are read as text strings:

.. code-block:: python

    from obspy import read_events

    cat = read_events('my_catalog.xml')
    print(cat.extra)

.. code-block:: python

    AttribDict({u'my_tag': {u'attrib': {'{http://some-page.de/xmlns/1.0}my_attrib2': '567',
                                        '{http://some-page.de/xmlns/1.0}my_attrib1': '123.4'},
                            u'namespace': u'http://some-page.de/xmlns/1.0',
                            u'value': 'true'},
                u'my_tag_4': {u'namespace': u'http://test.org/xmlns/0.1',
                              u'value': '2013-01-02T13:12:14.600000Z'},
                u'my_attribute': {u'type': u'attribute',
                                  u'namespace': u'http://test.org/xmlns/0.1',
                                  u'value': 'my_attribute_value'},
                u'my_tag_2': {u'namespace': u'http://some-page.de/xmlns/1.0',
                              u'value': 'True'},
                u'my_tag_3': {u'namespace': u'http://some-page.de/xmlns/1.0',
                              u'value': '1'}})

Custom tags can be nested:

.. code-block:: python

    from obspy import Catalog
    from obspy.core import AttribDict

    ns = 'http://some-page.de/xmlns/1.0'

    my_tag = AttribDict()
    my_tag.namespace = ns
    my_tag.value = AttribDict()

    my_tag.value.my_nested_tag1 = AttribDict()
    my_tag.value.my_nested_tag1.namespace = ns
    my_tag.value.my_nested_tag1.value = 1.23E+10

    my_tag.value.my_nested_tag2 = AttribDict()
    my_tag.value.my_nested_tag2.namespace = ns
    my_tag.value.my_nested_tag2.value = True

    cat = Catalog()
    cat.extra = AttribDict()
    cat.extra.my_tag = my_tag
    cat.write('my_catalog.xml', 'QUAKEML')

This will produce an xml output similar to the following:

.. code-block:: xml

    <?xml version='1.0' encoding='utf-8'?>
    <q:quakeml xmlns:q='http://quakeml.org/xmlns/quakeml/1.2'
               xmlns:ns0='http://some-page.de/xmlns/1.0'
               xmlns='http://quakeml.org/xmlns/bed/1.2'>
      <eventParameters publicID='smi:local/97d2b338-0701-41a4-9b6b-5903048bc341'>
        <ns0:my_tag>
          <ns0:my_nested_tag1>12300000000.0</ns0:my_nested_tag1>
          <ns0:my_nested_tag2>true</ns0:my_nested_tag2>
        </ns0:my_tag>
      </eventParameters>
    </q:quakeml>

The output xml can be read again using
:meth:`read_events() <obspy.core.event.read_events>` and the nested tags can be
retrieved in the following way:

.. code-block:: python

    from obspy import read_events

    cat = read_events('my_catalog.xml')
    print(cat.extra.my_tag.value.my_nested_tag1.value)
    print(cat.extra.my_tag.value.my_nested_tag2.value)

.. code-block:: python

    12300000000.0
    true

The order of extra tags can be controlled by using an
:py:class:`~collections.OrderedDict` for the extra attribute (using a plain
`dict` or :class:`~obspy.core.util.attribdict.AttribDict` can result in
arbitrary order of tags):

.. code-block:: python

    from collections import OrderedDict
    from obspy.core.event import Catalog, Event

    ns = 'http://some-page.de/xmlns/1.0'

    my_tag1 = {'namespace': ns, 'value': 'some value 1'}
    my_tag2 = {'namespace': ns, 'value': 'some value 2'}

    event = Event()
    cat = Catalog(events=[event])
    event.extra = OrderedDict()
    event.extra['myFirstExtraTag'] = my_tag2
    event.extra['mySecondExtraTag'] = my_tag1
    cat.write('my_catalog.xml', 'QUAKEML')
