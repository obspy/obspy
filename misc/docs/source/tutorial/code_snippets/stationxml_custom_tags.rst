.. _stationxml-extra:

=============================================================================
Handling custom defined tags in StationXML with the Obspy Inventory
=============================================================================

StationXML allows use of custom elements in addition to the 'usual' information
defined by the StationXML standard. It allows *a)* custom namespace attributes to
StationXML namespace tags and *b)* custom namespace subtags to StationXML namespace
elements.
ObsPy can handle both basic custom tags in all main elements
(Network, Station, Channel, etc.) (*a*) and custom attributes (*b*) during
input/output to/from StationXML. The following basic example illustrates how to
output a StationXML file that contains additional xml tags/attributes:

.. code-block:: python

    from obspy import Inventory, UTCDateTime
    from obspy.core.inventory import Network
    from obspy.core.util import AttribDict

    extra = AttribDict({
               'my_tag': {
                    'value': True,
                    'namespace': 'http://some-page.de/xmlns/1.0',
                    'attrib': {
                      '{http://some-page.de/xmlns/1.0}my_attrib1': '123.4',
                      '{http://some-page.de/xmlns/1.0}my_attrib2': '567'
                    }
                },
               'my_tag_2': {
                    'value': u'True',
                    'namespace': 'http://some-page.de/xmlns/1.0'
                },
               'my_tag_3': {
                    'value': 1,
                    'namespace': 'http://some-page.de/xmlns/1.0'
                },
               'my_tag_4': {
                    'value': UTCDateTime('2013-01-02T13:12:14.600000Z'),
                    'namespace': 'http://test.org/xmlns/0.1'
                },
               'my_attribute': {
                    'value': 'my_attribute_value',
                    'type': 'attribute',
                    'namespace': 'http://test.org/xmlns/0.1'
                }
            })

    inv = Inventory([Network('XX')], 'XX')
    inv[0].extra = extra
    inv.write('my_inventory.xml', format='STATIONXML',
              nsmap={'my_ns': 'http://test.org/xmlns/0.1',
                     'somepage_ns': 'http://some-page.de/xmlns/1.0'})

All custom information to be stored in the customized StationXML has to
be stored in form of a :class:`dict` or
:class:`~obspy.core.util.attribdict.AttribDict`
object as the ``extra`` attribute of the object that should carry the
additional custom information (e.g. ``Network``, ``Station``, ``Channel``). The
keys are used as the name of the xml tag, the content of the xml tag is defined
in a simple dictionary: ``'value'`` defines the content of the tag (the string
representation of the object gets stored in the textual xml output).
``'namespace'`` has to specify a custom namespace for the tag.
``'type'`` can be used to specify whether the extra information should be
stored as a subelement (``'element'``, default) or as an attribute
(``'attribute'``). Attributes to custom subelements can be provided in form of
a dictionary as ``'attrib'``.
If desired for better (human-)readability, namespace abbreviations in the
output xml can be specified during output as StationXML by providing a dictionary
of namespace abbreviation mappings as `nsmap` parameter to
:meth:`Inventory.write() <obspy.core.inventory.inventory.Inventory.write>`.
The xml output of the above example looks like:

.. code-block:: xml

    <?xml version='1.0' encoding='UTF-8'?>
    <FDSNStationXML xmlns:my_ns="http://test.org/xmlns/0.1" xmlns:somepage_ns="http://some-page.de/xmlns/1.0" xmlns="http://www.fdsn.org/xml/station/1" schemaVersion="1.0">
      <Source>XX</Source>
      <Module>ObsPy 1.0.2</Module>
      <ModuleURI>https://www.obspy.org</ModuleURI>
      <Created>2016-10-17T18:32:28.696287+00:00</Created>
      <Network code="XX">
        <somepage_ns:my_tag somepage_ns:my_attrib1="123.4" somepage_ns:my_attrib2="567">True</somepage_ns:my_tag>
        <my_ns:my_tag_4>2013-01-02T13:12:14.600000Z</my_ns:my_tag_4>
        <my_ns:my_attribute>my_attribute_value</my_ns:my_attribute>
        <somepage_ns:my_tag_2>True</somepage_ns:my_tag_2>
        <somepage_ns:my_tag_3>1</somepage_ns:my_tag_3>
      </Network>
    </FDSNStationXML>

When reading the above xml again, using
:meth:`read_inventory() <obspy.core.inventory.inventory.read_inventory>`, the custom tags get
parsed and attached to the respective Network type objects (in this example to
the Inventory object) as ``.extra``.
Note that all values are read as text strings:

.. code-block:: python

    from obspy import read_inventory
    
    inv = read_inventory('my_inventory.xml')
    print(inv[0].extra)

.. code-block:: python

    AttribDict({
        u'my_tag': AttribDict({
            'attrib': {
              '{http://some-page.de/xmlns/1.0}my_attrib2': '567',
              '{http://some-page.de/xmlns/1.0}my_attrib1': '123.4'
            },
            'namespace': 'http://some-page.de/xmlns/1.0',
            'value': 'True'
        }),
        u'my_tag_4': AttribDict({
            'namespace': 'http://test.org/xmlns/0.1',
            'value': '2013-01-02T13:12:14.600000Z'
        }),
        u'my_attribute': AttribDict({
            'namespace': 'http://test.org/xmlns/0.1',
            'value': 'my_attribute_value'
        }),
        u'my_tag_2': AttribDict({
            'namespace': 'http://some-page.de/xmlns/1.0',
            'value': 'True'
        }),
        u'my_tag_3': AttribDict({
            'namespace': 'http://some-page.de/xmlns/1.0',
            'value': '1'
        })
    })

Custom tags can be nested:

.. code-block:: python

    from obspy import Inventory
    from obspy.core.inventory import Network
    from obspy.core.util import AttribDict
    
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
    
    inv = Inventory([Network('XX')], 'XX')
    inv[0].extra = AttribDict()
    inv[0].extra.my_tag = my_tag
    inv.write('my_inventory.xml', format='STATIONXML',
              nsmap={'somepage_ns': 'http://some-page.de/xmlns/1.0'})

This will produce an xml output similar to the following:

.. code-block:: xml

    <?xml version='1.0' encoding='UTF-8'?>
    <FDSNStationXML xmlns:somepage_ns="http://some-page.de/xmlns/1.0" xmlns="http://www.fdsn.org/xml/station/1" schemaVersion="1.0">
      <Source>XX</Source>
      <Module>ObsPy 1.0.2</Module>
      <ModuleURI>https://www.obspy.org</ModuleURI>
      <Created>2016-10-17T18:45:14.302265+00:00</Created>
      <Network code="XX">
        <somepage_ns:my_tag>
          <somepage_ns:my_nested_tag1>12300000000.0</somepage_ns:my_nested_tag1>
          <somepage_ns:my_nested_tag2>True</somepage_ns:my_nested_tag2>
        </somepage_ns:my_tag>
      </Network>
    </FDSNStationXML>

The output xml can be read again using
:meth:`read_inventory() <obspy.core.inventory.inventory.read_inventory>` and the nested tags can be
retrieved in the following way:

.. code-block:: python

    from obspy import read_inventory

    inv = read_inventory('my_inventory.xml')
    print(inv[0].extra.my_tag.value.my_nested_tag1.value)
    print(inv[0].extra.my_tag.value.my_nested_tag2.value)

.. code-block:: python

    12300000000.0
    True
