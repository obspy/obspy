=============================================================================
Handling custom defined tags in QuakeML and the ObsPy Catalog/Event framework
=============================================================================

QuakeML allows use of custom xml tags in addition to the "usual" information
defined by the QuakeML standard. ObsPy can handle basic (non-nested) custom
tags in event type objects and input/output to/from QuakeML.
The following basic example illustrates how to output a valid QuakeML file
with custom xml tags:

.. code-block:: python

    from obspy import Catalog, UTCDateTime
    
    extra = {'public': {'value': True,
                        'namespace': r"http://some-page.de/xmlns/1.0"},
             'custom': {'value': u"True"},
             'new_tag': {'value': 1234,
                         'namespace': r"http://test.org/xmlns/0.1"},
             'tX': {'value': UTCDateTime('2013-01-02T13:12:14.600000Z')}}
    
    cat = Catalog()
    cat.extra = extra
    cat.write("my_catalog.xml", "QUAKEML",
              nsmap={"ns0": r"http://test.org/xmlns/0.1"})

All custom information to be stored in custom QuakeML conform xml tags has to
be stored in ``extra`` attribute of e.g. ``Catalog``, ``Event``, ``Pick``.  The
keys are used as the name of the xml tag, the content of the xml tag is defined
in a simple dictionary: ``'value'`` defines the content of the tag (the string
representation of the object gets stored in the textual xml output).
``'namespace'`` can be used to specify a custom namespace for the tag.
If the ``'namespace'`` key is missing a default ObsPy namespace is used.
Namespace abbreviations in the output xml can be specified during output as
QuakeML by providing a dictionary of namespace abbreviation mappings as
`nsmap` parameter to :meth:`Catalog.write() <obspy.core.event.Catalog.write>`.
The xml output of the above example looks like:

.. code-block:: xml

    <q:quakeml xmlns:q="http://quakeml.org/xmlns/quakeml/1.2"
               xmlns:ns0="http://test.org/xmlns/0.1"
               xmlns:obspy="http://obspy.org/xmlns/0.1"
               xmlns:ns1="http://some-page.de/xmlns/1.0"
               xmlns="http://quakeml.org/xmlns/bed/1.2">
      <eventParameters publicID="smi:local/0ea8f884-56a9-4a9f-8ebb-0e90e78a583b">
        <ns0:new_tag pythonType="int">1234</ns0:new_tag>
        <ns1:public pythonType="bool">true</ns1:public>
        <obspy:tX pythonType="obspy.core.utcdatetime.UTCDateTime">2013-01-02T13:12:14.600000Z</obspy:tX>
        <obspy:custom pythonType="unicode">True</obspy:custom>
      </eventParameters>
    </q:quakeml>

When reading the above xml again, the custom tags get parsed and attached to
the respective Event type objects as ``.extra``, Python types are restored
correctly when possible:

.. code-block:: python

    from obspy import readEvents
    
    cat = read("my_catalog.xml")
    print cat.extra

.. code-block:: python

    AttribDict({u'new_tag': {u'namespace': u'http://test.org/xmlns/0.1',
                             u'value': 1234},
                u'public': {u'namespace': u'http://some-page.de/xmlns/1.0',
                            u'value': True},
                u'tX': {u'namespace': u'http://obspy.org/xmlns/0.1',
                        u'value': UTCDateTime(2013, 1, 2, 13, 12, 14, 600000)},
                u'custom': {u'namespace': u'http://obspy.org/xmlns/0.1',
                            u'value': u'True'}})
