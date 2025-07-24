
{%- set command_mapping = {
    'obspy.scripts.flinnengdahl': 'obspy-flinn-engdahl',
    'obspy.scripts.runtests': 'obspy-runtests',
    'obspy.scripts.reftekrescue': 'obspy-reftek-rescue',
    'obspy.scripts.print': 'obspy-print',
    'obspy.scripts.sds_html_report': 'obspy-sds-report',
    'obspy.imaging.scripts.scan': 'obspy-scan',
    'obspy.imaging.scripts.plot': 'obspy-plot',
    'obspy.imaging.scripts.mopad': 'obspy-mopad',
    'obspy.io.mseed.scripts.recordanalyzer': 'obspy-mseed-recordanalyzer',
    'obspy.io.xseed.scripts.dataless2xseed': 'obspy-dataless2xseed',
    'obspy.io.xseed.scripts.xseed2dataless': 'obspy-xseed2dataless',
    'obspy.io.xseed.scripts.dataless2resp': 'obspy-dataless2resp'
} -%}
{%- set command_name = command_mapping.get(fullname) -%}

{{ fullname }}
{{ underline }}

.. note::
    This script automatically installs during the setup procedure with the
    name ``$ {{ command_name }}``. For more info on the command line options,
    please run ``$ {{ command_name }} --help``.
    Alternatively you can also execute ``$ python -m {{ fullname }}``.

.. currentmodule:: {{ fullname }}
.. automodule:: {{ fullname }}
