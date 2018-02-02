package obspy.io.win
====================

Copyright
---------
GNU Lesser General Public License, Version 3 (LGPLv3)

Copyright (c) 2012-2017 by:
    * Thomas Lecocq
    * Adolfo Inza
    * Philippe Lesage


Overview
--------
WIN read support for ObsPy.

This module provides read support for WIN data format. This module is
based on the code of Adolfo Inza and Philippe Lesage of the "Géophysique des
volcans" team of the "Institut des Sciences de la Terre de l'Université de
Savoie", France. This format is written by different dataloggers, including
Hakusan LS-7000XT Datamark dataloggers. There are two subformats, A0 and A1.
To our knowledge, A0 is the only one supported with the current code. A0
conforms to the data format of the WIN system developed by Earthquake Research
Institute (ERI), the University of Tokyo.

ObsPy is an open-source project dedicated to provide a Python framework for
processing seismological data. It provides parsers for common file formats and
seismological signal processing routines which allow the manipulation of
seismological time series (see Beyreuther et al. 2010, Megies et al. 2011).
The goal of the ObsPy project is to facilitate rapid application development
for seismology.

For more information visit https://www.obspy.org.
