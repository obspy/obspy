# -*- Mode: Python; -*-
#                            Package   : omniORBpy
# sslTP.py                   Created on: 2002/09/06
#                            Author    : Duncan Grisby (dgrisby)
#
#    Copyright (C) 2002 Apasphere Ltd.
#
#    This file is part of the omniORBpy library
#
#    The omniORBpy library is free software; you can redistribute it
#    and/or modify it under the terms of the GNU Lesser General
#    Public License as published by the Free Software Foundation;
#    either version 2.1 of the License, or (at your option) any later
#    version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free
#    Software Foundation, Inc., 59 Temple Place - Suite 330, Boston,
#    MA 02111-1307, USA
#
#
# Description:
#    Import this to enable the SSL transport.

# $Id: sslTP.py,v 1.1.4.1 2003/03/23 21:51:43 dgrisby Exp $
# $Log: sslTP.py,v $
# Revision 1.1.4.1  2003/03/23 21:51:43  dgrisby
# New omnipy3_develop branch.
#
# Revision 1.1.2.1  2002/09/06 21:34:26  dgrisby
# Add codesets and sslTP modules.
#

"""omniORB.sslTP

Import this module and set the files/passwords before calling
CORBA.ORB_init() to make the SSL transport available.

Functions:

  certificate_authority_file()
  key_file()
  key_file_password()
"""

import _omnipy
from _omnisslTP import *
