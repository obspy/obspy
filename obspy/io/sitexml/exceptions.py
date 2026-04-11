# -*- coding: utf-8 -*-
"""
Custom exceptions for the SiteXML module.
"""


class SiteXMLError(Exception):
    """
    Base class for SiteXML-specific exceptions.
    """


class SiteXMLValidationError(SiteXMLError, ValueError):
    """
    Raised when SiteXML content fails schema or structural validation.
    """


class SiteXMLImportError(SiteXMLError, ValueError):
    """
    Raised when SiteXML metadata imports cannot be completed.
    """


class SiteXMLIOError(SiteXMLError, OSError):
    """
    Raised when SiteXML-related input paths or files cannot be accessed.
    """
