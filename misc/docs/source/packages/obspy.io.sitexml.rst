.. currentmodule:: obspy.io.sitexml
.. automodule:: obspy.io.sitexml
    
    .. comment to end block
    
    Enums
    -----

    .. autosummary::
       :toctree: autogen
       :nosignatures:

       ~util.TopographySchemaA
       ~util.TopographySchemaB
       ~util.MorphologyType
       ~util.EC8Class
       ~util.ResonanceFrequencyMethod
       ~util.VelocityS30Method
       ~util.Vs30MethodCombined
       ~util.Vs30ManualIndex

    .. comment to end block

    Functions
    ---------
    
    .. autosummary::
       :toctree: autogen
       :nosignatures:
    
       ~sitexml._is_sitexml
       ~sitexml.validate_sitexml
       ~sitexml.read_sitexml
       ~sitexml.write_sitexml
       ~sitexml.sitedict_to_sitexml
       ~sitexml.sitexml_to_sitedict
       ~sitexml.add_sitexml_reference
       ~tabular.csv_to_sera_site
       ~tabular.excel_to_sera_site
       ~tabular.add_velocity_profiles
       ~quality_index.quality_index1
       ~quality_index.quality_index2
       ~quality_index.quality_index3
       ~quality_index.overall_quality_index
       ~quality_index.apply_quality_index_dataframe
       ~quality_index.apply_quality_index_csv
       ~quality_index.apply_quality_index_excel
       
    .. comment to end block

    Classes
    -------
    
    .. autosummary::
       :toctree: autogen
       :nosignatures:
    
       ~core.SERASite
       ~core.SERASiteOwner
       ~core.SiteDescription
       ~core.Analysis
       ~core.VelocityProfile
       ~core.VelocityProfileData
       ~core.Revision

    .. comment to end block

    Modules
    -------
    
    .. autosummary::
       :toctree: autogen
       :nosignatures:
    
       core
       sitexml
       tabular
       quality_index
       util

    .. comment to end block
