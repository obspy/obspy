import obspy
from obspy.core.inventory.util import (ExternalReference)
from obspy.io.sitexml.core import (SERASite, SiteDescription, SiteCharacterizationParameters, EC8, 
                                   LiteratureSource, ValueWithUncertainty, SERASiteOwner,
                                   ResonanceFrequency, VelocityS30)
from obspy.io.sitexml.write import _write_sitexml


sera_site=SERASite(station_code="ARG", 
                   created=obspy.UTCDateTime())

# Site Owner
sera_site.site_owner = SERASiteOwner(ownerID="quakeml:itsak.gr/siteOwner/001",
                                owner_codename="ITSAK",
                                owner_fullname="Institute of Engineering Seismology and Earthquake Engineering",
                                personID="quakeml:itsak.gr/person/001",
                                person_firstname="Nikolaos",
                                person_lastname="Theodoulides",
                                person_mbox="ntheo@itsak.gr",
                                institution_homepage="http://www.itsak.gr",
                                address_country="Greece",
                                affiliation_function="Senior researcher")

# Site Description
sera_site.site_description=SiteDescription(23.44, 45.33)
sera_site.site_description.topologyA = "T1"
sera_site.site_description.ec8="A"
sera_site.site_description.ec8.file_resource="http://some/uri"
sera_site.site_description.ec8.literature_source=LiteratureSource(title="Some title", year="2018")
sera_site.site_description.h800=300
sera_site.site_description.h800.literature_source=LiteratureSource(title="Some title", year="2018")
sera_site.site_description.bedrock_depth = ValueWithUncertainty(450, 10)

# Site Characterization
file_resource = ExternalReference(description="ambient noise records analysis with Geopsy Softaware", uri="")
literature_source = LiteratureSource(title="Some title", first_author="Some Author", year="2018", language="EN", doi="10.1007/s10518-017-0135-5")
rf=ResonanceFrequency(value="34.34", 
                      file_resource=file_resource,
                      literature_source=literature_source,
                      methods=["Active non-invasive S-wave methods","Passive non-invasive S-wave methods"])
vs30 = VelocityS30(value = ValueWithUncertainty(378, 10),
                   file_resource=file_resource,
                   literature_source=literature_source)
sera_site.site_characterization = SiteCharacterizationParameters(publicID="SomeID",
                                                            analysis_publicID="AnalysisID",
                                                            velocity_s30=vs30, 
                                                            resonance_frequency=rf, 
                                                            velocity_profile_count=5)
_write_sitexml(sera_site, "output.xml", validate=False)
