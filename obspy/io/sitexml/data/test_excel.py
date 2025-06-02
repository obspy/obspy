from obspy.io.sitexml.import_excel_v2 import (excel_to_sera_site, excel_to_sitexml)
import pandas as pd

#excel_file = "./example.xlsx"
excel_file = "./HI_SiteXML_all.xlsx"
output_folder = "/Users/kiriaki/Desktop/ITSAK/ORFEUS/SITEXML/HI"

df_dict = pd.read_excel(excel_file, None)

sera_site_dict = excel_to_sera_site(excel_file)
excel_to_sitexml(sera_site_dict, output_folder)


