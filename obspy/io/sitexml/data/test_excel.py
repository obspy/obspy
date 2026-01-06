
from obspy.io.sitexml.csv import (csv_to_sera_site, csv_to_sitexml, excel_to_sera_site)

#excel_file = "./example.xlsx"
#excel_file = "./HI_SiteXML_all.xlsx"
#output_folder = "/Users/kiriaki/Desktop/ITSAK/ORFEUS/SITEXML/HI"

#df_dict = pd.read_excel(excel_file, None)

#sera_site_dict = excel_to_sera_site(excel_file)
#excel_to_sitexml(sera_site_dict, output_folder)
#csv_file = "./site_description.csv"

output_folder = "./output_from_csv"
sera_site_dict = csv_to_sera_site(site_owner_csv="./site_owner.csv",
                 site_description_csv="./site_description.csv",
                 analysis_csv="./site_analysis.csv", 
                 velocity_profiles_csv="./vp_csv",
                 delim=';')
if sera_site_dict:
    csv_to_sitexml(sera_site_dict, output_folder)


output_folder = "./output_from_excel"
sera_site_dict = excel_to_sera_site("sera_site_all.xlsx", velocity_profiles="vp.xlsx")
if sera_site_dict:
    csv_to_sitexml(sera_site_dict, output_folder)

