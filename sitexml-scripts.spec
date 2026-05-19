# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, collect_data_files


datas = collect_data_files("obspy.io.sitexml")
datas += [("obspy/RELEASE-VERSION", "obspy")]

pandas_datas, pandas_binaries, pandas_hiddenimports = collect_all("pandas")
openpyxl_datas, openpyxl_binaries, openpyxl_hiddenimports = collect_all(
    "openpyxl")

datas += pandas_datas + openpyxl_datas
binaries = pandas_binaries + openpyxl_binaries
hiddenimports = pandas_hiddenimports + openpyxl_hiddenimports

excludes = [
    "IPython",
    "matplotlib",
    "scipy",
    "sqlalchemy",
    "tkinter",
]


csv_analysis = Analysis(
    ["obspy/io/sitexml/scripts/csv2serasite.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=None,
    runtime_hooks=None,
    excludes=excludes,
    noarchive=False,
)
csv_pyz = PYZ(csv_analysis.pure, csv_analysis.zipped_data)
csv_exe = EXE(
    csv_pyz,
    csv_analysis.scripts,
    exclude_binaries=True,
    name="csv2serasite",
    console=True,
)

excel_analysis = Analysis(
    ["obspy/io/sitexml/scripts/excel2serasite.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=None,
    runtime_hooks=None,
    excludes=excludes,
    noarchive=False,
)
excel_pyz = PYZ(excel_analysis.pure, excel_analysis.zipped_data)
excel_exe = EXE(
    excel_pyz,
    excel_analysis.scripts,
    exclude_binaries=True,
    name="excel2serasite",
    console=True,
)

coll = COLLECT(
    csv_exe,
    excel_exe,
    csv_analysis.binaries,
    csv_analysis.datas,
    excel_analysis.binaries,
    excel_analysis.datas,
    name="sitexml-scripts",
)
