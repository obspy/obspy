# -*- mode: python ; coding: utf-8 -*-

import os

from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_all, collect_data_files


os.makedirs(CONF["workpath"], exist_ok=True)
os.makedirs(CONF["distpath"], exist_ok=True)

datas = collect_data_files("obspy.io.sitexml", excludes=["internal/**"])
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

analysis = Analysis(
    ["obspy/io/sitexml/scripts/sitexml_standalone.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=None,
    runtime_hooks=None,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(analysis.pure, analysis.zipped_data)
csv_exe = EXE(
    pyz,
    analysis.scripts,
    exclude_binaries=True,
    name="csv2sitexml",
    console=True,
)

coll = COLLECT(
    csv_exe,
    [("excel2sitexml", csv_exe.name, "EXECUTABLE")],
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    name="sitexml-scripts",
)
