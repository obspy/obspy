from obspy.core.util import getExampleFile
from obspy.segy.segy import readSU
from obspy.segy.benchmark import plotBenchmark

files = [getExampleFile('seismic01_fdmpi_vz.su'),
         getExampleFile('seismic01_gemini_vz.su'),
         getExampleFile('seismic01_sofi2D_transformed_vz.su'),
         getExampleFile('seismic01_specfem_vz.su')]

sufiles = [readSU(file) for file in files]
plotBenchmark(sufiles, title="Homogenous halfspace", xmax=0.14)
