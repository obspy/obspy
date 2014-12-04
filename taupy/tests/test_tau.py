# Just check if the tau interface runs without upsets.
from taupy import tau
import os

try:
    #os.remove('iasp91.taup')  # optional
    pass
except FileNotFoundError:
    pass

i91 = tau.TauPyModel("iasp91")
i91.get_pierce_points(100, 10)
i91.get_ray_paths(5000, 180)
i91.get_travel_time(2000, 1)