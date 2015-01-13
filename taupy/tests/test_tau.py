# Manual test to just check if the tau interface runs without upsets.
from taupy import tau
import os

try:
    os.remove('iasp91.taup')  # optional
    pass
except FileNotFoundError:
    pass

i91 = tau.TauPyModel("iasp91", taup_model_path="./lalalalala")
i91.get_pierce_points(100, 10)
i91.get_ray_paths(5000, 180)
i91.get_travel_times(2000, 1)
i91.get_ray_paths(10, coordinate_list=[10, 20, 30, 40], print_output=True,
                  phase_list=["P", "PcS"])
os.remove('./lalalalala/iasp91.taup')
os.rmdir("./lalalalala")

