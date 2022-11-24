# -*- coding: utf-8 -*-
"""
Class to create new models.
"""
import glob
import inspect
import os
from math import pi
from pathlib import Path
from obspy.taup import _DEFAULT_VALUES
from obspy.taup.slowness_model import SlownessModel
from obspy.taup.tau_model import TauModel
from obspy.taup.velocity_model import VelocityModel

# Most generic way to get the data directory.
__DATA_DIR = Path(inspect.getfile(inspect.currentframe())).resolve()
__DATA_DIR = __DATA_DIR.parent / "data"


class TauPCreate(object):
    """
    The seismic travel time calculation method of [Buland1983]_.

    The calculation method is described in [Buland1983]_. This creates the
    SlownessModel and tau branches and saves them for later use.
    """
    def __init__(self, input_filename, output_filename, verbose=False,
                 min_delta_p=0.1, max_delta_p=11.0, max_depth_interval=115.0,
                 max_range_interval=2.5, max_interp_error=0.05,
                 allow_inner_core_s=True):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.debug = verbose
        self.min_delta_p = min_delta_p
        self.max_delta_p = max_delta_p
        self.max_depth_interval = max_depth_interval
        self.max_range_interval = max_range_interval
        self.max_interp_error = max_interp_error
        self.allow_inner_core_s = allow_inner_core_s

    def load_velocity_model(self):
        """
        Try to load a velocity model.
        """
        # Read the velocity model file.
        filename = self.input_filename
        if self.debug:
            print("filename =", filename)
        self.v_mod = VelocityModel.read_velocity_file(filename)
        if self.v_mod is None:
            raise IOError("Velocity model file not found: " + filename)
        # If model was read:
        if self.debug:
            print("Done reading velocity model.")
            print("Radius of model " + self.v_mod.model_name + " is " +
                  str(self.v_mod.radius_of_planet))
        # if self.debug:
        #    print("velocity mode: " + self.v_mod)
        return self.v_mod

    def create_tau_model(self, v_mod):
        """
        Create :class:`~.TauModel` from velocity model.

        First, a slowness model is created from the velocity model, and then it
        is passed to :class:`~.TauModel`.
        """
        if v_mod is None:
            raise ValueError("v_mod is None.")
        if v_mod.is_spherical is False:
            raise Exception("Flat slowness model not yet implemented.")
        SlownessModel.debug = self.debug
        if self.debug:
            print("Using parameters provided in TauP_config.ini (or defaults "
                  "if not) to call SlownessModel...")

        self.s_mod = SlownessModel(
            v_mod, self.min_delta_p, self.max_delta_p, self.max_depth_interval,
            self.max_range_interval * pi / 180.0, self.max_interp_error,
            self.allow_inner_core_s,
            _DEFAULT_VALUES["slowness_tolerance"])
        if self.debug:
            print("Parameters are:")
            print("taup.create.min_delta_p = " + str(self.s_mod.min_delta_p) +
                  " sec / radian")
            print("taup.create.max_delta_p = " + str(self.s_mod.max_delta_p) +
                  " sec / radian")
            print("taup.create.max_depth_interval = " +
                  str(self.s_mod.max_depth_interval) + " kilometers")
            print("taup.create.max_range_interval = " +
                  str(self.s_mod.max_range_interval) + " degrees")
            print("taup.create.max_interp_error = " +
                  str(self.s_mod.max_interp_error) + " seconds")
            print("taup.create.allow_inner_core_s = " +
                  str(self.s_mod.allow_inner_core_s))
            print("Slow model " + " " + str(self.s_mod.get_num_layers(True)) +
                  " P layers," + str(self.s_mod.get_num_layers(False)) +
                  " S layers")
        # if self.debug:
        #    print(self.s_mod)
        # set the debug flags to value given here:
        TauModel.debug = self.debug
        SlownessModel.debug = self.debug
        # Creates tau model from slownesses.
        return TauModel(self.s_mod, radius_of_planet=v_mod.radius_of_planet)

    def run(self):
        """
        Create a tau model from a velocity model.

        Called by :func:`build_taup_model` after :meth:`load_velocity_model`;
        calls :meth:`create_tau_model` and writes the result to a ``.npy``
        file.
        """
        try:
            self.tau_model = self.create_tau_model(self.v_mod)
            # this reassigns model! Used to be TauModel() class,
            # now it's an instance of it.
            if self.debug:
                print("Done calculating Tau branches.")

            dirname = os.path.dirname(self.output_filename)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)
            self.tau_model.serialize(self.output_filename)
            if self.debug:
                print("Done Saving " + self.output_filename)
        except IOError as e:
            print("Tried to write!\n Caught IOError. Do you have write "
                  "permission in this directory?", e)
        except KeyError as e:
            print('file not found or wrong key?', e)
        finally:
            if self.debug:
                print("Method run is done, but not necessarily successful.")


def get_builtin_model_files():
    """
    Get a list of paths to builtin files that can be used for models.

    These files reside in the ``<package-root>/obspy/taup/data`` directory.
    """
    files = glob.glob(os.path.join(__DATA_DIR, "*.tvel"))
    files.extend(glob.glob(os.path.join(__DATA_DIR, "*.nd")))
    return files


def build_taup_model(filename, output_folder=None, verbose=True):
    """
    Build an ObsPy model file from a "tvel" or "nd" file.

    The file is loaded into a :class:`~obspy.taup.tau_model.TauModel`
    instance and is then saved in ObsPy's own format, which can be loaded using
    :meth:`~obspy.taup.tau_model.TauModel.from_file`. The output file will have
    the same name as the input with ``'.npz'`` as file extension.

    :type filename: str
    :param filename: Absolute path of input file.
    :type output_folder: str
    :param output_folder: Directory in which the built
        :class:`~obspy.taup.tau_model.TauModel` will be stored. Defaults to
        the `taup/data` directory of the current obspy installation.
    """
    if output_folder is None:
        output_folder = __DATA_DIR

    model_name = os.path.splitext(os.path.basename(filename))[0]
    output_filename = os.path.join(output_folder, model_name + ".npz")

    if verbose:
        print("Building obspy.taup model for '%s' ..." % filename)
    mod_create = TauPCreate(input_filename=filename,
                            output_filename=output_filename,
                            verbose=verbose)
    mod_create.load_velocity_model()
    mod_create.run()


def build_all_taup_models():
    """
    Build all :class:`~obspy.taup.tau_model.TauModel` models in data directory.

    The data directory is defined to be ``<package-root>/obspy/taup/data``.
    """
    for model in get_builtin_model_files():
        build_taup_model(filename=model)


if __name__ == '__main__':
    build_all_taup_models()
