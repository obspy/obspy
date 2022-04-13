# -*- coding: utf-8 -*-
"""
Misc functionality.
"""
import inspect
from pathlib import Path

import numpy as np

ROOT = Path(Path(inspect.getfile(inspect.currentframe())).resolve()).parent


def parse_phase_list(phase_list):
    """
    Takes a list of phases, returns a list of individual phases. Performs e.g.
    replacing e.g. ``"ttall"`` with the relevant phases.
    """
    phase_names = []
    for phase_name in phase_list:
        phase_names += get_phase_names(phase_name)
    # Remove duplicates.
    return sorted(list(set(phase_names)))


def get_phase_names(phase_name):
    """
    Called by parse_phase_list to replace e.g. ttall with the relevant phases.
    """
    lphase = phase_name.lower()
    names = []
    if lphase in ("ttp", "tts", "ttbasic", "tts+", "ttp+", "ttall"):
        if lphase in ("ttp", "ttp+", "ttbasic", "ttall"):
            names.extend(["p", "P", "Pn", "Pdiff", "PKP", "PKiKP", "PKIKP"])

        if lphase in ("tts", "tts+", "ttbasic", "ttall"):
            names.extend(["s", "S", "Sn", "Sdiff", "SKS", "SKIKS"])

        if lphase in ("ttp+", "ttbasic", "ttall"):
            names.extend(["PcP", "pP", "pPdiff", "pPKP", "pPKIKP", "pPKiKP",
                          "sP", "sPdiff", "sPKP", "sPKIKP", "sPKiKP"])

        if lphase in ("tts+", "ttbasic", "ttall"):
            names.extend(["sS", "sSdiff", "sSKS", "sSKIKS", "ScS", "pS",
                          "pSdiff", "pSKS", "pSKIKS"])

        if lphase in ("ttbasic", "ttall"):
            names.extend(["ScP", "SKP", "SKIKP", "PKKP", "PKIKKIKP", "SKKP",
                          "SKIKKIKP", "PP", "PKPPKP", "PKIKPPKIKP"])

        if lphase == "ttall":
            names.extend(["SKiKP", "PP", "ScS", "PcS", "PKS", "PKIKS", "PKKS",
                          "PKIKKIKS", "SKKS", "SKIKKIKS", "SKSSKS",
                          "SKIKSSKIKS", "SS", "SP", "PS"])

    else:
        names.append(phase_name)

    return names


def split_ray_path(path, model):
    """
    Split and label ray path according to type of wave.

    :param path: Path taken by ray.
    :type path: :class:`~numpy.ndarray`
        (dtype = :class:`~obspy.taup.helper_classes.TimeDist`)
    :param model: The model used to calculate the ray path.
    :type model: :class:`obspy.taup.tau_model.TauModel`
    :returns: A list of paths and a list of wave types.
        Wave types are either "p", "s", or "diff".
    :rtype: list[:class:`~numpy.ndarray`]
        and list[str]

    The ray path is split at all discontinuities in the model
    and labelled according to wave type.
    """

    # Get discontinuity depths in the model in km
    discs = model.s_mod.v_mod.get_discontinuity_depths()[:-1]

    # Split path at discontinuity depths
    depths = path["depth"]
    is_disc = np.isin(depths, discs)
    is_disc[0] = False  # Don't split on first point
    idx = np.where(is_disc)[0]

    # Split ray path, including start and end points in each segment
    splitted = np.split(path, idx)
    paths = [
        np.append(s, splitted[i + 1][0]) for i, s in enumerate(splitted[:-1])
    ]

    # Classify the waves as p, s, or diff
    wave_types = [_classify_path(p, model) for p in paths]

    return paths, wave_types


def _expected_delay_time(ray_param, depth0, depth1, wave, model):
    """
    Expected delay time between two depths for a given wave type (p or s).
    """

    # Convert depths to radii
    radius0 = model.radius_of_planet - depth0
    radius1 = model.radius_of_planet - depth1

    # Velocity model from TauModel
    v_mod = model.s_mod.v_mod

    # Get velocities
    if depth1 >= depth0:
        v0 = v_mod.evaluate_below(depth0, wave)[0]
        v1 = v_mod.evaluate_above(depth1, wave)[0]
    else:
        v0 = v_mod.evaluate_above(depth0, wave)[0]
        v1 = v_mod.evaluate_below(depth1, wave)[0]

    # Calculate time for segment if velocity non-zero
    # - if velocity zero then return zero time
    if v0 > 0.0:

        eta0 = radius0 / v0
        eta1 = radius1 / v1

        def vertical_slowness(eta, p):
            y = eta**2 - p**2
            return np.sqrt(y * (y > 0))  # in s

        n0 = vertical_slowness(eta0, ray_param)
        n1 = vertical_slowness(eta1, ray_param)

        if ray_param == 0.0:
            return 0.5 * ((1.0 / v0) + (1.0 / v1)) * abs(radius1 - radius0)
        return 0.5 * (n0 + n1) * abs(np.log(radius1 / radius0))

    return 0.0


def _classify_path(path, model):
    """
    Determine whether we have a p or s-wave path by comparing delay times.
    """

    # Examine just the first two points near the shallowest part of the path
    if path[0]["depth"] < path[-1]["depth"]:
        point0 = path[0]
        point1 = path[1]
    else:
        point0 = path[-2]
        point1 = path[-1]

    # Ray parameter
    ray_param = point0["p"]

    # Depths
    depth0 = point0["depth"]
    depth1 = point1["depth"]

    # If no change in depth then this is a diffracted/head wave segment
    if depth0 == depth1:
        return "diff"

    # Delay time for this segment from ray path
    travel_time = point1["time"] - point0["time"]
    distance = abs(point0["dist"] - point1["dist"])
    delay_time = travel_time - ray_param * distance

    # Get the expected delay time for each wave type
    delay_p = _expected_delay_time(ray_param, depth0, depth1, "p", model)
    delay_s = _expected_delay_time(ray_param, depth0, depth1, "s", model)

    # Difference between predictions and given delay times
    error_p = (delay_p / delay_time) - 1.0
    error_s = (delay_s / delay_time) - 1.0

    # Check which wave type matches the given delay time the best
    if abs(error_p) < abs(error_s):
        return "p"
    return "s"
