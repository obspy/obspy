#!/usr/bin/env python
"""
SPEC-0 support-window planner for ObsPy.

Draws a Gantt chart of dependency support windows (SPEC-0000: 3 years for
Python, 2 years for core packages) and overlays *tentative* ObsPy release
dates, so you can read off which dependency versions each planned release
would have to support.

Edit OBSPY_RELEASES (and CORE_PACKAGES) below, then run:

    python obspy_spec0_plot.py                 # uses cached PyPI data if present
    python obspy_spec0_plot.py --refresh       # re-query PyPI
    python obspy_spec0_plot.py -o plan.png

Outputs: <name>.png  and  <name>.md (markdown table of the same info).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
import requests
from packaging.version import InvalidVersion, Version

# --------------------------------------------------------------------------
# CONFIGURATION -- this is the part you edit
# --------------------------------------------------------------------------

#: Tentative ObsPy releases: name -> date (YYYY-MM-DD).
#: Past releases are fine too, they just get a solid line instead of dashed.
OBSPY_RELEASES = {
    "1.5.0": "2026-03-01",
    "1.6.0": "2026-09-01",   # tentative
    "1.7.0": "2027-03-01",   # tentative
}

#: The release currently being worked on. Bars are faded when their SPEC-0
#: window has already closed by this date, so the shading does not silently
#: change meaning when you append a further release to OBSPY_RELEASES.
REFERENCE_RELEASE = "1.6.0"

#: Package colours. Deliberately no red in here -- red is reserved for the
#: "extended support" hatching, so adding packages never steals it.
PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#17becf",  # cyan
    "#bcbd22",  # olive
    "#7f7f7f",  # grey
]

#: What ObsPy *actually* declared as its floor, per release.
#: 1.5.0 values are verbatim from setup.py at tag 1.5.0:
#:   MIN_PYTHON_VERSION = (3, 8)  (classifiers list up to 3.14)
#:   numpy>=1.21, scipy>=1.7, matplotlib>=3.3
#: Anything below its SPEC-0 floor is drawn as "extended support".
OBSPY_ACTUAL_FLOOR = {
    "1.5.0": {
        "python": "3.8",
        "numpy": "1.21",
        "scipy": "1.7",
        "matplotlib": "3.3",
    },
}

#: Core dependencies queried from PyPI (2-year window under SPEC-0).
#: Add anything you like: "lxml", "sqlalchemy", "requests", "pandas", ...
CORE_PACKAGES = ["numpy", "scipy", "matplotlib"]

#: Python release dates (PyPI cannot tell us these).
PYTHON_RELEASES = {
    "3.8": "2019-10-14",
    "3.9": "2020-10-05",
    "3.10": "2021-10-04",
    "3.11": "2022-10-24",
    "3.12": "2023-10-02",
    "3.13": "2024-10-07",
    "3.14": "2025-10-07",
    "3.15": "2026-10-07",  # projected
    "3.16": "2027-10-05",  # projected
}

#: SPEC-0000 support windows.
WINDOW_PYTHON = timedelta(days=int(365.25 * 3))
WINDOW_CORE = timedelta(days=int(365.25 * 2))

#: How much history to draw before the first ObsPy release on the chart.
#: Versions whose support window ended before that are simply not shown.
HISTORY = timedelta(days=365 * 3)

#: Extrapolate future dependency releases from each project's median cadence,
#: so that the *upper* bound of a 2027 release is not artificially capped by
#: whatever is on PyPI today. Projected versions are hatched and marked "*".
PROJECT_FUTURE = True

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec0_cache.json")


# --------------------------------------------------------------------------
# DATA COLLECTION
# --------------------------------------------------------------------------


def _parse_upload_time(value: str):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def query_pypi(package: str) -> dict[str, str]:
    """Return {minor version: ISO release date} for a package on PyPI.

    Only X.Y.0 final releases are considered, and the release date is the
    upload time of the *first* file for that version.
    """
    print(f"Querying pypi.org for {package} ...", end="", flush=True)
    response = requests.get(
        f"https://pypi.org/simple/{package}",
        headers={"Accept": "application/vnd.pypi.simple.v1+json"},
        timeout=30,
    )
    response.raise_for_status()
    files = response.json()["files"]
    print(" OK")

    file_dates = collections.defaultdict(list)
    for f in files:
        try:
            raw = f["filename"].split("-")[1]
            version = Version(raw)
        except (IndexError, InvalidVersion):
            continue
        if version.is_prerelease or version.micro != 0:
            continue
        stamp = _parse_upload_time(f.get("upload-time", ""))
        if stamp is not None:
            file_dates[f"{version.major}.{version.minor}"].append(stamp)

    return {v: min(dates).isoformat() for v, dates in file_dates.items()}


def load_releases(refresh: bool = False) -> dict[str, dict[str, datetime]]:
    """Collect release dates for Python + all core packages, with caching."""
    cache = {}
    if os.path.exists(CACHE_FILE) and not refresh:
        with open(CACHE_FILE) as fh:
            cache = json.load(fh)

    missing = [p for p in CORE_PACKAGES if p not in cache]
    if refresh or missing:
        for package in CORE_PACKAGES if refresh else missing:
            cache[package] = query_pypi(package)
        with open(CACHE_FILE, "w") as fh:
            json.dump(cache, fh, indent=1, sort_keys=True)
        print(f"Cached PyPI data in {CACHE_FILE}")

    releases = {
        "python": {v: datetime.fromisoformat(d) for v, d in PYTHON_RELEASES.items()}
    }
    for package in CORE_PACKAGES:
        releases[package] = {
            v: datetime.fromisoformat(d) for v, d in cache[package].items()
        }
    return releases


def window_for(package: str) -> timedelta:
    return WINDOW_PYTHON if package == "python" else WINDOW_CORE


# --------------------------------------------------------------------------
# SPEC-0 LOGIC
# --------------------------------------------------------------------------


def project_future(releases: dict[str, datetime], until: datetime, n_hist: int = 6):
    """Append plausible future minor releases based on the median cadence.

    Returns the set of version strings that were invented (not real releases).
    """
    known = sorted(releases, key=Version)
    if len(known) < 3:
        return set()
    dates = [releases[v] for v in known[-n_hist:]]
    gaps = sorted((b - a).days for a, b in zip(dates, dates[1:]))
    cadence = timedelta(days=max(60, gaps[len(gaps) // 2]))

    version, date = Version(known[-1]), releases[known[-1]]
    projected = set()
    while date + cadence <= until:
        date += cadence
        version = Version(f"{version.major}.{version.minor + 1}")
        releases[str(version)] = date
        projected.add(str(version))
    return projected


def global_floor(package: str) -> str | None:
    """Lowest version any listed ObsPy release declares for this package."""
    floors = [
        f[package] for f in OBSPY_ACTUAL_FLOOR.values() if package in f
    ]
    return min(floors, key=Version) if floors else None


def trim(releases: dict[str, datetime], package: str, start: datetime, end: datetime):
    """Keep versions that are relevant to the chart.

    The lower bound is ObsPy's own declared floor when we know it (that is the
    whole point of the figure), otherwise the SPEC-0 history window.
    """
    floor = global_floor(package)
    window = window_for(package)
    return {
        v: d
        for v, d in releases.items()
        if d <= end
        and (Version(v) >= Version(floor) if floor else d + window >= start)
    }


def supported_at(releases: dict[str, datetime], date: datetime, window: timedelta):
    """Versions still supported at `date` under SPEC-0.

    A version is supported until `release_date + window`. If that would leave
    nothing (a very quiet upstream project), the most recent release is kept
    so the answer is never empty.
    """
    released = {v: d for v, d in releases.items() if d <= date}
    if not released:
        return []
    keep = [v for v, d in released.items() if d + window > date]
    if not keep:
        keep = [max(released, key=released.get)]
    return sorted(keep, key=Version)


def build_matrix(releases, obspy_releases):
    """{obspy version: {package: (min_version, max_version)}}."""
    matrix = {}
    for name, date in obspy_releases.items():
        row = {}
        for package, vers in releases.items():
            sup = supported_at(vers, date, window_for(package))
            row[package] = (sup[0], sup[-1]) if sup else (None, None)
        matrix[name] = row
    return matrix


# --------------------------------------------------------------------------
# PLOT
# --------------------------------------------------------------------------


def plot(releases, obspy_releases, matrix, projected, xlim, outfile):
    packages = list(releases)
    colors = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(packages)}
    extended_color = "#b2182b"

    # one row per (package, version): packages top to bottom, oldest first
    rows = []
    for package in packages:
        for version in sorted(releases[package], key=Version):
            rows.append((package, version))
    rows.reverse()
    y_of = {row: y for y, row in enumerate(rows)}

    reference = obspy_releases.get(REFERENCE_RELEASE, max(obspy_releases.values()))
    xmin, xmax = xlim
    n_rows = len(rows)

    height = 3.3 + 0.24 * n_rows
    fig, ax = plt.subplots(figsize=(15, height))
    fig.subplots_adjust(
        left=0.10, right=0.985,
        top=1 - 2.15 / height, bottom=0.75 / height,
    )

    # ---- SPEC-0 support windows -----------------------------------------
    labels = []
    for y, (package, version) in enumerate(rows):
        released = releases[package][version]
        dropped = released + window_for(package)
        # leader line from the tick label to where the bar actually starts
        ax.hlines(
            y, mdates.date2num(xmin), mdates.date2num(released),
            color=colors[package], lw=0.7, ls=(0, (1, 2.5)), alpha=0.8, zorder=1,
        )
        ax.barh(
            y,
            mdates.date2num(dropped) - mdates.date2num(released),
            left=mdates.date2num(released),
            height=0.60,
            color=colors[package],
            alpha=0.30 if dropped <= reference else 0.85,
            edgecolor=colors[package],
            linewidth=1.0,
            hatch="///" if version in projected[package] else None,
            zorder=3,
        )
        star = "*" if version in projected[package] else ""
        labels.append(f"{version}{star}")

    # ---- what ObsPy actually kept alive beyond the SPEC-0 window ---------
    for name, floors in OBSPY_ACTUAL_FLOOR.items():
        if name not in obspy_releases:
            continue
        obspy_date = obspy_releases[name]
        for package, floor in floors.items():
            if package not in releases:
                continue
            for version, released in releases[package].items():
                if Version(version) < Version(floor) or released > obspy_date:
                    continue
                dropped = released + window_for(package)
                if dropped >= obspy_date:
                    continue  # still inside its SPEC-0 window, nothing extended
                ax.barh(
                    y_of[(package, version)],
                    mdates.date2num(obspy_date) - mdates.date2num(dropped),
                    left=mdates.date2num(dropped),
                    height=0.60,
                    color=extended_color,
                    alpha=0.28,
                    edgecolor=extended_color,
                    linewidth=0.6,
                    hatch="\\\\",
                    zorder=2,
                )
            # bracket + label on the floor row
            y_floor = y_of[(package, floor)]
            n_supported = sum(
                1 for v, d in releases[package].items()
                if Version(v) >= Version(floor) and d <= obspy_date
            )
            ax.annotate(
                f"obspy {name}: {package} >= {floor}  ({n_supported} versions)",
                xy=(mdates.date2num(obspy_date), y_floor),
                xytext=(-7, 0), textcoords="offset points",
                va="center", ha="right", fontsize=8.5,
                color=extended_color, fontweight="bold", zorder=7,
            )

    # ---- SPEC-0 floor markers -------------------------------------------
    # a row can be the floor for several ObsPy releases; the version number is
    # written once, next to the earliest dot on that row.
    floor_dots = collections.defaultdict(list)
    for name, date in obspy_releases.items():
        for package in packages:
            floor = matrix[name][package][0]
            if floor is not None:
                floor_dots[(package, floor)].append(date)

    for (package, version), dates in floor_dots.items():
        y = y_of[(package, version)]
        for date in dates:
            ax.plot(
                mdates.date2num(date), y,
                marker="o", ms=7, color="white", mec="black", mew=1.4, zorder=6,
            )
        # label to the left of the first dot, inside the bar -- unless the bar
        # only just started there, in which case flip to the right
        first = min(dates)
        room = (first - releases[package][version]).days
        ax.annotate(
            version,
            xy=(mdates.date2num(first), y),
            xytext=(-8 if room > 220 else 8, 0),
            textcoords="offset points",
            ha="right" if room > 220 else "left",
            va="center", fontsize=7.5, fontweight="bold", color="black",
            zorder=7,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
        )

    # ---- ObsPy release lines --------------------------------------------
    for name, date in obspy_releases.items():
        future = date > datetime.now()
        is_ref = name == REFERENCE_RELEASE
        ax.axvline(
            mdates.date2num(date), color="black", lw=2.6 if is_ref else 1.4,
            ls="--" if future else "-", alpha=0.9 if is_ref else 0.65, zorder=5,
        )
        ax.text(
            mdates.date2num(date), 1.004,
            f"obspy {name}",
            transform=ax.get_xaxis_transform(),
            rotation=90, va="bottom", ha="center",
            fontsize=10.5 if is_ref else 9.5, fontweight="bold",
            zorder=8, clip_on=False,
        )

    ax.axvspan(
        mdates.date2num(datetime.now()), mdates.date2num(xmax),
        color="grey", alpha=0.07, zorder=0,
    )

    # ---- cosmetics -------------------------------------------------------
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=8.5, color="#333333")
    ax.tick_params(axis="y", length=0, pad=4)

    # package name once per block, as a bracket in the left margin
    trans = ax.get_yaxis_transform()
    for package in packages:
        ys = [y for y, (pkg, _) in enumerate(rows) if pkg == package]
        lo, hi = min(ys) - 0.42, max(ys) + 0.42
        ax.plot(
            [-0.043, -0.043], [lo, hi], transform=trans, clip_on=False,
            color=colors[package], lw=2.4, solid_capstyle="butt", zorder=9,
        )
        ax.text(
            -0.056, (lo + hi) / 2, package, transform=trans, clip_on=False,
            rotation=90, va="center", ha="center", fontsize=11,
            fontweight="bold", color=colors[package], zorder=9,
        )
        # faint separator between blocks
        ax.axhline(hi, color="0.85", lw=0.8, zorder=0)
    ax.set_ylim(-0.8, n_rows - 0.2)
    ax.set_xlim(mdates.date2num(xmin), mdates.date2num(xmax))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.grid(axis="x", which="major", ls=":", alpha=0.5, zorder=1)
    ax.set_axisbelow(True)
    ax.set_title(
        f"ObsPy releases vs. SPEC-0 support windows  (shading keyed to {REFERENCE_RELEASE}, in progress)\n"
        "bar = inside the SPEC-0 window (3 yr Python / 2 yr core)  ·  "
        "hatched red = support ObsPy 1.5.0 carried beyond it  ·  "
        "○ = SPEC-0 floor  ·  * = projected release",
        fontsize=11, pad=88,
    )
    # the first handle is a strip of all four package colours, so the legend
    # swatch actually looks like something present in the figure
    fig.legend(
        handles=[
            tuple(
                Patch(facecolor=colors[p], edgecolor=colors[p], alpha=0.85)
                for p in packages
            ),
            tuple(
                Patch(facecolor=colors[p], edgecolor=colors[p], alpha=0.30)
                for p in packages
            ),
            Patch(facecolor=extended_color, alpha=0.28, hatch="\\\\",
                  edgecolor=extended_color),
        ],
        labels=[
            f"SPEC-0 window still open at {REFERENCE_RELEASE}",
            f"window already closed by {REFERENCE_RELEASE}",
            "extended support carried by ObsPy 1.5.0",
        ],
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
        handlelength=4.0,
        loc="lower right", bbox_to_anchor=(0.985, 0.002),
        ncol=3, fontsize=9, framealpha=0.95,
    )

    fig.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")
    return fig


def format_range(bounds, projected):
    lo, hi = bounds
    if lo is None:
        return "-"
    hi_str = f"{hi}*" if hi in projected else f"{hi}"
    return hi_str if lo == hi else f"{lo} – {hi_str}"


def write_markdown(matrix, obspy_releases, packages, projected, outfile):
    lines = ["| ObsPy | date | " + " | ".join(packages) + " |"]
    lines.append("|" + "---|" * (len(packages) + 2))
    for name, date in sorted(obspy_releases.items(), key=lambda kv: kv[1]):
        cells = [
            format_range(matrix[name][p], projected[p]) for p in packages
        ]
        lines.append(
            f"| {name} | {date:%Y-%m-%d} | " + " | ".join(cells) + " |"
        )
    text = "\n".join(lines) + "\n"
    with open(outfile, "w") as fh:
        fh.write(text)
    print(f"Saved {outfile}\n")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-query PyPI")
    parser.add_argument("-o", "--output", default="obspy_spec0.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    releases = load_releases(refresh=args.refresh)
    obspy_releases = {
        k: datetime.fromisoformat(v) for k, v in OBSPY_RELEASES.items()
    }

    start = min(obspy_releases.values()) - HISTORY
    for name, floors in OBSPY_ACTUAL_FLOOR.items():
        if name not in obspy_releases:
            continue
        for package, floor in floors.items():
            if floor in releases.get(package, {}):
                start = min(start, releases[package][floor] - timedelta(days=90))
    end = max(obspy_releases.values()) + timedelta(days=270)

    projected = {}
    for package in releases:
        projected[package] = (
            project_future(releases[package], end) if PROJECT_FUTURE else set()
        )
        releases[package] = trim(releases[package], package, start, end)
        projected[package] &= set(releases[package])

    matrix = build_matrix(releases, obspy_releases)

    plot(releases, obspy_releases, matrix, projected, (start, end), args.output)
    write_markdown(
        matrix, obspy_releases, list(releases), projected,
        os.path.splitext(args.output)[0] + ".md",
    )
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()