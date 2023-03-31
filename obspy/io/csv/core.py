"""
.. note::
    This module can be easily extended to write and read more event
    information.
    If you are interested, please send a PR to the github repository.
      1. Add 'extended' or similar key to FIELDS dict, e.g. as a start use
          'extended': (
              '{time!s:.25} {lat:.6f} {lat_err:.6f} {lon:.6f} {lon_err:.6f} '
              '{dep:.3f} {dep_err:.3f} {mag:.2f} {mag_err:.2f} {magtype} {id}'
              )
         You can also test by just passing the string to `fields` option.
      2. Implement writing functionality in write_csv by adding the new
         properties to the dict d. Missing values have to be handled.
      3. Implement reading functionality for the new properties in read_csv.
      4. Write some tests testing all new properties. Check that an event with
         all new properties defined or missing can be written and read again.
    Similar can be done for the picks using PFIELDS, _write_picks, _read_picks
"""

import csv
from contextlib import contextmanager
import io
import math
import pathlib
from string import Formatter
from warnings import warn
import zipfile


import numpy as np
from obspy import UTCDateTime as UTC, __version__
from obspy.core import event as evmod
from obspy.core.util.decorator import map_example_filename


# for reading
DEFAULT = {'magtype': ''}
# for writing
FIELDS = {
    'basic': (
        '{id} {time!s:.25} {lat:.6f} {lon:.6f} {dep:.3f} '
        '{magtype} {mag:.2f}'),
    'eventtxt': (
        '{id},{time!s:.25},{lat:.6f},{lon:.6f},{dep:.3f},{author},,{contrib},,'
        '{magtype},{mag:.2f},{magauthor},{region}').split(',')
}
PFIELDS = {
    'basic': '{seedid} {phase} {time:.5f} {weight:.3f}'
}
CSZ_COMMENT = f'CSZ format ObsPy v{__version__} obspy_no_uncompress'.encode(
    'utf-8')
# for load_csv
DTYPE = {
    'time': 'datetime64[ms]',
    'lat': float,
    'lon': float,
    'dep': float,
    'mag': float,
    'magtype': 'U10',
    'id': 'U50'
}

# catalog and contribid are not used
EVENTTXT_NAMES = (  # for reading
    'id time lat lon dep author catalog contrib contribid '
    'magtype mag magauthor region')
EVENTTXT_HEADER = (  # for writing
    '#EventID | Time | Latitude | Longitude | Depth/km | Author | Catalog | '
    'Contributor | ContributorID | MagType | Magnitude | MagAuthor | '
    'EventLocationName')


def _is_csv(fname, **kwargs):
    try:
        return _read_csv(fname, format_check=True)
    except Exception:
        return False


def _is_eventtxt(fname, **kwargs):
    try:
        return _read_eventtxt(fname, format_check=True)
    except Exception:
        return False


def _is_csz(fname, **kwargs):
    try:
        if not zipfile.is_zipfile(fname):
            return False
        with zipfile.ZipFile(fname) as zipf:
            if not (zipf.comment.startswith(b'CSZ') and
                    zipf.comment.endswith(b'obspy_no_uncompress')):
                return False
        return True
    except Exception:
        return False


def _evid(event):
    return str(event.resource_id).split('/')[-1]


def _origin(event):
    return event.preferred_origin() or event.origins[0]


@contextmanager
def _open(filein, *args, **kwargs):
    """Accept files or file names or pathlib objects"""
    if isinstance(filein, (str, pathlib.PurePath)):  # filename
        with open(filein, *args, **kwargs) as f:
            yield f
    else:  # file-like object
        yield filein


def _names_sequence(names):
    if isinstance(names, dict):
        names = [names.get(i, '_') for i in range(max(names.keys())+1)]
    elif ' ' in names:
        names = names.split()
    return names


def _string(row, key):
    if key in row and row[key].strip() != '':
        return row[key].strip()


def _read_csv(fname, skipheader=0, default=None, names=None,
              format_check=False, **kwargs):
    """
    Read a CSV file and return ObsPy catalog

    :param fname: filename or file-like object to read from
    :param skipheader: skip first rows of file
    :param default: dictionary with default values, at the moment only
         magtype is supported,
         i.e. to set magtypes use ``default={'magtype': 'Ml'}``
    :param names: determined automatically from header line of file,
        otherwise can be specified as string, sequence or dict
    :param format_check: Check if the first event can be read
    :param **kargs: all other kwargs are passed to csv.DictReader,
        important additional arguments are fieldnames, dialect, delimiter, etc

    Example
    -------

    >>> from obspy import read_events
    >>> events = read_events('/path/to/catalog.csv')
    >>> print(events)  # doctest: +NORMALIZE_WHITESPACE
    3 Event(s) in Catalog:
    2012-04-04T14:21:42.300000Z | +41.818,  +79.689 | 4.4  mb
    2012-04-04T14:18:37.000000Z | +39.342,  +41.044 | 4.3  ML
    2012-04-04T14:08:46.000000Z | +38.017,  +37.736 | 3.0  ML

    You can read CSV files created by this module or external CSV files.
    Example reading an external CSV file:

    >>> from obspy import read_events
    >>> names = 'year mon day hour minu sec _ lat lon dep mag id'
    >>> events = read_events(
    ...     '/path/to/external.csv', 'CSV', skipheader=1, names=names)
    >>> print(events)  # doctest: +NORMALIZE_WHITESPACE
    1 Event(s) in Catalog:
    2023-05-06T19:55:01.300000Z | +10.194, +124.830 | 0.2
    """
    if default is None:
        default = DEFAULT
    events = []
    with _open(fname) as f:
        for _ in range(skipheader):
            f.readline()
        if names is not None:
            kwargs.setdefault('fieldnames', _names_sequence(names))
        reader = csv.DictReader(f, **kwargs)
        for row in reader:
            if 'time' in row:
                time = UTC(row['time'])
            else:
                time = UTC(
                    '{year}-{mon}-{day} {hour}:{minu}:{sec}'.format(**row))
            try:
                if 'depm' in row:
                    dep = float(row['depm'])
                else:
                    dep = float(row['dep']) * 1000
                if math.isnan(dep):
                    raise
            except Exception:
                dep = None
            author = _string(row, 'author')
            contrib = _string(row, 'contrib')
            if author is not None or contrib is not None:
                info = evmod.CreationInfo(author=author, agency_id=contrib)
            else:
                info = None
            origin = evmod.Origin(
                time=time,
                latitude=row['lat'],
                longitude=row['lon'],
                depth=dep,
                creation_info=info
            )
            try:
                # add zero to eliminate negative zeros in magnitudes
                mag = float(row['mag']) + 0
                if math.isnan(mag):
                    raise
            except Exception:
                magnitudes = []
            else:
                try:
                    magtype = row['magtype']
                    if magtype.lower() in ('', 'none', 'null', 'nan'):
                        raise
                except Exception:
                    magtype = default.get('magtype')
                magauthor = _string(row, 'magauthor')
                info = (evmod.CreationInfo(author=magauthor) if magauthor
                        else None)
                magnitudes = [evmod.Magnitude(
                    mag=mag, magnitude_type=magtype, creation_info=info)]
            try:
                id_ = evmod.ResourceIdentifier(row['id'].strip())
            except Exception:
                id_ = None
            region = _string(row, 'region')
            descs = ([evmod.EventDescription(region, 'region name')]
                     if region else [])
            event = evmod.Event(
                magnitudes=magnitudes,
                origins=[origin],
                resource_id=id_,
                event_descriptions=descs
            )
            events.append(event)
            if format_check:
                return True
    if format_check:
        # empty file will return an empty catalog,
        # but it is not detected as CSV file
        return False
    return evmod.Catalog(events=events)


def _write_csv(events, fname, fields='basic', depth_in_km=True, delimiter=',',
               header=None):
    """
    Write ObsPy catalog to CSV file

    :param events: catalog or list of events
    :param fname: filename or file-like object to write to
    :param fields: set format and header names of CSV file, see
       :const:`obspy.io.csv.FIELDS`, you can use your own format string here,
       just make sure to use the pre-defined header names
    :param depth_in_km: write depth in units of kilometer (default: True) or
        meter
    :param delimiter: defaults to `','`, if the delimiter is changed, ObsPy's
        read_events function will not automatically identify the file as
        CSV file
    :param header: Use a non-default header row

    .. warning::
        If the parameters `delimiter` or `header` are changed,
        ObsPy's read_events function will not automatically identify the file
        as CSV file

    Example
    -------

    >>> from obspy import read_events
    >>> events = read_events()  # get example catalog
    >>> events.write('local_catalog.csv', 'CSV')  # declare 'CSV' as format
    >>> with open('local_catalog.csv') as f: print(f.read())
    id,time,lat,lon,dep,magtype,mag
    20120404_0000041,2012-04-04T14:21:42.30000,41.818000,79.689000,1.000,mb,4.40
    20120404_0000038,2012-04-04T14:18:37.00000,39.342000,41.044000,14.400,ML,4.30
    20120404_0000039,2012-04-04T14:08:46.00000,38.017000,37.736000,7.000,ML,3.00
    <BLANKLINE>
    """
    fields = FIELDS.get(fields, fields)
    if ' ' in fields:
        fields = fields.split()
    fmtstr = delimiter.join(fields)
    if not depth_in_km and 'depm' not in fmtstr:
        fmtstr = fmtstr.replace('dep', 'depm')
    if header is None:
        fieldnames = [
            fn for _, fn, _, _ in Formatter().parse(fmtstr) if fn is not None]
        header = delimiter.join(fieldnames)
    with _open(fname, 'w') as f:
        f.write(header + '\n')
        for event in events:
            evid = str(event.resource_id).split('/')[-1]
            try:
                origin = _origin(event)
            except Exception:
                warn(f'No origin found -> do not write event {evid}')
                continue
            try:
                author = origin.creation_info.author
            except AttributeError:
                author = ''
            try:
                contrib = origin.creation_info.agency_id
            except AttributeError:
                contrib = ''
            try:
                magnitude = event.preferred_magnitude() or event.magnitudes[0]
            except Exception:
                warn(f'No magnitude found for event {evid}')
                mag = float('nan')
                magtype = ''
                magauthor = ''
            else:
                mag = magnitude.mag
                magtype = magnitude.magnitude_type or ''
                try:
                    magauthor = magnitude.creation_info.author
                except AttributeError:
                    magauthor = ''
            try:
                dep = origin.depth / (1000 if depth_in_km else 1)
            except Exception:
                warn(f'No depth set for event {evid}')
                dep = float('nan')
            for description in event.event_descriptions:
                if str(description.type) == 'region name':
                    region = description.text
                    break
            else:
                region = ''
            d = {'time': origin.time,
                 'lat': origin.latitude,
                 'lon': origin.longitude,
                 'dep' if depth_in_km else 'depm': dep,
                 'mag': mag,
                 'magtype': magtype,
                 'id': evid,
                 'author': author,
                 'contrib': contrib,
                 'magauthor': magauthor,
                 'region': region,
                 }
            f.write(fmtstr.format(**d).replace('nan', '') + '\n')


@map_example_filename('fname')
def load_csv(fname, skipheader=0, only=None, names=None,
             delimiter=',', **kw):
    """
    Load CSV or CSZ file into numpy array

    :param skipheader, names: see :func:`_read_csv`
    :param only: sequence, read only columns speified by name
    :param **kw: Other kwargs are passed to :func:`numpy.loadtxt`

    For an example see :mod:`obspy.io.csv`.
    """
    if isinstance(fname, str) and zipfile.is_zipfile(fname):
        with zipfile.ZipFile(fname) as zipf:
            with io.TextIOWrapper(
                    zipf.open('events.csv'), encoding='utf-8') as f:
                return load_csv(f)
    with _open(fname) as f:
        for _ in range(skipheader):
            f.readline()
        if names is None:
            names = f.readline().strip().split(',')
        names = _names_sequence(names)
        dtype = [(n, DTYPE[n]) for n in names if n in DTYPE and
                 (only is None or n in only)]
        usecols = [i for i, n in enumerate(names) if n in DTYPE and
                   (only is None or n in only)]
        kw.setdefault('usecols', usecols)
        kw.setdefault('dtype', dtype)
        return np.genfromtxt(f, delimiter=delimiter, **kw)


def _events2array(events, **kw):
    """
    Convert ObsPy catalog to numpy array

    All kwargs are passed to :func:`load_csv`, e.g. use
    `only=('lat', 'lon', 'mag')`
    to get an array with lat, lon, mag parameters.

    Example
    -------

    >>> from obspy import read_events
    >>> from obspy.io.csv import _events2array
    >>> events = read_events()
    >>> t = _events2array(events)
    >>> print(t.dtype.names)
    ('id', 'time', 'lat', 'lon', 'dep', 'magtype', 'mag')
    >>> print(t)   # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    [ ('20120404_0000041', '2012-04-04...', 41.818, 79.689, 1. , 'mb', 4.4)
     ('20120404_0000038', '2012-04-04T...', 39.342, 41.044, 14.4, 'ML', 4.3)
     ('20120404_0000039', '2012-04-04T...', 38.017, 37.736, 7. , 'ML', 3. )]
    >>> print(t['mag'])
    [ 4.4  4.3  3. ]
    """
    with io.StringIO() as f:
        _write_csv(events, f)
        f.seek(0)
        return load_csv(f, **kw)


def _read_eventtxt(fname, default=None, format_check=False):
    """
    Read EVENTTXT file and return ObsPy catalog

    :param fname: filename or file-like object to read from
    :param default: dictionary with default values, at the moment only
         magtype is supported,
         i.e. to set magtypes use ``default={'magtype': 'Ml'}``
    :param format_check: Check if the first event can be read

    For an example see :mod:`obspy.io.csv`.
    """
    return _read_csv(fname,
                     skipheader=1, names=EVENTTXT_NAMES, delimiter='|',
                     default=default, format_check=format_check)


def _write_eventtxt(events, fname):
    """
    Write ObsPy catalog to EVENTTXT file

    :param events: catalog or list of events
    :param fname: filename or file-like object to write to

    For an example see :mod:`obspy.io.csv`.
    """
    return _write_csv(events, fname, fields='eventtxt', header=EVENTTXT_HEADER,
                      delimiter='|')


@map_example_filename('fname')
def load_eventtxt(fname, **kw):
    """
    Load EVENTTXT file into numpy array

    For possible arguments see :func:`load_csv`.
    For an example see :mod:`obspy.io.csv`.
    """
    kw.setdefault('delimiter', '|')
    kw.setdefault('skipheader', 1)
    return load_csv(fname, names=EVENTTXT_NAMES, **kw)


def _read_picks(event, fname):
    """
    Read picks from CSV file and add them to event
    """
    otime = _origin(event).time
    picks = []
    arrivals = []
    with _open(fname) as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase = row['phase']
            seedid = row['seedid']
            wid = (evmod.WaveformStreamID(seed_string=seedid) if seedid
                   else None)
            pick = evmod.Pick(waveform_id=wid, phase_hint=phase,
                              time=otime + float(row['time']))
            arrival = evmod.Arrival(phase=phase, pick_id=pick.resource_id,
                                    time_weight=float(row['weight']))
            picks.append(pick)
            arrivals.append(arrival)
    event.picks = picks
    event.origins[0].arrivals = arrivals


def _write_picks(event, fname, fields_picks='basic', delimiter=','):
    """
    Write picks from event to a CSV file
    """
    fields = PFIELDS.get(fields_picks, fields_picks)
    if ' ' in fields:
        fields = fields.split()
    fmtstr = delimiter.join(fields)
    fieldnames = [
        fn for _, fn, _, _ in Formatter().parse(fmtstr) if fn is not None]
    origin = _origin(event)
    weights = {str(arrival.pick_id): arrival.time_weight
               for arrival in origin.arrivals if arrival.time_weight}
    phases = {str(arrival.pick_id): arrival.phase
              for arrival in origin.arrivals if arrival.phase}
    with _open(fname, 'w') as f:
        f.write(delimiter.join(fieldnames) + '\n')
        for pick in event.picks:
            pick_id = str(pick.resource_id)
            try:
                seedid = pick.waveform_id.id
                if seedid is None:
                    raise AttributeError
            except AttributeError:
                warn(f'No waveform id found for pick {pick_id}')
                seedid = ''
            d = {'time': pick.time - origin.time,
                 'seedid': seedid,
                 'phase': phases.get(pick_id, pick.phase_hint),
                 'weight': weights.get(pick_id, 1.)}
            f.write(fmtstr.format(**d) + '\n')


def _read_csz(fname, default=None):
    """
    Read a CSZ file and return ObsPy catalog with picks

    :param fname: filename or file-like object to read from
    :param default: dictionary with default values, at the moment only
         magtype is supported,
         i.e. to set magtypes use ``default={'magtype': 'Ml'}``

    For an example see :mod:`obspy.io.csv`.
    """
    with zipfile.ZipFile(fname) as zipf:
        with io.TextIOWrapper(zipf.open('events.csv'), encoding='utf-8') as f:
            events = _read_csv(f, default=default)
        for event in events:
            evid = _evid(event)
            fname = f'picks_{evid}.csv'
            if fname not in zipf.namelist():
                continue
            with io.TextIOWrapper(zipf.open(fname), encoding='utf-8') as f:
                _read_picks(event, f)
    return events


def _write_csz(events, fname, fields='basic', fields_picks='basic', **kwargs):
    """
    Write ObsPy catalog to CSZ file

    :param events: catalog or list of events
    :param fname: filename or file-like object to write to
    :param fields: set format and header names of CSV file, see
       :const:`obspy.io.csv.FIELDS`, you can use your own format string here,
       just make sure to use the pre-defined header names
    :param fields_picks: set format and header names of CSV pick files, see
       :const:`obspy.io.csv.PFIELDS`, you can use your own format string here,
       just make sure to use the pre-defined header names
    :param **kwargs:
 compression and compression level can be specified see
 `zipfile doc <https://docs.python.org/library/zipfile.html#zipfile.ZipFile>`_:
        ```
        events.write('CSZ', compression=True, compresslevel=9)
        ```

    For an example see :mod:`obspy.io.csv`.
    """
    # allow True as value for compression
    if kwargs.get('compression') is True:
        kwargs['compression'] = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(fname, mode='w', **kwargs) as zipf:
        zipf.comment = CSZ_COMMENT
        with io.StringIO() as f:
            _write_csv(events, f, fields=fields)
            zipf.writestr('events.csv', f.getvalue())
        for event in events:
            if len(event.picks) == 0:
                continue
            evid = str(event.resource_id).split('/')[-1]
            try:
                _origin(event)
            except Exception:
                continue
            with io.StringIO() as f:
                _write_picks(event, f, fields_picks=fields_picks)
                zipf.writestr(f'picks_{evid}.csv', f.getvalue())
