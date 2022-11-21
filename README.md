<img alt="ObsPy: A Python Toolbox for seismology/seismological observatories." class="right" style="width: 60%" src="https://raw.github.com/obspy/website/master/logo/obspy_logo_full_highres.png" />

[![NumFOCUS affiliated project](https://numfocus.org/wp-content/uploads/2018/01/optNumFocus_LRG.png)](https://numfocus.org/sponsored-projects/affiliated-projects)

[![Github Action Status](https://github.com/obspy/obspy/workflows/tests/badge.svg?event=push)](https://github.com/obspy/obspy/actions)
[![Coverage Status](https://codecov.io/gh/obspy/obspy/branch/master/graph/badge.svg)](https://codecov.io/gh/obspy/obspy)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/obspy.svg)](https://pypi.python.org/pypi/obspy/)

[![License](https://img.shields.io/pypi/l/obspy.svg)](https://pypi.python.org/pypi/obspy/)
[![LGPLv3](https://www.gnu.org/graphics/lgplv3-88x31.png)](https://www.gnu.org/licenses/lgpl.html)

[![PyPI Version](https://img.shields.io/pypi/v/obspy.svg)](https://pypi.python.org/pypi/obspy)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6645832.svg)](https://doi.org/10.5281/zenodo.6645832)
[![Conda](https://img.shields.io/conda/dn/conda-forge/obspy?label=conda%20downloads)](https://anaconda.org/conda-forge/obspy)

[![Discourse status](https://img.shields.io/discourse/status?server=https%3A%2F%2Fdiscourse.obspy.org)](https://discourse.obspy.org)
[![Gitter](https://badges.gitter.im/JoinChat.svg)](https://gitter.im/obspy/obspy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Announcements Mailing List](https://img.shields.io/badge/mailing%20list-announcements-blue)](https://mail.python.org/mailman3/lists/obspy.python.org/)

[![Twitter Follow](https://img.shields.io/twitter/follow/obspy?style=social)](https://twitter.com/obspy/)
<a rel="me" href="https://fosstodon.org/@obspy"><img src="https://img.shields.io/mastodon/follow/109309284431089274?domain=https%3A%2F%2Ffosstodon.org&style=social" alt="Mastodon Follow" /></a>

ObsPy is an open-source project dedicated to provide a **Python framework for processing seismological** data. It provides parsers for common file formats, clients to access data centers and seismological signal processing routines which allow the manipulation of seismological time series (see [Beyreuther et al. 2010](http://www.seismosoc.org/publications/SRL/SRL_81/srl_81-3_es/), [Megies et al. 2011](http://www.annalsofgeophysics.eu/index.php/annals/article/view/4838), [Krischer et al. 2015](http://iopscience.iop.org/article/10.1088/1749-4699/8/1/014003)).

The goal of the ObsPy project is to facilitate **rapid application development for seismology**.

ObsPy is licensed under the GNU Lesser General Public License (LGPL) v3.0.

A one-hour introduction to ObsPy is [available at YouTube](https://www.youtube.com/watch?v=kFwdjfiK4gk).

#### [Read more in our GitHub wiki](https://github.com/obspy/obspy/wiki)

### Installation

Installation instructions can be found in the [wiki](https://github.com/obspy/obspy/wiki#installation).

### Getting started

Read about how to get started in the [wiki](https://github.com/obspy/obspy/wiki#getting-started) and in our [**Tutorial** section in the documentation](http://docs.obspy.org/tutorial/).

ObsPy Tutorial notebooks -- and much more on specific seismology topics -- can also be found on [**Seismo-Live**](http://seismo-live.org/), both as a static preview and as interactively runnable version.

[![Link to Seismo-Live](https://user-images.githubusercontent.com/1842780/75337134-b4310a80-588c-11ea-8ed2-dbabdedaedfc.png)](http://seismo-live.org/)

```python
from obspy import read
st = read()  # load example seismogram
st.filter(type='highpass', freq=3.0)
st = st.select(component='Z')
st.plot()
```

![Example waveform Plot](https://user-images.githubusercontent.com/1842780/75334711-9d88b480-5888-11ea-8bc8-0bfe7021d79e.png)

### Documentation and Changelog

The **detailed changelog** is available [here](CHANGELOG.txt), our docs can be found at [docs.obspy.org](http://docs.obspy.org/).

### Contributing

Please see details on how to contribute to the project [here](CONTRIBUTING.md).

### References

  * Moritz Beyreuther, Robert Barsch, Lion Krischer, Tobias Megies, Yannik Behr and Joachim Wassermann (2010), [ObsPy: A Python Toolbox for Seismology](http://www.seismosoc.org/publications/SRL/SRL_81/srl_81-3_es/), _SRL_, 81(3), 530-533,  doi:`10.1785/gssrl.81.3.530`.
  * Tobias Megies, Moritz Beyreuther, Robert Barsch, Lion Krischer, Joachim Wassermann (2011), [ObsPy – What can it do for data centers and observatories?](http://www.annalsofgeophysics.eu/index.php/annals/article/view/4838) _Annals Of Geophysics_, 54(1), 47-58, doi:`10.4401/ag-4838`.
  * Lion Krischer, Tobias Megies, Robert Barsch, Moritz Beyreuther, Thomas Lecocq, Corentin Caudron and Joachim Wassermann (2015), [ObsPy: a bridge for seismology into the scientific Python ecosystem](http://iopscience.iop.org/1749-4699/8/1/014003/), _Computational Science & Discovery_, 8(1), 014003, doi:`10.1088/1749-4699/8/1/014003`
  * [concept DOI, that always resolves to latest ObsPy versions: `10.5281/zenodo.1040769`](https://doi.org/10.5281/zenodo.1040769) (see [Zenodo FAQ](http://help.zenodo.org/#versioning))
  * [specific version DOIs for all released ObsPy versions](https://zenodo.org/search?ln=en&p=obspy&action_search=)

### Impact

![ObsPy impact statistics](https://user-images.githubusercontent.com/1842780/70671351-0c884100-1c7c-11ea-81ed-7c477b7cf29c.png)
