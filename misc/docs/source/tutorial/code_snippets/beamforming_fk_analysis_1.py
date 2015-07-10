from obspy.core import read
from obspy.core.inventory import read_inventory
from obspy.signal.array_analysis import SeismicArray


# Read data (instrument response is already removed):
st = read('http://examples.obspy.org/agfa_corrected.mseed')
# Read matching inventory:
inv = read_inventory('http://examples.obspy.org/agfainventory.xml')

# Create an array:
the_array = SeismicArray("AGFA")
the_array.add_inventory(inv)

# Execute the beamforming:
kwargs2 = dict(
    # slowness grid: X min, X max, Y min, Y max, Slow Step
    slx=(-3.0, 3.0), sly=(-3.0, 3.0), sls=0.03,
    # sliding window properties
    wlen=1.0, wfrac=0.05,
    # frequency properties
    frqlow=1.0, frqhigh=8.0,
    filter=True, vel_corr=4.0)

results = the_array.fk_analysis(st, **kwargs2)

# Make plots:
results.plot_bf_results_over_time(show_immediately=False)
results.plot_baz_hist()
