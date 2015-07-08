from obspy.core import read, AttribDict, UTCDateTime
from obspy.signal.array_analysis import SeismicArray
from obspy.core.inventory import Station, Network, Inventory, Channel
from obspy.core.inventory.util import Latitude, Longitude
from obspy.signal import cornFreq2Paz


st = read("http://examples.obspy.org/agfa.mseed")

# Set up the inventory.
st.trim(UTCDateTime("20080217110515"), UTCDateTime("20080217110545"))
# Set PAZ for all 5 channels
st[0].stats.paz = AttribDict({
    'poles': [(-0.03736 - 0.03617j), (-0.03736 + 0.03617j)],
    'zeros': [0j, 0j],
    'sensitivity': 205479446.68601453,
    'gain': 1.0})
st[1].stats.paz = AttribDict({
    'poles': [(-0.03736 - 0.03617j), (-0.03736 + 0.03617j)],
    'zeros': [0j, 0j],
    'sensitivity': 205479446.68601453,
    'gain': 1.0})
st[2].stats.paz = AttribDict({
    'poles': [(-0.03736 - 0.03617j), (-0.03736 + 0.03617j)],
    'zeros': [0j, 0j],
    'sensitivity': 250000000.0,
    'gain': 1.0})
st[3].stats.paz = AttribDict({
    'poles': [(-4.39823 + 4.48709j), (-4.39823 - 4.48709j)],
    'zeros': [0j, 0j],
    'sensitivity': 222222228.10910088,
    'gain': 1.0})
st[4].stats.paz = AttribDict({
    'poles': [(-4.39823 + 4.48709j), (-4.39823 - 4.48709j), (-2.105 + 0j)],
    'zeros': [0j, 0j, 0j],
    'sensitivity': 222222228.10910088,
    'gain': 1.0})

# Station names must match those set in the trace headers.
stn0 = Station("BW01", Latitude(48.108589), Longitude(11.582967),
               elevation=.45,
               channels=[
                   Channel("ZH", "", Latitude(48.108589),
                           Longitude(11.582967), elevation=.45, depth=0)])
stn1 = Station("BW02", Latitude(48.108192), Longitude(11.583120),
               elevation=.45,
               channels=[
                   Channel("ZH", "", Latitude(48.108192),
                           Longitude(11.583120), elevation=.45, depth=0)])
stn2 = Station("BW03", Latitude(48.108692), Longitude(11.583414),
               elevation=.45,
               channels=[
                   Channel("ZH", "", Latitude(48.108692),
                           Longitude(11.583414), elevation=.45, depth=0)])
stn3 = Station("BW07", Latitude(48.108456), Longitude(11.583049),
               elevation=.45,
               channels=[
                   Channel("ZH", "", Latitude(48.108456),
                           Longitude(11.583049), elevation=.45, depth=0)])
stn4 = Station("BW08", Latitude(48.108730), Longitude(11.583157),
               elevation=.45,
               channels=[
                   Channel("ZH", "", Latitude(48.108730),
                           Longitude(11.583157), elevation=.45, depth=0)])
network = Network("BW", stations=(stn0, stn1, stn2, stn3, stn4))
# Also add the network name to the traces:
for tr in st:
    tr.stats.network = "BW"
inv = Inventory([network], "sender name")


# Instrument correction to 1Hz corner frequency
paz1hz = corn_freq_2_paz(1.0, damp=0.707)
st.simulate(paz_remove='self', paz_simulate=paz1hz)

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
results.plot_bf_results_over_time()
results.plot_baz_hist()
