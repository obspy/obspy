from obspy.taup import traveltime_plot
traveltime_plot(min_degree=0, max_degree=50, phases=['P', 'S', 'PP'],
                source_depth=120, model='iasp91')
