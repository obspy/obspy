from obspy.taup import TauPyModel


model = TauPyModel(model='iasp91')

arrivals = model.get_ray_paths(500, 140, phase_list=['PP', 'SSS'])
arrivals.plot(plot_type='cartesian', plot_all=False)
