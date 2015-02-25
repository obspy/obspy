from obspy.taup import TauPyModel


model = TauPyModel(model='iasp91')

arrivals = model.get_ray_paths(500, 140, phase_list=['Pdiff', 'SS'])

arrivals.plot(plot_type='spherical', legend=False, label_arrivals=True)
