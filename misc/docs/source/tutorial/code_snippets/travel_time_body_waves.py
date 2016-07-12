import numpy as np
import matplotlib.pyplot as plt

from obspy.taup import TauPyModel


PHASES = [
    # Phase, distance
    # Right half
    ('P', 26),
    ('PP', 60),
    ('PPP', 94),
    ('PPS', 155),
    ('p', 3),
    ('pPcP', 100),
    ('PKIKP', 170),
    ('PKJKP', 194),
    ('S', 65),
    ('SP', 85),
    ('SS', 134.5),
    ('SSS', 204),
    # Left half
    ('p', -10),
    ('pP', -37.5),
    ('s', -3),
    ('sP', -49),
    ('ScS', -44),
    ('SKS', -82),
    ('SKKS', -120),
]

PLOT_ALL = [
    'SKKS',
]


model = TauPyModel(model='iasp91')
phase_name_radius = model.model.radius_of_planet * 1.1

# ax_right is used for paths plotted on the right half.
fig, ax_right = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
ax_right.set_theta_zero_location('N')
ax_right.set_theta_direction(-1)
# ax_left is used for paths plotted on the left half.
ax_left = fig.add_axes(ax_right.get_position(), projection='polar',
                       label='twin', frameon=False)
ax_left.set_theta_zero_location('N')
ax_left.set_theta_direction(+1)
ax_left.xaxis.set_visible(False)
ax_left.yaxis.set_visible(False)

# Plot all pre-determined phases
for phase, distance in PHASES:
    if distance < 0:
        realdist = -distance
        ax = ax_left
    else:
        realdist = distance
        ax = ax_right

    arrivals = model.get_ray_paths(700, realdist, phase_list=[phase])
    if not len(arrivals):
        print('FAIL', phase, distance)
        continue
    arrivals.plot(plot_type='spherical', plot_all=phase in PLOT_ALL,
                  legend=False, label_arrivals=True,
                  show=False, ax=ax)

# Annotate regions
ax_right.text(0, 0, 'Solid\ninner\ncore',
              horizontalalignment='center', verticalalignment='center',
              bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
ocr = (model.model.radius_of_planet -
       (model.model.s_mod.v_mod.iocb_depth +
        model.model.s_mod.v_mod.cmb_depth) / 2)
ax_right.text(np.deg2rad(180), ocr, 'Fluid outer core',
              horizontalalignment='center',
              bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
mr = model.model.radius_of_planet - model.model.s_mod.v_mod.cmb_depth / 2
ax_right.text(np.deg2rad(180), mr, 'Solid mantle',
              horizontalalignment='center',
              bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

plt.show()
