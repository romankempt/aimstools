import matplotlib.pyplot as plt

from AIMS_tools.density_of_states import density_of_states
from AIMS_tools.bandstructures import bandstructure, brillouinezone
from AIMS_tools.bandstructures.bandstructure import plot_bs
#dos = density_of_states(".")
#dos.plot()
fig = plt.figure(figsize=(10,5))
#bs = bandstructure(".", soc=False)
#bs.plot()
ax1 = fig.add_subplot(1, 2, 1)
bs = bandstructure(".", soc=True)
ax1 = bs.plot(axes=ax1, show=False)

ax2 = fig.add_subplot(1, 2, 2) # projection='3d')
bz = brillouinezone(bs.structure, bs.bandpath.path, special_points=bs.bandpath.special_points)
ax2 = bz.plot(axes=ax2, show=False)
plt.show()
#plot_bs(".", soc=True, spin="none")
