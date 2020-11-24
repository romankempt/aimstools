from aimstools.density_of_states import TotalDOS
from aimstools.density_of_states import SpeciesProjectedDOS as spd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1,1)
tdos = TotalDOS(".")
tdos.plot(axes=axes, spin="dn", main=True)
tdos.plot(axes=axes, spin="up", main=True)
plt.show()

sdos = spd(".")
fig, axes = plt.subplots(1,1)
sdos.plot_one_species("Fe", spin="dn", axes=axes, main=True)
sdos.plot_one_species("Fe", spin="up", axes=axes, main=True)
plt.show()

fig, axes = plt.subplots(1,1)
sdos.plot_all_species(window=3, spin="dn", axes=axes, main=True)
sdos.plot_all_species(window=3, spin="up", axes=axes, main=True)
plt.show()

fig, axes = plt.subplots(1,1)
sdos.plot_all_angular_momenta(window=3, spin="dn", axes=axes, main=True)
sdos.plot_all_angular_momenta(window=3, spin="up", axes=axes, main=True)
plt.show()


fig, axes = plt.subplots(1,1)
con = sdos.spectrum.get_species_contributions("Fe")
sdos.plot_custom_contributions(con, l="s", spin="dn", axes=axes, main=True)
sdos.plot_custom_contributions(con, l="s", spin="up", axes=axes, main=True)
plt.show()
