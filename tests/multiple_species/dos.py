from aimstools.density_of_states import TotalDOS
from aimstools.density_of_states import SpeciesProjectedDOS as spd
from aimstools.density_of_states import AtomProjectedDOS as apd

sdos = apd(".")
sdos.plot_one_atom(0)
sdos.plot_one_species("B")
sdos.plot_all_species(window=5)
sdos.plot_all_angular_momenta(window=5)
