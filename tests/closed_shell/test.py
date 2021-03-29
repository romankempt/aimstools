from aimstools.density_of_states.total_dos import TotalDOS as TDOS
from aimstools.density_of_states.atom_proj_dos import AtomProjectedDOS as APD
from aimstools.density_of_states.species_proj_dos import SpeciesProjectedDOS as SPD
from aimstools.misc import *
import matplotlib.pyplot as plt
#tdos = TDOS(".")
#tdos.plot()
apd = APD(".")
spd = SPD(".")


set_global_plotsettings(target="paper")
#spd.plot_contributions(contributions=("Si", "tot"), labels=["Si_tot"], colors=["green"])
#spd.plot_all_species()

#fig, axes = plt.subplots(dpi=600)
spd.plot_all_angular_momenta()

