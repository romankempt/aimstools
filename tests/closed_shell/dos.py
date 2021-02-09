from aimstools.density_of_states import TotalDOS
from aimstools.density_of_states import SpeciesProjectedDOS as spd

#tdos = TotalDOS(".")
#tdos.plot()

sdos = spd(".")
#sdos.plot_one_species("Si")
sdos.plot_all_species(window=5, show_total=False)
#sdos.plot_all_angular_momenta()
#import matplotlib.pyplot as plt
#plt.show()
#sdos.plot_all_species()
