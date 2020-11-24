import numpy as np
import re
import matplotlib.pyplot as plt
from AIMS_tools.bandstructures.mulliken_bandstructure import mulliken_bandstructure as mbs

bs  = mbs(".", soc=False)
#bs.plot_contribution_of_one_atom(0)
#bs.plot_one_species("S", l=1)
#bs.plot_all_species()
con1 = bs.spectrum.get_species_contribution("Fe")
con2 = bs.spectrum.get_species_contribution("S")
#bs.plot_gradient_contributions(con1, con2)
bs.plot_majority_contributions()

fig, axes = plt.subplots(1,1)
bs.plot_custom_contribution(con1, axes=axes, color="crimson")
bs.plot_custom_contribution(con2, axes=axes, color="blue")
plt.show()
