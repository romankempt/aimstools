import numpy as np
import re
import matplotlib.pyplot as plt
from AIMS_tools import BandStructure
from AIMS_tools.bandstructures import MullikenBandStructure as mbs

bs = mbs(".", soc=True)

w = (-6,6)

#bs.plot(window=w)
#bs.plot_contribution_of_one_atom(0, window=w)
#bs.plot_one_species("B", window=w)
#bs.plot_all_species(window=w)


con1 = bs.spectrum.get_species_contribution("B")
con2 = bs.spectrum.get_species_contribution("N")
#bs.plot_gradient_contributions(con1, con2, window=w)

#bs.plot_majority_contributions(window=w)

bs.plot_custom_contribution(con1, window=w, color="red")
bs.plot_custom_contribution(con2, window=w, color="blue")

