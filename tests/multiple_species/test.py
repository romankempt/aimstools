from aimstools.bandstructures import RegularBandStructure as RBS
from aimstools.bandstructures import MullikenBandStructure as MBS

bs = MBS(".", soc=True)
con1 = bs.spectrum.get_species_contribution("B")
con2 = bs.spectrum.get_species_contribution("N")

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1,2)
bs.plot_all_species(window=10, axes=axes[1])
bs.plot_difference_contribution(con1, con2, window=10, axes=axes[0], show_colorbar=True)
plt.show()
