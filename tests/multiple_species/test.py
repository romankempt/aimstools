from aimstools.bandstructures import RegularBandStructure as RBS
from aimstools.bandstructures import MullikenBandStructure as MBS

bs = MBS(".", soc=True)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1,2)
bs.plot_majority_contribution(window=10, axes=axes[1])
bs.plot_difference_contribution("B", "N", window=10, axes=axes[0], show_colorbar=True)
plt.show()
