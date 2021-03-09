from aimstools.bandstructures import MullikenBandStructure as MBS
from aimstools.misc import *
import matplotlib.pyplot as plt


set_verbosity_level(2)

bs = MBS(".", soc=False)
bs.plot_all_species(reference="VBM")

plt.show()
