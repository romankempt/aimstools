import matplotlib.pyplot as plt

from AIMS_tools.density_of_states import density_of_states
from AIMS_tools.bandstructures import bandstructure, brillouinezone
from AIMS_tools.bandstructures.bandstructure import plot_bs
#dos = density_of_states(".")
#dos.plot()
#bs = bandstructure(".", soc=False)
#bs.plot()
plot_bs(".", soc=False, spin="none", bandpathstring="XGXGX")
