from aimstools.bandstructures import RegularBandStructure as RBS
from aimstools.bandstructures import MullikenBandStructure as MBS

bs = MBS(".", soc=True)
con1 = bs.spectrum.get_species_contribution("B")
con2 = bs.spectrum.get_species_contribution("N")

bs.plot_all_species(window=10)
bs.plot_difference_contribution(con1, con2, window=10)
