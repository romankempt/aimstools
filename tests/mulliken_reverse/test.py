from aimstools.bandstructures import MullikenBandStructure as MBS

bs = MBS(".", soc=False)
bs.plot_all_species(reference="VBM")
print(bs.fermi_level)
