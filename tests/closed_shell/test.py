from aimstools.density_of_states.total_dos import TotalDOS as TDOS

tdos = TDOS(".")
print(tdos.spectrum.contributions)
tdos.plot()
