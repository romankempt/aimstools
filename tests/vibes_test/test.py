
from aimstools.phonons import FHIVibesPhonons as FVP
phon = FVP("")

g = phon.get_gamma_point_frequencies()
print(g)

f = phon.get_irreducible_representations()
print(f)

