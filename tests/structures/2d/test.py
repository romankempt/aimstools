from aimstools.bandstructures import BrillouineZone as BZ
import ase.io
import matplotlib.pyplot as plt

atoms = ase.io.read("WS2_2H_1L.xyz")
bz = BZ(atoms)
bz.plot()

plt.show()

