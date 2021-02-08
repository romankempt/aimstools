from aimstools.dielectric_function import AbsorptionSpectrum as ABS

from pathlib import Path
import matplotlib.pyplot as plt

dirs = list(Path().cwd().glob("omega*"))
fig, axes = plt.subplots(1, 1, figsize=(10,10))
colors = ["red", "green", "blue", "orange", "purple"]
for i, d in enumerate(dirs):
    abs = ABS(d)
    ax = abs.plot(axes=axes, label=str(d.parts[-1]), energy_unit="nm", components="total", color=colors[i])
plt.show()
