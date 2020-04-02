# Plotting results

## Customizing and comparing band structures and densities of states

For simple cases, the **aims_plot** takes care of creating a suitable representation of a band structure or density of states for publication purposes. One of the main features of the scripts is the flexibility in arranging and customising these plots for specific purposes.

Each band structure (BS) or density of states (DOS) has an associated class with a plotting method. These plotting functions have limited memory. For example, changing the color or linewidth of a plot is saved internally to the class, which will be taken into account if the plot function is called again:

```python
from AIMS_tools import bandstructure
import matplotlib.pyplot as plt

bs = bandstructure.bandstructure(path)
fig1 = bs.plot(color="purple", linewidth=2, linestyle=":")
.
.
.
plt.close()
fig2 = bs.plot()
# The plot settings will remain the same as in fig1.
```

This can be used to assign highly customised plots to a grid in order to overlay and compare band structures and densities of states.

```python
from AIMS_tools import bandstructure, dos
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

bs1 = bandstructure.bandstructure(path1)
dos1 = dos.density_of_states(path2)
bs2 = bandstructure.bandstructure(path3)

# We just create some random plots with individual plot settings
f1 = bs1.plot(color="blue", kpath="G-M-G", fix_energy_limits=[-4,4])
f2 = dos1.plot_all_species(fill="constant")
f3 = bs2.plot(color="red", kpath="M-G-K-G", var_energy_limits=3)
plt.close()

# This initialises an empty figure with 3 columns and one row:
fig = plt.figure(constrained_layout=True, figsize=(12, 4))
spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig, width_ratios=[3, 1, 3])

# Then we assign the plots to specific axes:
ax1 = fig.add_subplot(spec[0])  # this is the first column
ax2 = fig.add_subplot(spec[1])  # this is the second column
ax3 = fig.add_subplot(spec[2])   # this is the third column

plt.sca(ax1) # sets axes to ax1
ax1 = bs1.plot(fig=fig, axes=ax1)

plt.sca(ax2) # sets axes to ax2
ax2 = dos1.plot_all_species(fig=fig, axes=ax2)
ax2.set_yticks([]) # removes ticks on y-axis
ax2.set_ylabel("") # removes label on y-axis

plt.sca(ax3) # sets axes to ax3
ax3 = bs2.plot(fig=fig, axes=ax3)
plt.show()
```

This becomes cumbersome for a large number of figures. Hence, exactly this procedure is wrapped into the **multiplots** module:

```python
from AIMS_tools import bandstructure, dos, multiplots
import matplotlib.pyplot as plt

fig = multiplots.combine(nrows=1, ncols=3, results=[bs1, dos1, bs2], ratios=[3,1,3], color="red", linewidth="2")
# Here, color and linewidth have been specified globally and will be applied to all plots.
```

## Customizing mulliken-projected band structures

The fatbandstructure class works the same way as the bandstructure class and inherits all of its functionalities. It has additional plotting functions:

```python
from AIMS_tools import bandstructure
import matplotlib.pyplot as plt

bs = bandstructure.bandstructure(path)          # this retrieves only the band structure
fatbs = bandstructure.fatbandstructure(path)    # this retrieves both the band structure and the fat band structure

f1 = fatbs.plot()                   # this will only plot the band structure
f2 = fatbs.plot_mlk("Mo", "tot")    # this will plot the contribution of an atom in the band structure
f3 = fatbs.plot_all_species()       # this will plot all total contributions of the same species
f4 = fatbs.plot_all_orbitals()      # this will plot all orbital characters of all atoms
```

The fatbandstructure module is currently not wrapped in the **aims_plot** command line, because it typically requires more customization. Especially for large structures, one typically needs to sum up the contributions of certain fragments of the system (for example one layer of a heterostructure), or one needs to plot different contributions on a grid. Both these operations are easily done in the same fashion as for band structures.

```python
from AIMS_tools import bandstructure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Let's assume we want to plot a heterostructure of MoS2 / WSe2.
fatbs = bandstructure.fatbandstructure(path)
fatbs.sum_all_species_contributions()
# The attribute fatbs.atoms_to_plot will now only contain Mo, S, W and Se.

fatbs.sum_contributions(["Mo", "S"])
# This sums up the contributions of Mo and S. They are saved under the label "MoS".
fatbs.sum_contributions(["W", "Se"])
# This sums up the contributions of W and Se. They are saved under the label "WSe".

# Now we plot the contribution MoS on the left and WSe on the right.
fig = plt.figure(constrained_layout=True, figsize=(10, 4))
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

ax1 = fig.add_subplot(spec[0])
plt.sca(ax1)
fatbs.plot(axes=ax1, fig=fig, linewidth=0.5, alpha=0.25, color="lightgray") # this adds the band structure in the background
fatbs.plot_mlk("MoS", "tot", axes=ax1, fig=fig, cmap="Oranges")
ax1.set_title("Mo$_2$ mulliken projection")

ax2 = fig.add_subplot(spec[1])
plt.sca(ax2)
fatbs.plot(axes=ax2, fig=fig, linewidth=0.5, alpha=0.25, color="lightgray") # this adds the band structure in the background
fatbs.plot_mlk("WSe", "tot", axes=ax2, fig=fig, cmap="Blues")
ax2.set_title("WSe$_2$ mulliken projection")
```



