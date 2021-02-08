from aimstools.misc import *
from aimstools.density_of_states.base import DOSBaseClass
from aimstools.density_of_states.utilities import DOSPlot, DOSSpectrum, Contribution

import matplotlib.pyplot as plt
import numpy as np


class TotalDOS(DOSBaseClass):
    def __init__(self, outputfile, soc=False) -> None:
        super().__init__(outputfile)

        assert any(
            x in ["total dos", "total dos tetrahedron"] for x in self.tasks
        ), "Total DOS was not specified as task in control.in ."

        dosfiles = self.get_dos_files(soc=soc)
        self.soc = soc
        self.dosfiles = dosfiles.total_dos
        self.spectrum = self.read_dosfiles()

    def __repr__(self):
        return "{}(outputfile={}, spin_orbit_coupling={})".format(
            self.__class__.__name__, repr(self.outputfile), self.soc
        )

    def read_dosfiles(self):
        assert (
            len(self.dosfiles) == 1
        ), "Too many DOS files found, something must have gone wrong."
        dosfile = self.dosfiles[0]
        d = np.loadtxt(dosfile, dtype=float, comments="#")
        energies, total_dos = d[:, 0], d[:, 1:]
        # This formatting might be complicated, but is consistent with the other DOS functions
        energies = np.stack([energies, energies], axis=1)
        total_dos = total_dos[:, :, np.newaxis]
        symbol = self.structure.get_chemical_formula()
        con = Contribution(symbol, total_dos)
        return DOSSpectrum(energies, [con], type="total")

    def _process_kwargs(self, **kwargs):
        kwargs = kwargs.copy()

        axargs = {}
        axargs["figsize"] = kwargs.pop("figsize", (3, 6))
        axargs["filename"] = kwargs.pop("filename", None)
        axargs["title"] = kwargs.pop("title", None)

        d = {}
        spin = kwargs.pop("spin", None)
        reference = kwargs.pop("reference", None)

        d["flip"] = kwargs.pop("flip", True)
        d["window"] = kwargs.pop("window", 3)
        d["mark_fermi_level"] = kwargs.pop("mark_fermi_level", fermi_color)
        d["broadening"] = kwargs.pop("broadening", 0.0)
        d["fill"] = kwargs.pop("fill", "gradient")
        _ = kwargs.pop("show_total", None)
        d["show_total"] = False

        self.set_energy_reference(reference, self.soc)
        ref, shift = self.energy_reference
        fermi_level = self.fermi_level.soc if self.soc else self.fermi_level.scalar
        be = self.band_extrema
        vbm = be.vbm_soc if self.soc else be.vbm_scalar
        cbm = be.cbm_soc if self.soc else be.cbm_scalar

        spin = self.spin2index(spin)
        if self.soc and spin == 1:
            raise Exception(
                "Spin channels are ill-defined for SOC calculations. A second spin channel does not exist."
            )

        d["spin"] = spin
        d["vbm"] = vbm
        d["cbm"] = cbm
        d["ref"] = ref
        d["shift"] = shift
        d["fermi_level"] = fermi_level
        return axargs, kwargs, d

    def plot(self, axes=None, color=mutedblack, main=True, **kwargs):
        axargs, kwargs, dosargs = self._process_kwargs(**kwargs)
        x = self.spectrum.energies
        con = self.spectrum.get_total_dos()

        with AxesContext(ax=axes, main=main, **axargs) as axes:
            dosplot = DOSPlot(x=x, con=con, l="tot", color=color, main=main, **dosargs)
            axes = dosplot.draw()
            x, y = dosplot.xy
            axes.plot(x, y, color=color, **kwargs)

        return axes
