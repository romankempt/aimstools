from aimstools.misc import *
from aimstools.density_of_states.base import DOSBaseClass
from aimstools.density_of_states.utilities import DOSPlot, DOSSpectrum, DOSContribution

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
        self.dos = self.read_dosfiles()
        self._spectrum = None

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
        return (energies, total_dos)

    def set_spectrum(self, reference=None):
        energies, total_dos = self.dos
        self.set_energy_reference(reference, self.soc)
        symbol = self.structure.get_chemical_formula()
        con = DOSContribution(symbol, total_dos)
        fermi_level = self.fermi_level.soc if self.soc else self.fermi_level.scalar
        reference, shift = self.energy_reference
        return DOSSpectrum(
            energies,
            [con],
            type="total",
            fermi_level=fermi_level,
            reference=reference,
            shift=shift,
        )

    @property
    def spectrum(self):
        if self._spectrum == None:
            self.set_spectrum(reference=None)
        return self._spectrum

    def get_spectrum(self, reference=None):
        self.set_spectrum(reference=reference)
        return self.spectrum

    def _process_kwargs(self, **kwargs):
        kwargs = kwargs.copy()
        spin = kwargs.pop("spin", None)

        deprecated = ["title", "mark_fermi_level", "mark_band_gap"]
        for dep in deprecated:
            if dep in kwargs.keys():
                kwargs.pop(dep)
                logger.warning(
                    f"Keyword {dep} is deprecated. Please do not use this anymore."
                )

        kwargs["spin"] = self.spin2index(spin)

        return kwargs

    def plot(
        self,
        axes=None,
        color=mutedblack,
        linewidth=mpllinewidth,
        linestyle="-",
        **kwargs,
    ):
        kwargs = self._process_kwargs(**kwargs)
        kwargs["show_total_dos"] = True
        kwargs["total_dos_color"] = color
        kwargs["total_dos_linewidth"] = linewidth
        kwargs["total_dos_linestyle"] = linestyle
        kwargs["colors"] = [color]
        kwargs["show_legend"] = False

        reference = kwargs.pop("reference", None)
        spectrum = self.get_spectrum(reference=reference)

        with AxesContext(ax=axes, **kwargs) as axes:
            dosplot = DOSPlot(ax=axes, spectrum=self.spectrum, **kwargs)
            dosplot.draw()

        return axes
