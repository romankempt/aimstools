from aimstools.misc import *
from aimstools.density_of_states.base import DOSBaseClass
from aimstools.density_of_states.utilities import DOSPlot, DOSSpectrum

import numpy as np


class TotalDOS(DOSBaseClass):
    def __init__(self, outputfile, soc=False) -> None:
        super().__init__(outputfile)
        self.soc = soc
        self._dos = None
        self._spectrum = None

    def __repr__(self):
        return "{}(outputfile={}, spin_orbit_coupling={})".format(
            self.__class__.__name__, repr(self.outputfile), self.soc
        )

    def _read_dosfiles(self):
        dosfiles = self.get_dos_files(soc=self.soc)
        dosfiles = dosfiles.total_dos
        assert (
            len(dosfiles) == 1
        ), "Too many DOS files found, something must have gone wrong."
        d = np.loadtxt(dosfiles[0], dtype=float, comments="#")
        energies, total_dos = d[:, 0], d[:, 1:]
        # This formatting might be complicated, but is consistent with the other DOS functions
        energies = energies[:, np.newaxis]
        total_dos = total_dos[:, :, np.newaxis]
        self._dos = (energies, (0, total_dos))

    def set_spectrum(self, reference=None):
        if self.dos == None:
            self._read_dosfiles()
        energies, total_dos = self.dos
        self.set_energy_reference(reference, self.soc)
        atoms = self.structure.copy()
        fermi_level = self.fermi_level.soc if self.soc else self.fermi_level.scalar
        reference, shift = self.energy_reference
        self._spectrum = DOSSpectrum(
            atoms=atoms,
            energies=energies,
            contributions=[total_dos],
            type="total",
            fermi_level=fermi_level,
            reference=reference,
            shift=shift,
        )

    @property
    def dos(self):
        if self._dos == None:
            self._dos = self._read_dosfiles()
        return self._dos

    @property
    def spectrum(self):
        ":class:`aimstools.density_of_states.utilities.DOSSpectrum`."
        if self._spectrum == None:
            self.set_spectrum(None)
        return self._spectrum

    def get_spectrum(self, reference=None):
        "Returns :class:`aimstools.density_of_states.utilities.DOSSpectrum`."
        self.set_spectrum(reference=reference)
        return self.spectrum

    def _process_kwargs(self, kwargs):
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

    def plot(self, axes=None, color=mutedblack, **kwargs):
        """Plots total density of states."""
        kwargs = self._process_kwargs(kwargs)
        kwargs["show_total_dos"] = False
        kwargs["show_legend"] = False

        reference = kwargs.pop("reference", None)
        spectrum = self.get_spectrum(reference=reference)
        contributions = self.spectrum.get_total_dos()

        with AxesContext(ax=axes, **kwargs) as axes:
            dosplot = DOSPlot(
                ax=axes,
                contributions=[contributions],
                colors=[color],
                spectrum=spectrum,
                **kwargs,
            )
            dosplot.draw()

        return axes
