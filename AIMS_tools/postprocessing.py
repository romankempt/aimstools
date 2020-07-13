import ase.io, ase.cell

from AIMS_tools.structuretools import structure
from AIMS_tools.misc import *


class postprocess:
    """ A base class that retrieves information from finished calculations.

    Args:
        path (pathlib object): Directory of outputfile.

    Attributes:
        success (bool): If output file contained "Have a nice day".
        calc_type (set): Set of requested calculation types.
        active_SOC (bool): If spin-orbit coupling was included in the control.in file.
        active_GW (bool): If GW was included in the control.in file.
        spin (int): Spin channel.
        fermi_level (float): Fermi level energy value in eV.
        smallest_direct_gap (str): Smallest direct gap from outputfile.
        VBM (float): Valence band maximum energy in eV.
        CBM (float): Conduction band minimum energy in eV.        
        structure (structure): AIMS_tools.structuretools.structure object.
        color_dict (dict): Dictionary of atom labels and JMOL color tuples.        
     """

    def __init__(self, outputfile, get_SOC=True, spin=None):
        self.success = self.__check_output(outputfile)
        if spin in ["up", 1]:
            self.spin = 1
        elif spin in ["down", 2, "dn"]:
            self.spin = 2
        else:
            self.spin = None

        self.__read_geometry()
        self.__read_control()
        if self.success == True:
            self.__read_output(get_SOC)
        self.color_dict = color_dict
        self.__set_global_plotproperties()
        self.__mplkwargs = {}

    def __repr__(self):
        return str(self.outputfile)

    def __check_output(self, outputfile):
        if Path(outputfile).is_file():
            check = os.popen(
                "tail -n 10 {filepath}".format(filepath=Path(outputfile))
            ).read()
            if "Have a nice day." in check:
                self.outputfile = Path(outputfile)
                self.path = self.outputfile.parent
                return True
            else:
                logging.error("Calculation did not converge!")
                return False

        elif Path(outputfile).is_dir():
            self.path = Path(outputfile)
            outfiles = list(Path(outputfile).glob("*.out"))
            if len(outfiles) == 0:
                logging.critical("Output file does not exist.")
                return False
            else:
                for i in outfiles:
                    check = os.popen(
                        "tail -n 10 {filepath}".format(filepath=str(i))
                    ).read()
                    if "Have a nice day." in check:
                        self.outputfile = i
                        self.path = self.outputfile.parent
                        return True
                else:
                    logging.error("Calculation did not converge!")
                    return False
        else:
            logging.critical("Could not find outputfile.")
            return False

    def __read_geometry(self):
        geometry = self.path.joinpath("geometry.in")
        if geometry.exists() == False:
            logging.critical("File geometry.in not found!")
            return None
        self.structure = structure(geometry)

    def __read_control(self):
        control = self.path.joinpath("control.in")
        if control.exists() == False:
            logging.critical("File control.in not found!")
            return None
        bandlines = []
        self.active_SOC = False
        self.active_GW = False
        self.calc_type = set()
        with open(control, "r") as file:
            for line in file.readlines():
                read = False if line.startswith("#") else True
                if read:
                    if "output band" in line:
                        bandlines.append(line.split())
                        self.calc_type.add("BS")
                    if "include_spin_orbit" in line:
                        self.active_SOC = True
                    if "k_grid" in line:
                        self.k_grid = (
                            int(line.split()[-3]),
                            int(line.split()[-2]),
                            int(line.split()[-1]),
                        )
                    if "qpe_calc" in line and "gw" in line:
                        self.active_GW = True
                    if (
                        ("spin" in line)
                        and ("collinear" in line)
                        and (self.spin == None)
                    ):
                        self.spin = 1
                    if "output atom_proj_dos" in line:
                        self.calc_type.add("DOS")
        ## band structure specific information
        if "BS" in self.calc_type:
            self.ksections = []
            self.special_points = {}
            for entry in bandlines:
                self.ksections.append((entry[-2], entry[-1]))
                self.special_points[entry[-1]] = np.array(
                    [entry[5], entry[6], entry[7]], dtype=float
                )
                self.special_points[entry[-2]] = np.array(
                    [entry[2], entry[3], entry[4]], dtype=float
                )

    def __read_output(self, get_SOC=True):
        # Retrieve information such as Fermi level and band gap from output file.
        self.smallest_direct_gap = "Direct gap not determined. This usually happens if the fundamental gap is direct."
        self.work_function = None
        fermi_levels = []
        soc, scalar = False, False
        with open(self.outputfile, "r") as file:
            for line in file.readlines():
                if "Chemical potential" in line:
                    if self.spin != None:
                        if "spin up" in line:
                            up_fermi_level = float(line.split()[-2])
                        elif "spin dn" in line:
                            down_fermi_level = float(line.split()[-2])
                            self.fermi_level = max([up_fermi_level, down_fermi_level])
                if "Chemical potential (Fermi level)" in line:
                    fermi_level = line.replace("eV", "")
                    fermi_levels.append(float(fermi_level.split()[-1]))
                if "Smallest direct gap :" in line:
                    self.smallest_direct_gap = line
                if "Number of k-points" in line:
                    self.k_points = int(line.split()[-1])
                if "Highest occupied state (VBM) at" in line:
                    self.VBM = float(line.split()[5])
                if "Lowest unoccupied state (CBM) at" in line:
                    self.CBM = float(line.split()[5])
                if ("Chemical potential is") in line:
                    # This is specific for SOC band structures. SOC fermi level is much higher!
                    fermi_levels.append(float(line.split()[-2]))
                if "Total energy uncorr" in line:
                    self.total_energy = float(line.split()[-2])
                if "Begin self-consistency iteration #" in line:
                    self.n_scf_cycles = int(line.split()[-1])
                if """Work function ("upper" slab surface)""" in line:
                    self.work_function = float(line.split()[-2])
                if """Potential vacuum level, "upper" slab surface:""" in line:
                    self.upper_vacuum_potential = float(line.split()[-2])
                if """Potential vacuum level, "lower" slab surface:""" in line:
                    self.lower_vacuum_potential = float(line.split()[-2])
                if """Scalar-relativistic "band gap" of total set of bands:""" in line:
                    scalar = True
                    soc = False
                if """Spin-orbit-coupled "band gap" of total set of bands:""" in line:
                    soc = True
                    scalar = False
                if "| Lowest unoccupied state:" in line and scalar:
                    self.CBM = float(line.split()[-2])
                if "| Lowest unoccupied state:" in line and soc and get_SOC:
                    self.CBM = float(line.split()[-2])
                if "| Highest occupied state :" in line and scalar:
                    self.VBM = float(line.split()[-2])
                if "| Highest occupied state :" in line and soc and get_SOC:
                    self.VBM = float(line.split()[-2])

        if self.active_SOC and get_SOC:
            # SOC calculations have two fermi levels, one with zora, one without. The last one should be with SOC.
            self.fermi_level = fermi_levels[-1]
        elif self.active_SOC and get_SOC == False:
            self.fermi_level = fermi_levels[-2]
        else:
            self.fermi_level = fermi_levels[-1]
        self.band_gap = np.abs(self.CBM - self.VBM)

    def __set_global_plotproperties(self):
        d = dict()

        # general plotsettings for bs, dos, fatbas
        d["title"] = ""
        d["color"] = "k"
        d["energy_reference"] = "middle"
        d["fix_energy_limits"] = []
        d["var_energy_limits"] = 1.0
        d["linewidths"] = 1.5

        # more specific to bs
        d["mark_fermi_level"] = "none"
        d["mark_gap"] = "lightgray"
        d["cmap"] = "Blues"
        d["nbands"] = False
        d["interpolation_step"] = False
        d["mode"] = "lines"
        d["capstyle"] = "round"
        d["kpath"] = None

        self.__global_plotproperties = d
        for key, item in d.items():
            setattr(self, key, item)


class hirshfeld(postprocess):
    """ A simple class to evaluate Hirshfeld charge analysis from AIMS.

    Args:
        outputfile (str): Path to outputfile.

    Attributes:
        charges (dict): Dictionary of (index, species) tuples and hirshfeld charges. 
        tot_charges (dict): Dictionary of species and summed hirshfeld charges.
    
    """

    def __init__(self, outputfile, get_SOC=True, spin=None):
        super().__init__(outputfile, get_SOC=get_SOC, spin=spin)
        self.charges = self.read_charges()
        self.tot_charges = self.sum_charges()

    def read_charges(self):
        with open(self.outputfile, "r") as file:
            ats = []
            charges = []
            read = False
            for line in file.readlines():
                if "Performing Hirshfeld analysis" in line:
                    read = True
                    i = 0
                if read:
                    if "Atom" in line:
                        ats.append(
                            (int(line.split()[-2].strip(":")) - 1, line.split()[-1])
                        )
                    if "Hirshfeld charge        :" in line:
                        charges.append(float(line.split()[-1]))
                        i += 1
                        if i == self.structure.atoms.get_global_number_of_atoms():
                            read = False
        charges = dict(zip(ats, charges))
        return charges

    def sum_charges(self, fragment_filter=[]):
        """ Sums charges of given indices of atoms for the same species.
        
        Args:
            filter (list): List of atom indices. If None, all atoms of same species will be summed up.
        
        Example:
            >>> from AIMS_tools.structuretools import structure
            >>> from AIMS_tools.postprocessing import hirshfeld
            >>> hs = hirshfeld.hirshfeld("outputfile")
            >>> frag1 = hs.structure.fragments[0][0]
            >>> hs.sum_charges(frag1)
            
        """
        indices = {i: j for i, j in self.charges.keys()}

        fragment_filter = (
            [i[0] for i in self.charges.keys()]
            if fragment_filter == []
            else fragment_filter
        )

        sum_charges = {}
        for atom in fragment_filter:
            species = indices[atom]
            if species not in sum_charges.keys():
                sum_charges[species] = self.charges[(atom, species)]
            else:
                sum_charges[species] += self.charges[(atom, species)]
        return sum_charges
