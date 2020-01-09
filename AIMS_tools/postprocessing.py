import numpy as np
import glob, sys, os, math
from pathlib import Path as Path
import ase.io, ase.cell
from AIMS_tools.structuretools import structure

Angstroem_to_bohr = 1.889725989


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
        self.__check_output(outputfile)
        if spin in ["up", 1]:
            self.spin = 1
        elif spin in ["down", 2, "dn"]:
            self.spin = 2
        else:
            self.spin = None

        self.__read_geometry()
        self.__read_control()
        self.__read_output()
        self.__def_color_dictionary()

    def __repr__(self):
        return repr(str(self.path.parts[-1]))

    def __check_output(self, outputfile):
        if Path(outputfile).is_file():
            check = os.popen(
                "tail -n 10 {filepath}".format(filepath=Path(outputfile))
            ).read()
            if "Have a nice day." in check:
                self.outputfile = Path(outputfile)
                self.path = self.outputfile.parent
                self.success = True
            else:
                print("Calculation did not converge!")
                self.success = False

        elif Path(outputfile).is_dir():
            outfiles = Path(outputfile).glob("*.out")
            found = False
            for i in outfiles:
                check = os.popen("tail -n 10 {filepath}".format(filepath=i)).read()
                if "Have a nice day." in check:
                    self.outputfile = i
                    self.path = self.outputfile.parent
                    self.success = True
                    found = True
            if found == False:
                print("Calculation did not converge!")
                self.success = False
        else:
            print("Could not find outputfile.")
            sys.exit()

    def __read_geometry(self):
        geometry = self.path.joinpath("geometry.in")
        self.structure = structure(geometry)

    def __read_control(self):
        control = self.path.joinpath("control.in")
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
            self.kvectors = {"G": np.array([0.0, 0.0, 0.0])}
            for entry in bandlines:
                self.ksections.append((entry[-2], entry[-1]))
                self.kvectors[entry[-1]] = np.array(
                    [entry[5], entry[6], entry[7]], dtype=float
                )
                self.kvectors[entry[-2]] = np.array(
                    [entry[2], entry[3], entry[4]], dtype=float
                )

    def __read_output(self):
        # Retrieve information such as Fermi level and band gap from output file.
        self.smallest_direct_gap = "Direct gap not determined. This usually happens if the fundamental gap is direct."
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
                    self.fermi_level = float(fermi_level.split()[-1])
                if "Smallest direct gap :" in line:
                    self.smallest_direct_gap = line
                if "Number of k-points" in line:
                    self.k_points = int(line.split()[-1])
                if "Highest occupied state (VBM) at" in line:
                    self.VBM = float(line.split()[5])
                if "Lowest unoccupied state (CBM) at" in line:
                    self.CBM = float(line.split()[5])
                if "Chemical potential is" in line:
                    self.fermi_level = float(line.split()[-2])

    def __def_color_dictionary(self):
        self.color_dict = {
            "H": (1, 1, 1),
            "He": (0.850980392, 1, 1),
            "Li": (0.8, 0.501960784, 1),
            "Be": (0.760784314, 1, 0),
            "B": (1, 0.709803922, 0.709803922),
            "C": (0.564705882, 0.564705882, 0.564705882),
            "N": (0.188235294, 0.31372549, 0.97254902),
            "O": (1, 0.050980392, 0.050980392),
            "F": (0.564705882, 0.878431373, 0.31372549),
            "Ne": (0.701960784, 0.890196078, 0.960784314),
            "Na": (0.670588235, 0.360784314, 0.949019608),
            "Mg": (0.541176471, 1, 0),
            "Al": (0.749019608, 0.650980392, 0.650980392),
            "Si": (0.941176471, 0.784313725, 0.62745098),
            "P": (1, 0.501960784, 0),
            "S": (1, 1, 0.188235294),
            "Cl": (0.121568627, 0.941176471, 0.121568627),
            "Ar": (0.501960784, 0.819607843, 0.890196078),
            "K": (0.560784314, 0.250980392, 0.831372549),
            "Ca": (0.239215686, 1, 0),
            "Sc": (0.901960784, 0.901960784, 0.901960784),
            "Ti": (0.749019608, 0.760784314, 0.780392157),
            "V": (0.650980392, 0.650980392, 0.670588235),
            "Cr": (0.541176471, 0.6, 0.780392157),
            "Mn": (0.611764706, 0.478431373, 0.780392157),
            "Fe": (0.878431373, 0.4, 0.2),
            "Co": (0.941176471, 0.564705882, 0.62745098),
            "Ni": (0.31372549, 0.815686275, 0.31372549),
            "Cu": (0.784313725, 0.501960784, 0.2),
            "Zn": (0.490196078, 0.501960784, 0.690196078),
            "Ga": (0.760784314, 0.560784314, 0.560784314),
            "Ge": (0.4, 0.560784314, 0.560784314),
            "As": (0.741176471, 0.501960784, 0.890196078),
            "Se": (1, 0.631372549, 0),
            "Br": (0.650980392, 0.160784314, 0.160784314),
            "Kr": (0.360784314, 0.721568627, 0.819607843),
            "Rb": (0.439215686, 0.180392157, 0.690196078),
            "Sr": (0, 1, 0),
            "Y": (0.580392157, 1, 1),
            "Zr": (0.580392157, 0.878431373, 0.878431373),
            "Nb": (0.450980392, 0.760784314, 0.788235294),
            "Mo": (0.329411765, 0.709803922, 0.709803922),
            "Tc": (0.231372549, 0.619607843, 0.619607843),
            "Ru": (0.141176471, 0.560784314, 0.560784314),
            "Rh": (0.039215686, 0.490196078, 0.549019608),
            "Pd": (0, 0.411764706, 0.521568627),
            "Ag": (0.752941176, 0.752941176, 0.752941176),
            "Cd": (1, 0.850980392, 0.560784314),
            "In": (0.650980392, 0.458823529, 0.450980392),
            "Sn": (0.4, 0.501960784, 0.501960784),
            "Sb": (0.619607843, 0.388235294, 0.709803922),
            "Te": (0.831372549, 0.478431373, 0),
            "I": (0.580392157, 0, 0.580392157),
            "Xe": (0.258823529, 0.619607843, 0.690196078),
            "Cs": (0.341176471, 0.090196078, 0.560784314),
            "Ba": (0, 0.788235294, 0),
            "La": (0.439215686, 0.831372549, 1),
            "Ce": (1, 1, 0.780392157),
            "Pr": (0.850980392, 1, 0.780392157),
            "Nd": (0.780392157, 1, 0.780392157),
            "Pm": (0.639215686, 1, 0.780392157),
            "Sm": (0.560784314, 1, 0.780392157),
            "Eu": (0.380392157, 1, 0.780392157),
            "Gd": (0.270588235, 1, 0.780392157),
            "Tb": (0.188235294, 1, 0.780392157),
            "Dy": (0.121568627, 1, 0.780392157),
            "Ho": (0, 1, 0.611764706),
            "Er": (0, 0.901960784, 0.458823529),
            "Tm": (0, 0.831372549, 0.321568627),
            "Yb": (0, 0.749019608, 0.219607843),
            "Lu": (0, 0.670588235, 0.141176471),
            "Hf": (0.301960784, 0.760784314, 1),
            "Ta": (0.301960784, 0.650980392, 1),
            "W": (0.129411765, 0.580392157, 0.839215686),
            "Re": (0.149019608, 0.490196078, 0.670588235),
            "Os": (0.149019608, 0.4, 0.588235294),
            "Ir": (0.090196078, 0.329411765, 0.529411765),
            "Pt": (0.815686275, 0.815686275, 0.878431373),
            "Au": (1, 0.819607843, 0.137254902),
            "Hg": (0.721568627, 0.721568627, 0.815686275),
            "Tl": (0.650980392, 0.329411765, 0.301960784),
            "Pb": (0.341176471, 0.349019608, 0.380392157),
            "Bi": (0.619607843, 0.309803922, 0.709803922),
            "Po": (0.670588235, 0.360784314, 0),
            "At": (0.458823529, 0.309803922, 0.270588235),
            "Rn": (0.258823529, 0.509803922, 0.588235294),
            "Fr": (0.258823529, 0, 0.4),
            "Ra": (0, 0.490196078, 0),
            "Ac": (0.439215686, 0.670588235, 0.980392157),
            "Th": (0, 0.729411765, 1),
            "Pa": (0, 0.631372549, 1),
            "U": (0, 0.560784314, 1),
            "Np": (0, 0.501960784, 1),
            "Pu": (0, 0.419607843, 1),
            "Am": (0.329411765, 0.360784314, 0.949019608),
            "Cm": (0.470588235, 0.360784314, 0.890196078),
            "Bk": (0.541176471, 0.309803922, 0.890196078),
            "Cf": (0.631372549, 0.211764706, 0.831372549),
            "Es": (0.701960784, 0.121568627, 0.831372549),
            "Fm": (0.701960784, 0.121568627, 0.729411765),
            "Md": (0.701960784, 0.050980392, 0.650980392),
            "No": (0.741176471, 0.050980392, 0.529411765),
            "Lr": (0.780392157, 0, 0.4),
            "Rf": (0.8, 0, 0.349019608),
            "Db": (0.819607843, 0, 0.309803922),
            "Sg": (0.850980392, 0, 0.270588235),
            "Bh": (0.878431373, 0, 0.219607843),
            "Hs": (0.901960784, 0, 0.180392157),
            "Mt": (0.921568627, 0, 0.149019608),
        }

