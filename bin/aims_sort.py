#!/usr/bin/env python
import os, sys
from AIMS_tools.preparation import prepare
from AIMS_tools.postprocessing import postprocess
from pathlib import Path
import shutil

cwd = Path.cwd()
dirs = [x for x in cwd.iterdir() if cwd.is_dir()]
dirs = [x for x in dirs if x.joinpath("control.in").exists()]
dirs = [x for x in dirs if x.joinpath("geometry.in").exists()]
# dirs = [postprocess(str(x)) for x in dirs]
converged = [x for x in dirs if postprocess(str(x)).success == True]
not_converged = [x for x in dirs if postprocess(str(x)).success == False]

print(
    "Converged: \t {}\nNot converged: \t {}".format(len(converged), len(not_converged))
)

do_sort = input("Sort into new directories?    (y/n)    ")
if do_sort in ["y", "Y", "yes", "Yes", "YES", None]:
    try:
        convs = str(cwd.joinpath("converged"))
        not_convs = str(cwd.joinpath("not_converged"))
        os.mkdir(convs)
        os.mkdir(not_convs)
    except FileExistsError:
        convs = str(cwd.joinpath("converged"))
        not_convs = str(cwd.joinpath("not_converged"))
    if len(converged) != 0:
        print("Moving converged calculations to {} ...".format(convs))
        for job in converged:
            shutil.move(str(job), convs)
    if len(not_converged) != 0:
        print("Moving other calculations to {} ...".format(not_convs))
        for job in not_converged:
            shutil.move(str(job), not_convs)
