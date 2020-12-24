.. aimstools documentation master file, created by
   sphinx-quickstart on Sun Nov 10 16:30:32 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Tools for FHI-aims
==========================

This library contains a personal collection of scripts to handle FHI-aims calculations. It's mainly meant for private use or to be shared with students and colleagues.

.. toctree::
   :maxdepth: 3
   :caption: Introduction:   

   intros/intro
   intros/index

.. toctree::
   :maxdepth: 3
   :caption: Command-line tools:
   
   cli/preparing_aims
   cli/workflows
   cli/visualization 

.. toctree::
   :maxdepth: 3
   :caption: Workflows:

   workflows/kpoint_convergence
   workflows/relaxation

.. toctree::
   :maxdepth: 3
   :caption: Examples:
   
   notebooks/structuretools.ipynb
   notebooks/bandstructures.ipynb
   notebooks/dosfigure.ipynb
   notebooks/fatbands.ipynb

.. toctree::
   :maxdepth: 5
   :caption: Modules

   aimstools/structuretools/structuretools
   aimstools/preparation/preparation
   aimstools/postprocessing/postprocessing
   aimstools/bandstructures/bandstructures
   aimstools/density_of_states/density_of_states
   aimstools/phonons/phonons
   aimstools/workflows/workflows

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
