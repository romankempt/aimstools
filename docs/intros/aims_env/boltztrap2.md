# Interface to BoltzTraP2

I've added an FHI-AIMS backend to the [BoltzTraP2 python library](https://gitlab.com/sousaw/BoltzTraP2) by Jesús Carrete Montaña.

The following guide only walks you through the installation with Anaconda on Unix or the Windows Subsystem for Linux (WSL). Installation on Windows or Mac might be difficult due to troubles with the C++-Compilers.

1. Install a C++-Compiler on your system:
    ```bash
    sudo apt-get update
    sudo apt-get install build-essential gdb
    whereis g++
    whereis gdb
    ```

2. Setup a conda environment for boltztrap2 and install the required packages:
    ```bash
    conda create -n boltztrap python=3.8
    conda activate boltztrap
    conda install -c conda-forge numpy scipy matplotlib spglib netcdf4 ase make cmake pytest
    ```

3. Download and unpack the [Fastest Fourier Transform in the West, FFTW](http://www.fftw.org/#documentation).
    Enter the directory where you unpacked it and run in your conda environment:
    ```bash
    bash configure
    make
    make install
    ```
    This will compile the C-subroutines. Then install the pyFFTW wrapper:
    ```bash
    conda install -c conda-forge pyfftw
    ```

4. Either clone or download and unpack [BoltzTraP2 python library](https://gitlab.com/sousaw/BoltzTraP2) and run in your anaconda environment:
    ```
    python setup.py install
    ```
    Or follow the installation instructions via pip.
    This should compile the binaries for BoltzTraP2 and install the python wrappers in your environment.


5. Test the installation.
    ```
    python setup.py develop
    pytest -v tests -k AIMS
    btp2 --help
    ```