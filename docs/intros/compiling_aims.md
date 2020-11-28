# Compiling a local version of FHI-aims

Having a local installation of FHI-aims can be useful to test keywords and develop new features. This guide walks you through the steps to compile your own installation of FHI-aims with the Anaconda Math Kernel Library (MKL).

## Prerequisites

You need to:
- have access to the [FHI-aims gitlab server](https://aims-git.rz-berlin.mpg.de/) to download the source code
- an installation of [Anaconda](https://aims-git.rz-berlin.mpg.de/).
- a working gcc compiler (on Unix or WSL, try `sudo apt-get install build-essential gdb`)

## Step-by-Step-Guide

1. **Install CMake, the MKL library and an openMPI compiler:**

    ```bash
    conda install -c conda-forge cmake 
    conda install -c anaconda mkl 
    conda install -c conda-forge openmpi-mpifort 
    ```

2. **Clone the FHI-aims repository:**

    ```bash
    git clone https://aims-git.rz-berlin.mpg.de/aims/FHIaims.git
    ```

3. **Enter the directory where you cloned the repository. Then:**

    ```bash
    mkdir build
    cd build
    cp ../initial_cache.example.cmake initital_cache.conda.cmake
    ```

4. **Edit the `initital_cache.conda.cmake` too look similar to this:**

    ```cmake
    set(CMAKE_Fortran_COMPILER "mpifort" CACHE STRING "" FORCE)
    set(CMAKE_Fortran_FLAGS "-O2 -ffree-line-length-none" CACHE STRING "" FORCE)
    set(Fortran_MIN_FLAGS "-O0 -ffree-line-length-none" CACHE STRING "" FORCE)

    set(CMAKE_C_COMPILER "gcc" CACHE STRING "" FORCE)

    set(CMAKE_C_FLAGS "-O2 -funroll-loops -std=gnu99" CACHE STRING "" FORCE)
    set(LIB_PATHS "~/miniconda3/pkgs/mkl-2020.2-256/lib" CACHE STRING "" FORCE)   # <-- this line is important
    set(LIBS "mkl_intel_lp64 mkl_sequential mkl_core" CACHE STRING "" FORCE)

    set(USE_MPI ON CACHE BOOL "" FORCE)
    set(USE_SCALAPACK OFF CACHE BOOL "" FORCE)
    set(USE_LIBXC ON CACHE BOOL "" FORCE)
    set(USE_HDF5 OFF CACHE BOOL "" FORCE)
    set(USE_RLSY ON CACHE BOOL "" FORCE)
    ```

5. **Find the path to the MKL library files.**

    Depending on the type of Anaconda installation you have and the operating system, this might differ from user to user. The files should be somewhere in the `/pkgs/mkl-versionnumber/lib directory`. Inside this folder, there should be many shared objects `.so`, such as `libmkl_intel_lp64.so`.

    Put this path in the `initital_cache.conda.cmake` in the `set(LIB_PATHS "/here/put/the/path" CACHE STRING "" FORCE)` line.

6. **Build & compile**

    Enter the build directory, then:

    ```bash
    cmake -C initial_cache.conda.cmake
    ```

    This will run CMake to prepare all files for linking and checks if the compilers work correctly. If everything configured correctly, run:

    ```bash
    make -j 4
    ```

    Where `-j 4` specifies the number of cores to use to speed up the compilation. This might take a while depending on your machine.

7. **Finishing up**

    When everything is done, you should have the `aims.VERSION.mpi.x` binary in the build directory. Just test it by running 

    ```bash
    mpirun aims.*.mpi.x
    ```

    To see that everything works. Afterwards, move it to the bin/ directory in the FHI-aims folder and add it to your path:

    ```bash
    mv aims.*.mpi.x ../bin
    ```

    And add theses line to your ~/.bashrc:

    ```bash
    export AIMS_EXECUTABLE="path/to/aims.201103.mpi.x"
    export ASE_AIMS_COMMAND="mpirun $AIMS_EXECUTABLE"
    ```
