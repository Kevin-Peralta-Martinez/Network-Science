# Network-Analysis
Numerical simulations and analysis of structural, spectral and eigenvector properties of random network models.
This repository contains code for visualizing and performing calculations on random graphs. It includes numerical computations and ensemble analysis.
## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## Usage
1. Install the required dependencies:
   ```bash
   pip install numpy networkx matplotlib
2. Open the notebook!
   ```bash
   jupyter notebook Random_network_models.ipynb
4. Run the cells to perform calculations and visualize graphs!.

### List of libraries and tools required to run the notebook:
-Python 3.x
-Numpy
-Networkx
-Matplotlib
-Jupyter notebook
-scipy
-seaborn

## Make sure to adress the script:
```bash
random_network_models.py
```

#### Here we find the classes and methods for the generation of three different random network models (directed models coming soon):
- Erdös Renyi
- Random geometric graphs (embedded in a unit square)
- Hyperbolic random graphs (embedded in a hyperbolic two-dimensional disk)

Execute the Notebook 
```bash
Networks_Visualization.ipynb
```
to visualize samples of the network models, their binary adjacency matrix, and degree distribution.

1. Execute the Notebook
```bsh
structural_properties.ipynb
```
to visualize the transition of the structural properties (topological indices, clustering coefficients) of the network models as they transit from isolated to complete graphs.

2. Execute the Notebook (coming soon) <br>
```bash
spectral_properties.ipnynb
```
to visualize the transition of the spectral and eigenvector properties (Shannon entropy, Participatio ratio, ratio between consecutive eigenvalue spacings) of the network models as they transit from isolated to complete graphs. <br> This, in the context of an ensemble of weighted adjacency matrices that correspond to the diluted full random matrix ensemble from Random Matrix Theory (RMT).


3. (Optional) Explore the Fortran 90 file (coming soon)
```bash
subroutines_structural_and_spectral_properties.f90
```

here you can find subroutines for the computation of the same structural, spectral, and eigenvector properties of the previous notebooks. <br>
Once you choose the model, you can use the subroutines of subroutines_structural_and_spectral_properties.f90 and perform in parallel for optimization <br>
```bash
ifort -O3 -qopenmp -o executable_name name_of_your_program.f90 -i8 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm
```

Choose the number of threads to use
```bash
OMP_NUM_THREADS=4
```
<br>


For running in the background, <br>
```bash
nohup ./executable_name & 
```
<br>
1. Software Requirements:<br>
   a. Intel Fortran Compiler (ifort)<br>
   The ifort command is part of the Intel oneAPI Toolkit (specifically the Intel oneAPI HPC Toolkit).

   You need to install the Intel Fortran Compiler on your machine.

   Download and install it from the Intel oneAPI website.

   b. Intel Math Kernel Library (MKL)<br>
   The -lmkl_intel_lp64, -lmkl_intel_thread, and -lmkl_core flags link the Intel MKL libraries.

   Intel MKL is included in the Intel oneAPI Base Toolkit.

   c. OpenMP Runtime
   The -qopenmp flag enables OpenMP support, which requires an OpenMP runtime library. <br>
   The Intel Fortran Compiler includes the necessary OpenMP runtime, so no additional installation is required.

   d. POSIX Threads (pthread)
   The -lpthread flag links the POSIX threads library, which is typically included in most Linux distributions.

   Ensure that the pthread library is available on your system (it usually is by default).

   e. Standard Math Library (libm)
   The -lm flag links the standard math library (libm), which is part of the GNU C Library (glibc).

   This library is included in all Linux distributions by default.

   f. Bash Shell
   The command is written for a Bash shell, which is the default shell on most Linux distributions.

   Ensure that Bash is installed and available on your system.

2. Hardware Requirements
   a. Multi-core CPU
   The OMP_NUM_THREADS=4 environment variable specifies that the program will use 4 threads for parallel execution.

   Your machine must have a multi-core CPU with at least 4 cores to fully utilize the parallelization.

   

3. Operating System
The command is designed for a Linux-based system (e.g., Ubuntu, CentOS, Fedora).

If you're using Windows, you can use the Windows Subsystem for Linux (WSL) or a virtual machine with a Linux distribution.

On macOS, you can use the Intel oneAPI Toolkit for macOS, but the command may need some modifications.

4. Alternative compiler:
   ```bash
gfortran -O3 -fopenmp -o P SpGralGOE.f90 -fdefault-integer-8 -llapack -lblas -lpthread -lm
```
Incorporation of BLAS and LAPACK packages is neccesary for numerical diagonalization for the computation of spectral and eigenvector properties.


