THIS IS NOT UP TO DATE.

Build requirements:
1. cmake 
2. make
3. a suitable c/c++ compiler
4. libeigen2-dev, libpng-dev, libpng++-dev, libtiff 3.9.4-5ubuntu6 (through ubuntu)
5. hdf5 (for pipeline), adios (for SCIO), boost, MPI

This package requires the following external libraries
CUDA  (need drivers, toolkit, and SDK from NVIDIA)
OpenCV 2.3.0 (require cmake, and optionally CUDA, TBB).  use system libpnginstead of the built-in. - important.  else get png version issues.  turned off video stuff, on with TBB and CUDA.  turn off QT-opengl. Do make install at the end
HDF5 4.1.2 shared 64bit for lnux 2.6

editing:
using eclipse is recommended but not required.
installed valgrind
eclipse Linux Tools is recommended for building with autoconf/automake


compiling:
