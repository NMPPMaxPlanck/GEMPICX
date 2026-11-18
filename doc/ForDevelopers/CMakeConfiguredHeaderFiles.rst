CMake-configured header files (.H.in files)
===========================================

CMake has the ability to fill out files with CMake variables at configure time. Such files can be useful for getting a number of constants, such as, for instance, version numbers.

For this as well we have guidelines:

#. The input file should be named ``GEMPIC_<...>.H.in`` and placed in the appropriate subfolder of ``Src``

#. The ``CMakeLists.txt`` file of the corresponding subfolder should call ``gempic_create_and_link_cmake_constants_file(<filename>)``

#. The output files are all in ``<Build folder>/generated`` (and will be if generated with ``gempic_create_and_link_cmake_constants_file``)

#. Files dependent on the produced header files simply include via ``#include "GEMPIC_<...>.H"``
