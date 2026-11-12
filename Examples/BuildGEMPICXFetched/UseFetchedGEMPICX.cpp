/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <AMReX_Print.H>

#include "GEMPICX.H"
#include "GEMPIC_Verbosity.H"
#include "GEMPIC_Version.H"

using namespace Gempicx;

int main (int argc, char* argv[])
{
    Gempic::initialize(argc, argv);
    {
        Verbosity::set_level(99);
        print_gempicx_version();

        amrex::Print() << "AMREX_SPACEDIM=" << AMREX_SPACEDIM << '\n'
                       << "verbosity:" << Verbosity::level() << '\n';
        amrex::Print() << "Git version: " << gempicx_git_version() << '\n'
                       << "CMake Package version: " << gempicx_pkg_version() << '\n'
                       << "Release number: " << gempicx_release_number() << '\n'
                       << "AMReX version: " << gempicx_amrex_version() << '\n'
                       << "Kokkos version: " << gempicx_kokkos_version() << '\n'
                       << "HYPRE version: " << gempicx_hypre_version() << '\n';
    }
    Gempic::finalize();
    return EXIT_SUCCESS;
}
