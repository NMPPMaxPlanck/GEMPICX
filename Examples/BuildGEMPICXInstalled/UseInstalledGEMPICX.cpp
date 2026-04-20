/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include "GEMPICX.H"
#include "GEMPIC_Verbosity.H"
#include "GEMPIC_Version.H"

using namespace Gempicx;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv, &overwrite_amrex_parser_defaults);
    {
        Verbosity::set_level(99);
        print_gempicx_version();

        amrex::Print() << "AMREX_SPACEDIM=" << AMREX_SPACEDIM << '\n'
                       << "verbosity:" << Verbosity::level() << '\n';
        amrex::Print() << "Git version: " << gempicx_git_version() << '\n'
                       << "CMake Package version: " << gempicx_pkg_version() << '\n'
                       << "Release number: " << gempicx_release_number() << '\n'
                       << "AMReX version: " << gempicx_amrex_version() << '\n'
                       << "HYPRE version: " << gempicx_hypre_version() << '\n';
    }
    amrex::Finalize();
    return EXIT_SUCCESS;
}
