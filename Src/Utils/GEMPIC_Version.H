/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef _GEMPIC_Version_H_
#define _GEMPIC_Version_H_

#include <string>

#include "GEMPIC_CMakeConstants.H"

namespace Gempic::Utils
{
/** @brief: Description of the status of the git repository at the time of compilation
 *  Note that AMReX should be initialized before calling this, or you'll get a SEGFAULT
 */
void print_gempicx_version();

inline std::string gempicx_git_version ()
{
#ifdef GEMPICX_GIT_VERSION
    return GEMPICX_GIT_VERSION;
#else
    return "Unknown (repository was not fetched/installed with git?)\n";
#endif
}

inline std::string gempicx_pkg_version ()
{
#ifdef GEMPICX_PKG_VERSION
    return GEMPICX_PKG_VERSION;
#else
    return "Unknown (repository was not fetched/installed with git?)\n";
#endif
}

inline std::string gempicx_release_number ()
{
#ifdef GEMPICX_RELEASE_NUMBER
    return GEMPICX_RELEASE_NUMBER;
#else
    return "Unknown (repository was not fetched/installed with git?)\n";
#endif
}

inline std::string gempicx_amrex_version ()
{
#ifdef GEMPICX_AMReX_VERSION_FOUND
    return GEMPICX_AMReX_VERSION_FOUND;
#else
    return "Unknown (AMReX was not fetched/installed with git?)\n";
#endif
}

inline std::string gempicx_hypre_version ()
{
#if AMReX_USE_HYPRE
#ifdef GEMPICX_HYPRE_VERSION_FOUND
    return GEMPICX_HYPRE_VERSION_FOUND;
#else
    return "Unknown (HYPRE was not fetched/installed with git?)\n";
#endif
#else
    return "Hypre not enabled\n";
#endif
}
} //namespace Gempic::Utils

#endif
