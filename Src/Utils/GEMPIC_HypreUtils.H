/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef HYPRE_UTILS
#define HYPRE_UTILS

#include <format>
#include <memory>

#include <AMReX_HypreSolver.H>

namespace Gempic::Utils
{
template <int stencilSize>
void check_solver_convergence (amrex::HypreSolver<stencilSize> const& hypreSolver,
                               int maxiter,
                               amrex::Real accTol)
{
    if (hypreSolver.getNumIters() == maxiter)
    {
        amrex::Real norm{hypreSolver.getFinalResidualNorm()};
        std::string toleranceMessage = std::format(
            "Hypre solver: Required tolerance not reached\nGot {:e}. The accepted tolerance is "
            "{:e}",
            norm, accTol);
        GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(norm < accTol, toleranceMessage);
        // Otherwise
        if (hypre_error_flag)
        {
            char msg[100];
            HYPRE_DescribeError(hypre_error_flag, msg);
            std::string message{std::format("Warning: {} {}.\nAccepting", msg, toleranceMessage)};
            amrex::Warning(message.c_str());
            HYPRE_ClearAllErrors();
        }
    }
}

template <int stencilSize>
void check_solver_convergence (std::unique_ptr<amrex::HypreSolver<stencilSize>> const& hypreSolver,
                               int maxiter,
                               amrex::Real accTol)
{
    check_solver_convergence(*hypreSolver, maxiter, accTol);
}

// Function needed for building all HYPRE matrices
inline void assign_or_add_col (int gdtmp,
                               int& icol,
                               HYPRE_Int& ncols,
                               HYPRE_Int* cols,
                               HYPRE_Real* mat,
                               std::map<int, int>& mapColsGeneral)
{
    if (mapColsGeneral.find(gdtmp) != mapColsGeneral.end()) // element exists already
    {
        icol = mapColsGeneral[gdtmp];
    }
    else // element does not exist, and is newly added
    {
        icol = ncols++;
        cols[icol] = gdtmp;
        mapColsGeneral.insert(std::make_pair(cols[icol], icol));
        mat[icol] = 0.0; // matrix element is initialized to 0, and then incremented by filler
                         // functions
    }
}
} //namespace Gempic::Utils
#endif
