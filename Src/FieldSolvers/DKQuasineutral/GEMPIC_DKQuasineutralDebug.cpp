/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/

#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "GEMPIC_DKQuasineutralDebug.H"

#ifndef AMREX_USE_HYPRE
#error "HYPRE not enabled"
#endif

// Functions made for debugging start
namespace Gempic::FieldSolvers
{
using namespace Forms;
using namespace Particle;

void check_rho_related_norms_dkqn (DeRhamField<Forms::Grid::dual, Forms::Space::cell>& rho,
                                   std::vector<amrex::Real> rhoAnVec,
                                   DeRhamField<Forms::Grid::dual, Forms::Space::face>& J,
                                   DeRhamField<Forms::Grid::dual, Forms::Space::cell>& divJ,
                                   ComputationalDomain& mInfra)
{
    Kokkos::Profiling::ScopedRegion region("check_rho_related_norms_DKQN()");

    amrex::Print() << "\nError checks for rho and J values:";
    div(divJ, J);
    amrex::Print() << "\nDivergence of J: " << Utils::gempic_norm(divJ.m_data, mInfra, 2);

    amrex::Print() << "\nrho error %: "
                   << (Utils::gempic_norm(rho.m_data, mInfra, 1) / rhoAnVec[1]) * 100.0 << "\n";
}

} //namespace Gempic::FieldSolvers