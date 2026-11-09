/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#ifndef GEMPIC_DKQUASINEUTRALIMPL_H_
#define GEMPIC_DKQUASINEUTRALIMPL_H_

#include <AMReX.H>
#include <AMReX_MultiFab.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"

namespace Gempic::FieldSolvers::Impl
{

void interpolate_from_rhs_and_swap_directions(
    Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& rhsDummy,
    Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& rhs,
    amrex::GpuArray<amrex::Real, 3> dxi,
    std::array<int, 2> const& pdir);

} // namespace Gempic::FieldSolvers::Impl

#endif
