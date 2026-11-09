/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <AMReX.H>
#include <AMReX_MultiFab.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"

namespace Gempic::FieldSolvers::Impl
{

void interpolate_from_rhs_and_swap_directions (
    Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& rhsDummy,
    Forms::DeRhamField<Forms::Grid::dual, Forms::Space::face>& rhs,
    amrex::GpuArray<amrex::Real, 3> dxi,
    std::array<int, 2> const& pdir)
{
    amrex::IntVect ng{rhs.m_deRham->get_n_ghost()};

    std::array<int, 2> in = {0, 0};
    std::array<int, 2> jn = {0, 0};
#if AMREX_SPACEDIM == 3
    std::array<int, 2> kn = {0, 0};
#endif

    for (int n = 0; n < 2; n++)
    {
        in[n] = (pdir[n] == xDir);
        jn[n] = (pdir[n] == yDir);
#if AMREX_SPACEDIM == 3
        kn[n] = (pdir[n] == zDir);
#endif
    }

    for (amrex::MFIter mfi(rhs.m_data[pdir[1]]); mfi.isValid(); ++mfi)
    {
        amrex::Array4<amrex::Real> const& rhsDummyArrPdir0 =
            (rhsDummy.m_data[pdir[0]])[mfi].array();
        amrex::Array4<amrex::Real> const& rhsArrPdir1 = (rhs.m_data[pdir[1]])[mfi].array();

        amrex::Array4<amrex::Real> const& rhsDummyArrPdir1 =
            (rhsDummy.m_data[pdir[1]])[mfi].array();
        amrex::Array4<amrex::Real> const& rhsArrPdir0 = (rhs.m_data[pdir[0]])[mfi].array();

        auto bx = (rhsDummy.m_data[pdir[0]])[mfi].box();
        bx.grow({AMREX_D_DECL(-ng[yDir] + 1, -ng[xDir] + 1, -ng[zDir] + 1)});
        int nc = rhsDummyArrPdir0.nComp();

        for (int n = 0; n < nc; n++)
        {
            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    // Project rhs[pdir[1]] to rhs_dummy[pdir[0]]
                    rhsDummyArrPdir0(AMREX_D_DECL(i, j, k), n) =
                        0.25 * (dxi[pdir[0]] / dxi[pdir[1]]) *
                        (rhsArrPdir1(AMREX_D_DECL(i, j, k), n) +
                         rhsArrPdir1(AMREX_D_DECL(i + in[0], j + jn[0], k + kn[0]), n) +
                         rhsArrPdir1(AMREX_D_DECL(i - in[1], j - jn[1], k - kn[1]), n) +
                         rhsArrPdir1(
                             AMREX_D_DECL(i + in[0] - in[1], j + jn[0] - jn[1], k + kn[0] - kn[1]),
                             n));

                    // Project rhs[pdir[0]] to rhs_dummy[pdir[1]]
                    rhsDummyArrPdir1(AMREX_D_DECL(i, j, k), n) =
                        0.25 * (dxi[pdir[1]] / dxi[pdir[0]]) *
                        (rhsArrPdir0(AMREX_D_DECL(i, j, k), n) +
                         rhsArrPdir0(AMREX_D_DECL(i + in[1], j + jn[1], k + kn[1]), n) +
                         rhsArrPdir0(AMREX_D_DECL(i - in[0], j - jn[0], k - kn[0]), n) +
                         rhsArrPdir0(
                             AMREX_D_DECL(i + in[1] - in[0], j + jn[1] - jn[0], k + kn[1] - kn[0]),
                             n));
                });
        }
    }

    rhsDummy.fill_boundary();
}

} // namespace Gempic::FieldSolvers::Impl
