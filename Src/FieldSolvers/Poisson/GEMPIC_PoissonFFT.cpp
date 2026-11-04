/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <AMReX_FFT.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_FieldMethods.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_PoissonFFT.H"
#include "GEMPIC_PoissonSolver.H"

using namespace Gempic::FieldSolvers;
using namespace Gempic::Forms;

namespace
{
/**
 * Eigenvalues of the circulant matrix defined by a symmetric 1D stencil on n periodic points:
 * \f$ \sum_d s_d \cos(2\pi k (d - c) / n) \f$ with c the stencil center
 */
template <unsigned int stencilLength>
amrex::Vector<amrex::Real> stencil_eigenvalues (
    amrex::GpuArray<amrex::Real, stencilLength> const& stencil, int n)
{
    amrex::Vector<amrex::Real> eigenvalues(n, 0.0);
    int const center = (stencilLength - 1) / 2;
    for (int k = 0; k < n; k++)
    {
        for (unsigned int d = 0; d < stencilLength; d++)
        {
            eigenvalues[k] += stencil[d] * std::cos(2 * M_PI * k * (int(d) - center) / n);
        }
    }
    return eigenvalues;
}

/**
 * Computes the eigenvalues of both 1D Hodge stencils used by
 * hodge_operator(Dual2Form&, Primal1Form const&): stencilAlong is applied along the component
 * direction, stencilAcross along the other directions.
 */
template <HodgeScheme hodgeScheme, int hodgeDegreeIdx = 0>
void hodge_eigenvalues (int hodgeDegree,
                        int n,
                        amrex::Vector<amrex::Real>& along,
                        amrex::Vector<amrex::Real>& across)
{
    constexpr auto validDegreesArray = valid_degrees<hodgeScheme>();
    if constexpr (hodgeDegreeIdx < validDegreesArray.size())
    {
        if (hodgeDegree == validDegreesArray[hodgeDegreeIdx])
        {
            auto [stencilAcross, stencilAlong] =
                get_hodge_stencils<validDegreesArray[hodgeDegreeIdx], hodgeScheme>();
            along = stencil_eigenvalues(stencilAlong, n);
            across = stencil_eigenvalues(stencilAcross, n);
        }
        else
        {
            hodge_eigenvalues<hodgeScheme, hodgeDegreeIdx + 1>(hodgeDegree, n, along, across);
        }
    }
    else
    {
        GEMPIC_ERROR("Degree " + std::to_string(hodgeDegree) +
                     " not implemented for the selected Hodge in the FFT Poisson solver");
    }
}
} // namespace

PoissonFFTSolver::PoissonFFTSolver(ComputationalDomain const& compDom,
                                   int const hodgeDegree,
                                   HodgeScheme hodgeScheme) :
    m_compDom{compDom}
{
    Kokkos::Profiling::ScopedRegion region(
        "Gempic::FieldSolvers::PoissonFFTSolver::PoissonFFTSolver()");
    if (hodgeScheme != HodgeScheme::FDHodge && hodgeScheme != HodgeScheme::GDECHodge)
    {
        GEMPIC_ERROR(
            "The FFT Poison solver only works with Finite Difference Hodge and Galerkin Difference "
            "Hodge");
    }

    // Check for periodic boundaries, required for FFT solver
    if (!compDom.geometry().periodicity().isAllPeriodic())
    {
        GEMPIC_ERROR("FFT Poisson solver requires periodic boundary conditions");
    }

    m_r2c = std::make_unique<amrex::FFT::R2C<amrex::Real>>(compDom.box());
    m_rhoFft.define(compDom.m_grid, compDom.m_distriMap, 1, 0);
    m_phiFft.define(compDom.m_grid, compDom.m_distriMap, 1, 0);

    // sizes in x, y and z directions
    auto dx = compDom.cell_size_array();

    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        int const n = compDom.box().length(dir);
        amrex::Vector<amrex::Real> along;
        amrex::Vector<amrex::Real> across;
        if (hodgeScheme == HodgeScheme::FDHodge)
        {
            hodge_eigenvalues<HodgeScheme::FDHodge>(hodgeDegree, n, along, across);
        }
        else if (hodgeScheme == HodgeScheme::GDECHodge)
        {
            hodge_eigenvalues<HodgeScheme::GDECHodge>(hodgeDegree, n, along, across);
        }
        else
        {
            GEMPIC_ERROR("Hodge scheme not implemented for FFT Poisson solver");
        }

        // eigenvalues of the 1D Laplacian -D G, times Hodge eigenvalues and scaling
        for (int iter = 0; iter < n; iter++)
        {
            along[iter] *=
                (2 - 2 * std::cos(2 * M_PI * iter / n)) * dx.product() / (dx[dir] * dx[dir]);
        }

        // Copy eigenvalues to GPU
        m_eigenvaluesAlongGpu[dir].resize(n);
        m_eigenvaluesAcrossGpu[dir].resize(n);
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, along.begin(), along.end(),
                              m_eigenvaluesAlongGpu[dir].begin());
        amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, across.begin(), across.end(),
                              m_eigenvaluesAcrossGpu[dir].begin());
    }
    amrex::Gpu::streamSynchronize();
}

void PoissonFFTSolver::solve (DeRhamField<Grid::primal, Space::node>& phi,
                             DeRhamField<Grid::dual, Space::cell> const& rho)
{
    Kokkos::Profiling::ScopedRegion region("Gempic::FieldSolvers::PoissonFFTSolver::solve()");

    AMREX_ALWAYS_ASSERT(phi.m_data.nComp() == rho.m_data.nComp());
    check_charge_neutrality(rho, m_compDom);

    // AMReX FFT requires cell-centered input. The source/destination may not be cell-centered,
    // which on a periodic domain means the high-boundary node is a duplicate of the low-boundary
    // node.
    // In the following we drop the duplicated high-boundary node by copying data into a
    // cell-centered MultiFab. The FFT is applied to the MultiFab with cell-centered data. Finally
    // the data is written to the result array and an OverrideSync is used to (in MPI terms)
    // broadcast the result back to the duplicated nodes.
    amrex::iMultiFab ownermask{phi.m_data.boxArray(), phi.m_data.DistributionMap(), 1, 0};
    ownermask.setVal(0);

    for (int comp{0}; comp < rho.m_data.nComp(); ++comp)
    {
        // assign values to containers from rho.m_data
        for (amrex::MFIter mfi(m_rhoFft); mfi.isValid(); ++mfi)
        {
            auto const& rhoFftPtr = m_rhoFft.array(mfi);
            auto const& rhoMDataPtr = rho.m_data.array(mfi);
            auto const& owner = ownermask.array(mfi);

            amrex::ParallelFor(mfi.validbox(),
                               [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                               {
                                   rhoFftPtr(i, j, k) = rhoMDataPtr(i, j, k, comp);
                                   // The owner mask is the same for the src and dst MultiFab
                                   // since the src and dst MF of the FFT are the same.
                                   owner(i, j, k) = 1;
                               });
        }

        // local variables necessary for CUDA
        // acrossX is only needed from 2D on (in 1D the eigenvalue is just alongX)
        AMREX_D_TERM(amrex::Real const* alongX = m_eigenvaluesAlongGpu[xDir].dataPtr();
                     , amrex::Real const* acrossX = m_eigenvaluesAcrossGpu[xDir].dataPtr();
                     amrex::Real const* alongY = m_eigenvaluesAlongGpu[yDir].dataPtr();
                     amrex::Real const* acrossY = m_eigenvaluesAcrossGpu[yDir].dataPtr();
                     , amrex::Real const* alongZ = m_eigenvaluesAlongGpu[zDir].dataPtr();
                     amrex::Real const* acrossZ = m_eigenvaluesAcrossGpu[zDir].dataPtr();)
        int cellNum = m_compDom.box().length3d().product();

        m_r2c->forwardThenBackward(
            m_rhoFft, m_phiFft,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, auto& sp)
            {
                GEMPIC_D_EXCL(UNUSED(j);, UNUSED(k);, )
                auto eigenvalueSum{GEMPIC_D_ADD(GEMPIC_D_MULT(alongX[i], acrossY[j], acrossZ[k]),
                                                GEMPIC_D_MULT(acrossX[i], alongY[j], acrossZ[k]),
                                                GEMPIC_D_MULT(acrossX[i], acrossY[j], alongZ[k]))};
                if (eigenvalueSum == 0)
                {
                    sp = 0;
                }
                else
                {
                    sp /= eigenvalueSum * cellNum;
                }
            });

        // Iterate over cell-centered m_phiFft boxes to avoid accessing the periodic boundary
        // node that lies outside the cell-centered valid region; OverrideSync handles it below.
        for (amrex::MFIter mfi(m_phiFft); mfi.isValid(); ++mfi)
        {
            auto const& phiFftPtr = m_phiFft.array(mfi);
            auto const& phiMDataPtr = phi.m_data.array(mfi);

            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                               { phiMDataPtr(i, j, k, comp) = phiFftPtr(i, j, k); });
        }
    }
    phi.m_data.OverrideSync(ownermask, rho.m_deRham->get_periodicity());
}
