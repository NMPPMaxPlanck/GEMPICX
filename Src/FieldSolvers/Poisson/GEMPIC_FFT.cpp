/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <AMReX_FFT.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_FFT.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_PoissonSolver.H"

using namespace Gempic::FieldSolvers;
using namespace Gempic::Forms;

FFTSolver::FFTSolver(ComputationalDomain const& compDom,
                     int const hodgeDegree,
                     HodgeScheme hodgeScheme) :
    m_compDom{compDom}
{
    BL_PROFILE("Gempic::FieldSolvers::FFTSolver::FFTSolver()");
    if (hodgeScheme != HodgeScheme::FDHodge)
    {
        GEMPIC_ERROR("The FFT Poison solver only works with Finite Difference Hodge");
    }

    // Check for periodic boundaries, required for FFT solver
    if (!compDom.geometry().periodicity().isAllPeriodic())
    {
        GEMPIC_ERROR("FFT Poisson solver requires periodic boundary conditions");
    }

    m_r2c = std::make_unique<amrex::FFT::R2C<amrex::Real>>(compDom.box());
    // define cell-centered containers for storing data, declaring 1 ghost cell to be able to
    // copy back to the node-centered MultiFab, phi, without data loss.
    m_rhoFft.define(compDom.m_grid, compDom.m_distriMap, 1, 1);
    m_phiFft.define(compDom.m_grid, compDom.m_distriMap, 1, 1);

    //dimensions in x, y and z directions
    GEMPIC_D_EXCL(int Ny = 0;, int Nz = 0;, ) // nvcc issues a warning when using const
    AMREX_D_TERM(int const Nx = compDom.box().length(xDir);
                 , int const Ny = compDom.box().length(yDir);
                 , int const Nz = compDom.box().length(zDir);)

    // sizes in x, y and z directions
    auto dx = compDom.cell_size_array();

    auto eigenH0x = amrex::Vector<amrex::Real>(Nx, 1.0);
    auto eigenH0y = amrex::Vector<amrex::Real>(Ny, 1.0);
    auto eigenH0z = amrex::Vector<amrex::Real>(Nz, 1.0);

    switch (hodgeDegree)
    {
        case 2:
            for (int iter = 0; iter < Nx; iter++)
            {
                eigenH0x[iter] = dx.product() / (dx[xDir] * dx[xDir]);
            }
            for (int iter = 0; iter < Ny; iter++)
            {
                eigenH0y[iter] = dx.product() / (dx[yDir] * dx[yDir]);
            }
            for (int iter = 0; iter < Nz; iter++)
            {
                eigenH0z[iter] = dx.product() / (dx[zDir] * dx[zDir]);
            }
            break;
        case 4:
            for (int iter = 0; iter < Nx; iter++)
            {
                eigenH0x[iter] =
                    (13.0 - cos(2 * M_PI * iter / Nx)) * dx.product() / (12 * dx[xDir] * dx[xDir]);
            }
            for (int iter = 0; iter < Ny; iter++)
            {
                eigenH0y[iter] =
                    (13.0 - cos(2 * M_PI * iter / Ny)) * dx.product() / (12 * dx[yDir] * dx[yDir]);
            }
            for (int iter = 0; iter < Nz; iter++)
            {
                eigenH0z[iter] =
                    (13.0 - cos(2 * M_PI * iter / Nz)) * dx.product() / (12 * dx[zDir] * dx[zDir]);
            }
            break;
        case 6:
            for (int iter = 0; iter < Nx; iter++)
            {
                eigenH0x[iter] =
                    (1067.0 - 116.0 * cos(2 * M_PI * iter / Nx) + 9.0 * cos(4 * M_PI * iter / Nx)) *
                    dx.product() / (960 * dx[xDir] * dx[xDir]);
            }
            for (int iter = 0; iter < Ny; iter++)
            {
                eigenH0y[iter] =
                    (1067.0 - 116.0 * cos(2 * M_PI * iter / Ny) + 9.0 * cos(4 * M_PI * iter / Ny)) *
                    dx.product() / (960 * dx[yDir] * dx[yDir]);
            }
            for (int iter = 0; iter < Nz; iter++)
            {
                eigenH0z[iter] =
                    (1067.0 - 116.0 * cos(2 * M_PI * iter / Nz) + 9.0 * cos(4 * M_PI * iter / Nz)) *
                    dx.product() / (960 * dx[zDir] * dx[zDir]);
            }
            break;
        default:
            GEMPIC_ERROR("Degree not implemented for three dimensional Hodge in FFT");
            break;
    }

    // calculate eigenvalues of the lhs matrix
    auto eigenvalues0x = amrex::Vector<amrex::Real>(Nx, 1.0);
    auto eigenvalues0y = amrex::Vector<amrex::Real>(Ny, 1.0);
    auto eigenvalues0z = amrex::Vector<amrex::Real>(Nz, 1.0);

    for (int iter = 0; iter < Nx; iter++)
    {
        eigenvalues0x[iter] = (2 - 2 * cos(2 * M_PI * iter / Nx)) * eigenH0x[iter];
    }
    for (int iter = 0; iter < Ny; iter++)
    {
        eigenvalues0y[iter] = (2 - 2 * cos(2 * M_PI * iter / Ny)) * eigenH0y[iter];
    }
    for (int iter = 0; iter < Nz; iter++)
    {
        eigenvalues0z[iter] = (2 - 2 * cos(2 * M_PI * iter / Nz)) * eigenH0z[iter];
    }

    // Handle the gpu parallel for for gpu running
    m_eigenvalues0xGpu.resize(Nx);
    m_eigenvalues0yGpu.resize(Ny);
    m_eigenvalues0zGpu.resize(Nz);

    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, eigenvalues0x.begin(), eigenvalues0x.end(),
                          m_eigenvalues0xGpu.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, eigenvalues0y.begin(), eigenvalues0y.end(),
                          m_eigenvalues0yGpu.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, eigenvalues0z.begin(), eigenvalues0z.end(),
                          m_eigenvalues0zGpu.begin());

    m_eigenvalues0x = m_eigenvalues0xGpu.dataPtr();
    m_eigenvalues0y = m_eigenvalues0yGpu.dataPtr();
    m_eigenvalues0z = m_eigenvalues0zGpu.dataPtr();
}

void FFTSolver::solve (DeRhamField<Grid::primal, Space::node>& phi,
                      DeRhamField<Grid::dual, Space::cell> const& rho)
{
    BL_PROFILE("Gempic::FieldSolvers::FFTSolver::solve()");

    AMREX_ALWAYS_ASSERT(phi.m_data.nComp() == rho.m_data.nComp());
    check_charge_neutrality(rho, m_compDom);

    for (int comp{0}; comp < rho.m_data.nComp(); ++comp)
    {
        // assign values to containers from rho.m_data
        for (amrex::MFIter mfi(m_rhoFft); mfi.isValid(); ++mfi)
        {
            auto const& rhoFftPtr = m_rhoFft.array(mfi);

            auto const& rhoMDataPtr = rho.m_data.array(mfi);

            amrex::Box const& bx = mfi.fabbox();

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                               { rhoFftPtr(i, j, k) = rhoMDataPtr(i, j, k, comp); });
        }

        // local variables necessary for CUDA
        AMREX_D_TERM(amrex::Real* eigenvalues0x = m_eigenvalues0x;
                     , amrex::Real* eigenvalues0y = m_eigenvalues0y;
                     , amrex::Real* eigenvalues0z = m_eigenvalues0z;)
        int cellNum = m_compDom.box().length3d().product();

        m_r2c->forwardThenBackward(m_rhoFft, m_phiFft,
                                   [=] AMREX_GPU_DEVICE(int i, int j, int k, auto& sp)
                                   {
                                       auto eigenvalueSum{GEMPIC_D_ADD(
                                           eigenvalues0x[i], eigenvalues0y[j], eigenvalues0z[k])};
                                       if (eigenvalueSum == 0)
                                       {
                                           sp = 0;
                                       }
                                       else
                                       {
                                           sp /= eigenvalueSum * cellNum;
                                       }
                                   });

        //boundary handling
        m_phiFft.FillBoundary(rho.m_deRham->get_periodicity());

        for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
        {
            auto const& phiFftPtr = m_phiFft.array(mfi);

            auto const& phiMDataPtr = phi.m_data.array(mfi);

            amrex::Box const& bx = mfi.validbox();

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                               { phiMDataPtr(i, j, k, comp) = phiFftPtr(i, j, k); });
        }
    }
}
