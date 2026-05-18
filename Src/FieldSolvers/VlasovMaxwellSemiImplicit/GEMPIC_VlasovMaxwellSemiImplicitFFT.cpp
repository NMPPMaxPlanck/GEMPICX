/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <AMReX_FFT.H>
#include <AMReX_GpuComplex.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_MatrixOperations.H"
#include "GEMPIC_VlasovMaxwellSemiImplicitFFT.H"

using namespace Gempic::FieldSolvers;
using namespace Gempic::Forms;

VlasovMaxwellSemiImplicitFFTSolver::VlasovMaxwellSemiImplicitFFTSolver(
    ComputationalDomain const& compDom,
    std::shared_ptr<FDDeRhamComplex> deRham,
    int const hodgeDegree) :
    m_compDom{compDom}, m_deRham{deRham}, m_hodgeDegree{hodgeDegree}
{
    BL_PROFILE(
        "Gempic::FieldSolvers::VlasovMaxwellSemiImplicitFFTSolver::"
        "VlasovMaxwellSemiImplicitFFTSolver()");

    // Check for periodic boundaries, required for FFT solver
    if (!compDom.geometry().periodicity().isAllPeriodic())
    {
        GEMPIC_ERROR("VlasovMaxwellSemiImplicit FFT solver requires periodic boundary conditions");
    }

    m_r2c = std::make_unique<amrex::FFT::R2C<amrex::Real>>(compDom.box());
    setup_eigenvalues();
}

void VlasovMaxwellSemiImplicitFFTSolver::setup_eigenvalues ()
{
    BL_PROFILE("Gempic::FieldSolvers::VlasovMaxwellSemiImplicitFFTSolver::setup_eigenvalues()");

    setup_derivative_eigenvalues();
    setup_mass_matrix_eigenvalues();
}

void VlasovMaxwellSemiImplicitFFTSolver::setup_derivative_eigenvalues ()
{
    GEMPIC_D_EXCL(int Ny = 0;, int Nz = 0;, ) // nvcc issues a warning when using const
    AMREX_D_TERM(int const Nx = m_compDom.box().length(xDir);
                 , int const Ny = m_compDom.box().length(yDir);
                 , int const Nz = m_compDom.box().length(zDir);)

    // Initialize eigenvalue vectors
    auto eigenDx = amrex::Vector<amrex::GpuComplex<amrex::Real>>(Nx, 1.0);
    auto eigenDy = amrex::Vector<amrex::GpuComplex<amrex::Real>>(Ny, 1.0);
    auto eigenDz = amrex::Vector<amrex::GpuComplex<amrex::Real>>(Nz, 1.0);
    auto eigenDTx = amrex::Vector<amrex::GpuComplex<amrex::Real>>(Nx, 1.0);
    auto eigenDTy = amrex::Vector<amrex::GpuComplex<amrex::Real>>(Ny, 1.0);
    auto eigenDTz = amrex::Vector<amrex::GpuComplex<amrex::Real>>(Nz, 1.0);

    std::complex<amrex::Real> I(0.0, 1.0);
    std::complex<amrex::Real> val(0.0, 0.0);

    // Compute eigenvalues for derivative matrices
    for (int iter = 0; iter < Nx; iter++)
    {
        val = 1.0 - std::exp(I * (2.0 * M_PI * iter / Nx));
        eigenDx[iter] = amrex::GpuComplex<amrex::Real>(val.real(), val.imag());
        eigenDTx[iter] = amrex::GpuComplex<amrex::Real>(val.real(), -val.imag());
    }

    for (int iter = 0; iter < Ny; iter++)
    {
        val = 1.0 - std::exp(I * (2.0 * M_PI * iter / Ny));
        eigenDy[iter] = amrex::GpuComplex<amrex::Real>(val.real(), val.imag());
        eigenDTy[iter] = amrex::GpuComplex<amrex::Real>(val.real(), -val.imag());
    }

    for (int iter = 0; iter < Nz; iter++)
    {
        val = 1.0 - std::exp(I * (2.0 * M_PI * iter / Nz));
        eigenDz[iter] = amrex::GpuComplex<amrex::Real>(val.real(), val.imag());
        eigenDTz[iter] = amrex::GpuComplex<amrex::Real>(val.real(), -val.imag());
    }

    // Handle the gpu parallel for gpu running
    m_eigenDxGpu.resize(Nx);
    m_eigenDyGpu.resize(Ny);
    m_eigenDzGpu.resize(Nz);
    m_eigenDTxGpu.resize(Nx);
    m_eigenDTyGpu.resize(Ny);
    m_eigenDTzGpu.resize(Nz);

    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, eigenDx.begin(), eigenDx.end(),
                          m_eigenDxGpu.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, eigenDy.begin(), eigenDy.end(),
                          m_eigenDyGpu.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, eigenDz.begin(), eigenDz.end(),
                          m_eigenDzGpu.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, eigenDTx.begin(), eigenDTx.end(),
                          m_eigenDTxGpu.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, eigenDTy.begin(), eigenDTy.end(),
                          m_eigenDTyGpu.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, eigenDTz.begin(), eigenDTz.end(),
                          m_eigenDTzGpu.begin());

    m_eigenDx = m_eigenDxGpu.dataPtr();
    m_eigenDy = m_eigenDyGpu.dataPtr();
    m_eigenDz = m_eigenDzGpu.dataPtr();
    m_eigenDTx = m_eigenDTxGpu.dataPtr();
    m_eigenDTy = m_eigenDTyGpu.dataPtr();
    m_eigenDTz = m_eigenDTzGpu.dataPtr();
}

void VlasovMaxwellSemiImplicitFFTSolver::setup_mass_matrix_eigenvalues ()
{
    GEMPIC_D_EXCL(int Ny = 0;, int Nz = 0;, )
    AMREX_D_TERM(int const Nx = m_compDom.box().length(xDir);
                 , int const Ny = m_compDom.box().length(yDir);
                 , int const Nz = m_compDom.box().length(zDir);)

    // Initialize mass matrix eigenvalue vectors
    auto eigenM1x = amrex::Vector<amrex::Real>(Nx, 1.0);
    auto eigenM1y = amrex::Vector<amrex::Real>(Ny, 1.0);
    auto eigenM1z = amrex::Vector<amrex::Real>(Nz, 1.0);
    auto eigenM2x = amrex::Vector<amrex::Real>(Nx, 1.0);
    auto eigenM2y = amrex::Vector<amrex::Real>(Ny, 1.0);
    auto eigenM2z = amrex::Vector<amrex::Real>(Nz, 1.0);

    auto dx = m_compDom.cell_size_array();

    // Compute eigenvalues for mass matrices depending on the polynomial degree
    switch (m_hodgeDegree)
    {
        case 2:
            for (int iter = 0; iter < Nx; iter++)
            {
                eigenM1x[iter] = (2.0 + std::cos(2.0 * M_PI * iter / Nx)) * (dx[xDir]) / 3;
                eigenM2x[iter] = 1.0 / (dx[xDir]);
            }
            for (int iter = 0; iter < Ny; iter++)
            {
                eigenM1y[iter] = (2.0 + std::cos(2.0 * M_PI * iter / Ny)) * (dx[yDir]) / 3;
                eigenM2y[iter] = 1.0 / (dx[yDir]);
            }
            for (int iter = 0; iter < Nz; iter++)
            {
                eigenM1z[iter] = (2.0 + std::cos(2.0 * M_PI * iter / Nz)) * (dx[zDir]) / 3;
                eigenM2z[iter] = 1.0 / (dx[zDir]);
            }
            break;

        case 4:
            for (int iter = 0; iter < Nx; iter++)
            {
                eigenM1x[iter] = (733.0 / 945.0 + 257.0 * std::cos(2.0 * M_PI * iter / Nx) / 840.0 -
                                  3.0 * std::cos(4.0 * M_PI * iter / Nx) / 35.0 +
                                  31.0 * std::cos(6.0 * M_PI * iter / Nx) / 7560.0) *
                                 (dx[xDir]);

                eigenM2x[iter] = (378.0 - 2.0 * 2.0 * std::cos(2.0 * M_PI * iter / Nx) -
                                  2.0 * 7.0 * std::cos(4.0 * M_PI * iter / Nx)) /
                                 (360.0 * dx[xDir]);
            }
            for (int iter = 0; iter < Ny; iter++)
            {
                eigenM1y[iter] = (733.0 / 945.0 + 257.0 * std::cos(2.0 * M_PI * iter / Ny) / 840.0 -
                                  3.0 * std::cos(4.0 * M_PI * iter / Ny) / 35.0 +
                                  31.0 * std::cos(6.0 * M_PI * iter / Ny) / 7560.0) *
                                 (dx[yDir]);

                eigenM2y[iter] = (378.0 - 2.0 * 2.0 * std::cos(2.0 * M_PI * iter / Ny) -
                                  2.0 * 7.0 * std::cos(4.0 * M_PI * iter / Ny)) /
                                 (360.0 * dx[yDir]);
            }
            for (int iter = 0; iter < Nz; iter++)
            {
                eigenM1z[iter] = (733.0 / 945.0 + 257.0 * std::cos(2.0 * M_PI * iter / Nz) / 840.0 -
                                  3.0 * std::cos(4.0 * M_PI * iter / Nz) / 35.0 +
                                  31.0 * std::cos(6.0 * M_PI * iter / Nz) / 7560.0) *
                                 (dx[zDir]);

                eigenM2z[iter] = (378.0 - 2.0 * 2.0 * std::cos(2.0 * M_PI * iter / Nz) -
                                  2.0 * 7.0 * std::cos(4.0 * M_PI * iter / Nz)) /
                                 (360.0 * dx[zDir]);
            }
            break;

        case 6:
            for (int iter = 0; iter < Nx; iter++)
            {
                eigenM1x[iter] =
                    (455963.0 / 554400.0 + 906919.0 * std::cos(2 * M_PI * iter / Nx) / 3326400.0 -
                     12421.0 * std::cos(4.0 * M_PI * iter / Nx) / 103950.0 +
                     59053.0 * std::cos(6.0 * M_PI * iter / Nx) / 2217600.0 -
                     3937.0 * std::cos(8.0 * M_PI * iter / Nx) / 1663200.0 +
                     313.0 * std::cos(10.0 * M_PI * iter / Nx) / 2217600.0) *
                    (dx[xDir]);

                eigenM2x[iter] =
                    (99307.0 / 90720.0 - 9691.0 * std::cos(2.0 * M_PI * iter / Nx) / 226800.0 -
                     4003.0 * std::cos(4.0 * M_PI * iter / Nx) / 56700.0 +
                     4547.0 * std::cos(6.0 * M_PI * iter / Nx) / 226800.0 -
                     89.0 * std::cos(8.0 * M_PI * iter / Nx) / 64800.0) /
                    (dx[xDir]);
            }
            for (int iter = 0; iter < Ny; iter++)
            {
                eigenM1y[iter] =
                    (455963.0 / 554400.0 + 906919.0 * std::cos(2 * M_PI * iter / Ny) / 3326400.0 -
                     12421.0 * std::cos(4.0 * M_PI * iter / Ny) / 103950.0 +
                     59053.0 * std::cos(6.0 * M_PI * iter / Ny) / 2217600.0 -
                     3937.0 * std::cos(8.0 * M_PI * iter / Ny) / 1663200.0 +
                     313.0 * std::cos(10.0 * M_PI * iter / Ny) / 2217600.0) *
                    (dx[yDir]);

                eigenM2y[iter] =
                    (99307.0 / 90720.0 - 9691.0 * std::cos(2.0 * M_PI * iter / Ny) / 226800.0 -
                     4003.0 * std::cos(4.0 * M_PI * iter / Ny) / 56700.0 +
                     4547.0 * std::cos(6.0 * M_PI * iter / Ny) / 226800.0 -
                     89.0 * std::cos(8.0 * M_PI * iter / Ny) / 64800.0) /
                    (dx[yDir]);
            }
            for (int iter = 0; iter < Nz; iter++)
            {
                eigenM1z[iter] =
                    (455963.0 / 554400.0 + 906919.0 * std::cos(2 * M_PI * iter / Nz) / 3326400.0 -
                     12421.0 * std::cos(4.0 * M_PI * iter / Nz) / 103950.0 +
                     59053.0 * std::cos(6.0 * M_PI * iter / Nz) / 2217600.0 -
                     3937.0 * std::cos(8.0 * M_PI * iter / Nz) / 1663200.0 +
                     313.0 * std::cos(10.0 * M_PI * iter / Nz) / 2217600.0) *
                    (dx[zDir]);

                eigenM2z[iter] =
                    (99307.0 / 90720.0 - 9691.0 * std::cos(2.0 * M_PI * iter / Nz) / 226800.0 -
                     4003.0 * std::cos(4.0 * M_PI * iter / Nz) / 56700.0 +
                     4547.0 * std::cos(6.0 * M_PI * iter / Nz) / 226800.0 -
                     89.0 * std::cos(8.0 * M_PI * iter / Nz) / 64800.0) /
                    (dx[zDir]);
            }
            break;

        default:
            AMREX_ASSERT(
                "Degree not implemented for three dimensional Hodge in VlasovMaxwellSemiImplicit "
                "FFT");
            break;
    }

    // Handle the gpu parallel for for gpu running
    m_eigenM1xGpu.resize(Nx);
    m_eigenM1yGpu.resize(Ny);
    m_eigenM1zGpu.resize(Nz);

    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, eigenM1x.begin(), eigenM1x.end(),
                          m_eigenM1xGpu.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, eigenM1y.begin(), eigenM1y.end(),
                          m_eigenM1yGpu.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, eigenM1z.begin(), eigenM1z.end(),
                          m_eigenM1zGpu.begin());

    m_eigenM1x = m_eigenM1xGpu.dataPtr();
    m_eigenM1y = m_eigenM1yGpu.dataPtr();
    m_eigenM1z = m_eigenM1zGpu.dataPtr();

    m_eigenM2xGpu.resize(Nx);
    m_eigenM2yGpu.resize(Ny);
    m_eigenM2zGpu.resize(Nz);

    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, eigenM2x.begin(), eigenM2x.end(),
                          m_eigenM2xGpu.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, eigenM2y.begin(), eigenM2y.end(),
                          m_eigenM2yGpu.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, eigenM2z.begin(), eigenM2z.end(),
                          m_eigenM2zGpu.begin());

    m_eigenM2x = m_eigenM2xGpu.dataPtr();
    m_eigenM2y = m_eigenM2yGpu.dataPtr();
    m_eigenM2z = m_eigenM2zGpu.dataPtr();
}

void VlasovMaxwellSemiImplicitFFTSolver::solve_implicit_step (
    DeRhamField<Grid::primal, Space::edge>& E,
    DeRhamField<Grid::primal, Space::face>& B,
    amrex::Real dt)
{
    BL_PROFILE("Gempic::FieldSolvers::VlasovMaxwellSemiImplicitFFTSolver::solve_implicit_step()");

    // Create temporary fields for the solve
    DeRhamField<Grid::primal, Space::edge> eOld(m_deRham);
    DeRhamField<Grid::dual, Space::face> rhs(m_deRham);
    DeRhamField<Grid::dual, Space::cell> e1(m_deRham);
    DeRhamField<Grid::dual, Space::cell> e2(m_deRham);
    DeRhamField<Grid::dual, Space::cell> e3(m_deRham);

    // Grid dimensions
    GEMPIC_D_EXCL(int Ny = 0;, int Nz = 0;, )
    AMREX_D_TERM(int const Nx = m_compDom.box().length(xDir);
                 , int const Ny = m_compDom.box().length(yDir);
                 , int const Nz = m_compDom.box().length(zDir);)

    // Save old E field
    copy(eOld, E);

    // Setup RHS for the linear system
    setup_rhs(rhs, E, B, dt);

    // Setup FFT containers
    amrex::Box dom0 = m_compDom.box();
    amrex::BoxArray ba0 = m_compDom.m_grid;

    amrex::FFT::R2C myFft(dom0);

    amrex::FFT::R2C<amrex::Real>::cMF e1Fft(ba0, m_compDom.m_distriMap, e1.m_data.nComp(), 0);
    amrex::FFT::R2C<amrex::Real>::MF e1FftContainer(ba0, m_compDom.m_distriMap, e1.m_data.nComp(),
                                                    1);

    amrex::FFT::R2C<amrex::Real>::cMF e2Fft(ba0, m_compDom.m_distriMap, e2.m_data.nComp(), 0);
    amrex::FFT::R2C<amrex::Real>::MF e2FftContainer(ba0, m_compDom.m_distriMap, e2.m_data.nComp(),
                                                    1);

    amrex::FFT::R2C<amrex::Real>::cMF e3Fft(ba0, m_compDom.m_distriMap, e3.m_data.nComp(), 0);
    amrex::FFT::R2C<amrex::Real>::MF e3FftContainer(ba0, m_compDom.m_distriMap, e3.m_data.nComp(),
                                                    1);

    // Copy RHS to FFT containers
    for (amrex::MFIter mfi(e1FftContainer); mfi.isValid(); ++mfi)
    {
        auto const& e1FftContainerPtr = e1FftContainer.array(mfi);
        auto const& f1MDataPtr = rhs.m_data[0].array(mfi);
        auto const& e2FftContainerPtr = e2FftContainer.array(mfi);
        auto const& f2MDataPtr = rhs.m_data[1].array(mfi);
        auto const& e3FftContainerPtr = e3FftContainer.array(mfi);
        auto const& f3MDataPtr = rhs.m_data[2].array(mfi);

        amrex::Box const& bx = mfi.validbox();

        amrex::ParallelFor(bx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                           {
                               e1FftContainerPtr(i, j, k) = f1MDataPtr(i, j, k);
                               e2FftContainerPtr(i, j, k) = f2MDataPtr(i, j, k);
                               e3FftContainerPtr(i, j, k) = f3MDataPtr(i, j, k);
                           });
    }

    // Fill boundaries and apply forward FFT
    e1FftContainer.FillBoundary(rhs.m_deRham->get_periodicity());
    e2FftContainer.FillBoundary(rhs.m_deRham->get_periodicity());
    e3FftContainer.FillBoundary(rhs.m_deRham->get_periodicity());

    myFft.forward(e1FftContainer, e1Fft);
    myFft.forward(e2FftContainer, e2Fft);
    myFft.forward(e3FftContainer, e3Fft);

    // Solve in frequency domain
    for (amrex::MFIter mfi(e1Fft); mfi.isValid(); ++mfi)
    {
        auto const& e1FftPtr = e1Fft.array(mfi);
        auto const& e2FftPtr = e2Fft.array(mfi);
        auto const& e3FftPtr = e3Fft.array(mfi);

        amrex::Box const& bx = mfi.validbox();

        // local variables necessary for CUDA
        amrex::Real* eigenM1x = m_eigenM1x;
        amrex::Real* eigenM1y = m_eigenM1y;
        amrex::Real* eigenM1z = m_eigenM1z;
        amrex::Real* eigenM2x = m_eigenM2x;
        amrex::Real* eigenM2y = m_eigenM2y;
        amrex::Real* eigenM2z = m_eigenM2z;
        amrex::GpuComplex<amrex::Real>* eigenDx = m_eigenDx;
        amrex::GpuComplex<amrex::Real>* eigenDy = m_eigenDy;
        amrex::GpuComplex<amrex::Real>* eigenDz = m_eigenDz;
        amrex::GpuComplex<amrex::Real>* eigenDTx = m_eigenDTx;
        amrex::GpuComplex<amrex::Real>* eigenDTy = m_eigenDTy;
        amrex::GpuComplex<amrex::Real>* eigenDTz = m_eigenDTz;

        amrex::ParallelFor(
            bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                // Set up 3x3 system matrix
                auto m11 = eigenM2x[i] * eigenM1y[j] * eigenM1z[k];
                auto m12 = eigenM1x[i] * eigenM2y[j] * eigenM1z[k];
                auto m13 = eigenM1x[i] * eigenM1y[j] * eigenM2z[k];
                auto m21 = eigenM1x[i] * eigenM2y[j] * eigenM2z[k];
                auto m22 = eigenM2x[i] * eigenM1y[j] * eigenM2z[k];
                auto m23 = eigenM2x[i] * eigenM2y[j] * eigenM1z[k];

                auto a11 =
                    m11 + 0.25 * (dt * dt) *
                              (eigenDTz[k] * m22 * eigenDz[k] + eigenDTy[j] * m23 * eigenDy[j]);

                auto a22 =
                    m12 + 0.25 * (dt * dt) *
                              (eigenDTz[k] * m21 * eigenDz[k] + eigenDTx[i] * m23 * eigenDx[i]);

                auto a33 =
                    m13 + 0.25 * (dt * dt) *
                              (eigenDTy[j] * m21 * eigenDy[j] + eigenDTx[i] * m22 * eigenDx[i]);

                auto a12 = -0.25 * (dt * dt) * (eigenDTy[j] * m23 * eigenDx[i]);
                auto a21 = -0.25 * (dt * dt) * (eigenDTx[i] * m23 * eigenDy[j]);
                auto a13 = -0.25 * (dt * dt) * (eigenDTz[k] * m22 * eigenDx[i]);
                auto a31 = -0.25 * (dt * dt) * (eigenDTx[i] * m22 * eigenDz[k]);
                auto a23 = -0.25 * (dt * dt) * (eigenDTz[k] * m21 * eigenDy[j]);
                auto a32 = -0.25 * (dt * dt) * (eigenDTy[j] * m21 * eigenDz[k]);

                auto sp = sol3x3(a11, a12, a13, a21, a22, a23, a31, a32, a33, e1FftPtr(i, j, k),
                                 e2FftPtr(i, j, k), e3FftPtr(i, j, k));

                e1FftPtr(i, j, k) = sp[0] / static_cast<amrex::Real>(Nx * Ny * Nz);
                e2FftPtr(i, j, k) = sp[1] / static_cast<amrex::Real>(Nx * Ny * Nz);
                e3FftPtr(i, j, k) = sp[2] / static_cast<amrex::Real>(Nx * Ny * Nz);
            });
    }

    // Apply inverse FFT
    myFft.backward(e1Fft, e1FftContainer);
    e1FftContainer.FillBoundary(rhs.m_deRham->get_periodicity());

    myFft.backward(e2Fft, e2FftContainer);
    e2FftContainer.FillBoundary(rhs.m_deRham->get_periodicity());

    myFft.backward(e3Fft, e3FftContainer);
    e3FftContainer.FillBoundary(rhs.m_deRham->get_periodicity());

    // Copy solution back to E field
    for (amrex::MFIter mfi(e1FftContainer); mfi.isValid(); ++mfi)
    {
        auto const& e1FftContainerPtr = e1FftContainer.array(mfi);
        auto const& e2FftContainerPtr = e2FftContainer.array(mfi);
        auto const& e3FftContainerPtr = e3FftContainer.array(mfi);

        auto const& e1MDataPtr = E.m_data[0].array(mfi);
        auto const& e2MDataPtr = E.m_data[1].array(mfi);
        auto const& e3MDataPtr = E.m_data[2].array(mfi);

        amrex::Box const& bx = mfi.fabbox();

        amrex::ParallelFor(bx,
                           [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                           {
                               e1MDataPtr(i, j, k) = e1FftContainerPtr(i, j, k);
                               e2MDataPtr(i, j, k) = e2FftContainerPtr(i, j, k);
                               e3MDataPtr(i, j, k) = e3FftContainerPtr(i, j, k);
                           });
    }

    // Fill boundaries
    E.fill_boundary();
    eOld.fill_boundary();
    B.fill_boundary();

    // Update magnetic field using Crank-Nicolson scheme
    update_magnetic_field(eOld, E, B, dt);
}

void VlasovMaxwellSemiImplicitFFTSolver::setup_rhs (DeRhamField<Grid::dual, Space::face>& rhs,
                                                   DeRhamField<Grid::primal, Space::edge>& E,
                                                   DeRhamField<Grid::primal, Space::face>& B,
                                                   amrex::Real dt)
{
    BL_PROFILE("Gempic::FieldSolvers::VlasovMaxwellSemiImplicitFFTSolver::setup_rhs()");

    // This implements the RHS calculation from rhs_op1 in VlasovMaxwellSemiImplicit.cpp
    DeRhamField<Grid::dual, Space::edge> bstar(m_deRham);
    DeRhamField<Grid::primal, Space::face> v2fieldTmp(m_deRham);

    // rhs = M1 * E + dt * curl^T * M2 * B - dt^2/4 * curl^T * M2 * curl * E
    hodge(bstar, B);
    hodge(rhs, E);
    add_dt_curl(rhs, bstar, dt);

    curl(v2fieldTmp, E);
    hodge(bstar, v2fieldTmp);
    add_dt_curl(rhs, bstar, -dt * dt / 4);
}

void VlasovMaxwellSemiImplicitFFTSolver::update_magnetic_field (
    DeRhamField<Grid::primal, Space::edge>& eOld,
    DeRhamField<Grid::primal, Space::edge>& eNew,
    DeRhamField<Grid::primal, Space::face>& B,
    amrex::Real dt)
{
    BL_PROFILE("Gempic::FieldSolvers::VlasovMaxwellSemiImplicitFFTSolver::update_magnetic_field()");

    // Crank-Nicolson update: B^{n+1} = B^n - dt/2 * curl(E^n) - dt/2 * curl(E^{n+1})
    add_dt_curl(B, eOld, -0.5 * dt);
    add_dt_curl(B, eNew, -0.5 * dt);
}
