#include <AMReX_FFT.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_FFT.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_PoissonSolver.H"

using namespace Gempic::FieldSolvers;
using namespace Gempic::Forms;

FFTSolver::FFTSolver(const ComputationalDomain& compDom,
                     const int hodgeDegree,
                     HodgeScheme hodgeScheme) :
    m_compDom{compDom}
{
    BL_PROFILE("Gempic::FieldSolvers::FFTSolver::FFTSolver()");
    if (hodgeScheme != HodgeScheme::FDHodge)
    {
        amrex::Abort("The FFT Poison solver only works with Finite Difference Hodge");
    }

    // Check for periodic boundaries, required for FFT solver
    if (!compDom.geometry().periodicity().isAllPeriodic())
    {
        amrex::Abort("FFT Poisson solver requires periodic boundary conditions");
    }

    m_r2c = std::make_unique<amrex::FFT::R2C<amrex::Real>>(compDom.box());

    //dimensions in x, y and z directions
    GEMPIC_D_EXCL(const int Ny = 0;, const int Nz = 0;, )
    AMREX_D_TERM(const int Nx = compDom.box().length(xDir);
                 , const int Ny = compDom.box().length(yDir);
                 , const int Nz = compDom.box().length(zDir);)

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
            AMREX_ASSERT("Degree not implemented for three dimensional Hodge in FFT");
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
    AMREX_D_TERM(m_eigenvalues0xGpu.resize(Nx);, m_eigenvalues0yGpu.resize(Ny);
                 , m_eigenvalues0zGpu.resize(Nz);)

    AMREX_D_TERM(amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, eigenvalues0x.begin(),
                                       eigenvalues0x.end(), m_eigenvalues0xGpu.begin());
                 , amrex::Gpu::copyAsync (amrex::Gpu::hostToDevice, eigenvalues0y.begin(),
                                         eigenvalues0y.end(), m_eigenvalues0yGpu.begin());
                 , amrex::Gpu::copyAsync (amrex::Gpu::hostToDevice, eigenvalues0z.begin(),
                                         eigenvalues0z.end(), m_eigenvalues0zGpu.begin());)

    AMREX_D_TERM(m_eigenvalues0x = m_eigenvalues0xGpu.dataPtr();
                 , m_eigenvalues0y = m_eigenvalues0yGpu.dataPtr();
                 , m_eigenvalues0z = m_eigenvalues0zGpu.dataPtr();)
}

void FFTSolver::solve (DeRhamField<Grid::primal, Space::node>& phi,
                      DeRhamField<Grid::dual, Space::cell>& rho)
{
    BL_PROFILE("Gempic::FieldSolvers::FFTSolver::solve()");

    subtract_constant_part(rho, m_compDom);

    // define cell-centered containers for storing data, declaring 1 ghost cell to be able to
    // copy back to the node-centered MultiFab, phi, without data loss.
    amrex::MultiFab rhoFft(m_compDom.m_grid, m_compDom.m_distriMap, rho.m_data.nComp(), 1);
    amrex::MultiFab phiFft(m_compDom.m_grid, m_compDom.m_distriMap, phi.m_data.nComp(), 1);

    // assign values to containers from rho.m_data
    for (amrex::MFIter mfi(rhoFft); mfi.isValid(); ++mfi)
    {
        auto const& rhoFftPtr = rhoFft.array(mfi);

        auto const& rhoMDataPtr = rho.m_data.array(mfi);

        const amrex::Box& bx = mfi.fabbox();

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                           { rhoFftPtr(i, j, k) = rhoMDataPtr(i, j, k); });
    }

    // local variables necessary for CUDA
    AMREX_D_TERM(amrex::Real* eigenvalues0x = m_eigenvalues0x;
                 , amrex::Real* eigenvalues0y = m_eigenvalues0y;
                 , amrex::Real* eigenvalues0z = m_eigenvalues0z;)
    int cellNum = m_compDom.box().length3d().product();
    // Amrex FFT only fourier transforms the first component, so we need to do hacks
    for (int comp{0}; comp < rho.m_data.nComp(); ++comp)
    {
        amrex::MultiFab rhoFftComp(rhoFft, amrex::make_alias, comp, 1);
        amrex::MultiFab phiFftComp(phiFft, amrex::make_alias, comp, 1);
        m_r2c->forwardThenBackward(rhoFftComp, phiFftComp,
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
    }

    //boundary handling
    phiFft.FillBoundary(rho.m_deRham->get_periodicity());

    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        auto const& phiFftPtr = phiFft.array(mfi);

        auto const& phiMDataPtr = phi.m_data.array(mfi);

        const amrex::Box& bx = mfi.validbox();

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
                           { phiMDataPtr(i, j, k) = phiFftPtr(i, j, k); });
    }

    phi.fill_boundary();
}
