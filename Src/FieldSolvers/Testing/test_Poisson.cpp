#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_PoissonSolver.H"

using namespace Gempic::Forms;

/**
 * @brief Tests the Poisson solver for an analytical rho of 1.0 + cos(x)
 *
 * @todo: Use our Hodge and compare to exact analytical functon
 */
int main (int argc, char *argv[])
{
    amrex::Initialize(argc, argv);
    {
        BL_PROFILE("main()");
        /* Initialize the infrastructure */
        // const amrex::RealBox realBox({AMREX_D_DECL(-M_PI, -M_PI, -M_PI)},{AMREX_D_DECL( M_PI,
        // M_PI, M_PI)});
        const amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(-M_PI, -M_PI, -M_PI)};
        const amrex::Vector<amrex::Real> k{AMREX_D_DECL(1.0, 1.0, 1.0)};
        const amrex::Vector<int> nCell{AMREX_D_DECL(64, 64, 64)};
        // const amrex::IntVect nCell{AMREX_D_DECL(128, 128, 128)};
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(64, 64, 64)};

        const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};
        const int degree = 2;
        const int maxSplineDegree = 1;
        const int nGhostExtra = 0;

        Gempic::Io::Parameters parameters{};

        parameters.set("domainLo", domainLo);
        parameters.set("k", k);
        parameters.set("nCellVector", nCell);
        parameters.set("maxGridSizeVector", maxGridSize);
        parameters.set("isPeriodicVector", isPeriodic);
        parameters.set("nGhostExtra", nGhostExtra);

        // Initialize computational_domain
        Gempic::ComputationalDomain infra;

        // Initialize the De Rham Complex
        auto deRham =
            std::make_shared<FDDeRhamComplex>(infra, degree, maxSplineDegree, HodgeScheme::FDHodge);

        auto const dr = infra.m_dx;

        // Declare both rho and phi
        DeRhamField<Grid::dual, Space::cell> rho(deRham);
        DeRhamField<Grid::primal, Space::node> phi(deRham);
        DeRhamField<Grid::primal, Space::node> anPhi(deRham);
        phi.m_data.setVal(0.0);

        // Analytical rho and phi such that -Delta phi = rho
#if GEMPIC_SPACEDIM == 1
        const std::string analyticalRho = "cos(x)";
        const std::string analyticalPhi = "cos(x)";
#elif GEMPIC_SPACEDIM == 2
        const std::string analyticalRho = "2*cos(x)*cos(y)";
        const std::string analyticalPhi = "cos(x)*cos(y)";
#elif GEMPIC_SPACEDIM == 3
        const std::string analyticalRho = "3*cos(x)*cos(y)*cos(z)";
        const std::string analyticalPhi = "cos(x)*cos(y)*cos(z)";
#endif

        const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
        amrex::Parser parserRho, parserPhi;
        amrex::ParserExecutor<nVar> funcRho, funcPhi;

        parserRho.define(analyticalRho);
        parserRho.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcRho = parserRho.compile<nVar>();
        parserPhi.define(analyticalPhi);
        parserPhi.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcPhi = parserPhi.compile<nVar>();

        // Compute the projection of the fields
        deRham->projection(funcRho, 0.0, rho);
        deRham->projection(funcPhi, 0.0, anPhi);

        // solve Poisson
        Gempic::FieldSolvers::PoissonSolver poisson(deRham);
        poisson.solve(infra, rho, phi);

        for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::Array4<amrex::Real> const &anPhiMF = (anPhi.m_data)[mfi].array();
            const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = infra.m_geom.ProbLoArray();

            ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r = {AMREX_D_DECL(
                        r0[xDir] + i * dr[xDir], r0[yDir] + j * dr[yDir], r0[zDir] + k * dr[zDir])};

                    // Right mathematically. Rho is a volume integral of analyticalRho in the dual
                    // grid. But it doesn't use our Hodge, nor it is the exact analytical function
                    // to which it should compare. Change later.
                    anPhiMF(i, j, k) =
                        (GEMPIC_D_MULT(dr[xDir], dr[yDir], dr[zDir])) *
                        ((GEMPIC_D_MULT(std::cos(r[xDir]), std::cos(r[yDir]), std::cos(r[zDir])) +
                          (1. / 4.) * GEMPIC_D_MULT(std::cos(2 * r[xDir]), std::cos(2 * r[yDir]),
                                                    std::cos(2 * r[zDir]))));
                });
        }

        anPhi.average_sync();

        /*
         * Has a Print on ParallelFor, run it only on CPU
        for (amrex::MFIter mfi(anPhi.data); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::Array4<amrex::Real> const &phiMF = (phi.data)[mfi].array();
            amrex::Array4<amrex::Real> const &anPhiMF = (anPhi.data)[mfi].array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::Print() << "(" << i << "," << j << "," << k << ") phi: " << phiMF(i, j, k) <<
        " anPhi: " << anPhiMF(i, j, k) << std::endl;
            });
        }
        */

        // Calculate errors
        DeRhamField<Grid::primal, Space::node> errorPhi(deRham);

        for (amrex::MFIter mfi(errorPhi.m_data); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::Array4<amrex::Real> const &phiMF = (phi.m_data)[mfi].array();
            amrex::Array4<amrex::Real> const &anPhiMF = (anPhi.m_data)[mfi].array();
            amrex::Array4<amrex::Real> const &errorPhiMF = (errorPhi.m_data)[mfi].array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                        { errorPhiMF(i, j, k) = std::abs(phiMF(i, j, k) - anPhiMF(i, j, k)); });
        }

        amrex::Print() << "max errorPhi: " << errorPhi.m_data.norm0() << std::endl;
    }
    amrex::Finalize();
}
