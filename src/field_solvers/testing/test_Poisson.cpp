#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_PoissonSolver.H>

using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;
using namespace GEMPIC_PoissonSolver;

int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);

    /* Initialize the infrastructure */
    const amrex::RealBox realBox({AMREX_D_DECL(-M_PI, -M_PI, -M_PI)},{AMREX_D_DECL( M_PI, M_PI, M_PI)});
	const amrex::IntVect nCell = {AMREX_D_DECL(64, 64, 64)};
    const amrex::IntVect maxGridSize = {AMREX_D_DECL(64, 64, 64)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {1, 1, 1};
    const int degree = 2;

	Parameters params(realBox, nCell, maxGridSize, isPeriodic, degree);
    auto deRham = std::make_shared<FDDeRhamComplex>(params);

    auto const dr = params.dr();

    // Declare both rho and phi
    DeRhamField<Grid::dual, Space::cell> rho(deRham);
    DeRhamField<Grid::primal, Space::node> phi(deRham);
    DeRhamField<Grid::primal, Space::node> anPhi(deRham);

    // Analytical rho
    //const std::string analyticalRho = "3.0*cos(x)*cos(y)*cos(z) + (3.0)*cos(2*x)*cos(2*y)*cos(2*z)";
    const std::string analyticalRho = "1.0 + cos(x)";

    const int nVar = 4; //x, y, z, t
    amrex::ParserExecutor<nVar> func; 
    amrex::Parser parser;

    parser.define(analyticalRho);
    parser.registerVariables({"x", "y", "z", "t"});
    func = parser.compile<4>();

    // Compute the projection of the field
    deRham -> projection(func, 0.0, rho);

    PoissonSolver poisson;

    poisson.solve(params, rho, phi);

    for (amrex::MFIter mfi(phi.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::Array4<amrex::Real> const &phiMF = (phi.data)[mfi].array();
        amrex::Array4<amrex::Real> const &anPhiMF = (anPhi.data)[mfi].array();
        const amrex::RealVect dr = params.dr();
        const amrex::GpuArray<amrex::Real, 3> r0 = params.geometry().ProbLoArray();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
                amrex::GpuArray<amrex::Real, 3> r =
                {
                 r0[0] + i*dr[0],
                 r0[1] + j*dr[1],
                 r0[2] + k*dr[2]
                };

                // Right mathematically. Rho is a volume integral of analyticalRho in the dual grid.
                // But it doesn't use our Hodge, nor it is the exact analytical function to which it should compare. Change later.
                anPhiMF(i, j, k) =  (dr[0]*dr[1]*dr[2])*((std::cos(r[0])*std::cos(r[1])*std::cos(r[2]) + (1./4.)*std::cos(2*r[0])*std::cos(2*r[1])*std::cos(2*r[2])));

        });
    }

    anPhi.averageSync();

    /*
     * Has a Print on ParallelFor, run it only on CPU
    for (amrex::MFIter mfi(anPhi.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::Array4<amrex::Real> const &phiMF = (phi.data)[mfi].array();
        amrex::Array4<amrex::Real> const &anPhiMF = (anPhi.data)[mfi].array();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::Print() << "(" << i << "," << j << "," << k << ") phi: " << phiMF(i, j, k) << " anPhi: " << anPhiMF(i, j, k) << std::endl;
        });
    }
    */

    // Calculate errors
    DeRhamField<Grid::primal, Space::node> errorPhi(deRham);

    for (amrex::MFIter mfi(errorPhi.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::Array4<amrex::Real> const &phiMF = (phi.data)[mfi].array();
        amrex::Array4<amrex::Real> const &anPhiMF = (anPhi.data)[mfi].array();
        amrex::Array4<amrex::Real> const &errorPhiMF = (errorPhi.data)[mfi].array();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            errorPhiMF(i, j, k) = std::abs(phiMF(i, j, k) - anPhiMF(i, j, k));
        });
    }

    amrex::Print() << "max errorPhi: " << errorPhi.data.norm0() << std::endl;

    amrex::Finalize();
}
