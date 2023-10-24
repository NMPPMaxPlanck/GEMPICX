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

/**
 * @brief Tests the Poisson solver for an analytical rho of 1.0 + cos(x)
 * 
 * @todo: Use our Hodge and compare to exact analytical functon
*/
int main(int argc, char *argv[])
{
    amrex::Initialize(argc, argv);

    /* Initialize the infrastructure */
    const amrex::RealBox realBox({AMREX_D_DECL(-M_PI, -M_PI, -M_PI)},{AMREX_D_DECL( M_PI, M_PI, M_PI)});
    const amrex::IntVect nCell{AMREX_D_DECL(64, 64, 64)};
    //const amrex::IntVect nCell{AMREX_D_DECL(128, 128, 128)};
    const amrex::IntVect maxGridSize{AMREX_D_DECL(64, 64, 64)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};
    const int degree = 2;

    Parameters params(realBox, nCell, maxGridSize, isPeriodic, degree);
    auto deRham = std::make_shared<FDDeRhamComplex>(params);

    auto const dr = params.dr();

    // Declare both rho and phi
    DeRhamField<Grid::dual, Space::cell> rho(deRham);
    DeRhamField<Grid::primal, Space::node> phi(deRham);
    DeRhamField<Grid::primal, Space::node> anPhi(deRham);
    phi.data.setVal(0.0);

    // Analytical rho
    //const std::string analyticalRho = "3.0*cos(x)*cos(y)*cos(z) + (3.0)*cos(2*x)*cos(2*y)*cos(2*z)";
    const std::string analyticalRho = "cos(x)";

    const int nVar = GEMPIC_SPACEDIM + 1; //x, y, z, t
    amrex::ParserExecutor<nVar> func; 
    amrex::Parser parser;

    parser.define(analyticalRho);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Compute the projection of the field
    deRham -> projection(func, 0.0, rho);

    PoissonSolver poisson;

    poisson.solve(params, rho, phi);

    for (amrex::MFIter mfi(phi.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::Array4<amrex::Real> const &anPhiMF = (anPhi.data)[mfi].array();
        const amrex::RealVect dr = params.dr();
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = params.geometry().ProbLoArray();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
                {
                 AMREX_D_DECL(r0[xDir] + i*dr[xDir],
                 r0[yDir] + j*dr[yDir],
                 r0[zDir] + k*dr[zDir])
                };

                // Right mathematically. Rho is a volume integral of analyticalRho in the dual grid.
                // But it doesn't use our Hodge, nor it is the exact analytical function to which it should compare. Change later.
                anPhiMF(i, j, k) =  (GEMPIC_D_MULT(dr[xDir],dr[yDir],dr[zDir]))*((GEMPIC_D_MULT(std::cos(r[xDir]),std::cos(r[yDir]),std::cos(r[zDir])) + (1./4.)*GEMPIC_D_MULT(std::cos(2*r[xDir]),std::cos(2*r[yDir]),std::cos(2*r[zDir]))));

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
