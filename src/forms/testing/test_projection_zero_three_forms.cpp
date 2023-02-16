#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_FDDeRhamComplex.H>

using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

int main (int argc, char *argv[]) 
{
	amrex::Initialize(argc, argv); 

    /* Initialize the infrastructure */
    const amrex::RealBox realBox({AMREX_D_DECL(-1, -1, -1)},{AMREX_D_DECL( 2, 2, 2)});
	const amrex::IntVect nCell = {AMREX_D_DECL(10, 10, 10)};
    const amrex::IntVect maxGridSize = {AMREX_D_DECL(10, 10, 10)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {1, 1, 1};
    const int hodgeDegree = 2;

	Parameters params(realBox, nCell, maxGridSize, isPeriodic, hodgeDegree);
    const amrex::Geometry geom = params.geometry();

    auto deRham = std::make_shared<FDDeRhamComplex>(params);

	// Declare the fields 
	DeRhamField<Grid::primal, Space::node> Q(deRham);
    
    // Using (xyz)^5 polynomials as test case
    const std::string analyticalQ = "(x^5) * (y^5) * (z^5)";

    const int nVar = 4; //x, y, z, t
    amrex::ParserExecutor<nVar> func; 
    amrex::Parser parser;

    parser.define(analyticalQ);
    parser.registerVariables({"x", "y", "z", "t"});
    func = parser.compile<4>();

    // Compute the projection of the field
    deRham -> projection(func, 0.0, Q);

    // Compute the analytical result of the field
    DeRhamField<Grid::primal, Space::node> pointVals(deRham);

    for (amrex::MFIter mfi(pointVals.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::Array4<amrex::Real> const &zeroForm = (pointVals.data)[mfi].array();

        const amrex::RealVect dr = params.dr();
        const amrex::GpuArray<amrex::Real, 3> r0 = params.geometry().ProbLoArray(); // Put ProbLo in parameters as a getter. Domainbounds...

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            amrex::GpuArray<amrex::Real, 3> r =
            {
             r0[0] + i*dr[0],
             r0[1] + j*dr[1],
             r0[2] + k*dr[2]
            };

            
            zeroForm(i, j, k) = pow(r[0], 5.0) * pow(r[1], 5.0) * pow(r[2], 5.0);
            
        });

    }

    pointVals.averageSync();

    // Calculate error
	DeRhamField<Grid::primal, Space::node> errorQ(deRham);
    for (amrex::MFIter mfi(Q.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        const auto lo = lbound(bx);
        const auto hi = ubound(bx);

        amrex::Array4<amrex::Real> const &projectionMF = (Q.data)[mfi].array();
        amrex::Array4<amrex::Real> const &analyticalMF = (pointVals.data)[mfi].array();
        amrex::Array4<amrex::Real> const &errorMF = (errorQ.data)[mfi].array();

        // Values at the boundaries are set to zero artificially in order to avoid problems
        // with the Boundary Conditions
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            if ((i < lo.x + 2 || j < lo.y + 2 || k < lo.z + 2) || (i >= hi.x - 2 || j >= hi.y - 2 || k >= hi.z - 2))
                errorMF(i, j, k) = 0.0;
            else
                errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
        });
    }

   
    // Same test for dual, node

	DeRhamField<Grid::dual, Space::node> QDual(deRham);

    // Using (xyz)^5 polynomials as test case
    const std::string analyticalQDual = "(x^5) * (y^5) * (z^5)";

    parser.define(analyticalQDual);
    parser.registerVariables({"x", "y", "z", "t"});
    func = parser.compile<4>();

    // Compute the projection of the field
    deRham -> projection(func, 0.0, QDual);

    // Compute the analytical result of the field
    DeRhamField<Grid::dual, Space::node> pointValsDual(deRham);

    for (amrex::MFIter mfi(pointValsDual.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::Array4<amrex::Real> const &zeroForm = (pointValsDual.data)[mfi].array();

        const amrex::RealVect dr = params.dr();
        const amrex::GpuArray<amrex::Real, 3> r0 = params.geometry().ProbLoArray(); // Put ProbLo in parameters as a getter. Domainbounds...

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            amrex::GpuArray<amrex::Real, 3> r =
            {
             (r0[0] + 0.5*dr[0]) + i*dr[0],
             (r0[1] + 0.5*dr[1]) + j*dr[1],
             (r0[2] + 0.5*dr[2]) + k*dr[2]
            };

            
            zeroForm(i, j, k) = pow(r[0], 5.0) * pow(r[1], 5.0) * pow(r[2], 5.0);
            
        });

    }

    pointValsDual.averageSync();
    pointValsDual.fillBoundary();

    // Calculate error
	DeRhamField<Grid::dual, Space::node> errorQDual(deRham);
    for (amrex::MFIter mfi(QDual.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        const auto lo = lbound(bx);
        const auto hi = ubound(bx);

        amrex::Array4<amrex::Real> const &projectionMF = (QDual.data)[mfi].array();
        amrex::Array4<amrex::Real> const &analyticalMF = (pointValsDual.data)[mfi].array();
        amrex::Array4<amrex::Real> const &errorMF = (errorQDual.data)[mfi].array();

        // Values at the boundaries are set to zero artificially in order to avoid problems
        // with the Boundary Conditions
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            if ((i < lo.x + 2 || j < lo.y + 2 || k < lo.z + 2) || (i >= hi.x - 2 || j >= hi.y - 2 || k >= hi.z - 2))
                errorMF(i, j, k) = 0.0;
            else
                errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
        });

        // This visualization is only suitable for CPU
        /*
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::Print() << "(" << i << ", " << j << ", " << k << ") an: " << analyticalMF(i, j, k) << " pr: " << projectionMF(i, j, k) << std::endl; 
        });
        */
    }

    // Test projection for primal three form
    DeRhamField<Grid::primal, Space::cell> rho(deRham);

    // Using (xyz)^5 polynomials as test case
    const std::string analyticalRho = "(x^5) * (y^5) * (z^5)";

    parser.define(analyticalRho);
    parser.registerVariables({"x", "y", "z", "t"});
    func = parser.compile<4>();

    // Compute the projection of the field
    deRham -> projection(func, 0.0, rho);

    // Compute the analytical result of the field
    DeRhamField<Grid::primal, Space::cell> rhoAn(deRham);

    for (amrex::MFIter mfi(rhoAn.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::Array4<amrex::Real> const &threeForm = (rhoAn.data)[mfi].array();

        const amrex::RealVect dr = params.dr();
        const amrex::GpuArray<amrex::Real, 3> r0 = params.geometry().ProbLoArray(); // Put ProbLo in parameters as a getter. Domainbounds...

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            amrex::GpuArray<amrex::Real, 3> r =
            {
             r0[0] + i*dr[0],
             r0[1] + j*dr[1],
             r0[2] + k*dr[2]
            };

            
            threeForm(i, j, k) = (1./6.)*(pow(r[0] + dr[0], 6.0) - pow(r[0], 6.0)) * (1./6.)*(pow(r[1] + dr[1], 6.0) - pow(r[1], 6.0)) *
                                 (1./6.)*(pow(r[2] + dr[2], 6.0) - pow(r[2], 6.0));
            
        });

    }

    rhoAn.averageSync();
    rhoAn.fillBoundary();

    // Calculate error
	DeRhamField<Grid::dual, Space::cell> errorRho(deRham);
    for (amrex::MFIter mfi(rho.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        const auto lo = lbound(bx);
        const auto hi = ubound(bx);

        amrex::Array4<amrex::Real> const &projectionMF = (rho.data)[mfi].array();
        amrex::Array4<amrex::Real> const &analyticalMF = (rhoAn.data)[mfi].array();
        amrex::Array4<amrex::Real> const &errorMF = (errorRho.data)[mfi].array();

        // Values at the boundaries are set to zero artificially in order to avoid problems
        // with the Boundary Conditions
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            if ((i < lo.x + 2 || j < lo.y + 2 || k < lo.z + 2) || (i >= hi.x - 2 || j >= hi.y - 2 || k >= hi.z - 2))
                errorMF(i, j, k) = 0.0;
            else
                errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
        });

        // This visualization is only suitable for CPU
        /*
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::Print() << "(" << i << ", " << j << ", " << k << ") an: " << analyticalMF(i, j, k) << " pr: " << projectionMF(i, j, k) << std::endl; 
        });
        */
    }


    // Test three form projection dual
    DeRhamField<Grid::dual, Space::cell> rhoDual(deRham);

    // Using (xyz)^5 polynomials as test case
    const std::string analyticalRhoDual = "(x^5) * (y^5) * (z^5)";

    parser.define(analyticalRhoDual);
    parser.registerVariables({"x", "y", "z", "t"});
    func = parser.compile<4>();

    // Compute the projection of the field
    deRham -> projection(func, 0.0, rhoDual);

    // Compute the analytical result of the field
    DeRhamField<Grid::dual, Space::cell> rhoAnDual(deRham);

    for (amrex::MFIter mfi(rhoAnDual.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::Array4<amrex::Real> const &threeForm = (rhoAnDual.data)[mfi].array();

        const amrex::RealVect dr = params.dr();
        const amrex::GpuArray<amrex::Real, 3> r0 = params.geometry().ProbLoArray(); // Put ProbLo in parameters as a getter. Domainbounds...

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            amrex::GpuArray<amrex::Real, 3> r =
            {
             (r0[0] + 0.5*dr[0]) + i*dr[0],
             (r0[1] + 0.5*dr[1]) + j*dr[1],
             (r0[2] + 0.5*dr[2]) + k*dr[2]
            };

            
            threeForm(i, j, k) = (1./6.)*(pow(r[0], 6.0) - pow(r[0] - dr[0], 6.0)) * (1./6.)*(pow(r[1], 6.0) - pow(r[1] - dr[1], 6.0)) *
                                 (1./6.)*(pow(r[2], 6.0) - pow(r[2] - dr[2], 6.0));
            
        });

    }

    rhoAnDual.averageSync();
    rhoAnDual.fillBoundary();

    // Calculate error
	DeRhamField<Grid::dual, Space::cell> errorRhoDual(deRham);
    for (amrex::MFIter mfi(rhoDual.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        const auto lo = lbound(bx);
        const auto hi = ubound(bx);

        amrex::Array4<amrex::Real> const &projectionMF = (rhoDual.data)[mfi].array();
        amrex::Array4<amrex::Real> const &analyticalMF = (rhoAnDual.data)[mfi].array();
        amrex::Array4<amrex::Real> const &errorMF = (errorRhoDual.data)[mfi].array();

        // Values at the boundaries are set to zero artificially in order to avoid problems
        // with the Boundary Conditions
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            if ((i < lo.x + 2 || j < lo.y + 2 || k < lo.z + 2) || (i >= hi.x - 2 || j >= hi.y - 2 || k >= hi.z - 2))
                errorMF(i, j, k) = 0.0;
            else
                errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
        });

        // This visualization is only suitable for CPU
        /*
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::Print() << "(" << i << ", " << j << ", " << k << ") an: " << analyticalMF(i, j, k) << " pr: " << projectionMF(i, j, k) << std::endl; 
        });
        */
    }

    // Maximum absolute error must be zero. Integrals must be of order of numeric_limit of Real, 1e2 is a higher tolerance as the dual error is ~10^-15
    bool passQ, passQDual, passRho, passRhoDual = false;
    if (errorQ.data.norm0() == 0 && errorQDual.data.norm0() == 0 && (errorRho.data.norm0() < std::numeric_limits<amrex::Real>::epsilon()) && 
       (errorRhoDual.data.norm0() < std::numeric_limits<amrex::Real>::epsilon() * 1e2))
    {
        passQ = true;
        passQDual = true;
        passRho = true;
        passRhoDual = true;
    }

    if (passQ == true && passQDual == true && passRho == true && passRhoDual == true)
    {
        amrex::PrintToFile("test_projection_zero_three_forms.output") << std::endl;
        amrex::PrintToFile("test_projection_zero_three_forms.output") << "test passed" << std::endl;
    }
    else
    {
        amrex::PrintToFile("test_projection_zero_three_forms.output") << std::endl;
        amrex::PrintToFile("test_projection_zero_three_forms.output") << "test failed" << std::endl;
    }

    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename("test_projection_zero_three_forms.output.0", "test_projection_zero_three_forms.output");
    amrex::Print() << "IOProcessorNumber " << amrex::ParallelDescriptor::IOProcessorNumber() << std::endl;

    amrex::Finalize();
}
