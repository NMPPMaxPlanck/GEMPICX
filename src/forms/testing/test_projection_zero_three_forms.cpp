/*------------------------------------------------------------------------------
 Test the restriction operators R_0, R_3 on both primal and dual grid for
 periodic boundary conditions.
    Integration uses enough Gauss nodes so that the quadrature error can be
    smaller thant 1e-15.
    Test passes if all projected DOFs are within 1e-15 of the analytical value.
------------------------------------------------------------------------------*/
#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_FDDeRhamComplex.H>

using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

int main (int argc, char *argv[]) 
{
	amrex::Initialize(argc, argv);

    // error tolerance
    const amrex::Real tol = 1e-15;

    // number of quadrature points
    int gaussNodes = 6;

    /* Initialize the infrastructure */
    const amrex::RealBox realBox({AMREX_D_DECL(-M_PI + 0.3, -M_PI + 0.6, -M_PI + 0.4)},{AMREX_D_DECL(M_PI + 0.3, M_PI + 0.6, M_PI + 0.4)});
	const amrex::IntVect nCell{AMREX_D_DECL(9, 8, 7)};
    const amrex::IntVect maxGridSize{AMREX_D_DECL(3, 4, 5)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};
    const int hodgeDegree = 2;

	Parameters params(realBox, nCell, maxGridSize, isPeriodic, hodgeDegree);
    const amrex::Geometry geom = params.geometry();

    auto deRham = std::make_shared<FDDeRhamComplex>(params);

	// Declare the fields 
	DeRhamField<Grid::primal, Space::node> Q(deRham);

#if (GEMPIC_SPACEDIM == 1)
    const std::string analyticalQ = "cos(x)";
#endif
#if (GEMPIC_SPACEDIM == 2)
    const std::string analyticalQ = "cos(x) * cos(y)";
#endif
#if (GEMPIC_SPACEDIM == 3)
    const std::string analyticalQ = "cos(x) * cos(y) * cos(z)";
#endif

    const int nVar = GEMPIC_SPACEDIM + 1; //x, y, z, t
    amrex::ParserExecutor<nVar> func; 
    amrex::Parser parser;

    parser.define(analyticalQ);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Compute the projection of the field
    deRham->projection(func, 0.0, Q);

    // Compute the analytical result of the field
    DeRhamField<Grid::primal, Space::node> pointVals(deRham);

    for (amrex::MFIter mfi(pointVals.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::Array4<amrex::Real> const &zeroForm = (pointVals.data)[mfi].array();

        const amrex::RealVect dr = params.dr();
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = params.geometry().ProbLoArray();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[0] + i * dr[0], r0[1] + j * dr[1], r0[2] + k * dr[2])
            };

            zeroForm(i, j, k) = GEMPIC_D_MULT(std::cos(r[0]), std::cos(r[1]), std::cos(r[2]));
        });
    }

    pointVals.averageSync();
    pointVals.fillBoundary();

    // Calculate error
	DeRhamField<Grid::primal, Space::node> errorQ(deRham);
    for (amrex::MFIter mfi(Q.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real> const &projectionMF = (Q.data)[mfi].array();
        amrex::Array4<amrex::Real> const &analyticalMF = (pointVals.data)[mfi].array();
        amrex::Array4<amrex::Real> const &errorMF = (errorQ.data)[mfi].array();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        { 
            errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
        });
    }

   
    // Same test for dual, node
	DeRhamField<Grid::dual, Space::node> QDual(deRham);

    #if (GEMPIC_SPACEDIM == 1)
    const std::string analyticalQDual = "cos(x)";
#endif
#if (GEMPIC_SPACEDIM == 2)
    const std::string analyticalQDual = "cos(x) * cos(y)";
#endif
#if (GEMPIC_SPACEDIM == 3)
    const std::string analyticalQDual = "cos(x) * cos(y) * cos(z)";
#endif

    parser.define(analyticalQDual);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Compute the projection of the field
    deRham->projection(func, 0.0, QDual);

    // Compute the analytical result of the field
    DeRhamField<Grid::dual, Space::node> pointValsDual(deRham);

    for (amrex::MFIter mfi(pointValsDual.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::Array4<amrex::Real> const &zeroForm = (pointValsDual.data)[mfi].array();

        const amrex::RealVect dr = params.dr();
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = params.geometry().ProbLoArray();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[0] + 0.5*dr[0] + i * dr[0], r0[1] + 0.5*dr[1] + j * dr[1], r0[2] + 0.5*dr[2] + k * dr[2])
            };
            
            zeroForm(i, j, k) = GEMPIC_D_MULT(std::cos(r[0]), std::cos(r[1]), std::cos(r[2]));
        });

    }

    pointValsDual.averageSync();
    pointValsDual.fillBoundary();

    // Calculate error
	DeRhamField<Grid::dual, Space::node> errorQDual(deRham);
    for (amrex::MFIter mfi(QDual.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real> const &projectionMF = (QDual.data)[mfi].array();
        amrex::Array4<amrex::Real> const &analyticalMF = (pointValsDual.data)[mfi].array();
        amrex::Array4<amrex::Real> const &errorMF = (errorQDual.data)[mfi].array();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        { 
            errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
        });
    }

    // Test projection for primal three form
    DeRhamField<Grid::primal, Space::cell> rho(deRham);

#if (GEMPIC_SPACEDIM == 1)
    const std::string analyticalRho = "cos(x)";
#endif
#if (GEMPIC_SPACEDIM == 2)
    const std::string analyticalRho = "cos(x) * cos(y)";
#endif
#if (GEMPIC_SPACEDIM == 3)
    const std::string analyticalRho = "cos(x) * cos(y) * cos(z)";
#endif

    parser.define(analyticalRho);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Compute the projection of the field
    deRham->projection(func, 0.0, rho, gaussNodes);

    // Compute the analytical result of the field
    DeRhamField<Grid::primal, Space::cell> rhoAn(deRham);

    for (amrex::MFIter mfi(rhoAn.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::Array4<amrex::Real> const &threeForm = (rhoAn.data)[mfi].array();

        const amrex::RealVect dr = params.dr();
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = params.geometry().ProbLoArray();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[0] + i * dr[0], r0[1] + j * dr[1], r0[2] + k * dr[2])
            };
            
            threeForm(i, j, k) = GEMPIC_D_MULT((std::sin(r[0] + dr[0]) - std::sin(r[0])), (std::sin(r[1] + dr[1]) - std::sin(r[1])), (std::sin(r[2] + dr[2]) - std::sin(r[2])));
        });
    }

    rhoAn.fillBoundary();

    // Calculate error
	DeRhamField<Grid::primal, Space::cell> errorRho(deRham);
    for (amrex::MFIter mfi(rho.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<amrex::Real> const &projectionMF = (rho.data)[mfi].array();
        amrex::Array4<amrex::Real> const &analyticalMF = (rhoAn.data)[mfi].array();
        amrex::Array4<amrex::Real> const &errorMF = (errorRho.data)[mfi].array();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        { 
            errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
        });
    }


    // Test three form projection dual
    DeRhamField<Grid::dual, Space::cell> rhoDual(deRham);

#if (GEMPIC_SPACEDIM == 1)
    const std::string analyticalRhoDual = "cos(x)";
#endif
#if (GEMPIC_SPACEDIM == 2)
    const std::string analyticalRhoDual = "cos(x) * cos(y)";
#endif
#if (GEMPIC_SPACEDIM == 3)
    const std::string analyticalRhoDual = "cos(x) * cos(y) * cos(z)";
#endif

    parser.define(analyticalRhoDual);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Compute the projection of the field
    deRham->projection(func, 0.0, rhoDual, gaussNodes);

    // Compute the analytical result of the field
    DeRhamField<Grid::dual, Space::cell> rhoAnDual(deRham);

    for (amrex::MFIter mfi(rhoAnDual.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::Array4<amrex::Real> const &threeForm = (rhoAnDual.data)[mfi].array();

        const amrex::RealVect dr = params.dr();
        const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = params.geometry().ProbLoArray();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
            {
                AMREX_D_DECL(r0[0] + 0.5*dr[0] + i * dr[0], r0[1] + 0.5*dr[1] + j * dr[1], r0[2] + 0.5*dr[2] + k * dr[2])
            };

            threeForm(i, j, k) = GEMPIC_D_MULT((std::sin(r[0]) - std::sin(r[0] - dr[0])), (std::sin(r[1]) - std::sin(r[1] - dr[1])), (std::sin(r[2]) - std::sin(r[2] - dr[2])));
        });

    }

    rhoAnDual.averageSync();
    rhoAnDual.fillBoundary();

    // Calculate error
	DeRhamField<Grid::dual, Space::cell> errorRhoDual(deRham);
    for (amrex::MFIter mfi(rhoDual.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        
        amrex::Array4<amrex::Real> const &projectionMF = (rhoDual.data)[mfi].array();
        amrex::Array4<amrex::Real> const &analyticalMF = (rhoAnDual.data)[mfi].array();
        amrex::Array4<amrex::Real> const &errorMF = (errorRhoDual.data)[mfi].array();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        { 
            errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
        });
    }

    bool passQ, passQDual, passRho, passRhoDual = false;
    amrex::Real errorQ_norm0 = errorQ.data.norm0();
    amrex::Real errorQDual_norm0 = errorQDual.data.norm0();
    amrex::Real errorRho_norm0 = errorRho.data.norm0();
    amrex::Real errorRhoDual_norm0 = errorRhoDual.data.norm0();

    /*
    amrex::Print() << "errorQ_norm0 = " << errorQ_norm0 << std::endl;
    amrex::Print() << "errorQDual_norm0 = " << errorQDual_norm0 << std::endl;
    amrex::Print() << "errorRho_norm0 = " << errorRho_norm0 << std::endl;
    amrex::Print() << "errorRhoDual_norm0 = " << errorRhoDual_norm0 << std::endl;
    */

    if (std::max({errorQ_norm0, errorQDual_norm0, errorRho_norm0, errorRhoDual_norm0}) < tol)
    {
        passQ = true;
        passQDual = true;
        passRho = true;
        passRhoDual = true;
    }

    if (passQ == true && passQDual == true && passRho == true && passRhoDual == true)
    {
        amrex::PrintToFile("test_projection_zero_three_forms.output") << std::endl;
        amrex::PrintToFile("test_projection_zero_three_forms.output") << GEMPIC_SPACEDIM << "D test passed" << std::endl;
    }
    else
    {
        amrex::PrintToFile("test_projection_zero_three_forms.output") << std::endl;
        amrex::PrintToFile("test_projection_zero_three_forms.output") << GEMPIC_SPACEDIM << "D test failed" << std::endl;
    }

    amrex::PrintToFile("test_projection_zero_three_forms.output") << "max Error Q = " << errorQ_norm0 << std::endl;
    amrex::PrintToFile("test_projection_zero_three_forms.output") << "max Error QDual = " << errorQDual_norm0 << std::endl;
    amrex::PrintToFile("test_projection_zero_three_forms.output") << "max Error Rho = " << errorRho_norm0 << std::endl;
    amrex::PrintToFile("test_projection_zero_three_forms.output") << "max Error RhoDual = " << errorRhoDual_norm0 << std::endl;

    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename("test_projection_zero_three_forms.output.0", "test_projection_zero_three_forms.output");
    amrex::Print() << "IOProcessorNumber " << amrex::ParallelDescriptor::IOProcessorNumber() << std::endl;

    amrex::Finalize();
}
