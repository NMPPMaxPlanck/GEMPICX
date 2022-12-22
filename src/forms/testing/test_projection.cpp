#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_FDDeRhamComplex.H>

using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

int main (int argc, char *argv[]) 
{
	amrex::Initialize(argc, argv); 

    /* Initialize the infrastructure */
    const amrex::RealBox realBox({AMREX_D_DECL(-M_PI, -M_PI, -M_PI)},{AMREX_D_DECL( M_PI, M_PI, M_PI)});
	const amrex::IntVect nCell = {AMREX_D_DECL(8, 8, 8)};
    const amrex::IntVect maxGridSize = {AMREX_D_DECL(4, 4, 4)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {1, 1, 1};
    const int hodgeDegree = 2;

	Parameters params(realBox, nCell, maxGridSize, isPeriodic, hodgeDegree);
    const amrex::Geometry geom = params.geometry();

    auto deRham = std::make_shared<FDDeRhamComplex>(params);

	// Declare the fields 
	DeRhamField<Grid::primal, Space::edge> E(deRham);
    
    // Parse analytical fields and and initialize parserEval
    const amrex::Array<std::string, 3> analyticalE = {"cos (z)", 
                                                      "cos (y)",
                                                      "cos (x)"};
    const int nVar = 4; //x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> func; 
    amrex::Array<amrex::Parser, GEMPIC_SPACEDIM> parser;
    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalE[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }

    // Compute the analytical result of the integral
    DeRhamField<Grid::primal, Space::edge> integral(deRham);

    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(integral.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::Array4<amrex::Real> const &oneForm = (integral.data[comp])[mfi].array();

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

                
                if (comp == 0)
                    oneForm(i, j, k) = dr[0] * std::cos(r[2]);

                if (comp == 1)
                    oneForm(i, j, k) = std::sin(r[1] + dr[1]) - std::sin(r[1]);

                if (comp == 2)
                    oneForm(i, j, k) = dr[2] * std::cos(r[0]); 

                
            });

        }
    }

    integral.averageSync();
    integral.fillBoundary(); 

    // Compute the projection of the field
    deRham -> projection(func, 0.0, E);

    // Visualize projection and analytical line integral
    /*
    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(E.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &projectionMF = (E.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &analyticalMF = (integral.data[comp])[mfi].array();

            amrex::Print() << "comp: " << comp << std::endl;
            for (int k = lo.z; k < hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                    for (int i = lo.x; i <= hi.x; ++i) 
                     { 
                         amrex::Print() << "("<< i << "," << j << "," << k <<
                         ") E: " << projectionMF(i, j, k) << " anE: " << analyticalMF(i, j, k) << std::endl; 
                     }
        }
    }
    */

    // Test passes if error < GEMPIC_CTEST_TOL
    bool passE = false;

	DeRhamField<Grid::primal, Space::edge> errorE(deRham);
    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(E.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::Array4<amrex::Real> const &projectionMF = (E.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &analyticalMF = (integral.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorMF= (errorE.data[comp])[mfi].array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                    { 
                        errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
                    });
        }
    }

    if (errorE.data[0].norm0() + errorE.data[1].norm0() + errorE.data[2].norm0() < GEMPIC_CTEST_TOL)
        passE = true;

    // Run the same test for face integrals
    // Declare the fields 
	DeRhamField<Grid::primal, Space::face> B(deRham);
	DeRhamField<Grid::primal, Space::face> faceIntegral(deRham);
    
    // Parse analytical fields and and initialize parserEval
    const amrex::Array<std::string, 3> analyticalB = {"cos (z)", 
                                                      "cos (y)",
                                                      "cos (x)"};
    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalE[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }

    // Analytical face integral representing the projection
    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(faceIntegral.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::Array4<amrex::Real> const &oneForm = (faceIntegral.data[comp])[mfi].array();

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

                
                if (comp == 0)
                    oneForm(i, j, k) = dr[1] * (std::sin(r[2] + dr[2]) - std::sin(r[2]));

                if (comp == 1)
                    oneForm(i, j, k) = dr[0] * dr[2] * std::cos(r[1]);

                if (comp == 2)
                    oneForm(i, j, k) = dr[0] * (std::sin(r[0] + dr[0]) - std::sin(r[0]));
                
            });

        }
    }

    faceIntegral.averageSync();
    faceIntegral.fillBoundary(); 

    // Compute the projection of B
    deRham -> projection(func, 0.0, B);

    // Visualize projection of B and analytical face integral
    /*
    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(B.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &projectionMF = (B.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &analyticalMF = (faceIntegral.data[comp])[mfi].array();

            amrex::Print() << "comp: " << comp << std::endl;
            for (int k = lo.z; k < hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                    for (int i = lo.x; i <= hi.x; ++i) 
                     { 
                         amrex::Print() << "("<< i << "," << j << "," << k <<
                         ") B: " << projectionMF(i, j, k) << " anB: " << analyticalMF(i, j, k) << std::endl; 
                     }
        }
    }
    */

    // Test passes if error < GEMPIC_CTEST_TOL
    bool passB = false;

	DeRhamField<Grid::primal, Space::face> errorB(deRham);
    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(B.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::Array4<amrex::Real> const &projectionMF = (B.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &analyticalMF = (faceIntegral.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorMF= (errorB.data[comp])[mfi].array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            { 
                errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
            });
        }
    }

    if (errorB.data[0].norm0() + errorB.data[1].norm0() + errorB.data[2].norm0() < GEMPIC_CTEST_TOL)
        passB = true;

    if (passE == true && passB == true)
    {
        amrex::PrintToFile("test_projection.output") << std::endl;
        amrex::PrintToFile("test_projection.output") << "test passed" << std::endl;
    }
    else
    {
        amrex::PrintToFile("test_projection.output") << std::endl;
        amrex::PrintToFile("test_projection.output") << "test failed" << std::endl;
    }

    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename("test_projection.output.0", "test_projection.output");
    amrex::Print() << "IOProcessorNumber " << amrex::ParallelDescriptor::IOProcessorNumber() << std::endl;

    amrex::Finalize();
}
