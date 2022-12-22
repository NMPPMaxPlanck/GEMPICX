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
	DeRhamField<Grid::primal, Space::edge> E(deRham);
    
    // Using (xyz)^5 polynomials as test case
    const amrex::Array<std::string, 3> analyticalE = {"(x^5) * (y^5) * (z^5)", 
                                                      "(x^5) * (y^5) * (z^5)",
                                                      "(x^5) * (y^5) * (z^5)"};
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
                    oneForm(i, j, k) = (1./6.) * (pow(r[0] + dr[0], 6.0) - pow(r[0], 6.0)) * pow(r[1], 5.0) * pow(r[2], 5.0);
                if (comp == 1)
                    oneForm(i, j, k) = (1./6.) * (pow(r[1] + dr[1], 6.0) - pow(r[1], 6.0)) * pow(r[0], 5.0) * pow(r[2], 5.0);
                if (comp == 2)
                    oneForm(i, j, k) = (1./6.) * (pow(r[2] + dr[2], 6.0) - pow(r[2], 6.0)) * pow(r[0], 5.0) * pow(r[1], 5.0);
                
            });

        }
    }

    // Compute the projection of the field
    deRham -> projection(func, 0.0, E);

    // Calculate error
	DeRhamField<Grid::primal, Space::edge> errorE(deRham);
    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(E.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &projectionMF = (E.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &analyticalMF = (integral.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorMF = (errorE.data[comp])[mfi].array();

            // Pseudo-L2 norm because values at the boundaries are set to zero artificially in order to avoid problems
            // with the Boundary Conditions
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                if ((i < lo.x + 2 || j < lo.y + 2 || k < lo.z + 2) || (i >= hi.x - 2 || j >= hi.y - 2 || k >= hi.z - 2))
                    errorMF(i, j, k) = 0.0;
                else
                    errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
            });
        }
    }

    // L2 norm of the error must be of the order of machine precision ( ~ e-16 for amrex::Real)
    bool passE = false;
    if (errorE.data[0].norm0() < std::numeric_limits<amrex::Real>::epsilon() * 1e1 &&
        errorE.data[1].norm0() < std::numeric_limits<amrex::Real>::epsilon() * 1e1 &&
        errorE.data[2].norm0() < std::numeric_limits<amrex::Real>::epsilon() * 1e1)
        passE = true;

    // Test 2-form
	DeRhamField<Grid::primal, Space::face> B(deRham);
	DeRhamField<Grid::primal, Space::face> faceIntegral(deRham);

    // Using (xyz)^5 polynomials as test case
    const amrex::Array<std::string, 3> analyticalB = {"(x^5) * (y^5) * (z^5)", 
                                                      "(x^5) * (y^5) * (z^5)",
                                                      "(x^5) * (y^5) * (z^5)"};
    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalB[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }


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
                    oneForm(i, j, k) = (1./6.) * (pow(r[1] + dr[1], 6.0) - pow(r[1], 6.0)) * (1./6.) * (pow(r[2] + dr[2], 6.0) - pow(r[2], 6.0)) * pow(r[0], 5.0);
                if (comp == 1)
                    oneForm(i, j, k) = (1./6.) * (pow(r[0] + dr[0], 6.0) - pow(r[0], 6.0)) * (1./6.) * (pow(r[2] + dr[2], 6.0) - pow(r[2], 6.0)) * pow(r[1], 5.0);
                if (comp == 2)
                    oneForm(i, j, k) = (1./6.) * (pow(r[0] + dr[0], 6.0) - pow(r[0], 6.0)) * (1./6.) * (pow(r[1] + dr[1], 6.0) - pow(r[1], 6.0)) * pow(r[2], 5.0);
                
            });

        }
    }

    // Compute the projection of the field
    deRham -> projection(func, 0.0, B);

    // Calculate error
	DeRhamField<Grid::primal, Space::face> errorB(deRham);
    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(B.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &projectionMF = (B.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &analyticalMF = (faceIntegral.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorMF = (errorB.data[comp])[mfi].array();

            // Pseudo-L2 norm because values at the boundaries are set to zero artificially in order to avoid problems
            // with the Boundary Conditions
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                if (i < lo.x + 2 || j < lo.y + 2 || k < lo.z + 2 || (i >= hi.x - 2 || j >= hi.y - 2 || k >= hi.z - 2))
                    errorMF(i, j, k) = 0.0;
                else
                    errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
            });
        }
    }

    // L2 norm of the error must be of the order of machine precision ( ~ e-16 for amrex::Real)
    bool passB = false;
    if (errorB.data[0].norm0() < std::numeric_limits<amrex::Real>::epsilon() * 1e1 &&
        errorB.data[1].norm0() < std::numeric_limits<amrex::Real>::epsilon() * 1e1 &&
        errorB.data[2].norm0() < std::numeric_limits<amrex::Real>::epsilon() * 1e1)
        passB = true;

    if (passE == true && passB == true)
    {
        amrex::PrintToFile("test_projection_polynomial.output") << std::endl;
        amrex::PrintToFile("test_projection_polynomial.output") << "test passed" << std::endl;
    }
    else
    {
        amrex::PrintToFile("test_projection_polynomial.output") << std::endl;
        amrex::PrintToFile("test_projection_polynomial.output") << "test failed" << std::endl;
    }

    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename("test_projection_polynomial.output.0", "test_projection_polynomial.output");
    amrex::Print() << "IOProcessorNumber " << amrex::ParallelDescriptor::IOProcessorNumber() << std::endl;

    amrex::Finalize();
}
