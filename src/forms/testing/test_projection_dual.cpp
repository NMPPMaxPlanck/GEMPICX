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
	DeRhamField<Grid::dual, Space::edge> H(deRham);

    // Using (xyz)^5 polynomials as test case
    const amrex::Array<std::string, 3> analyticalH = {"(x^5) * (y^5) * (z^5)", 
                                                      "(x^5) * (y^5) * (z^5)",
                                                      "(x^5) * (y^5) * (z^5)"};

    const int nVar = 4; //x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> func; 
    amrex::Array<amrex::Parser, GEMPIC_SPACEDIM> parser;
    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalH[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }

    // Compute the projection of the field
    deRham -> projection(func, 0.0, H);

    // Compute the analytical solution
    DeRhamField<Grid::dual, Space::edge> integral(deRham);

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
                 (r0[0] + 0.5*dr[0]) + i*dr[0],
                 (r0[1] + 0.5*dr[1]) + j*dr[1],
                 (r0[2] + 0.5*dr[2]) + k*dr[2]
                };


                
                if (comp == 0)
                    oneForm(i, j, k) = (1./6.) * (pow(r[0], 6.0) - pow(r[0] - dr[0], 6.0)) * pow(r[1], 5.0) * pow(r[2], 5.0);
                if (comp == 1)
                    oneForm(i, j, k) = (1./6.) * (pow(r[1], 6.0) - pow(r[1] - dr[1], 6.0)) * pow(r[0], 5.0) * pow(r[2], 5.0);
                if (comp == 2)
                    oneForm(i, j, k) = (1./6.) * (pow(r[2], 6.0) - pow(r[2] - dr[2], 6.0)) * pow(r[0], 5.0) * pow(r[1], 5.0);
                
            });

        }
    }
    
    integral.fillBoundary();

    // Calculate error
	DeRhamField<Grid::dual, Space::edge> errorH(deRham);
    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(H.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &projectionMF = (H.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &analyticalMF = (integral.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorMF = (errorH.data[comp])[mfi].array();

            // Values at the boundaries are set to zero artificially in order to avoid problems
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

    amrex::Print() << "max error H: " << errorH.data[0].norm0() << std::endl;

    /*
    // Visualize dual 1-form
    for (amrex::MFIter mfi(H.data[0]); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        const auto lo = lbound(bx);
        const auto hi = ubound(bx);

        amrex::Array4<amrex::Real> const &projMF = (H.data[0])[mfi].array();
        amrex::Array4<amrex::Real> const &anMF = (integral.data[0])[mfi].array();
        for (int k = lo.z + 2; k <= hi.z - 2; ++k)
            for (int j = lo.y + 2; j <= hi.y - 2; ++j)
                for (int i = lo.x + 2; i <= hi.x - 2; ++i)
                {
                    amrex::Print() << "(" << i << "," << j << "," << k << ") H " << projMF(i, j, k) << " an: " << anMF(i, j, k) << std::endl;
                }
    }
    */

    // L2 norm of the error must be of the order of machine precision ( ~ e-16 for amrex::Real)
    bool passH = false;
    if (errorH.data[0].norm0() < std::numeric_limits<amrex::Real>::epsilon() * 1e1 &&
        errorH.data[1].norm0() < std::numeric_limits<amrex::Real>::epsilon() * 1e1 &&
        errorH.data[2].norm0() < std::numeric_limits<amrex::Real>::epsilon() * 1e1)
        passH = true;

    // Declare the fields 
	DeRhamField<Grid::dual, Space::face> D(deRham);

    // Using (xyz)^5 polynomials as test case
    const amrex::Array<std::string, 3> analyticalD = {"(x^5) * (y^5) * (z^5)", 
                                                      "(x^5) * (y^5) * (z^5)",
                                                      "(x^5) * (y^5) * (z^5)"};

    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalD[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }

    // Compute the projection of the field
    deRham -> projection(func, 0.0, D);

    // Compute the analytical solution
    DeRhamField<Grid::dual, Space::face> faceIntegral(deRham);

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
                 (r0[0] + 0.5*dr[0]) + i*dr[0],
                 (r0[1] + 0.5*dr[1]) + j*dr[1],
                 (r0[2] + 0.5*dr[2]) + k*dr[2]
                };

                if (comp == 0)
                    oneForm(i, j, k) = (1./6.) * (pow(r[1], 6.0) - pow(r[1] - dr[1], 6.0)) * (1./6.) * (pow(r[2], 6.0) - pow(r[2] - dr[2], 6.0)) * pow(r[0], 5.0);
                if (comp == 1)
                    oneForm(i, j, k) = (1./6.) * (pow(r[0], 6.0) - pow(r[0] - dr[0], 6.0)) * (1./6.) * (pow(r[2], 6.0) - pow(r[2] - dr[2], 6.0)) * pow(r[1], 5.0);
                if (comp == 2)
                    oneForm(i, j, k) = (1./6.) * (pow(r[0], 6.0) - pow(r[0] - dr[0], 6.0)) * (1./6.) * (pow(r[1], 6.0) - pow(r[1] - dr[1], 6.0)) * pow(r[2], 5.0);
            });

        }
    }

    faceIntegral.fillBoundary();

    // Calculate error
	DeRhamField<Grid::dual, Space::face> errorD(deRham);
    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(D.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &projectionMF = (D.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &analyticalMF = (faceIntegral.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorMF = (errorD.data[comp])[mfi].array();

            // Values at the boundaries are set to zero artificially in order to avoid problems
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

    amrex::Print() << "max error D: " << errorD.data[0].norm0() << std::endl;

    // L2 norm of the error must be of the order of machine precision ( ~ e-16 for amrex::Real)
    bool passD = false;
    if (errorD.data[0].norm0() < std::numeric_limits<amrex::Real>::epsilon() * 1e1 &&
        errorD.data[1].norm0() < std::numeric_limits<amrex::Real>::epsilon() * 1e1 &&
        errorD.data[2].norm0() < std::numeric_limits<amrex::Real>::epsilon() * 1e1)
        passD = true;

    // Visualize dual 2-form
    /*
    for (amrex::MFIter mfi(D.data[0]); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        const auto lo = lbound(bx);
        const auto hi = ubound(bx);

        amrex::Array4<amrex::Real> const &projMF = (D.data[0])[mfi].array();
        amrex::Array4<amrex::Real> const &anMF = (faceIntegral.data[0])[mfi].array();
        for (int k = lo.z + 2; k <= hi.z - 2; ++k)
            for (int j = lo.y + 2; j <= hi.y - 2; ++j)
                for (int i = lo.x + 2; i <= hi.x - 2; ++i)
                {
                    amrex::Print() << "(" << i << "," << j << "," << k << ") D " << projMF(i, j, k) << " an: " << anMF(i, j, k) << std::endl;
                }
    }
    */

    if (passH == true && passD == true)
    {
        amrex::PrintToFile("test_projection_dual.output") << std::endl;
        amrex::PrintToFile("test_projection_dual.output") << "test passed" << std::endl;
    }
    else
    {
        amrex::PrintToFile("test_projection_dual.output") << std::endl;
        amrex::PrintToFile("test_projection_dual.output") << "test failed" << std::endl;
    }

    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename("test_projection_dual.output.0", "test_projection_dual.output");
    amrex::Print() << "IOProcessorNumber " << amrex::ParallelDescriptor::IOProcessorNumber() << std::endl;

    amrex::Finalize();
}

