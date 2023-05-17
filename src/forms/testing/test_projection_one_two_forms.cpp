/*------------------------------------------------------------------------------
 Test the restriction operators R_1, R_2 on both primal and dual grid for
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
	DeRhamField<Grid::primal, Space::edge> E(deRham);

    // Parse analytical fields and and initialize parserEval
#if (GEMPIC_SPACEDIM == 1)
    const amrex::Array<std::string, 3> analyticalE = {"cos(x)", 
                                                      "cos(x)",
                                                      "cos(x)"};
#endif
#if (GEMPIC_SPACEDIM == 2)
    const amrex::Array<std::string, 3> analyticalE = {"cos(x) * cos(y)", 
                                                      "cos(x) * cos(y)",
                                                      "cos(x) * cos(y)"};
#endif

#if (GEMPIC_SPACEDIM == 3)
    const amrex::Array<std::string, 3> analyticalE = {"cos(x) * cos(y) * cos(z)", 
                                                      "cos(x) * cos(y) * cos(z)",
                                                      "cos(x) * cos(y) * cos(z)"};
#endif

    const int nVar = GEMPIC_SPACEDIM + 1; //x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> func; 
    amrex::Array<amrex::Parser, 3> parser;
    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(analyticalE[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        func[i] = parser[i].compile<nVar>();
    }

    // Compute the analytical result of the integral
    DeRhamField<Grid::primal, Space::edge> lineIntegral(deRham);

    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(lineIntegral.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::Array4<amrex::Real> const &oneForm = (lineIntegral.data[comp])[mfi].array();

            const amrex::RealVect dr = params.dr();
            const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = params.geometry().ProbLoArray();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {

                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
                {
                    AMREX_D_DECL(r0[0] + i * dr[0], r0[1] + j * dr[1], r0[2] + k * dr[2])
                };

                if (comp == 0)
                    oneForm(i, j, k) = GEMPIC_D_MULT(std::sin(r[0] + dr[0]) - std::sin(r[0]), std::cos(r[1]), std::cos(r[2]));

                if (comp == 1)
                    oneForm(i, j, k) = GEMPIC_D_MULT(std::cos(r[0]), std::sin(r[1] + dr[1]) - std::sin(r[1]), std::cos(r[2]));

                if (comp == 2)
                    oneForm(i, j, k) = GEMPIC_D_MULT(std::cos(r[0]), std::cos(r[1]), std::sin(r[2] + dr[2]) - std::sin(r[2]));
            });

        }
    }

    lineIntegral.averageSync();
    lineIntegral.fillBoundary(); 

    // Compute the projection of the field
    deRham->projection(func, 0.0, E, gaussNodes);


    // Test passes if error < GEMPIC_CTEST_TOL
    bool passE = false;

	DeRhamField<Grid::primal, Space::edge> errorE(deRham);
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(E.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::Array4<amrex::Real> const &projectionMF = (E.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &analyticalMF = (lineIntegral.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorMF= (errorE.data[comp])[mfi].array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            { 
                errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
            });
        }
    }

    amrex::Real errorEx_norm0 = errorE.data[0].norm0();
    amrex::Real errorEy_norm0 = errorE.data[1].norm0();
    amrex::Real errorEz_norm0 = errorE.data[2].norm0();

    if (std::max({errorEx_norm0, errorEy_norm0, errorEz_norm0}) < tol)
        passE = true;

    // Run the same test for face integrals
    // Declare the fields 
	DeRhamField<Grid::primal, Space::face> B(deRham);
	DeRhamField<Grid::primal, Space::face> faceIntegral(deRham);

    // Parse analytical fields and and initialize parserEval
#if (GEMPIC_SPACEDIM == 1)
    const amrex::Array<std::string, 3> analyticalB = {"cos(x)", 
                                                      "cos(x)",
                                                      "cos(x)"};
#endif
#if (GEMPIC_SPACEDIM == 2)
    const amrex::Array<std::string, 3> analyticalB= {"cos(x) * cos(y)", 
                                                     "cos(x) * cos(y)",
                                                     "cos(x) * cos(y)"};
#endif
#if (GEMPIC_SPACEDIM == 3)
    const amrex::Array<std::string, 3> analyticalB = {"cos(x) * cos(y) * cos(z)", 
                                                      "cos(x) * cos(y) * cos(z)",
                                                      "cos(x) * cos(y) * cos(z)"};
#endif

    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(analyticalB[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        func[i] = parser[i].compile<nVar>();
    }

    // Analytical face integral representing the projection
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(faceIntegral.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::Array4<amrex::Real> const &oneForm = (faceIntegral.data[comp])[mfi].array();

            const amrex::RealVect dr = params.dr();
            const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = params.geometry().ProbLoArray();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
                {
                    AMREX_D_DECL(r0[0] + i * dr[0], r0[1] + j * dr[1], r0[2] + k * dr[2])
                };

                if (comp == 0)
                    oneForm(i, j, k) = GEMPIC_D_MULT(std::cos(r[0]), std::sin(r[1] + dr[1]) - std::sin(r[1]), std::sin(r[2] + dr[2]) - std::sin(r[2]));

                if (comp == 1)
                    oneForm(i, j, k) = GEMPIC_D_MULT(std::sin(r[0] + dr[0]) - std::sin(r[0]), std::cos(r[1]), std::sin(r[2] + dr[2]) - std::sin(r[2]));

                if (comp == 2)
                    oneForm(i, j, k) = GEMPIC_D_MULT(std::sin(r[0] + dr[0]) - std::sin(r[0]), std::sin(r[1] + dr[1]) - std::sin(r[1]), std::cos(r[2]));
            });

        }
    }

    faceIntegral.averageSync();
    faceIntegral.fillBoundary(); 

    // Compute the projection of B
    deRham->projection(func, 0.0, B, gaussNodes);

    // Test passes if error < GEMPIC_CTEST_TOL
    bool passB = false;

	DeRhamField<Grid::primal, Space::face> errorB(deRham);
    for (int comp = 0; comp < 3; ++comp)
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


    amrex::Real errorBx_norm0 = errorB.data[0].norm0();
    amrex::Real errorBy_norm0 = errorB.data[1].norm0();
    amrex::Real errorBz_norm0 = errorB.data[2].norm0();

    if (std::max({errorBx_norm0, errorBy_norm0, errorBz_norm0}) < tol)
        passB = true;

    // same test on dual grid
    DeRhamField<Grid::dual, Space::edge> H(deRham);

#if (GEMPIC_SPACEDIM == 1)
    const amrex::Array<std::string, 3> analyticalH = {"cos(x)", 
                                                      "cos(x)",
                                                      "cos(x)"};
#endif
#if (GEMPIC_SPACEDIM == 2)
    const amrex::Array<std::string, 3> analyticalH = {"cos(x) * cos(y)", 
                                                      "cos(x) * cos(y)",
                                                      "cos(x) * cos(y)"};
#endif
#if (GEMPIC_SPACEDIM == 3)
    const amrex::Array<std::string, 3> analyticalH = {"cos(x) * cos(y) * cos(z)", 
                                                      "cos(x) * cos(y) * cos(z)",
                                                      "cos(x) * cos(y) * cos(z)"};
#endif

    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(analyticalH[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        func[i] = parser[i].compile<nVar>();
    }

    // Compute the projection of the field
    deRham->projection(func, 0.0, H, gaussNodes);

    // Compute the analytical solution
    DeRhamField<Grid::dual, Space::edge> lineIntegralDual(deRham);

    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(lineIntegralDual.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::Array4<amrex::Real> const &oneForm = (lineIntegralDual.data[comp])[mfi].array();

            const amrex::RealVect dr = params.dr();
            const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = params.geometry().ProbLoArray();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
                {
                    AMREX_D_DECL(r0[0] + 0.5*dr[0] + i * dr[0], r0[1] + 0.5*dr[1] + j * dr[1], r0[2] + 0.5*dr[2] + k * dr[2])
                };
                
                if (comp == 0)
                    oneForm(i, j, k) = GEMPIC_D_MULT(std::sin(r[0]) - std::sin(r[0] - dr[0]), std::cos(r[1]), std::cos(r[2]));

                if (comp == 1)
                    oneForm(i, j, k) = GEMPIC_D_MULT(std::cos(r[0]), std::sin(r[1]) - std::sin(r[1] - dr[1]), std::cos(r[2]));

                if (comp == 2)
                    oneForm(i, j, k) = GEMPIC_D_MULT(std::cos(r[0]), std::cos(r[1]), std::sin(r[2]) - std::sin(r[2] - dr[2]));
            });

        }
    }
    
    lineIntegralDual.averageSync();
    lineIntegralDual.fillBoundary();

    // Calculate error
	DeRhamField<Grid::dual, Space::edge> errorH(deRham);
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(H.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            amrex::Array4<amrex::Real> const &projectionMF = (H.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &analyticalMF = (lineIntegralDual.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorMF = (errorH.data[comp])[mfi].array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            { 
                errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
            });
        }
    }

    bool passH = false;
    
    amrex::Real errorHx_norm0 = errorH.data[0].norm0();
    amrex::Real errorHy_norm0 = errorH.data[1].norm0();
    amrex::Real errorHz_norm0 = errorH.data[2].norm0();

    if (std::max({errorHx_norm0, errorHy_norm0, errorHz_norm0}) < tol)
        passH = true;

    // Declare the fields 
	DeRhamField<Grid::dual, Space::face> D(deRham);

#if (GEMPIC_SPACEDIM == 1)
    const amrex::Array<std::string, 3> analyticalD = {"cos(x)", 
                                                      "cos(x)",
                                                      "cos(x)"};
#endif
#if (GEMPIC_SPACEDIM == 2)
    const amrex::Array<std::string, 3> analyticalD = {"cos(x) * cos(y)", 
                                                      "cos(x) * cos(y)",
                                                      "cos(x) * cos(y)"};
#endif
#if (GEMPIC_SPACEDIM == 3)
    const amrex::Array<std::string, 3> analyticalD = {"cos(x) * cos(y) * cos(z)", 
                                                      "cos(x) * cos(y) * cos(z)",
                                                      "cos(x) * cos(y) * cos(z)"};
#endif

    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(analyticalD[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        func[i] = parser[i].compile<nVar>();
    }

    // Compute the projection of the field
    deRham->projection(func, 0.0, D, gaussNodes);

    // Compute the analytical solution
    DeRhamField<Grid::dual, Space::face> faceIntegralDual(deRham);

    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(faceIntegralDual.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::Array4<amrex::Real> const &oneForm = (faceIntegralDual.data[comp])[mfi].array();

            const amrex::RealVect dr = params.dr();
            const amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r0 = params.geometry().ProbLoArray();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> r =
                {
                    AMREX_D_DECL(r0[0] + 0.5*dr[0] + i * dr[0], r0[1] + 0.5*dr[1] + j * dr[1], r0[2] + 0.5*dr[2] + k * dr[2])
                };

                if (comp == 0)
                    oneForm(i, j, k) = GEMPIC_D_MULT(std::cos(r[0]), std::sin(r[1]) - std::sin(r[1] - dr[1]), std::sin(r[2]) - std::sin(r[2] - dr[2]));

                if (comp == 1)
                    oneForm(i, j, k) = GEMPIC_D_MULT(std::sin(r[0]) - std::sin(r[0] - dr[0]), std::cos(r[1]), std::sin(r[2]) - std::sin(r[2] - dr[2]));

                if (comp == 2)
                    oneForm(i, j, k) = GEMPIC_D_MULT(std::sin(r[0]) - std::sin(r[0] - dr[0]), std::sin(r[1]) - std::sin(r[1] - dr[1]), std::cos(r[2]));
            });

        }
    }

    faceIntegralDual.averageSync();
    faceIntegralDual.fillBoundary();

    // Calculate error
	DeRhamField<Grid::dual, Space::face> errorD(deRham);
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(D.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            amrex::Array4<amrex::Real> const &projectionMF = (D.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &analyticalMF = (faceIntegralDual.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorMF = (errorD.data[comp])[mfi].array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            { 
                errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
            });
        }
    }

    bool passD = false;
    
    amrex::Real errorDx_norm0 = errorD.data[0].norm0();
    amrex::Real errorDy_norm0 = errorD.data[1].norm0();
    amrex::Real errorDz_norm0 = errorD.data[2].norm0();

    /*
    amrex::Print() << "errorEx_norm0 = " << errorEx_norm0 << ", errorEy_norm0 = " << errorEy_norm0 << ", errorEz_norm0 = " << errorEz_norm0 << std::endl;
    amrex::Print() << "errorBx_norm0 = " << errorBx_norm0 << ", errorBy_norm0 = " << errorBy_norm0 << ", errorBz_norm0 = " << errorBz_norm0 << std::endl;
    amrex::Print() << "errorHx_norm0 = " << errorHx_norm0 << ", errorHy_norm0 = " << errorHy_norm0 << ", errorHz_norm0 = " << errorHz_norm0 << std::endl;
    amrex::Print() << "errorDx_norm0 = " << errorDx_norm0 << ", errorDy_norm0 = " << errorDy_norm0 << ", errorDz_norm0 = " << errorDz_norm0 << std::endl;
    */

    if (std::max({errorDx_norm0, errorDy_norm0, errorDz_norm0}) < tol)
        passD = true;

    if (passE == true && passB == true && passD == true && passH == true)
    {
        amrex::PrintToFile("test_projection_one_two_forms.output") << std::endl;
        amrex::PrintToFile("test_projection_one_two_forms.output") << GEMPIC_SPACEDIM << "D test passed" << std::endl;
    }
    else
    {
        amrex::PrintToFile("test_projection_one_two_forms.output") << std::endl;
        amrex::PrintToFile("test_projection_one_two_forms.output") << GEMPIC_SPACEDIM << "D test failed" << std::endl;
    }

    amrex::PrintToFile("test_projection_one_two_forms.output") << "max Error E[0] = " << errorEx_norm0 << std::endl;
    amrex::PrintToFile("test_projection_one_two_forms.output") << "max Error E[1] = " << errorEy_norm0 << std::endl;
    amrex::PrintToFile("test_projection_one_two_forms.output") << "max Error E[2] = " << errorEz_norm0 << std::endl;
    amrex::PrintToFile("test_projection_one_two_forms.output") << "max Error B[0] = " << errorBx_norm0 << std::endl;
    amrex::PrintToFile("test_projection_one_two_forms.output") << "max Error B[1] = " << errorBy_norm0 << std::endl;
    amrex::PrintToFile("test_projection_one_two_forms.output") << "max Error B[2] = " << errorBz_norm0 << std::endl;
    amrex::PrintToFile("test_projection_one_two_forms.output") << "max Error H[0] = " << errorHx_norm0 << std::endl;
    amrex::PrintToFile("test_projection_one_two_forms.output") << "max Error H[1] = " << errorHy_norm0 << std::endl;
    amrex::PrintToFile("test_projection_one_two_forms.output") << "max Error H[2] = " << errorHz_norm0 << std::endl;
    amrex::PrintToFile("test_projection_one_two_forms.output") << "max Error D[0] = " << errorDx_norm0 << std::endl;
    amrex::PrintToFile("test_projection_one_two_forms.output") << "max Error D[1] = " << errorDy_norm0 << std::endl;
    amrex::PrintToFile("test_projection_one_two_forms.output") << "max Error D[2] = " << errorDz_norm0 << std::endl;

    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename("test_projection_one_two_forms.output.0", "test_projection_one_two_forms.output");
    amrex::Print() << "IOProcessorNumber " << amrex::ParallelDescriptor::IOProcessorNumber() << std::endl;

    amrex::Finalize();
}
