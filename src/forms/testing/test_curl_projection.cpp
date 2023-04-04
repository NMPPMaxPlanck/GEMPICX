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
	const amrex::IntVect nCell{AMREX_D_DECL(8, 8, 8)};
    const amrex::IntVect maxGridSize{AMREX_D_DECL(8, 8, 8)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};
    const int degree = 2;

	Parameters params(realBox, nCell, maxGridSize, isPeriodic, degree);
    const amrex::Geometry geom = params.geometry();

    auto deRham = std::make_shared<FDDeRhamComplex>(params);

	// Declare the fields 
	DeRhamField<Grid::primal, Space::edge> E(deRham);
	DeRhamField<Grid::primal, Space::face> curlE(deRham);
    
    // Parse analytical fields and and initialize func
#if (GEMPIC_SPACEDIM == 1)
    const amrex::Array<std::string, 3> analyticalE = {"-cos(x)",
                                                      "sin(x)",
                                                      "-sin(x)"};
#endif
#if (GEMPIC_SPACEDIM == 2)
    const amrex::Array<std::string, 3> analyticalE = {"-cos(x)*cos(y)",
                                                      "sin(x)*cos(y)",
                                                      "-sin(x)*sin(y)"};
#endif

#if (GEMPIC_SPACEDIM == 3)
    const amrex::Array<std::string, 3> analyticalE = {"-cos(x)*cos(y)*sin(z)",
                                                      "sin(x)*cos(y)*sin(z)",
                                                      "-sin(x)*sin(y)*cos(z)"};
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

    // Compute the projection of the field
    deRham -> projection(func, 0.0, E);

    // Calculate curlE from E
    deRham -> curl(E, curlE);

    // Analytical curlE
#if (GEMPIC_SPACEDIM == 1)
    const amrex::Array<std::string, 3> analyticalCurlE = {"0.",
                                                          "cos(x)",
                                                          "cos(x)"};
#endif
#if (GEMPIC_SPACEDIM == 2)
    const amrex::Array<std::string, 3> analyticalCurlE = {"-sin(x)*cos(y)",
                                                          "cos(x)*sin(y)",
                                                          "cos(x)*cos(y) - cos(x)*sin(y)"};
#endif
#if (GEMPIC_SPACEDIM == 3)
    const amrex::Array<std::string, 3> analyticalCurlE = {"-2*sin(x)*cos(y)*cos(z)",
                                                          "-cos(x)*cos(y)*cos(z) + cos(x)*sin(y)*cos(z)",
                                                          "cos(x)*cos(y)*sin(z) - cos(x)*sin(y)*sin(z)"};
#endif
    
    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(analyticalCurlE[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        func[i] = parser[i].compile<nVar>();
    }
	
    // Declare B field
    DeRhamField<Grid::primal, Space::face> B(deRham);
    deRham -> projection(func, 0.0, B);

    // Calculate errorE
    bool passE = false;
    DeRhamField<Grid::primal, Space::face> errorE(deRham);

    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(B.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::Array4<amrex::Real> const &BMF = (B.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &curlEMF = (curlE.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorEMF = (errorE.data[comp])[mfi].array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            { 
                errorEMF(i, j, k) = std::abs(BMF(i, j, k) - curlEMF(i, j, k));
            }); 

        }
    }


    // Test curl projection = projection curl for dual
    // Declare the fields 
	DeRhamField<Grid::dual, Space::edge> H(deRham);
	DeRhamField<Grid::dual, Space::face> curlH(deRham);
    
    // Parse analytical fields and and initialize parserEval
#if (GEMPIC_SPACEDIM == 1)
    const amrex::Array<std::string, 3> analyticalH = {"-cos(x)",
                                                      "sin(x)",
                                                      "-sin(x)"};
#endif
#if (GEMPIC_SPACEDIM == 2)
    const amrex::Array<std::string, 3> analyticalH = {"-cos(x)*cos(y)",
                                                      "sin(x)*cos(y)",
                                                      "-sin(x)*sin(y)"};
#endif

#if (GEMPIC_SPACEDIM == 3)
    const amrex::Array<std::string, 3> analyticalH = {"-cos(x)*cos(y)*sin(z)",
                                                      "sin(x)*cos(y)*sin(z)",
                                                      "-sin(x)*sin(y)*cos(z)"};
#endif
    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(analyticalH[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        func[i] = parser[i].compile<nVar>();
    }

    // Compute the projection of the field
    deRham -> projection(func, 0.0, H);

    // Calculate curlH from H
    deRham -> curl(H, curlH);

    // Analytical curlH
    #if (GEMPIC_SPACEDIM == 1)
    const amrex::Array<std::string, 3> analyticalCurlH = {"0.",
                                                          "cos(x)",
                                                          "cos(x)"};
#endif
#if (GEMPIC_SPACEDIM == 2)
    const amrex::Array<std::string, 3> analyticalCurlH = {"-sin(x)*cos(y)",
                                                          "cos(x)*sin(y)",
                                                          "cos(x)*cos(y) - cos(x)*sin(y)"};
#endif
#if (GEMPIC_SPACEDIM == 3)
    const amrex::Array<std::string, 3> analyticalCurlH = {"-2*sin(x)*cos(y)*cos(z)",
                                                          "-cos(x)*cos(y)*cos(z) + cos(x)*sin(y)*cos(z)",
                                                          "cos(x)*cos(y)*sin(z) - cos(x)*sin(y)*sin(z)"};
#endif
    
    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(analyticalCurlH[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        func[i] = parser[i].compile<nVar>();
    }
	
    // Declare D field
    DeRhamField<Grid::dual, Space::face> D(deRham);
    deRham -> projection(func, 0.0, D);

    // Calculate errorH
    bool passH = false;
    DeRhamField<Grid::dual, Space::face> errorH(deRham);

    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(D.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::Array4<amrex::Real> const &DMF = (D.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &curlHMF = (curlH.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorHMF = (errorH.data[comp])[mfi].array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            { 
                errorHMF(i, j, k) = std::abs(DMF(i, j, k) - curlHMF(i, j, k));
            }); 

            // Visualization only suitable for CPU
            /*
            if (comp == 0)
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            { 
                amrex::Print() << "(" << i << ", " << j << ", " << k << ") D: " << DMF(i, j, k) << " curlH: " << curlHMF(i, j, k) << " error: " << errorHMF(i, j, k) << std::endl;
            });
            */
        }
    }

    amrex::Real errorCurlEx_norm0 = errorE.data[0].norm0();
    amrex::Real errorCurlEy_norm0 = errorE.data[1].norm0();
    amrex::Real errorCurlEz_norm0 = errorE.data[2].norm0();

    amrex::Real errorCurlHx_norm0 = errorH.data[0].norm0();
    amrex::Real errorCurlHy_norm0 = errorH.data[1].norm0();
    amrex::Real errorCurlHz_norm0 = errorH.data[2].norm0();

    /*
    amrex::Print() << "errorCurlEx_norm0: " << errorCurlEx_norm0 << std::endl;
    amrex::Print() << "errorCurlEy_norm0: " << errorCurlEy_norm0 << std::endl;
    amrex::Print() << "errorCurlEz_norm0: " << errorCurlEz_norm0 << std::endl;
    amrex::Print() << "errorCurlHx_norm0: " << errorCurlHx_norm0 << std::endl;
    amrex::Print() << "errorCurlHy_norm0: " << errorCurlHy_norm0 << std::endl;
    amrex::Print() << "errorCurlHz_norm0: " << errorCurlHz_norm0 << std::endl;
    */

    if (std::max({errorCurlEx_norm0, errorCurlEy_norm0, errorCurlEz_norm0}) < 1e-6)
        passE = true;
    if (std::max({errorCurlHx_norm0, errorCurlHy_norm0, errorCurlHz_norm0}) < 1e-6)
        passH = true;

    if (passE == true && passH == true)
    {
        amrex::PrintToFile("test_curl_projection.output") << std::endl;
        amrex::PrintToFile("test_curl_projection.output") << GEMPIC_SPACEDIM << "D test passed" << std::endl;
    }
    else
    {
        amrex::PrintToFile("test_curl_projection.output") << std::endl;
        amrex::PrintToFile("test_curl_projection.output") << GEMPIC_SPACEDIM << "D test failed" << std::endl;
    }

    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename("test_curl_projection.output.0", "test_curl_projection.output");
    amrex::Print() << "IOProcessorNumber " << amrex::ParallelDescriptor::IOProcessorNumber() << std::endl;


    // Visualize errorE 
    /*
    for (int comp = 0; comp < 3; ++comp)
    for (amrex::MFIter mfi(errorE.data[comp]); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::Array4<amrex::Real> const &errorEMF = (errorE.data[1])[mfi].array();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) { amrex::Print() << "comp: " << comp << " ,("<< i << "," << j << "," << k <<
                 ") errorE: " << errorEMF(i, j, k) << std::endl; });
    }
    */

    amrex::Finalize();
}
