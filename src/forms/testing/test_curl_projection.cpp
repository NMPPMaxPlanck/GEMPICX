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
    const amrex::IntVect maxGridSize = {AMREX_D_DECL(8, 8, 8)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {1, 1, 1};
    const int degree = 2;

	Parameters params(realBox, nCell, maxGridSize, isPeriodic, degree);
    const amrex::Geometry geom = params.geometry();

    auto deRham = std::make_shared<FDDeRhamComplex>(params);

	// Declare the fields 
	DeRhamField<Grid::primal, Space::edge> E(deRham);
	DeRhamField<Grid::primal, Space::face> curlE(deRham);
    
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

    // Compute the projection of the field
    deRham -> projection(func, 0.0, E);

    // Calculate curlE from E
    deRham -> curl(E, curlE);

    // Analytical curlE
    const amrex::Array<std::string, 3> analyticalCurlE = {"0.0", 
                                                          "sin (x) - sin (z)",
                                                          "0.0"};
    
    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalCurlE[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }
	
    // Declare B field
    DeRhamField<Grid::primal, Space::face> B(deRham);
    deRham -> projection(func, 0.0, B);

    // Calculate error
    bool pass = false;
    DeRhamField<Grid::primal, Space::face> error(deRham);

    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(B.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            amrex::Array4<amrex::Real> const &BMF = (B.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &curlEMF = (curlE.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorMF = (error.data[comp])[mfi].array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            { 
                errorMF(i, j, k) = std::abs(BMF(i, j, k) - curlEMF(i, j, k));
            }); 

        }
    }

    pass = ((error.data[0].norm0() < GEMPIC_CTEST_TOL) && (error.data[1].norm0() < GEMPIC_CTEST_TOL) && (error.data[2].norm0() < GEMPIC_CTEST_TOL));

    if (pass == true)
    {
        amrex::PrintToFile("test_curl_projection.output") << std::endl;
        amrex::PrintToFile("test_curl_projection.output") << true << std::endl;
        amrex::PrintToFile("test_curl_projection.output") << std::endl;
        for (int comp = 0; comp < 3; ++comp)
            for (amrex::MFIter mfi(error.data[comp]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                const auto lo = lbound(bx);
                const auto hi = ubound(bx);

                amrex::Array4<amrex::Real> const &errorMF = (error.data[1])[mfi].array();

                for (int i = lo.x; i < hi.x; ++i)
                {
                    amrex::PrintToFile("test_curl_projection.output") << "(" << i << "," << 0 << "," << "0) error(curl.proj(E) - proj.curl(E)) [" << comp << "] = "
                        << errorMF(i, 0, 0) << std::endl;
                }

            }
    }

    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename("test_curl_projection.output.0", "test_curl_projection.output");
    amrex::Print() << "IOProcessorNumber " << amrex::ParallelDescriptor::IOProcessorNumber() << std::endl;


    // Visualize error 
    /*
    for (int comp = 0; comp < 3; ++comp)
    for (amrex::MFIter mfi(error.data[comp]); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::Array4<amrex::Real> const &errorMF = (error.data[1])[mfi].array();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) { amrex::Print() << "comp: " << comp << " ,("<< i << "," << j << "," << k <<
                 ") error: " << errorMF(i, j, k) << std::endl; });
    }
    */

    amrex::Finalize();
}
