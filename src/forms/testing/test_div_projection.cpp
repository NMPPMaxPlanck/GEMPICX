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
	DeRhamField<Grid::primal, Space::face> B(deRham);
	DeRhamField<Grid::primal, Space::cell> divB(deRham);
    
    // Parse analytical fields and and initialize func
    const amrex::Array<std::string, 3> analyticalB = {"sin(x)*sin(y)*sin(z)", 
                                                      "sin(x)*sin(y)*sin(z)",
                                                      "sin(x)*sin(y)*sin(z)"};
    const int nVar = 4; //x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcB; 
    amrex::Array<amrex::Parser, GEMPIC_SPACEDIM> parserB;
    for (int i=0; i<3; ++i)
    {
        parserB[i].define(analyticalB[i]);
        parserB[i].registerVariables({"x", "y", "z", "t"});
        funcB[i] = parserB[i].compile<4>();
    }

    // Compute the projection of the field
    deRham -> projection(funcB, 0.0, B);

    // Calculate divB from B
    deRham -> div(B, divB);

    // Analytical divB
    const std::string analyticalDivB = "sin(z)*sin(x + y) + sin(x)*sin(y)*cos(z)"; 
    
    amrex::ParserExecutor<nVar> funcDivB; 
    amrex::Parser parserDivB;

    parserDivB.define(analyticalDivB);
    parserDivB.registerVariables({"x", "y", "z", "t"});
    funcDivB = parserDivB.compile<4>();
	
    // Declare rho
    DeRhamField<Grid::primal, Space::cell> rho(deRham);
    deRham -> projection(funcDivB, 0.0, rho);

    // Calculate errorE
    bool passRho = false;
    DeRhamField<Grid::primal, Space::cell> errorRho(deRham);

    for (amrex::MFIter mfi(rho.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::Array4<amrex::Real> const &analyticalMF = (rho.data)[mfi].array();
        amrex::Array4<amrex::Real> const &projectionMF = (divB.data)[mfi].array();
        amrex::Array4<amrex::Real> const &errorRhoMF = (errorRho.data)[mfi].array();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        { 
            errorRhoMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
        }); 
        
    }

    //amrex::Print() << errorRho.data.norm0() << std::endl;

    // Test div projection for Dual
    // Declare the fields 
	DeRhamField<Grid::dual, Space::face> D(deRham);
	DeRhamField<Grid::dual, Space::cell> divD(deRham);
    
    // Parse analytical fields and and initialize func
    const amrex::Array<std::string, 3> analyticalD = {"sin(x)*sin(y)*sin(z)", 
                                                      "sin(x)*sin(y)*sin(z)",
                                                      "sin(x)*sin(y)*sin(z)"};
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcD; 
    amrex::Array<amrex::Parser, GEMPIC_SPACEDIM> parserD;
    for (int i=0; i<3; ++i)
    {
        parserD[i].define(analyticalD[i]);
        parserD[i].registerVariables({"x", "y", "z", "t"});
        funcD[i] = parserD[i].compile<4>();
    }

    // Compute the projection of the field
    deRham -> projection(funcD, 0.0, D);

    // Calculate divB from B
    deRham -> div(D, divD);

    // Analytical divB
    const std::string analyticalDivD = "sin(z)*sin(x + y) + sin(x)*sin(y)*cos(z)"; 
    
    amrex::ParserExecutor<nVar> funcDivD; 
    amrex::Parser parserDivD;

    parserDivD.define(analyticalDivD);
    parserDivD.registerVariables({"x", "y", "z", "t"});
    funcDivD = parserDivD.compile<4>();
	
    // Declare rho
    DeRhamField<Grid::dual, Space::cell> rhoDual(deRham);
    deRham -> projection(funcDivD, 0.0, rhoDual);

    // Calculate errorE
    bool passRhoDual = false;
    DeRhamField<Grid::dual, Space::cell> errorRhoDual(deRham);

    for (amrex::MFIter mfi(errorRhoDual.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::Array4<amrex::Real> const &analyticalMF = (rhoDual.data)[mfi].array();
        amrex::Array4<amrex::Real> const &projectionMF = (divD.data)[mfi].array();
        amrex::Array4<amrex::Real> const &errorRhoMF = (errorRhoDual.data)[mfi].array();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        { 
            errorRhoMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
        }); 
        
    }

    passRho = (errorRho.data.norm0() < GEMPIC_CTEST_TOL);
    passRhoDual = (errorRhoDual.data.norm0() < GEMPIC_CTEST_TOL);

    if (passRho == true)
    {
        amrex::PrintToFile("test_div_projection.output") << std::endl;
        amrex::PrintToFile("test_div_projection.output") << true << std::endl;
        amrex::PrintToFile("test_div_projection.output") << std::endl;
        for (amrex::MFIter mfi(errorRho.data); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &errorRhoMF = (errorRho.data)[mfi].array();

            for (int i = lo.x; i < hi.x; ++i)
            {
                amrex::PrintToFile("test_div_projection.output") << "(" << i << "," << 0 << "," << "0) errorQ(div.proj(B) - proj.div(B)): = "
                    << errorRhoMF(i, 0, 0) << std::endl;
            }

        }
    }

    if (passRhoDual == true)
    {
        amrex::PrintToFile("test_div_projection.output") << std::endl;
        amrex::PrintToFile("test_div_projection.output") << true << std::endl;
        amrex::PrintToFile("test_div_projection.output") << std::endl;
        for (amrex::MFIter mfi(errorRhoDual.data); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &errorRhoDualMF = (errorRhoDual.data)[mfi].array();

            for (int i = lo.x; i < hi.x; ++i)
            {
                amrex::PrintToFile("test_div_projection.output") << "(" << i << "," << 7 << "," << "0) errorRhoDual(div.proj(RhoDual) - proj.div(RhoDual)): = "
                    << errorRhoDualMF(i, 7, 0) << std::endl;
            }

        }
    }

    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename("test_div_projection.output.0", "test_div_projection.output");
    amrex::Print() << "IOProcessorNumber " << amrex::ParallelDescriptor::IOProcessorNumber() << std::endl;

    amrex::Finalize();
}
