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
	DeRhamField<Grid::primal, Space::node> Q(deRham);
	DeRhamField<Grid::primal, Space::edge> gradQ(deRham);
    
    // Parse analytical fields and and initialize func 
    const std::string analyticalQ = "sin(x + y + z)*sin(x + y + z)*sin(x + y + z)";

    const int nVar = 4; //x, y, z, t
    amrex::ParserExecutor<nVar> funcQ; 
    amrex::Parser parserQ;

    parserQ.define(analyticalQ);
    parserQ.registerVariables({"x", "y", "z", "t"});
    funcQ = parserQ.compile<4>();

    // Compute the projection of the field
    deRham -> projection(funcQ, 0.0, Q);

    // Calculate curlQ from Q
    deRham -> grad(Q, gradQ);

    // Analytical curlQ
    const amrex::Array<std::string, 3> analyticalGradQ = {"3*sin(x + y + z)*sin(x + y + z)*cos(x + y + z)", 
                                                          "3*sin(x + y + z)*sin(x + y + z)*cos(x + y + z)",
                                                          "3*sin(x + y + z)*sin(x + y + z)*cos(x + y + z)"};
    
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcGradQ; 
    amrex::Array<amrex::Parser, GEMPIC_SPACEDIM> parserGradQ;
    for (int i=0; i<3; ++i)
    {
        parserGradQ[i].define(analyticalGradQ[i]);
        parserGradQ[i].registerVariables({"x", "y", "z", "t"});
        funcGradQ[i] = parserGradQ[i].compile<4>();
    }
	
    // Declare B field
    DeRhamField<Grid::primal, Space::edge> E(deRham);
    deRham -> projection(funcGradQ, 0.0, E);

    // Calculate errorE
    bool passQ = false;
    DeRhamField<Grid::primal, Space::edge> errorQ(deRham);

    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(errorQ.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            amrex::Array4<amrex::Real> const &projectionMF = (E.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &analyticalMF = (gradQ.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorMF = (errorQ.data[comp])[mfi].array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            { 
                errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
            }); 

        }
    }


    // Test grad for dual
    // Declare the fields 
	DeRhamField<Grid::dual, Space::node> QDual(deRham);
	DeRhamField<Grid::dual, Space::edge> gradDualQ(deRham);
    
    // Parse analytical fields and and initialize func 
    const std::string analyticalDualQ = "sin(x + y + z)*sin(x + y + z)*sin(x + y + z)";

    amrex::ParserExecutor<nVar> funcDualQ; 
    amrex::Parser parserDualQ;

    parserDualQ.define(analyticalDualQ);
    parserDualQ.registerVariables({"x", "y", "z", "t"});
    funcDualQ = parserDualQ.compile<4>();

    // Compute the projection of the field
    deRham -> projection(funcDualQ, 0.0, QDual);

    // Calculate curlQ from Q
    deRham -> grad(QDual, gradDualQ);

    // Analytical curlE
    const amrex::Array<std::string, 3> analyticalGradDualQ = {"3*sin(x + y + z)*sin(x + y + z)*cos(x + y + z)", 
                                                              "3*sin(x + y + z)*sin(x + y + z)*cos(x + y + z)",
                                                              "3*sin(x + y + z)*sin(x + y + z)*cos(x + y + z)"};
    
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcGradDualQ; 
    amrex::Array<amrex::Parser, GEMPIC_SPACEDIM> parserGradDualQ;
    for (int i=0; i<3; ++i)
    {
        parserGradDualQ[i].define(analyticalGradDualQ[i]);
        parserGradDualQ[i].registerVariables({"x", "y", "z", "t"});
        funcGradDualQ[i] = parserGradDualQ[i].compile<4>();
    }
	
    // Declare B field
    DeRhamField<Grid::dual, Space::edge> H(deRham);
    deRham -> projection(funcGradDualQ, 0.0, H);

    // Calculate errorE
    bool passDualQ = false;
    DeRhamField<Grid::dual, Space::edge> errorDualQ(deRham);

    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(errorDualQ.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            amrex::Array4<amrex::Real> const &projectionMF = (H.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &analyticalMF = (gradDualQ.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorMF = (errorDualQ.data[comp])[mfi].array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            { 
                errorMF(i, j, k) = std::abs(projectionMF(i, j, k) - analyticalMF(i, j, k));
            }); 

        }
    }


    passQ = ((errorQ.data[0].norm0() < GEMPIC_CTEST_TOL) && (errorQ.data[1].norm0() < GEMPIC_CTEST_TOL) && (errorQ.data[2].norm0() < GEMPIC_CTEST_TOL));
    passDualQ = ((errorDualQ.data[0].norm0() < GEMPIC_CTEST_TOL) && (errorDualQ.data[1].norm0() < GEMPIC_CTEST_TOL) && (errorDualQ.data[2].norm0() < GEMPIC_CTEST_TOL));
    
    if (passQ == true)
    {
        amrex::PrintToFile("test_grad_projection.output") << std::endl;
        amrex::PrintToFile("test_grad_projection.output") << true << std::endl;
        amrex::PrintToFile("test_grad_projection.output") << std::endl;
        for (int comp = 0; comp < 3; ++comp)
            for (amrex::MFIter mfi(errorQ.data[comp]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                const auto lo = lbound(bx);
                const auto hi = ubound(bx);

                amrex::Array4<amrex::Real> const &errorQMF = (errorQ.data[0])[mfi].array();

                for (int i = lo.x; i < hi.x; ++i)
                {
                    amrex::PrintToFile("test_grad_projection.output") << "(" << i << "," << 0 << "," << "0) errorQ(grad.proj(Q) - proj.grad(Q)) [" << comp << "] = "
                        << errorQMF(i, 0, 0) << std::endl;
                }

            }
    }

    if (passDualQ == true)
    {
        amrex::PrintToFile("test_grad_projection.output") << std::endl;
        amrex::PrintToFile("test_grad_projection.output") << true << std::endl;
        amrex::PrintToFile("test_grad_projection.output") << std::endl;
        for (int comp = 0; comp < 3; ++comp)
            for (amrex::MFIter mfi(errorDualQ.data[comp]); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                const auto lo = lbound(bx);
                const auto hi = ubound(bx);

                amrex::Array4<amrex::Real> const &errorQMF = (errorDualQ.data[0])[mfi].array();

                for (int i = lo.x; i < hi.x; ++i)
                {
                    amrex::PrintToFile("test_grad_projection.output") << "(" << i << "," << 0 << "," << "0) errorQ(grad.proj(QDual) - proj.grad(QDual)) [" << comp << "] = "
                        << errorQMF(i, 0, 0) << std::endl;
                }

            }
    }

    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename("test_grad_projection.output.0", "test_grad_projection.output");
    amrex::Print() << "IOProcessorNumber " << amrex::ParallelDescriptor::IOProcessorNumber() << std::endl;

    amrex::Finalize();
}
