#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_FDDeRhamComplex.H>

using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

int main (int argc, char *argv[]) 
{
	amrex::Initialize(argc, argv); 

    /* Initialize the infrastructure */
    const amrex::RealBox realBox({AMREX_D_DECL(-M_PI + 0.3, -M_PI + 0.6, -M_PI + 0.4)},{AMREX_D_DECL(M_PI + 0.3, M_PI + 0.6, M_PI + 0.4)});
	const amrex::IntVect nCell{AMREX_D_DECL(19, 15, 17)};
    const amrex::IntVect maxGridSize{AMREX_D_DECL(3, 4, 5)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};
    const int degree = 2;

	Parameters params(realBox, nCell, maxGridSize, isPeriodic, degree);
    const amrex::Geometry geom = params.geometry();

    auto deRham = std::make_shared<FDDeRhamComplex>(params);

	// Declare the fields 
	DeRhamField<Grid::primal, Space::node> Q(deRham);
	DeRhamField<Grid::primal, Space::edge> gradQ(deRham);
    
    // Parse analytical fields and and initialize func
#if (GEMPIC_SPACEDIM == 1)
    const std::string analyticalQ = "sin(x)";
#endif
#if (GEMPIC_SPACEDIM == 2)
    const std::string analyticalQ = "sin(x + 2*y)";
#endif
#if (GEMPIC_SPACEDIM == 3)
    const std::string analyticalQ = "sin(x + 2*y - z)";
#endif

    const int nVar = GEMPIC_SPACEDIM + 1; //x, y, z, t
    amrex::ParserExecutor<nVar> funcQ; 
    amrex::Parser parserQ;

    parserQ.define(analyticalQ);
    parserQ.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    funcQ = parserQ.compile<nVar>();

    // Compute the projection of the field
    deRham -> projection(funcQ, 0.0, Q);

    // Calculate gradQ from Q
    deRham -> grad(Q, gradQ);

    // Analytical gradQ
#if (GEMPIC_SPACEDIM == 1)
    const amrex::Array<std::string, 3> analyticalGradQ = {"cos(x)", 
                                                          "0.",
                                                          "0."};
#endif
#if (GEMPIC_SPACEDIM == 2)
    const amrex::Array<std::string, 3> analyticalGradQ = {"cos(x + 2*y)", 
                                                          "2*cos(x + 2*y)",
                                                          "0."};
#endif
#if (GEMPIC_SPACEDIM == 3)
    const amrex::Array<std::string, 3> analyticalGradQ = {"cos(x + 2*y - z)", 
                                                          "2*cos(x + 2*y - z)",
                                                          "-cos(x + 2*y - z)"};
#endif
    
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcGradQ; 
    amrex::Array<amrex::Parser, 3> parserGradQ;
    for (int i = 0; i < 3; ++i)
    {
        parserGradQ[i].define(analyticalGradQ[i]);
        parserGradQ[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcGradQ[i] = parserGradQ[i].compile<nVar>();
    }
	
    // Declare E field
    DeRhamField<Grid::primal, Space::edge> E(deRham);
    deRham -> projection(funcGradQ, 0.0, E);

    // Calculate errorQ
    bool passQ{false};
    DeRhamField<Grid::primal, Space::edge> errorQ(deRham);

    for (int comp = 0; comp < 3; ++comp)
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
#if (GEMPIC_SPACEDIM == 1)
    const std::string analyticalDualQ = "sin(x)";
#endif
#if (GEMPIC_SPACEDIM == 2)
    const std::string analyticalDualQ = "sin(x + 2*y)";
#endif
#if (GEMPIC_SPACEDIM == 3)
    const std::string analyticalDualQ = "sin(x + 2*y - z)";
#endif

    amrex::ParserExecutor<nVar> funcDualQ; 
    amrex::Parser parserDualQ;

    parserDualQ.define(analyticalDualQ);
    parserDualQ.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    funcDualQ = parserDualQ.compile<nVar>();

    // Compute the projection of the field
    deRham -> projection(funcDualQ, 0.0, QDual);

    // Calculate gradQ from Q
    deRham -> grad(QDual, gradDualQ);

    // Analytical gradE
 #if (GEMPIC_SPACEDIM == 1)
    const amrex::Array<std::string, 3> analyticalGradDualQ = {"cos(x)", 
                                                              "0.",
                                                              "0."};
#endif
#if (GEMPIC_SPACEDIM == 2)
    const amrex::Array<std::string, 3> analyticalGradDualQ = {"cos(x + 2*y)", 
                                                              "2*cos(x + 2*y)",
                                                              "0."};
#endif
#if (GEMPIC_SPACEDIM == 3)
    const amrex::Array<std::string, 3> analyticalGradDualQ = {"cos(x + 2*y - z)", 
                                                              "2*cos(x + 2*y - z)",
                                                              "-cos(x + 2*y - z)"};
#endif
    
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcGradDualQ; 
    amrex::Array<amrex::Parser, 3> parserGradDualQ;
    for (int i = 0; i < 3; ++i)
    {
        parserGradDualQ[i].define(analyticalGradDualQ[i]);
        parserGradDualQ[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcGradDualQ[i] = parserGradDualQ[i].compile<nVar>();
    }
	
    // Declare H field
    DeRhamField<Grid::dual, Space::edge> H(deRham);
    deRham -> projection(funcGradDualQ, 0.0, H);

    // Calculate errorQDual
    bool passDualQ{false};
    DeRhamField<Grid::dual, Space::edge> errorDualQ(deRham);

    for (int comp = 0; comp < 3; ++comp)
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

    amrex::Real errorGradQx_norm0 = errorQ.data[0].norm0();
    amrex::Real errorGradQy_norm0 = errorQ.data[1].norm0();
    amrex::Real errorGradQz_norm0 = errorQ.data[2].norm0();

    amrex::Real errorGradQDualx_norm0 = errorDualQ.data[0].norm0();
    amrex::Real errorGradQDualy_norm0 = errorDualQ.data[1].norm0();
    amrex::Real errorGradQDualz_norm0 = errorDualQ.data[2].norm0();

    /*
    amrex::Print() << "errorGradQx_norm0 = " << errorGradQx_norm0 << std::endl;
    amrex::Print() << "errorGradQy_norm0 = " << errorGradQy_norm0 << std::endl;
    amrex::Print() << "errorGradQz_norm0 = " << errorGradQz_norm0 << std::endl;
    amrex::Print() << "errorGradQDualx_norm0 = " << errorGradQDualx_norm0 << std::endl;
    amrex::Print() << "errorGradQDualy_norm0 = " << errorGradQDualy_norm0 << std::endl;
    amrex::Print() << "errorGradQDualz_norm0 = " << errorGradQDualz_norm0 << std::endl;
    */

    if (std::max({errorGradQx_norm0, errorGradQy_norm0, errorGradQz_norm0}) < 1e-6)
        passQ = true;
    if (std::max({errorGradQDualx_norm0, errorGradQDualy_norm0, errorGradQDualz_norm0}) < 1e-6)
        passDualQ = true;
    
    if (passQ == true && passDualQ == true)
    {
        amrex::PrintToFile("test_grad_projection.output") << std::endl;
        amrex::PrintToFile("test_grad_projection.output") << GEMPIC_SPACEDIM << "D test passed" << std::endl;
    }
    else
    {
        amrex::PrintToFile("test_grad_projection.output") << std::endl;
        amrex::PrintToFile("test_grad_projection.output") << GEMPIC_SPACEDIM << "D test failed" << std::endl;
    }

    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename("test_grad_projection.output.0", "test_grad_projection.output");
    amrex::Print() << "IOProcessorNumber " << amrex::ParallelDescriptor::IOProcessorNumber() << std::endl;

    amrex::Finalize();
}
