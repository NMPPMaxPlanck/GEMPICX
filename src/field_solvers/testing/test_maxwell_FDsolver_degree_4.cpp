/*------------------------------------------------------------------------------
 Test Maxwell solver convergence rates with Finite Difference Hodges.

  Performs 10 time steps with dt = 0.0001 so that spatial discretization error dominates.
  Analytical solutions are
  in 1D:                           (     0    )
                          D(x,t) = ( cos(x-t) )
                                   ( cos(x-t) )

                                   (     0     )
                          B(x,t) = ( -cos(x-t) )
                                   (  cos(x-t) )

  in 2D:                           (  cos(x+y-sqrt(2)*t)         )
                        D(x,y,t) = ( -cos(x+y-sqrt(2)*t)         )
                                   ( -sqrt(2)*cos(x+y-sqrt(2)*t) )

                                   ( -cos(x+y-sqrt(2)*t)         )
                        B(x,y,t) = (  cos(x+y-sqrt(2)*t)         )
                                   ( -sqrt(2)*cos(x+y-sqrt(2)*t) )

  in 3D:                           (  cos(x+y+z-sqrt(3)*t)   )
                      D(x,y,z,t) = ( -2*cos(x+y+z-sqrt(3)*t) )
                                   (  cos(x+y+z-sqrt(3)*t)   )

                                   (  sqrt(3)*cos(x+y+z-sqrt(3)*t) )
                      B(x,y,z,t) = (                0              )
                                   ( -sqrt(3)*cos(x+y+z-sqrt(3)*t) ).

  And epsilon = mu = 1.
  They are computed for 16 and 32 nodes in each direction. The convergence rate is estimated by log_2 (error_16 / error_32)
------------------------------------------------------------------------------*/

#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_FDDeRhamComplex.H>

using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

const int hodgeDegree = 4;

std::tuple<amrex::Real, amrex::Real> maxwell(const int n)
{
    /* Initialize the infrastructure */
    const amrex::RealBox realBox({AMREX_D_DECL(-M_PI + 0.3, -M_PI + 0.6, -M_PI + 0.4)},{AMREX_D_DECL(M_PI + 0.3, M_PI + 0.6, M_PI + 0.4)});
    const amrex::IntVect nCell{AMREX_D_DECL(n, n, n)};
    const amrex::IntVect maxGridSize{AMREX_D_DECL(8, 6, 7)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};

    const amrex::Real dt = 0.0001;
    const int Nt = 10;

    Parameters params(realBox, nCell, maxGridSize, isPeriodic, hodgeDegree);
    const amrex::Geometry geom = params.geometry();
    
    auto deRham = std::make_shared<FDDeRhamComplex>(params);

    // Declare the fields 
    DeRhamField<Grid::dual, Space::face> D(deRham);
    DeRhamField<Grid::primal, Space::face> B(deRham);
    DeRhamField<Grid::primal, Space::edge> E(deRham);
    DeRhamField<Grid::dual, Space::edge> H(deRham);
    
    DeRhamField<Grid::dual, Space::cell> divD(deRham);
    DeRhamField<Grid::primal, Space::cell> divB(deRham);

    DeRhamField<Grid::primal, Space::face> auxPrimalF2(deRham);
    DeRhamField<Grid::dual, Space::face> auxDualF2(deRham);

    // Parse analytical fields and and initialize parserEval
#if (GEMPIC_SPACEDIM == 1)
    const amrex::Array<std::string, 3> analyticalD = {"0.",
                                                      "cos(x-t)",
                                                      "cos(x-t)"};

    const amrex::Array<std::string, 3> analyticalB = {"0.",
                                                      "-cos(x-t)",
                                                      "cos(x-t)"};
#endif
#if (GEMPIC_SPACEDIM == 2)
    const amrex::Array<std::string, 3> analyticalD = {"cos(x+y-sqrt(2.0)*t)",
                                                      "-cos(x+y-sqrt(2.0)*t)",
                                                      "-sqrt(2)*cos(x+y-sqrt(2.0)*t)"};

    const amrex::Array<std::string, 3> analyticalB = {"-cos(x+y-sqrt(2.0)*t)",
                                                      "cos(x+y-sqrt(2.0)*t)",
                                                      "-sqrt(2)*cos(x+y-sqrt(2.0)*t)"};
#endif
#if (GEMPIC_SPACEDIM == 3)
    const amrex::Array<std::string, 3> analyticalD = {"cos(x+y+z-sqrt(3.0)*t)",
                                                      "-2*cos(x+y+z-sqrt(3.0)*t)",
                                                      "cos(x+y+z-sqrt(3.0)*t)"};

    const amrex::Array<std::string, 3> analyticalB = {"sqrt(3)*cos(x+y+z-sqrt(3.0)*t)",
                                                      "0.0",
                                                      "-sqrt(3)*cos(x+y+z-sqrt(3.0)*t)"};
#endif
    
    // Project B and D to a primal and dual two form respectively
    const int nVar = GEMPIC_SPACEDIM + 1; //x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcD;  
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB; 
    amrex::Array<amrex::Parser, 3> parserD;
    amrex::Array<amrex::Parser, 3> parserB;
    for (int i = 0; i < 3; ++i)
    {
        parserD[i].define(analyticalD[i]);
        parserD[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcD[i] = parserD[i].compile<nVar>();
    }

    for (int i = 0; i < 3; ++i)
    {
        parserB[i].define(analyticalB[i]);
        parserB[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcB[i] = parserB[i].compile<nVar>();
    }

    deRham -> projection(funcD, 0.0, D);
    deRham -> projection(funcB, 0.0, B);
    
    // Advance Maxwell equations using second-order Hamiltonian splitting
    //amrex::Print() << "Start computing " << Nt << " time steps with " << n << " nodes in each direction....." << std::endl;
    for (int i = 0; i < Nt; ++i)
    {
        // Compute E from D 
        deRham -> hodgeFD<hodgeDegree>(D, E); // E = Hodge(D)

        // Faraday's law
        deRham -> curl(E, auxPrimalF2); // f2 = curl(E)
        auxPrimalF2 *= dt/2; // f2 = dt/2 * curl(E)

        //B -= f2;  // B -= dt/2 * curl(E)
        B -= auxPrimalF2;
        
        // Compute H from B 
        deRham -> hodgeFD<hodgeDegree>(B, H); // H = Hodge(B)
        
        // Ampere-Maxwell law 
        deRham -> curl(H, auxDualF2); // df2 = curl(H)
        auxDualF2 *= dt; // df2 = dt * curl(H)
                                
        //D += df2; // D += dt * curl(H)
        D += auxDualF2;

        // Compute E from D 
        deRham -> hodgeFD<hodgeDegree>(D, E); // E = Hodge(D)

        // Faraday's law
        deRham -> curl(E, auxPrimalF2); // f2 = curl(E)
        auxPrimalF2 *= dt/2; // f2 = dt/2 * curl(E)

        //B -= f2;  // B -= dt/2 * curl(E)
        B -= auxPrimalF2;
    }
    
    deRham -> div(D, divD);
    deRham -> div(B, divB);
    amrex::Print() << "Gauss errors: max(div D) = " << divD.data.norm0() << ", max(div B) = " << divB.data.norm0() << std::endl;
    
    // Calculate max error of D and B
    amrex::Real de = 0;
    amrex::Real be = 0;
    for (int comp = 0; comp < 3; ++comp)
    {
        de += deRham -> maxErrorMidpoint<hodgeDegree>(geom, funcD[comp], D.data[comp], params.dr(), 2, true, comp, Nt*dt);
    }
    for (int comp = 0; comp < 3; ++comp)
    {
        be += deRham -> maxErrorMidpoint<hodgeDegree>(geom, funcB[comp], B.data[comp], params.dr(), 2, false, comp, Nt*dt);
    }
    
    return std::make_tuple(de,be);
}

int main (int argc, char *argv[]) 
{
	amrex::Initialize(argc, argv);
    {
        const int coarse = 16;
        const int fine = 32;
        amrex::Real error_coarseD;
        amrex::Real error_coarseB;
        amrex::Real error_fineD;
        amrex::Real error_fineB;

        amrex::GpuArray<amrex::Real, 2> rate;

        std::tie(error_coarseD, error_coarseB) = maxwell(coarse);
        std::tie(error_fineD, error_fineB) = maxwell(fine);
        rate[0] = std::log2(error_coarseD / error_fineD);
        rate[1] = std::log2(error_coarseB / error_fineB);

        amrex::PrintToFile("test_maxwell_FDsolver_degree_4.output") << std::endl;
        amrex::PrintToFile("test_maxwell_FDsolver_degree_4.output") << GEMPIC_SPACEDIM << "D Maxwell degree 4 convergence test:" << std::endl;
        amrex::PrintToFile("test_maxwell_FDsolver_degree_4.output") << std::endl;
        amrex::PrintToFile("test_maxwell_FDsolver_degree_4.output").SetPrecision(3) << "D field rate: " << rate[0] << std::endl;
        amrex::PrintToFile("test_maxwell_FDsolver_degree_4.output").SetPrecision(3) << "B field rate: " << rate[1] << std::endl;
        
        
        if (amrex::ParallelDescriptor::MyProc() == 0)
            std::rename("test_maxwell_FDsolver_degree_4.output.0", "test_maxwell_FDsolver_degree_4.output");

    }
    amrex::Finalize();
}
