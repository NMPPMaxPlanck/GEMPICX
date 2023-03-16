#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_FDDeRhamComplex.H>

using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;


int main (int argc, char *argv[]) 
{
	amrex::Initialize(argc, argv); 

    /* Initialize the infrastructure */
    const amrex::RealBox realBox({AMREX_D_DECL(-M_PI,-M_PI,-M_PI)},{AMREX_D_DECL(M_PI, M_PI, M_PI)});
	const amrex::IntVect nCell = {AMREX_D_DECL(32, 32, 32)};
    const amrex::IntVect maxGridSize = {AMREX_D_DECL(32, 32, 32)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {1, 1, 1};
    const int hodgeDegree = 2;

	const amrex::Real dt = 0.01;
    const int Nsteps[7] = {0, 1, 2, 5, 10, 20, 50};

	Parameters params(realBox, nCell, maxGridSize, isPeriodic, hodgeDegree);
    const amrex::Geometry geom = params.geometry();

    auto deRham = std::make_shared<FDDeRhamComplex>(params);
    auto NGhost = deRham -> getNGhost();

	// Declare the fields 
	DeRhamField<Grid::dual, Space::face> D(deRham);
	DeRhamField<Grid::primal, Space::face> B(deRham);
	DeRhamField<Grid::primal, Space::edge> E(deRham);
	DeRhamField<Grid::dual, Space::edge> H(deRham);

    // Parse analytical fields and and initialize parserEval
    const amrex::Array<std::string, 3> analyticalFuncD = {"cos(x+y+z-sqrt(3.0)*t)", 
                                                          "-2*cos(x+y+z-sqrt(3.0)*t)",
                                                          "cos(x+y+z-sqrt(3.0)*t)"};

    const amrex::Array<std::string, 3> analyticalFuncB = {"sqrt(3)*cos(x+y+z-sqrt(3.0)*t)", 
                                                          "0.0",
                                                          "-sqrt(3)*cos(x+y+z-sqrt(3.0)*t)"};
    
    // Project B and D to a primal and dual two form respectively
    const int nVar = 4; //x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcD; 
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcB; 
    amrex::Parser parser;
    for (int i=0; i<3; ++i)
    {
        parser.define(analyticalFuncD[i]);
        parser.registerVariables({"x", "y", "z", "t"});
        funcD[i] = parser.compile<4>();
    }

   	deRham -> projection(funcD, 0.0, D);

    for (int i=0; i<3; ++i)
    {
        parser.define(analyticalFuncB[i]);
        parser.registerVariables({"x", "y", "z", "t"});
        funcB[i] = parser.compile<4>();
    }

   	deRham -> projection(funcB, 0.0, B);

    // Advance Maxwell equations using first-order Hamiltonian splitting
    // Initialize auxiliary forms
    DeRhamField<Grid::primal, Space::face> auxPrimalF2(deRham);
    DeRhamField<Grid::dual, Space::face> auxDualF2(deRham);
    
    for (const int &Nt : Nsteps)
    {
        amrex::Print() << "Number of time steps: " << Nt << std::endl;
        for (int i=0; i<Nt; ++i)
        {
            
          // Compute E from D 
          deRham -> hodgeFD<hodgeDegree>(D, E); // E = Hodge(D)

          // Faraday's Law with DeRhamComplex methods 
          deRham -> curl(E, auxPrimalF2); // f2 = curl(E)
          auxPrimalF2 *= dt; // f2 = dt * curl(E)
          
          //B -= f2;  // B -= dt * curl(E)
          B -= auxPrimalF2;
          
          // Compute H from B 
          deRham -> hodgeFD<hodgeDegree>(B, H); // H = Hodge(B)
          
          // Ampere-Maxwell law 
          deRham -> curl(H, auxDualF2); // df2 = curl(H)
          auxDualF2 *= dt; // df2 = dt * curl(H)
                                   
          //D += df2; // D += dt * curl(H)
          D += auxDualF2;
          
        }
        
        // Calculate L2 norm of D and B after all timesteps.
        for (int i=0; i<3; ++i)
        {
            parser.define(analyticalFuncD[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcD[i] = parser.compile<4>();
        }

        deRham -> projection(funcD, Nt*dt, auxDualF2);

        for (int i=0; i<3; ++i)
        {
            parser.define(analyticalFuncB[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcB[i] = parser.compile<4>();
        }

        deRham -> projection(funcB, Nt*dt, auxPrimalF2);

        auxDualF2 -= D;
        auxPrimalF2 -= B;

        for (int comp = 0; comp < 3; ++comp)
            amrex::Print() << "errorD[" << comp << "]: " << (1/std::sqrt(nCell[0]))*(1/std::sqrt(nCell[1]))*(1/std::sqrt(nCell[2]))*(auxDualF2.data[comp].norm0()) << std::endl;
        for (int comp = 0; comp < 3; ++comp)
            amrex::Print() << "errorB[" << comp << "]: " << (1/std::sqrt(nCell[0]))*(1/std::sqrt(nCell[1]))*(1/std::sqrt(nCell[2]))*(auxPrimalF2.data[comp].norm0()) << std::endl;

        amrex::Print() << std::endl;
    }

    /*
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(D.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &DMF = (D.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &exactDMF = (auxDualF2.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorDMF = (errorD.data[comp])[mfi].array();


            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                errorDMF(i, j, k) = (DMF(i, j, k) - exactDMF(i, j, k));
            });

        }
        
        amrex::Print() << "error[" << comp << "]: " << (1/std::sqrt(nCell[0]))*(1/std::sqrt(nCell[1]))*(1/std::sqrt(nCell[2]))*(errorD.data[comp].norm2()) << std::endl;

    }
    */


    amrex::Finalize();
}

