#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_FDDeRhamComplex.H>

using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

int main (int argc, char *argv[]) 
{
	amrex::Initialize(argc, argv); 

    /* Initialize the infrastructure */
    const amrex::RealBox realBox({AMREX_D_DECL(-1.0,-1.0,-1.0)},{AMREX_D_DECL( 1.0, 1.0, 1.0)});
	const amrex::IntVect nCell = {AMREX_D_DECL(32, 32, 32)};
    const amrex::IntVect maxGridSize = {AMREX_D_DECL(32, 32, 32)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {1, 1, 1};
    const int hodgeDegree = 2;

	const amrex::Real dt = 0.01;
    const int Nsteps = 1;

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
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> func; 
    amrex::Parser parser;
    for (int i=0; i<3; ++i)
    {
        parser.define(analyticalFuncD[i]);
        parser.registerVariables({"x", "y", "z", "t"});
        func[i] = parser.compile<4>();
    }

   	deRham -> projection(func, 0.0, D);

    for (int i=0; i<3; ++i)
    {
        parser.define(analyticalFuncB[i]);
        parser.registerVariables({"x", "y", "z", "t"});
        func[i] = parser.compile<4>();
    }

   	deRham -> projection(func, 0.0, B);

    // Advance Maxwell equations using first-order Hamiltonian splitting
    // Initialize auxiliary forms
    DeRhamField<Grid::primal, Space::face> auxPrimalF2(deRham);
    DeRhamField<Grid::dual, Space::face> auxDualF2(deRham);
    
    for (amrex::MFIter mfi(D.data[0]); mfi.isValid(); ++mfi)
      {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &DMF = (D.data[0])[mfi].array();
            amrex::Array4<amrex::Real> const &EMF = (E.data[0])[mfi].array();

            for (int k = lo.z; k <= hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                    for (int i = lo.x; i <= hi.x; ++i) 
                        amrex::Print() << "(" << i << "," << j << "," << k << ") D0: " << DMF(i, j, k) << " E0: " << EMF(i, j, k) << std::endl;
      }

      // Visualize B -> H
      for (amrex::MFIter mfi(H.data[0]); mfi.isValid(); ++mfi)
      {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &BMF= (B.data[0])[mfi].array();
            amrex::Array4<amrex::Real> const &HMF= (H.data[0])[mfi].array();

            for (int k = lo.z; k <= hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                    for (int i = lo.x; i <= hi.x; ++i) 
                        amrex::Print() << "(" << i << "," << j << "," << k << ") B0: " << BMF(i, j, k) << " H0: " << HMF(i, j, k) << std::endl;
      }

	for (int i=0; i<Nsteps; ++i)
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

      // Visualize D -> E
      for (amrex::MFIter mfi(D.data[0]); mfi.isValid(); ++mfi)
      {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &DMF = (D.data[0])[mfi].array();
            amrex::Array4<amrex::Real> const &EMF = (E.data[0])[mfi].array();

            for (int k = lo.z; k <= hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                    for (int i = lo.x; i <= hi.x; ++i) 
                        amrex::Print() << "(" << i << "," << j << "," << k << ") D: " << DMF(i, j, k) << " E: " << EMF(i, j, k) << std::endl;
      }

      // Visualize B -> H
      for (amrex::MFIter mfi(H.data[0]); mfi.isValid(); ++mfi)
      {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &BMF= (B.data[0])[mfi].array();
            amrex::Array4<amrex::Real> const &HMF= (H.data[0])[mfi].array();

            for (int k = lo.z; k <= hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                    for (int i = lo.x; i <= hi.x; ++i) 
                        amrex::Print() << "(" << i << "," << j << "," << k << ") B: " << BMF(i, j, k) << " H: " << HMF(i, j, k) << std::endl;
      }
    }
    
    amrex::Finalize();
}
