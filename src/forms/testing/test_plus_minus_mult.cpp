#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_FDDeRhamComplex.H>

using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

int main (int argc, char *argv[]) 
{
	amrex::Initialize(argc, argv); 

    /* Initialize the infrastructure */
    const amrex::RealBox realBox({AMREX_D_DECL(0, 0, 0)},{AMREX_D_DECL( 10, 10, 10)});
	const amrex::IntVect nCell = {AMREX_D_DECL(10, 10, 10)};
    const amrex::IntVect maxGridSize = {AMREX_D_DECL(10, 10, 10)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {1, 1, 1};
    const int hodgeDegree = 2;

	Parameters params(realBox, nCell, maxGridSize, isPeriodic, hodgeDegree);
    const amrex::Geometry geom = params.geometry();

    auto deRham = std::make_shared<FDDeRhamComplex>(params);

	// Declare the fields 
	DeRhamField<Grid::primal, Space::edge> E1(deRham);
	DeRhamField<Grid::primal, Space::edge> E2(deRham);
	DeRhamField<Grid::primal, Space::edge> E3(deRham);
    
    const amrex::Array<std::string, 3> analyticalFuncE1 = {"x - y - z", 
                                                           "x - y - z",
                                                           "x - y - z"};

    const amrex::Array<std::string, 3> analyticalFuncE2 = {"x - y - z", 
                                                           "x - y - z",
                                                           "x - y - z"};

    const amrex::Array<std::string, 3> analyticalFuncE3 = {"1.0", 
                                                           "1.0",
                                                           "1.0"};


    const int nVar = 4; //x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> func; 
    amrex::Array<amrex::Parser, GEMPIC_SPACEDIM> parser;
    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalFuncE1[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }

    deRham -> projection(func, 0.0, E1);

    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalFuncE2[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }

    deRham -> projection(func, 0.0, E2);

    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalFuncE3[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }

    deRham -> projection(func, 0.0, E3);

    amrex::Real scalar = 2.;

    E2 += E1;
    E3 -= E1;
    E2 *= scalar;

    // Visualize fields
    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(E1.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &E1MF = (E1.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &E2MF = (E2.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &E3MF = (E3.data[comp])[mfi].array();

            for (int k = lo.z; k <= hi.z; ++k) 
                for (int j = lo.y; j <= hi.y; ++j)
                    for (int i = lo.x; i <= hi.x; ++i)
                        amrex::Print() << "(" << i << "," << j << "," << k << ") E1: " << E1MF(i, j, k) << " E2: " << E2MF(i, j, k) << " E3: " << E3MF(i, j, k) << std::endl;
        }
    }
}
