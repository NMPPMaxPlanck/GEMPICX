#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_FDDeRhamComplex.H>

using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

int main (int argc, char *argv[]) 
{
	amrex::Initialize(argc, argv); 

    /* Initialize the infrastructure */
    const amrex::RealBox realBox({AMREX_D_DECL(-M_PI, -M_PI, -M_PI)},{AMREX_D_DECL(M_PI, M_PI, M_PI)});
	const amrex::IntVect nCell = {AMREX_D_DECL(32, 32, 32)};
    const amrex::IntVect maxGridSize = {AMREX_D_DECL(32, 32, 32)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {1, 1, 1};
    const int hodgeDegree = 2;

	Parameters params(realBox, nCell, maxGridSize, isPeriodic, hodgeDegree);
    const amrex::Geometry geom = params.geometry();

    auto deRham = std::make_shared<FDDeRhamComplex>(params);

    // Declare the fields 
	DeRhamField<Grid::primal, Space::face> B(deRham);
	DeRhamField<Grid::dual, Space::edge> H(deRham);
	DeRhamField<Grid::dual, Space::face> D(deRham);
	DeRhamField<Grid::primal, Space::edge> E(deRham);

    const amrex::Array<std::string, 3> analyticalB = {"cos(x)*cos(y)*cos(z)", 
                                                      "cos(x)*cos(y)*cos(z)",
                                                      "cos(x)*cos(y)*cos(z)"};

    const amrex::Array<std::string, 3> analyticalD = {"cos(x)*cos(y)*cos(z)", 
                                                      "cos(x)*cos(y)*cos(z)",
                                                      "cos(x)*cos(y)*cos(z)"};


    const int nVar = 4; //x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> func; 
    amrex::Array<amrex::Parser, GEMPIC_SPACEDIM> parser;
    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalB[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }


    deRham -> projection(func, 0.0, B);
    //amrex::Print() << "Before hodge B -> H" << std::endl;
    deRham -> hodgeFD<hodgeDegree>(B,H);

    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalD[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }

    deRham -> projection(func, 0.0, D);
    //amrex::Print() << "Before hodge D -> E" << std::endl;
    //deRham -> hodgeFD<hodgeDegree>(D,E);

    const amrex::Array<std::string, 3> analyticalE = {"cos(x)*cos(y)*cos(z)", 
                                                      "cos(x)*cos(y)*cos(z)",
                                                      "cos(x)*cos(y)*cos(z)"};

    const amrex::Array<std::string, 3> analyticalH = {"cos(x)*cos(y)*cos(z)", 
                                                      "cos(x)*cos(y)*cos(z)",
                                                      "cos(x)*cos(y)*cos(z)"};

    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalH[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }
    
	DeRhamField<Grid::primal, Space::edge> primalOneForm(deRham);
    deRham -> projection(func, 0.0, primalOneForm);

    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalE[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }

	DeRhamField<Grid::dual, Space::edge> dualOneForm(deRham);
    deRham -> projection(func, 0.0, dualOneForm);
    
    // Visualize fields
    /*
    for (int comp = 0; comp < 3; ++comp)
    {
        amrex::Print() << "comp: " << comp << std::endl;
        for (amrex::MFIter mfi(E.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &DMF = (D.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &hodgeMF = (E.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &oneFormMF = (primalOneForm.data[comp])[mfi].array();

            for (int k = lo.z; k < hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                    for (int i = lo.x; i <= hi.x; ++i) 
                        amrex::Print() << "(" << i << "," << j << "," << k << ") D: " << DMF(i, j, k) << " HD: " << hodgeMF(i, j, k) << " E: " << oneFormMF(i, j, k) << std::endl;
        }
    }
    */

    for (int comp = 0; comp < 3; ++comp)
    {
        amrex::Print() << "comp: " << comp << std::endl;
        for (amrex::MFIter mfi(H.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &BMF = (B.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &hodgeMF = (H.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &oneFormMF = (primalOneForm.data[comp])[mfi].array();

            for (int k = lo.z; k < hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                    for (int i = lo.x; i <= hi.x; ++i) 
                        amrex::Print() << "(" << i << "," << j << "," << k << ") B: " << BMF(i, j, k) << " HB: " << hodgeMF(i, j, k) << " H: " << oneFormMF(i, j, k) << std::endl;
        }
    }
   
    // Calculate errors
    /*
	DeRhamField<Grid::dual, Space::edge> errorH(deRham);
	DeRhamField<Grid::primal, Space::edge> errorE(deRham);

    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(H.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &hodgeMF = (H.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &analyticalMF = (dualOneForm.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorMF = (errorH.data[comp])[mfi].array();

            // Pseudo-L2 norm because values at the boundaries are set to zero artificially in order to avoid problems
            // with the Boundary Conditions
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                if ((i < lo.x + 2 || j < lo.y + 2 || k < lo.z + 2) || (i >= hi.x - 2 || j >= hi.y - 2 || k >= hi.z - 2))
                    errorMF(i, j, k) = 0.0;
                else
                    errorMF(i, j, k) = std::abs(hodgeMF(i, j, k) - analyticalMF(i, j, k));
            });
        }

        for (amrex::MFIter mfi(E.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &hodgeMF = (E.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &analyticalMF = (primalOneForm.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorMF = (errorE.data[comp])[mfi].array();

            // Pseudo-L2 norm because values at the boundaries are set to zero artificially in order to avoid problems
            // with the Boundary Conditions
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                if ((i < lo.x + 2 || j < lo.y + 2 || k < lo.z + 2) || (i >= hi.x - 2 || j >= hi.y - 2 || k >= hi.z - 2))
                    errorMF(i, j, k) = 0.0;
                else
                    errorMF(i, j, k) = std::abs(hodgeMF(i, j, k) - analyticalMF(i, j, k));
            });
        }
    }

    for (int comp = 0; comp < 3; ++comp)
    {
        amrex::Print() << "comp: " << comp << " MaxError H: " << errorH.data[comp].norm0() << std::endl;
        amrex::Print() << "comp: " << comp << " MaxError E: " << errorE.data[comp].norm0() << std::endl;
    }
    */

    // Print additional visualization of output
    /*
    // D, HD and E
    amrex::PrintToFile("test_hodge_sinusoidal.output") << std::endl;
    for (int comp = 0; comp < 2; ++comp)
    {
        amrex::PrintToFile("test_hodge_sinusoidal.output") << "comp: " << comp << std::endl;
        for (amrex::MFIter mfi(D.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &DMF= (D.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &hodgeMF= (E.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &oneFormMF = (primalOneForm.data[comp])[mfi].array();

            for (int i = lo.x; i <= hi.x; ++i) 
                        amrex::PrintToFile("test_hodge_sinusoidal.output") << "(" << i << "," << 0 << "," << 0 << ") D: " << DMF(i, 0, 0) << " HD: " << hodgeMF(i, 0, 0) << " E: " << oneFormMF(i, 0, 0) << std::endl;
        }
    }

    // B, HB and H
    amrex::PrintToFile("test_hodge_sinusoidal.output") << std::endl;
    for (int comp = 0; comp < 2; ++comp)
    {
        amrex::PrintToFile("test_hodge_sinusoidal.output") << "comp: " << comp << std::endl;
        for (amrex::MFIter mfi(H.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &BMF= (B.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &hodgeMF= (H.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &oneFormMF = (dualOneForm.data[comp])[mfi].array();

            for (int i = lo.x; i <= hi.x; ++i) 
                        amrex::PrintToFile("test_hodge_sinusoidal.output") << "(" << i << "," << 0 << "," << 0 << ") B: " << BMF(i, 0, 0) << " HB: " << hodgeMF(i, 0, 0) << " H: " << oneFormMF(i, 0, 0) << std::endl;
        }
    }
    */

    /*
    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename("test_hodge_sinusoidal.output.0", "test_hodge_sinusoidal.output");
    amrex::Print() << "IOProcessorNumber " << amrex::ParallelDescriptor::IOProcessorNumber() << std::endl;
    */
    amrex::Finalize();
}
