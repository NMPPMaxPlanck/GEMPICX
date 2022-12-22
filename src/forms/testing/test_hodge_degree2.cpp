#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_FDDeRhamComplex.H>

using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

int main (int argc, char *argv[]) 
{
	amrex::Initialize(argc, argv); 

    /* Initialize the infrastructure */
    const amrex::RealBox realBox({AMREX_D_DECL(0, 0, 0)},{AMREX_D_DECL( 10.0, 10.0, 10.0)});
	const amrex::IntVect nCell = {AMREX_D_DECL(10, 10, 10)};
    const amrex::IntVect maxGridSize = {AMREX_D_DECL(10, 10, 10)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {1, 1, 1};
    const int hodgeDegree = 2;

	Parameters params(realBox, nCell, maxGridSize, isPeriodic, hodgeDegree);
    const amrex::Geometry geom = params.geometry();

    auto deRham = std::make_shared<FDDeRhamComplex>(params);

    // Declare the fields 
	DeRhamField<Grid::primal, Space::edge> E(deRham);
	DeRhamField<Grid::primal, Space::face> B(deRham);
	DeRhamField<Grid::dual, Space::edge> H(deRham);
	DeRhamField<Grid::dual, Space::face> D(deRham);

    const amrex::Array<std::string, 3> analyticalB = {"(x^2)*(y^0)*(z^0)", 
                                                      "(x^2)*(y^0)*(z^0)",
                                                      "(x^2)*(y^0)*(z^0)"};

    const amrex::Array<std::string, 3> analyticalH = {"(x^2)*(y^0)*(z^0)", 
                                                      "(x^2)*(y^0)*(z^0)",
                                                      "(x^2)*(y^0)*(z^0)"};

    const amrex::Array<std::string, 3> analyticalE = {"(x^2)*(y^0)*(z^0)", 
                                                      "(x^2)*(y^0)*(z^0)",
                                                      "(x^2)*(y^0)*(z^0)"};

    const amrex::Array<std::string, 3> analyticalD = {"(x^2)*(y^0)*(z^0)", 
                                                      "(x^2)*(y^0)*(z^0)",
                                                      "(x^2)*(y^0)*(z^0)"};



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
    deRham -> hodgeFD<hodgeDegree>(B,H);
 
    
    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalH[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }
    
    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalD[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }

    deRham -> projection(func, 0.0, D);
    deRham -> hodgeFD<hodgeDegree>(D,E);

    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalE[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }

	DeRhamField<Grid::dual, Space::edge> dualOneForm(deRham);
	DeRhamField<Grid::primal, Space::edge> primalOneForm(deRham);
    deRham -> projection(func, 0.0, dualOneForm);
    deRham -> projection(func, 0.0, primalOneForm);
    
    // Visualize D -> E
    for (int comp = 0; comp < 3; ++comp)
    {
        amrex::Print() << "comp: " << comp << std::endl;
        for (amrex::MFIter mfi(D.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &DMF= (D.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &hodgeMF= (E.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &oneFormMF = (primalOneForm.data[comp])[mfi].array();

            for (int k = lo.z; k <= hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                    for (int i = lo.x; i <= hi.x; ++i) 
                        amrex::Print() << "(" << i << "," << j << "," << k << ") D: " << DMF(i, j, k) << " HD: " << hodgeMF(i, j, k) << " E: " << oneFormMF(i, j, k) << std::endl;
        }
    }

    // Visualize B -> H
    for (int comp = 0; comp < 3; ++comp)
    {
        amrex::Print() << "comp: " << comp << std::endl;
        for (amrex::MFIter mfi(H.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &BMF= (B.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &hodgeMF= (H.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &oneFormMF = (dualOneForm.data[comp])[mfi].array();

            for (int k = lo.z; k <= hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                    for (int i = lo.x; i <= hi.x; ++i) 
                        amrex::Print() << "(" << i << "," << j << "," << k << ") B: " << BMF(i, j, k) << " HB: " << hodgeMF(i, j, k) << " H: " << oneFormMF(i, j, k) << std::endl;
        }
    }

    // Calculate error
    /*
	DeRhamField<Grid::primal, Space::edge> errorE(deRham);
	DeRhamField<Grid::dual, Space::edge> errorH(deRham);

    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(errorE.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &hodgeMF = (E.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &oneFormMF = (primalOneForm.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorMF = (errorE.data[comp])[mfi].array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                if ((i < lo.x + 2 || j < lo.y + 2 || k < lo.z + 2) || (i >= hi.x - 2 || j >= hi.y - 2 || k >= hi.z - 2))
                    errorMF(i, j, k) = 0.0;
                else
                    errorMF(i, j, k) = std::abs(hodgeMF(i, j, k) - oneFormMF(i, j, k));
            });
        }

        for (amrex::MFIter mfi(errorH.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &hodgeMF = (H.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &oneFormMF = (dualOneForm.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorMF = (errorH.data[comp])[mfi].array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                if ((i < lo.x + 2 || j < lo.y + 2 || k < lo.z + 2) || (i >= hi.x - 2 || j >= hi.y - 2 || k >= hi.z - 2))
                    errorMF(i, j, k) = 0.0;
                else
                    errorMF(i, j, k) = std::abs(hodgeMF(i, j, k) - oneFormMF(i, j, k));
            });
        }
    }

    bool passE = false;
    if (errorE.data[0].norm0() < GEMPIC_CTEST_TOL &&
        errorE.data[1].norm0() < GEMPIC_CTEST_TOL &&
        errorE.data[2].norm0() < GEMPIC_CTEST_TOL)
        passE = true;
    
    bool passH = false;
    if (errorH.data[0].norm0() < GEMPIC_CTEST_TOL &&
        errorH.data[1].norm0() < GEMPIC_CTEST_TOL &&
        errorH.data[2].norm0() < GEMPIC_CTEST_TOL)
        passH = true;

    if (passE == true && passH == true)
    {
        amrex::PrintToFile("test_hodge_degree2.output") << std::endl;
        amrex::PrintToFile("test_hodge_degree2.output") << true << std::endl;
    }
    else
    {
        amrex::PrintToFile("test_hodge_degree2.output") << std::endl;
        for (int comp = 0; comp < 3; ++comp)
            amrex::PrintToFile("test_hodge_degree2.output") << "errorE[" << comp << "] = " << errorE.data[comp].norm0() <<  " || errorH[" << comp << "] = " << errorH.data[comp].norm0() << std::endl;
        amrex::PrintToFile("test_hodge_degree2.output") << false << std::endl;
    }
    */

    // Print additional visualization of output
    // D, HD and E
    amrex::PrintToFile("test_hodge_degree2.output") << std::endl;
    for (int comp = 0; comp < 2; ++comp)
    {
        amrex::PrintToFile("test_hodge_degree2.output") << "comp: " << comp << std::endl;
        for (amrex::MFIter mfi(D.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &DMF= (D.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &hodgeMF= (E.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &oneFormMF = (primalOneForm.data[comp])[mfi].array();

            for (int i = lo.x; i <= hi.x; ++i) 
                        amrex::PrintToFile("test_hodge_degree2.output") << "(" << i << "," << 0 << "," << 0 << ") D: " << DMF(i, 0, 0) << " HD: " << hodgeMF(i, 0, 0) << " E: " << oneFormMF(i, 0, 0) << std::endl;
        }
    }

    // B, HB and H
    amrex::PrintToFile("test_hodge_degree2.output") << std::endl;
    for (int comp = 0; comp < 2; ++comp)
    {
        amrex::PrintToFile("test_hodge_degree2.output") << "comp: " << comp << std::endl;
        for (amrex::MFIter mfi(H.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &BMF= (B.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &hodgeMF= (H.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &oneFormMF = (dualOneForm.data[comp])[mfi].array();

            for (int i = lo.x; i <= hi.x; ++i) 
                        amrex::PrintToFile("test_hodge_degree2.output") << "(" << i << "," << 0 << "," << 0 << ") B: " << BMF(i, 0, 0) << " HB: " << hodgeMF(i, 0, 0) << " H: " << oneFormMF(i, 0, 0) << std::endl;
        }
    }

    if (amrex::ParallelDescriptor::MyProc() == 0)
        std::rename("test_hodge_degree2.output.0", "test_hodge_degree2.output");
    amrex::Print() << "IOProcessorNumber " << amrex::ParallelDescriptor::IOProcessorNumber() << std::endl;

    amrex::Finalize();
}
