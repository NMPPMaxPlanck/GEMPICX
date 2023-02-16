#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_FDDeRhamComplex.H>

using namespace GEMPIC_Fields;
using namespace GEMPIC_FDDeRhamComplex;

void hodgeC2ToC1()
{
    /* Initialize the infrastructure */
    const amrex::RealBox realBox({AMREX_D_DECL(-M_PI, -M_PI, -M_PI)},{AMREX_D_DECL(M_PI, M_PI, M_PI)});
	const amrex::IntVect nCell = {AMREX_D_DECL(32, 32, 32)};
    const amrex::IntVect maxGridSize = {AMREX_D_DECL(32, 32, 32)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {1, 1, 1};
    const int hodgeDegree = 4;

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
    deRham -> hodgeFD<hodgeDegree>(B,H);

    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalD[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }

    deRham -> projection(func, 0.0, D);
    deRham -> hodgeFD<hodgeDegree>(D,E);

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
    
    // Calculate errors
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

            // Values at the boundaries are set to zero artificially in order to avoid problems
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

            // Values at the boundaries are set to zero artificially in order to avoid problems
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
        amrex::Print() << "errorH[" << comp << "]: " << errorH.data[comp].norm0() << std::endl;
        amrex::Print() << "errorE[" << comp << "]: " << errorE.data[comp].norm0() << std::endl;
    }
}

void hodgeC1ToC2()
{
    /* Initialize the infrastructure */
    const amrex::RealBox realBox({AMREX_D_DECL(-M_PI, -M_PI, -M_PI)},{AMREX_D_DECL(M_PI, M_PI, M_PI)});
	const amrex::IntVect nCell = {AMREX_D_DECL(32, 32, 32)};
    const amrex::IntVect maxGridSize = {AMREX_D_DECL(32, 32, 32)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {1, 1, 1};
    const int hodgeDegree = 4;

	Parameters params(realBox, nCell, maxGridSize, isPeriodic, hodgeDegree);
    const amrex::Geometry geom = params.geometry();

    auto deRham = std::make_shared<FDDeRhamComplex>(params);

	DeRhamField<Grid::dual, Space::edge> H(deRham);
	DeRhamField<Grid::primal, Space::face> B(deRham);
	DeRhamField<Grid::primal, Space::edge> E(deRham);
	DeRhamField<Grid::dual, Space::face> D(deRham);

    const amrex::Array<std::string, 3> analyticalH = {"cos(x)*cos(y)*cos(z)", 
                                                      "cos(x)*cos(y)*cos(z)",
                                                      "cos(x)*cos(y)*cos(z)"};
    
    const amrex::Array<std::string, 3> analyticalE = {"cos(x)*cos(y)*cos(z)", 
                                                      "cos(x)*cos(y)*cos(z)",
                                                      "cos(x)*cos(y)*cos(z)"};


    const int nVar = 4; //x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> func; 
    amrex::Array<amrex::Parser, GEMPIC_SPACEDIM> parser;
    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalH[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }
    
    deRham -> projection(func, 0.0, H);
    deRham -> hodgeFD<hodgeDegree>(H,B);
    
    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalE[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }

    deRham -> projection(func, 0.0, E);
    deRham -> hodgeFD<hodgeDegree>(E,D);

    const amrex::Array<std::string, 3> analyticalB = {"cos(x)*cos(y)*cos(z)", 
                                                      "cos(x)*cos(y)*cos(z)",
                                                      "cos(x)*cos(y)*cos(z)"};

    const amrex::Array<std::string, 3> analyticalD = {"cos(x)*cos(y)*cos(z)", 
                                                      "cos(x)*cos(y)*cos(z)",
                                                      "cos(x)*cos(y)*cos(z)"};
	for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalB[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }

    DeRhamField<Grid::primal, Space::face> primalTwoForm(deRham);
    deRham -> projection(func, 0.0, primalTwoForm);

    for (int i=0; i<3; ++i)
    {
        parser[i].define(analyticalD[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        func[i] = parser[i].compile<4>();
    }
	
    DeRhamField<Grid::dual, Space::face> dualTwoForm(deRham);
    deRham -> projection(func, 0.0, dualTwoForm);

    DeRhamField<Grid::primal, Space::face> errorB(deRham);
    DeRhamField<Grid::dual, Space::face> errorD(deRham);

    for (int comp = 0; comp < GEMPIC_SPACEDIM; ++comp)
    {
        for (amrex::MFIter mfi(B.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &hodgeMF = (B.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &analyticalMF = (primalTwoForm.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorMF = (errorB.data[comp])[mfi].array();

            // Values at the boundaries are set to zero artificially in order to avoid problems
            // with the Boundary Conditions
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                errorMF(i, j, k) = std::abs(hodgeMF(i, j, k) - analyticalMF(i, j, k));
            });
        }

        for (amrex::MFIter mfi(D.data[comp]); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto lo = lbound(bx);
            const auto hi = ubound(bx);

            amrex::Array4<amrex::Real> const &hodgeMF = (D.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &analyticalMF = (dualTwoForm.data[comp])[mfi].array();
            amrex::Array4<amrex::Real> const &errorMF = (errorD.data[comp])[mfi].array();

            // Values at the boundaries are set to zero artificially in order to avoid problems
            // with the Boundary Conditions
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                errorMF(i, j, k) = std::abs(hodgeMF(i, j, k) - analyticalMF(i, j, k));
            });
        }
        amrex::Print() << "errorB[" << comp << "] = " << errorB.data[comp].norm0() << std::endl;
        amrex::Print() << "errorD[" << comp << "] = " << errorD.data[comp].norm0() << std::endl;
    }

}

void hodgeC3ToC0()
{
    /* Initialize the infrastructure */
    const amrex::RealBox realBox({AMREX_D_DECL(-M_PI, -M_PI, -M_PI)},{AMREX_D_DECL(M_PI, M_PI, M_PI)});
	const amrex::IntVect nCell = {AMREX_D_DECL(32, 32, 32)};
    const amrex::IntVect maxGridSize = {AMREX_D_DECL(32, 32, 32)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {1, 1, 1};
    const int hodgeDegree = 4;

	Parameters params(realBox, nCell, maxGridSize, isPeriodic, hodgeDegree);
    const amrex::Geometry geom = params.geometry();

    auto deRham = std::make_shared<FDDeRhamComplex>(params);

    // Declare the fields 
	DeRhamField<Grid::dual, Space::cell> df3(deRham);
	DeRhamField<Grid::primal, Space::node> f0(deRham);
	DeRhamField<Grid::primal, Space::cell> f3(deRham);
	DeRhamField<Grid::dual, Space::node> df0(deRham);
    
    // Using (xyz)^5 polynomials as test case
    const std::string analyticalDF3 = "cos(x)*cos(y)*cos(z)";
    const std::string analyticalF3 = "cos(x)*cos(y)*cos(z)";

    const int nVar = 4; //x, y, z, t
    amrex::ParserExecutor<nVar> func; 
    amrex::Parser parser;

    parser.define(analyticalDF3);
    parser.registerVariables({"x", "y", "z", "t"});
    func = parser.compile<4>();

    deRham -> projection(func, 0.0, df3);
    deRham -> hodgeFD<hodgeDegree>(df3,f0);
    
    parser.define(analyticalF3);
    parser.registerVariables({"x", "y", "z", "t"});
    func = parser.compile<4>();

    deRham -> projection(func, 0.0, f3);
    deRham -> hodgeFD<hodgeDegree>(f3,df0);

    const std::string analyticalF0 = "cos(x)*cos(y)*cos(z)";
    const std::string analyticalDF0 = "cos(x)*cos(y)*cos(z)";

    parser.define(analyticalF0);
    parser.registerVariables({"x", "y", "z", "t"});
    func = parser.compile<4>();

    DeRhamField<Grid::primal, Space::node> primalZeroForm(deRham);
    deRham -> projection(func, 0.0, primalZeroForm);

    parser.define(analyticalDF0);
    parser.registerVariables({"x", "y", "z", "t"});
    func = parser.compile<4>();

    DeRhamField<Grid::dual, Space::node> dualZeroForm(deRham);
    deRham -> projection(func, 0.0, dualZeroForm);

    DeRhamField<Grid::primal, Space::node> errorF0(deRham);
    DeRhamField<Grid::dual, Space::node> errorDF0(deRham);

    for (amrex::MFIter mfi(f0.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        const auto lo = lbound(bx);
        const auto hi = ubound(bx);

        amrex::Array4<amrex::Real> const &hodgeMF = (f0.data)[mfi].array();
        amrex::Array4<amrex::Real> const &analyticalMF = (primalZeroForm.data)[mfi].array();
        amrex::Array4<amrex::Real> const &errorMF = (errorF0.data)[mfi].array();

        // Values at the boundaries are set to zero artificially in order to avoid problems
        // with the Boundary Conditions
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            if ((i < lo.x + 2 || j < lo.y + 2 || k < lo.z + 2) || (i >= hi.x - 2 || j >= hi.y - 2 || k >= hi.z - 2))
                errorMF(i, j, k) = 0.0;
            else
                errorMF(i, j, k) = std::abs(hodgeMF(i, j, k) - analyticalMF(i, j, k));
        });

    }

    for (amrex::MFIter mfi(df0.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        const auto lo = lbound(bx);
        const auto hi = ubound(bx);

        amrex::Array4<amrex::Real> const &hodgeMF = (df0.data)[mfi].array();
        amrex::Array4<amrex::Real> const &analyticalMF = (dualZeroForm.data)[mfi].array();
        amrex::Array4<amrex::Real> const &errorMF = (errorDF0.data)[mfi].array();

        // Values at the boundaries are set to zero artificially in order to avoid problems
        // with the Boundary Conditions
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            if ((i < lo.x + 2 || j < lo.y + 2 || k < lo.z + 2) || (i >= hi.x - 2 || j >= hi.y - 2 || k >= hi.z - 2))
                errorMF(i, j, k) = 0.0;
            else
                errorMF(i, j, k) = std::abs(hodgeMF(i, j, k) - analyticalMF(i, j, k));
        });

    }

    amrex::Print() << "errorF0: " << errorF0.data.norm0() << std::endl;
    amrex::Print() << "errorDF0: " << errorDF0.data.norm0() << std::endl;
}


void hodgeC0ToC3()
{
    /* Initialize the infrastructure */
    const amrex::RealBox realBox({AMREX_D_DECL(-M_PI, -M_PI, -M_PI)},{AMREX_D_DECL(M_PI, M_PI, M_PI)});
	const amrex::IntVect nCell = {AMREX_D_DECL(32, 32, 32)};
    const amrex::IntVect maxGridSize = {AMREX_D_DECL(32, 32, 32)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {1, 1, 1};
    const int hodgeDegree = 4;

	Parameters params(realBox, nCell, maxGridSize, isPeriodic, hodgeDegree);
    const amrex::Geometry geom = params.geometry();

    auto deRham = std::make_shared<FDDeRhamComplex>(params);

    // Declare the fields 
	DeRhamField<Grid::dual, Space::node> df0(deRham);
	DeRhamField<Grid::primal, Space::cell> f3(deRham);
	DeRhamField<Grid::primal, Space::node> f0(deRham);
	DeRhamField<Grid::dual, Space::cell> df3(deRham);
    
    // Using (xyz)^5 polynomials as test case
    const std::string analyticalDF0 = "cos(x)*cos(y)*cos(z)";
    const std::string analyticalF0 = "cos(x)*cos(y)*cos(z)";

    const int nVar = 4; //x, y, z, t
    amrex::ParserExecutor<nVar> func; 
    amrex::Parser parser;

    parser.define(analyticalDF0);
    parser.registerVariables({"x", "y", "z", "t"});
    func = parser.compile<4>();

    deRham -> projection(func, 0.0, df0);
    deRham -> hodgeFD<hodgeDegree>(df0,f3);
    
    parser.define(analyticalF0);
    parser.registerVariables({"x", "y", "z", "t"});
    func = parser.compile<4>();

    deRham -> projection(func, 0.0, f0);
    deRham -> hodgeFD<hodgeDegree>(f0,df3);

    const std::string analyticalF3 = "cos(x)*cos(y)*cos(z)";
    const std::string analyticalDF3 = "cos(x)*cos(y)*cos(z)";

    parser.define(analyticalF3);
    parser.registerVariables({"x", "y", "z", "t"});
    func = parser.compile<4>();

    DeRhamField<Grid::primal, Space::cell> primalThreeForm(deRham);
    deRham -> projection(func, 0.0, primalThreeForm);

    parser.define(analyticalDF3);
    parser.registerVariables({"x", "y", "z", "t"});
    func = parser.compile<4>();

    DeRhamField<Grid::dual, Space::cell> dualThreeForm(deRham);
    deRham -> projection(func, 0.0, dualThreeForm);

    DeRhamField<Grid::primal, Space::cell> errorF3(deRham);
    DeRhamField<Grid::dual, Space::cell> errorDF3(deRham);

    for (amrex::MFIter mfi(f3.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        const auto lo = lbound(bx);
        const auto hi = ubound(bx);

        amrex::Array4<amrex::Real> const &hodgeMF = (f3.data)[mfi].array();
        amrex::Array4<amrex::Real> const &analyticalMF = (primalThreeForm.data)[mfi].array();
        amrex::Array4<amrex::Real> const &errorMF = (errorF3.data)[mfi].array();

        // Values at the boundaries are set to zero artificially in order to avoid problems
        // with the Boundary Conditions
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            if ((i < lo.x + 2 || j < lo.y + 2 || k < lo.z + 2) || (i >= hi.x - 2 || j >= hi.y - 2 || k >= hi.z - 2))
                errorMF(i, j, k) = 0.0;
            else
                errorMF(i, j, k) = std::abs(hodgeMF(i, j, k) - analyticalMF(i, j, k));
        });

    }

    for (amrex::MFIter mfi(df3.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        const auto lo = lbound(bx);
        const auto hi = ubound(bx);

        amrex::Array4<amrex::Real> const &hodgeMF = (df3.data)[mfi].array();
        amrex::Array4<amrex::Real> const &analyticalMF = (dualThreeForm.data)[mfi].array();
        amrex::Array4<amrex::Real> const &errorMF = (errorDF3.data)[mfi].array();

        // Values at the boundaries are set to zero artificially in order to avoid problems
        // with the Boundary Conditions
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            if ((i < lo.x + 2 || j < lo.y + 2 || k < lo.z + 2) || (i >= hi.x - 2 || j >= hi.y - 2 || k >= hi.z - 2))
                errorMF(i, j, k) = 0.0;
            else
                errorMF(i, j, k) = std::abs(hodgeMF(i, j, k) - analyticalMF(i, j, k));
        });

    }

    amrex::Print() << "errorF3: " << errorF3.data.norm0() << std::endl;
    amrex::Print() << "errorDF3: " << errorDF3.data.norm0() << std::endl;
}

int main (int argc, char *argv[]) 
{
	amrex::Initialize(argc, argv); 

    hodgeC2ToC1();
    
    hodgeC1ToC2();
    
    hodgeC3ToC0();
    
    hodgeC0ToC3();

    amrex::Finalize();
}

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
            amrex::Array4<amrex::Real> const &oneFormMF = (dualOneForm.data[comp])[mfi].array();

            for (int k = lo.z; k < hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                    for (int i = lo.x; i <= hi.x; ++i) 
                        amrex::Print() << "(" << i << "," << j << "," << k << ") B: " << BMF(i, j, k) << " HB: " << hodgeMF(i, j, k) << " H: " << oneFormMF(i, j, k) << std::endl;
        }
    }

    for (amrex::MFIter mfi(f0.data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        const auto lo = lbound(bx);
        const auto hi = ubound(bx);

        amrex::Array4<amrex::Real> const &df3MF = (df3.data)[mfi].array();
        amrex::Array4<amrex::Real> const &hodgeMF = (f0.data)[mfi].array();
        amrex::Array4<amrex::Real> const &zeroFormMF = (primalZeroForm.data)[mfi].array();

        for (int k = lo.z; k < hi.z; ++k)
            for (int j = lo.y; j <= hi.y; ++j)
                for (int i = lo.x; i <= hi.x; ++i) 
                    amrex::Print() << "(" << i << "," << j << "," << k << ") df3: " << df3MF(i, j, k) << " H(df3): " << hodgeMF(i, j, k) << " f0: " << zeroFormMF(i, j, k) << std::endl;
    }
    */
