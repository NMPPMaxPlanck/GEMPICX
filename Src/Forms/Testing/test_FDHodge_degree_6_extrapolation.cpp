/*------------------------------------------------------------------------------
 Test Finite Difference Hodge convergence rates with non periodic boundaries (extrapolation)

  Computes the max norm ||I_{3-k} H_k R_k f - 1/omega f|| at the cell mid points
  where f(x,y,z) =  cos(2*pi*(x-0.2))*sin(4*pi*(y-0.6))*cos(2*pi*z) + sin(2*pi*x -
0.4)*sin(2*pi*(y-0.1))*sin(4*pi*z + 0.3) for k = 0,3

  or                cos(2*pi*(x-0.2))*sin(4*pi*(y-0.6))*cos(2*pi*z) + sin(2*pi*x -
0.4)*sin(2*pi*(y-0.1))*sin(4*pi*z + 0.3) f(x,y,z) = (sin(2*pi*(x+0.3))*cos(2*pi*(y-0.6))*cos(4*pi*z)
+ cos(2*pi*x - 0.2)*sin(2*pi*(y+0.6))*sin(2*pi*z - 0.4)) for k = 1,2
                    cos(4*pi*(x-0.6))*cos(2*pi*(y+0.3))*sin(2*pi*z) + sin(2*pi*x +
0.3)*sin(2*pi*(y-0.3))*cos(2*pi*z + 0.6)

  for 16 and 32 nodes (1D and 2D: 64/128) in each direction. The convergence rate is estimated
by log_2 (error_coarse / error_fine)
------------------------------------------------------------------------------*/
#include <cmath>
#include <map>
#include <string>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Interpolation.H"
#include "GEMPIC_Parameters.H"

using namespace Gempic::Forms;

const int hodgeDegree = 6;
const int maxSplineDegree = 1;

std::map<int, std::string> hodges{
    {0, "Hodge 0 -> 3"}, {1, "Hodge 1 -> 2"}, {2, "Hodge 2 -> 1"}, {3, "Hodge 3 -> 0"}};

amrex::Real test12 (int n)
{
    BL_PROFILE("Gempic::Forms::amrex::Real test12()");
    // Initialize the infrastructure
    // const amrex::RealBox realBox({AMREX_D_DECL(0.3, 0.6, 0.4)},{AMREX_D_DECL(1.3, 1.6, 1.4)});
    const amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.3, 0.6, 0.4)};
    const amrex::Vector<amrex::Real> k{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
    const amrex::Vector<int> nCell{AMREX_D_DECL(n, n, n)};
    const amrex::Vector<int> maxGridSize{AMREX_D_DECL(8, 6, 9)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(0, 0, 0)};

    Gempic::Io::Parameters parameters{};
    parameters.set("domainLo", domainLo);
    parameters.set("k", k);
    parameters.set("nCellVector", nCell);
    parameters.set("maxGridSizeVector", maxGridSize);
    parameters.set("isPeriodicVector", isPeriodic);

    // Initialize computational_domain
    Gempic::ComputationalDomain infra;

    const amrex::Geometry geom = infra.m_geom;

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    // Declare the fields
    DeRhamField<Grid::dual, Space::face> dualTwoForm(deRham);
    DeRhamField<Grid::primal, Space::edge> primalOneForm(deRham);

#if (GEMPIC_SPACEDIM == 1)
    const amrex::Array<std::string, 3> func = {
        "pi = 3.141592653589793; (cos(2*pi*x) + sin(2*pi*x - 0.2))",
        "pi = 3.141592653589793; (sin(2*pi*x) + cos(2*pi*x - 0.2))",
        "pi = 3.141592653589793; (cos(2*2*pi*x) + sin(2*pi*x - 0.2))",
    };
#endif
#if (GEMPIC_SPACEDIM == 2)
    const amrex::Array<std::string, 3> func = {
        "pi = 3.141592653589793; (cos(2*pi*x)*sin(2*2*pi*y) + sin(2*pi*x - 0.2)*sin(2*pi*y))",
        "pi = 3.141592653589793; (sin(2*pi*x)*cos(2*pi*y) + cos(2*pi*x - 0.2)*sin(2*pi*y))",
        "pi = 3.141592653589793; (cos(2*2*pi*x)*cos(2*pi*y) + sin(2*pi*x - 0.2)*sin(2*pi*y))",
    };
#endif
#if (GEMPIC_SPACEDIM == 3)
    const amrex::Array<std::string, 3> func = {
        "pi = 3.141592653589793; (cos(2*pi*x)*sin(2*2*pi*y)*cos(2*pi*z) + sin(2*pi*x - "
        "0.2)*sin(2*pi*y)*sin(2*2*pi*z + 1.3))",
        "pi = 3.141592653589793; (sin(2*pi*x)*cos(2*pi*y)*cos(2*2*pi*z) + cos(2*pi*x - "
        "0.2)*sin(2*pi*y)*sin(2*2*pi*z + 1.3))",
        "pi = 3.141592653589793; (cos(2*2*pi*x)*cos(2*pi*y)*sin(2*pi*z) + sin(2*pi*x - "
        "0.2)*sin(2*pi*y)*cos(2*2*pi*z + 1.3))",
    };
#endif

    const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcP;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(func[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcP[i] = parser[i].compile<GEMPIC_SPACEDIM + 1>();
    }

    deRham->projection(funcP, 0.0, primalOneForm);
    primalOneForm.apply_bc();
    deRham->hodge(dualTwoForm, primalOneForm);
    dualTwoForm.apply_bc();
    deRham->hodge(primalOneForm, dualTwoForm);
    primalOneForm.apply_bc();

    amrex::Real e = 0;
    for (int comp = 0; comp < 3; ++comp)
    {
        e += max_error_midpoint<hodgeDegree>(
            geom, funcP[comp], primalOneForm.m_data[comp],
            amrex::RealVect{AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])}, 1,
            false, comp);
    }
    return e;
}

amrex::Real test21 (int n)
{
    BL_PROFILE("Gempic::Forms::amrex::Real test21()");
    // Initialize the infrastructure
    // const amrex::RealBox realBox({AMREX_D_DECL(0.3, 0.6, 0.4)},{AMREX_D_DECL(1.3, 1.6, 1.4)});
    const amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.3, 0.6, 0.4)};
    const amrex::Vector<amrex::Real> k{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
    const amrex::Vector<int> nCell{AMREX_D_DECL(n, n, n)};
    const amrex::Vector<int> maxGridSize{AMREX_D_DECL(8, 6, 9)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(0, 0, 0)};

    Gempic::Io::Parameters parameters{};
    parameters.set("domainLo", domainLo);
    parameters.set("k", k);
    parameters.set("nCellVector", nCell);
    parameters.set("maxGridSizeVector", maxGridSize);
    parameters.set("isPeriodicVector", isPeriodic);

    // Initialize computational_domain
    Gempic::ComputationalDomain infra;

    const amrex::Geometry geom = infra.m_geom;

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    // Declare the fields
    DeRhamField<Grid::primal, Space::face> primalTwoForm(deRham);
    DeRhamField<Grid::dual, Space::edge> dualOneForm(deRham);

#if (GEMPIC_SPACEDIM == 1)
    const amrex::Array<std::string, 3> func = {
        "pi = 3.141592653589793; (cos(2*pi*x) + sin(2*pi*x - 0.2))",
        "pi = 3.141592653589793; (sin(2*pi*x) + cos(2*pi*x - 0.2))",
        "pi = 3.141592653589793; (cos(2*2*pi*x) + sin(2*pi*x - 0.2))",
    };
#endif
#if (GEMPIC_SPACEDIM == 2)
    const amrex::Array<std::string, 3> func = {
        "pi = 3.141592653589793; (cos(2*pi*x)*sin(2*2*pi*y) + sin(2*pi*x - 0.2)*sin(2*pi*y))",
        "pi = 3.141592653589793; (sin(2*pi*x)*cos(2*pi*y) + cos(2*pi*x - 0.2)*sin(2*pi*y))",
        "pi = 3.141592653589793; (cos(2*2*pi*x)*cos(2*pi*y) + sin(2*pi*x - 0.2)*sin(2*pi*y))",
    };
#endif
#if (GEMPIC_SPACEDIM == 3)
    const amrex::Array<std::string, 3> func = {
        "pi = 3.141592653589793; (cos(2*pi*x)*sin(2*2*pi*y)*cos(2*pi*z) + sin(2*pi*x - "
        "0.2)*sin(2*pi*y)*sin(2*2*pi*z + 1.3))",
        "pi = 3.141592653589793; (sin(2*pi*x)*cos(2*pi*y)*cos(2*2*pi*z) + cos(2*pi*x - "
        "0.2)*sin(2*pi*y)*sin(2*2*pi*z + 1.3))",
        "pi = 3.141592653589793; (cos(2*2*pi*x)*cos(2*pi*y)*sin(2*pi*z) + sin(2*pi*x - "
        "0.2)*sin(2*pi*y)*cos(2*2*pi*z + 1.3))",
    };
#endif

    const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcP;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(func[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcP[i] = parser[i].compile<GEMPIC_SPACEDIM + 1>();
    }

    deRham->projection(funcP, 0.0, primalTwoForm);
    primalTwoForm.apply_bc();
    deRham->hodge(dualOneForm, primalTwoForm);
    dualOneForm.apply_bc();
    deRham->hodge(primalTwoForm, dualOneForm);
    primalTwoForm.apply_bc();

    amrex::Real e = 0;
    for (int comp = 0; comp < 3; ++comp)
    {
        e += max_error_midpoint<hodgeDegree>(
            geom, funcP[comp], primalTwoForm.m_data[comp],
            amrex::RealVect{AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])}, 2,
            false, comp);
    }
    return e;
}

amrex::Real test03 (int n)
{
    BL_PROFILE("Gempic::Forms::amrex::Real test03()");
    // Initialize the infrastructure
    // const amrex::RealBox realBox({AMREX_D_DECL(0.3, 0.6, 0.4)},{AMREX_D_DECL(1.3, 1.6, 1.4)});
    const amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.3, 0.6, 0.4)};
    const amrex::Vector<amrex::Real> k{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
    const amrex::Vector<int> nCell{AMREX_D_DECL(n, n, n)};
    const amrex::Vector<int> maxGridSize{AMREX_D_DECL(8, 6, 9)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(0, 0, 0)};

    Gempic::Io::Parameters parameters{};
    parameters.set("domainLo", domainLo);
    parameters.set("k", k);
    parameters.set("nCellVector", nCell);
    parameters.set("maxGridSizeVector", maxGridSize);
    parameters.set("isPeriodicVector", isPeriodic);

    // Initialize computational_domain
    Gempic::ComputationalDomain infra;

    const amrex::Geometry geom = infra.m_geom;

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    // Declare the fields
    DeRhamField<Grid::primal, Space::node> primalZeroForm(deRham);
    DeRhamField<Grid::dual, Space::cell> dualThreeForm(deRham);

#if (GEMPIC_SPACEDIM == 1)
    const std::string func = "pi = 3.141592653589793; (cos(2*pi*x) + sin(2*pi*x - 0.2))";
#endif
#if (GEMPIC_SPACEDIM == 2)
    const std::string func =
        "pi = 3.141592653589793; (cos(2*pi*x)*sin(2*2*pi*y) + sin(2*pi*x - 0.2)*sin(2*pi*y))";
#endif
#if (GEMPIC_SPACEDIM == 3)
    const std::string func =
        "pi = 3.141592653589793; (cos(2*pi*x)*sin(2*2*pi*y)*cos(2*pi*z) + sin(2*pi*x - "
        "0.2)*sin(2*pi*y)*sin(2*2*pi*z + 1.3))";
#endif

    const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
    amrex::ParserExecutor<nVar> funcP;
    amrex::Parser parser;

    parser.define(func);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    funcP = parser.compile<GEMPIC_SPACEDIM + 1>();

    deRham->projection(funcP, 0.0, primalZeroForm);
    primalZeroForm.apply_bc();
    deRham->hodge(dualThreeForm, primalZeroForm);
    dualThreeForm.apply_bc();
    deRham->hodge(primalZeroForm, dualThreeForm);
    primalZeroForm.apply_bc();

    amrex::Real e = max_error_midpoint<hodgeDegree>(
        geom, funcP, primalZeroForm.m_data,
        amrex::RealVect{AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])}, 0,
        false);
    return e;
}

amrex::Real test30 (int n)
{
    BL_PROFILE("Gempic::Forms::amrex::Real test30()");
    // Initialize the infrastructure
    // const amrex::RealBox realBox({AMREX_D_DECL(0.3, 0.6, 0.4)},{AMREX_D_DECL(1.3, 1.6, 1.4)});
    const amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.3, 0.6, 0.4)};
    const amrex::Vector<amrex::Real> k{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
    const amrex::Vector<int> nCell{AMREX_D_DECL(n, n, n)};
    const amrex::Vector<int> maxGridSize{AMREX_D_DECL(8, 6, 9)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(0, 0, 0)};

    Gempic::Io::Parameters parameters{};
    parameters.set("domainLo", domainLo);
    parameters.set("k", k);
    parameters.set("nCellVector", nCell);
    parameters.set("maxGridSizeVector", maxGridSize);
    parameters.set("isPeriodicVector", isPeriodic);

    // Initialize computational_domain
    Gempic::ComputationalDomain infra;

    const amrex::Geometry geom = infra.m_geom;

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    // Declare the fields
    DeRhamField<Grid::primal, Space::cell> primalThreeForm(deRham);
    DeRhamField<Grid::dual, Space::node> dualZeroForm(deRham);

#if (GEMPIC_SPACEDIM == 1)
    const std::string func = "pi = 3.141592653589793; (cos(2*pi*x) + sin(2*pi*x - 0.2))";
#endif
#if (GEMPIC_SPACEDIM == 2)
    const std::string func =
        "pi = 3.141592653589793; (cos(2*pi*x)*sin(2*2*pi*y) + sin(2*pi*x - 0.2)*sin(2*pi*y))";
#endif
#if (GEMPIC_SPACEDIM == 3)
    const std::string func =
        "pi = 3.141592653589793; (cos(2*pi*x)*sin(2*2*pi*y)*cos(2*pi*z) + sin(2*pi*x - "
        "0.2)*sin(2*pi*y)*sin(2*2*pi*z + 1.3))";
#endif

    const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
    amrex::ParserExecutor<nVar> funcP;
    amrex::Parser parser;

    parser.define(func);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    funcP = parser.compile<GEMPIC_SPACEDIM + 1>();

    deRham->projection(funcP, 0.0, primalThreeForm);
    primalThreeForm.apply_bc();
    deRham->hodge(dualZeroForm, primalThreeForm);
    dualZeroForm.apply_bc();
    deRham->hodge(primalThreeForm, dualZeroForm);
    primalThreeForm.apply_bc();

    amrex::Real e = max_error_midpoint<hodgeDegree>(
        geom, funcP, primalThreeForm.m_data,
        amrex::RealVect{AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])}, 3,
        false);
    return e;
}

int main (int argc, char *argv[])
{
    amrex::Initialize(argc, argv);
    {
        BL_PROFILE("main()");

#if GEMPIC_SPACEDIM == 3
        const int coarse = 16;
        const int fine = 32;
#else
        const int coarse = 64;
        const int fine = 128;
#endif

        amrex::Real errorCoarse;
        amrex::Real errorFine;

        const int len = 4;
        amrex::GpuArray<amrex::Real, len> rate;

        errorCoarse = test03(coarse);
        errorFine = test03(fine);
        rate[0] = std::log2(errorCoarse / errorFine);

        errorCoarse = test12(coarse);
        errorFine = test12(fine);
        rate[1] = std::log2(errorCoarse / errorFine);

        errorCoarse = test21(coarse);
        errorFine = test21(fine);
        rate[2] = std::log2(errorCoarse / errorFine);

        errorCoarse = test30(coarse);
        errorFine = test30(fine);
        rate[3] = std::log2(errorCoarse / errorFine);

        amrex::PrintToFile("test_FDHodge_degree_6_extrapolation.output") << std::endl;
        amrex::PrintToFile("test_FDHodge_degree_6_extrapolation.output")
            << GEMPIC_SPACEDIM
            << "D Finite Difference Hodge degree 6 extrapolation convergence test:" << std::endl;
        amrex::PrintToFile("test_FDHodge_degree_6_extrapolation.output") << std::endl;
        for (int i = 0; i < len; ++i)
        {
            amrex::PrintToFile("test_FDHodge_degree_6_extrapolation.output").SetPrecision(3)
                << hodges[i] + " rate: " << rate[i] << std::endl;
        }

        if (amrex::ParallelDescriptor::MyProc() == 0)
        {
            std::rename("test_FDHodge_degree_6_extrapolation.output.0",
                        "test_FDHodge_degree_6_extrapolation.output");
        }
    }
    amrex::Finalize();
}
