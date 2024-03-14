/*------------------------------------------------------------------------------
 Test Finite Difference Hodge convergence rates

  Computes the max norm ||I_{3-k} H_k R_k f - 1/omega f|| at the cell mid points
  where f(x,y,z) =  cos(2*pi*(x-0.2))*sin(4*pi*(y-0.6))*cos(2*pi*z) + sin(2*pi*x -
0.4)*sin(2*pi*(y-0.1))*sin(4*pi*z + 0.3) for k = 0,3

  or                cos(2*pi*(x-0.2))*sin(4*pi*(y-0.6))*cos(2*pi*z) + sin(2*pi*x -
0.4)*sin(2*pi*(y-0.1))*sin(4*pi*z + 0.3) f(x,y,z) = (sin(2*pi*(x+0.3))*cos(2*pi*(y-0.6))*cos(4*pi*z)
+ cos(2*pi*x - 0.2)*sin(2*pi*(y+0.6))*sin(2*pi*z - 0.4)) for k = 1,2
                    cos(4*pi*(x-0.6))*cos(2*pi*(y+0.3))*sin(2*pi*z) + sin(2*pi*x +
0.3)*sin(2*pi*(y-0.3))*cos(2*pi*z + 0.6)

  for 16 and 32 nodes in each direction. The convergence rate is estimated by log_2 (error_16 /
error_32)
------------------------------------------------------------------------------*/
#include <cmath>
#include <map>
#include <string>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Interpolation.H"
#include "GEMPIC_Parameters.H"

using namespace Gempic::Forms;

const int hodgeDegree = 4;
const int maxSplineDegree = 1;

std::map<int, std::string> hodges{
    {0, "Hodge 1 -> 2"}, {1, "Hodge 2 -> 1"}, {2, "Hodge 0 -> 3"}, {3, "Hodge 3 -> 0"}};

amrex::Real test12 (int n)
{
    // Initialize the infrastructure
    // const amrex::RealBox realBox({AMREX_D_DECL(0.3, 0.6, 0.4)},{AMREX_D_DECL(1.3, 1.6, 1.4)});
    const amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.3, 0.6, 0.4)};
    const amrex::Vector<amrex::Real> k{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
    const amrex::Vector<int> nCell{AMREX_D_DECL(n, n, n)};
    const amrex::Vector<int> maxGridSize{AMREX_D_DECL(8, 6, 9)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};

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
    DeRhamField<Grid::dual, Space::face> dualTwoForm(deRham);
    DeRhamField<Grid::primal, Space::edge> primalOneForm(deRham);

    const amrex::Real weight = 2. / 3.;

#if (GEMPIC_SPACEDIM == 1)
    const amrex::Array<std::string, 3> func = {
        "pi = 3.141592653589793; w * (cos(2*pi*x) + sin(2*pi*x - 0.2))",
        "pi = 3.141592653589793; w * (sin(2*pi*x) + cos(2*pi*x - 0.2))",
        "pi = 3.141592653589793; w * (cos(2*2*pi*x) + sin(2*pi*x - 0.2))",
    };
#endif
#if (GEMPIC_SPACEDIM == 2)
    const amrex::Array<std::string, 3> func = {
        "pi = 3.141592653589793; w * (cos(2*pi*x)*sin(2*2*pi*y) + sin(2*pi*x - 0.2)*sin(2*pi*y))",
        "pi = 3.141592653589793; w * (sin(2*pi*x)*cos(2*pi*y) + cos(2*pi*x - 0.2)*sin(2*pi*y))",
        "pi = 3.141592653589793; w * (cos(2*2*pi*x)*cos(2*pi*y) + sin(2*pi*x - 0.2)*sin(2*pi*y))",
    };
#endif
#if (GEMPIC_SPACEDIM == 3)
    const amrex::Array<std::string, 3> func = {
        "pi = 3.141592653589793; w * (cos(2*pi*x)*sin(2*2*pi*y)*cos(2*pi*z) + sin(2*pi*x - "
        "0.2)*sin(2*pi*y)*sin(2*2*pi*z + 1.3))",
        "pi = 3.141592653589793; w * (sin(2*pi*x)*cos(2*pi*y)*cos(2*2*pi*z) + cos(2*pi*x - "
        "0.2)*sin(2*pi*y)*sin(2*2*pi*z + 1.3))",
        "pi = 3.141592653589793; w * (cos(2*2*pi*x)*cos(2*pi*y)*sin(2*pi*z) + sin(2*pi*x - "
        "0.2)*sin(2*pi*y)*cos(2*2*pi*z + 1.3))",
    };
#endif

    const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcP;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(func[i]);
        parser[i].setConstant("w", 1.);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcP[i] = parser[i].compile<GEMPIC_SPACEDIM + 1>();
    }

    deRham->projection(funcP, 0.0, dualOneForm);
    deRham->hodge(dualOneForm, primalTwoForm, weight);

    deRham->projection(funcP, 0.0, primalOneForm);
    deRham->hodge(primalOneForm, dualTwoForm, weight);

    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(func[i]);
        parser[i].setConstant("w", 1 / weight);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcP[i] = parser[i].compile<GEMPIC_SPACEDIM + 1>();
    }

    amrex::Real e1 = 0;
    amrex::Real e2 = 0;
    for (int comp = 0; comp < 3; ++comp)
    {
        e1 += max_error_midpoint<hodgeDegree>(
            geom, funcP[comp], primalTwoForm.m_data[comp],
            amrex::RealVect{AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])}, 2,
            false, comp);
        e2 += max_error_midpoint<hodgeDegree>(
            geom, funcP[comp], dualTwoForm.m_data[comp],
            amrex::RealVect{AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])}, 2,
            true, comp);
    }
    return std::max((e1), (e2));
}

amrex::Real test21 (int n)
{
    // Initialize the infrastructure
    // const amrex::RealBox realBox({AMREX_D_DECL(0.3, 0.6, 0.4)},{AMREX_D_DECL(1.3, 1.6, 1.4)});
    const amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.3, 0.6, 0.4)};
    const amrex::Vector<amrex::Real> k{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
    const amrex::Vector<int> nCell{AMREX_D_DECL(n, n, n)};
    const amrex::Vector<int> maxGridSize{AMREX_D_DECL(8, 6, 9)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};

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
    DeRhamField<Grid::dual, Space::face> dualTwoForm(deRham);
    DeRhamField<Grid::primal, Space::edge> primalOneForm(deRham);

    const amrex::Real weight = 2. / 3.;

#if (GEMPIC_SPACEDIM == 1)
    const amrex::Array<std::string, 3> func = {
        "pi = 3.141592653589793; w * (cos(2*pi*x) + sin(2*pi*x - 0.2))",
        "pi = 3.141592653589793; w * (sin(2*pi*x) + cos(2*pi*x - 0.2))",
        "pi = 3.141592653589793; w * (cos(2*2*pi*x) + sin(2*pi*x - 0.2))",
    };
#endif
#if (GEMPIC_SPACEDIM == 2)
    const amrex::Array<std::string, 3> func = {
        "pi = 3.141592653589793; w * (cos(2*pi*x)*sin(2*2*pi*y) + sin(2*pi*x - 0.2)*sin(2*pi*y))",
        "pi = 3.141592653589793; w * (sin(2*pi*x)*cos(2*pi*y) + cos(2*pi*x - 0.2)*sin(2*pi*y))",
        "pi = 3.141592653589793; w * (cos(2*2*pi*x)*cos(2*pi*y) + sin(2*pi*x - 0.2)*sin(2*pi*y))",
    };
#endif
#if (GEMPIC_SPACEDIM == 3)
    const amrex::Array<std::string, 3> func = {
        "pi = 3.141592653589793; w * (cos(2*pi*x)*sin(2*2*pi*y)*cos(2*pi*z) + sin(2*pi*x - "
        "0.2)*sin(2*pi*y)*sin(2*2*pi*z + 1.3))",
        "pi = 3.141592653589793; w * (sin(2*pi*x)*cos(2*pi*y)*cos(2*2*pi*z) + cos(2*pi*x - "
        "0.2)*sin(2*pi*y)*sin(2*2*pi*z + 1.3))",
        "pi = 3.141592653589793; w * (cos(2*2*pi*x)*cos(2*pi*y)*sin(2*pi*z) + sin(2*pi*x - "
        "0.2)*sin(2*pi*y)*cos(2*2*pi*z + 1.3))",
    };
#endif

    const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcP;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(func[i]);
        parser[i].setConstant("w", 1.);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcP[i] = parser[i].compile<GEMPIC_SPACEDIM + 1>();
    }

    deRham->projection(funcP, 0.0, primalTwoForm);
    deRham->hodge(primalTwoForm, dualOneForm, weight);

    deRham->projection(funcP, 0.0, dualTwoForm);
    deRham->hodge(dualTwoForm, primalOneForm, weight);

    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(func[i]);
        parser[i].setConstant("w", 1 / weight);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcP[i] = parser[i].compile<GEMPIC_SPACEDIM + 1>();
    }

    amrex::Real e1 = 0;
    amrex::Real e2 = 0;
    for (int comp = 0; comp < 3; ++comp)
    {
        e1 += max_error_midpoint<hodgeDegree>(
            geom, funcP[comp], primalOneForm.m_data[comp],
            amrex::RealVect{AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])}, 1,
            false, comp);
        e2 += max_error_midpoint<hodgeDegree>(
            geom, funcP[comp], dualOneForm.m_data[comp],
            amrex::RealVect{AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])}, 1,
            true, comp);
    }
    return std::max((e1), (e2));
}

amrex::Real test03 (int n)
{
    // Initialize the infrastructure
    // const amrex::RealBox realBox({AMREX_D_DECL(0.3, 0.6, 0.4)},{AMREX_D_DECL(1.3, 1.6, 1.4)});
    const amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.3, 0.6, 0.4)};
    const amrex::Vector<amrex::Real> k{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
    const amrex::Vector<int> nCell{AMREX_D_DECL(n, n, n)};
    const amrex::Vector<int> maxGridSize{AMREX_D_DECL(8, 6, 9)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};

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
    DeRhamField<Grid::dual, Space::cell> dualThreeForm(deRham);
    DeRhamField<Grid::primal, Space::node> primalZeroForm(deRham);

    const amrex::Real weight = 2. / 3.;

#if (GEMPIC_SPACEDIM == 1)
    const std::string func = "pi = 3.141592653589793; w * (cos(2*pi*x) + sin(2*pi*x - 0.2))";
#endif
#if (GEMPIC_SPACEDIM == 2)
    const std::string func =
        "pi = 3.141592653589793; w * (cos(2*pi*x)*sin(2*2*pi*y) + sin(2*pi*x - 0.2)*sin(2*pi*y))";
#endif
#if (GEMPIC_SPACEDIM == 3)
    const std::string func =
        "pi = 3.141592653589793; w * (cos(2*pi*x)*sin(2*2*pi*y)*cos(2*pi*z) + sin(2*pi*x - "
        "0.2)*sin(2*pi*y)*sin(2*2*pi*z + 1.3))";
#endif

    const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
    amrex::ParserExecutor<nVar> funcP;
    amrex::Parser parser;

    parser.define(func);
    parser.setConstant("w", 1.);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    funcP = parser.compile<GEMPIC_SPACEDIM + 1>();

    deRham->projection(funcP, 0.0, dualZeroForm);
    deRham->hodge(dualZeroForm, primalThreeForm, weight);

    deRham->projection(funcP, 0.0, primalZeroForm);
    deRham->hodge(primalZeroForm, dualThreeForm, weight);

    parser.define(func);
    parser.setConstant("w", 1 / weight);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    funcP = parser.compile<GEMPIC_SPACEDIM + 1>();

    amrex::Real e1 = max_error_midpoint<hodgeDegree>(
        geom, funcP, primalThreeForm.m_data,
        amrex::RealVect{AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])}, 3,
        false);
    amrex::Real e2 = max_error_midpoint<hodgeDegree>(
        geom, funcP, dualThreeForm.m_data,
        amrex::RealVect{AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])}, 3,
        true);
    return std::max((e1), (e2));
}

amrex::Real test30 (int n)
{
    // Initialize the infrastructure
    // const amrex::RealBox realBox({AMREX_D_DECL(0.3, 0.6, 0.4)},{AMREX_D_DECL(1.3, 1.6, 1.4)});
    const amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.3, 0.6, 0.4)};
    const amrex::Vector<amrex::Real> k{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
    const amrex::Vector<int> nCell{AMREX_D_DECL(n, n, n)};
    const amrex::Vector<int> maxGridSize{AMREX_D_DECL(8, 6, 9)};
    const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};

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
    DeRhamField<Grid::dual, Space::cell> dualThreeForm(deRham);
    DeRhamField<Grid::primal, Space::node> primalZeroForm(deRham);

    const amrex::Real weight = 2. / 3.;

#if (GEMPIC_SPACEDIM == 1)
    const std::string func = "pi = 3.141592653589793; w * (cos(2*pi*x) + sin(2*pi*x - 0.2))";
#endif
#if (GEMPIC_SPACEDIM == 2)
    const std::string func =
        "pi = 3.141592653589793; w * (cos(2*pi*x)*sin(2*2*pi*y) + sin(2*pi*x - 0.2)*sin(2*pi*y))";
#endif
#if (GEMPIC_SPACEDIM == 3)
    const std::string func =
        "pi = 3.141592653589793; w * (cos(2*pi*x)*sin(2*2*pi*y)*cos(2*pi*z) + sin(2*pi*x - "
        "0.2)*sin(2*pi*y)*sin(2*2*pi*z + 1.3))";
#endif

    const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
    amrex::ParserExecutor<nVar> funcP;
    amrex::Parser parser;

    parser.define(func);
    parser.setConstant("w", 1.);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    funcP = parser.compile<GEMPIC_SPACEDIM + 1>();

    deRham->projection(funcP, 0.0, dualThreeForm);
    deRham->hodge(dualThreeForm, primalZeroForm, weight);

    deRham->projection(funcP, 0.0, primalThreeForm);
    deRham->hodge(primalThreeForm, dualZeroForm, weight);

    parser.define(func);
    parser.setConstant("w", 1 / weight);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    funcP = parser.compile<GEMPIC_SPACEDIM + 1>();

    amrex::Real e1 = max_error_midpoint<hodgeDegree>(
        geom, funcP, primalZeroForm.m_data,
        amrex::RealVect{AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])}, 0,
        false);
    amrex::Real e2 = max_error_midpoint<hodgeDegree>(
        geom, funcP, dualZeroForm.m_data,
        amrex::RealVect{AMREX_D_DECL(infra.m_dx[xDir], infra.m_dx[yDir], infra.m_dx[zDir])}, 0,
        true);
    return std::max((e1), (e2));
}

int main (int argc, char *argv[])
{
    amrex::Initialize(argc, argv);
    {
        const int coarse = 16;
        const int fine = 32;
        amrex::Real errorCoarse;
        amrex::Real errorFine;

        const int len = 4;
        amrex::GpuArray<amrex::Real, len> rate;

        errorCoarse = test12(coarse);
        errorFine = test12(fine);
        rate[0] = std::log2(errorCoarse / errorFine);

        errorCoarse = test21(coarse);
        errorFine = test21(fine);
        rate[1] = std::log2(errorCoarse / errorFine);

        errorCoarse = test03(coarse);
        errorFine = test03(fine);
        rate[2] = std::log2(errorCoarse / errorFine);

        errorCoarse = test30(coarse);
        errorFine = test30(fine);
        rate[3] = std::log2(errorCoarse / errorFine);

        amrex::PrintToFile("test_FDHodge_degree_4.output") << std::endl;
        amrex::PrintToFile("test_FDHodge_degree_4.output")
            << GEMPIC_SPACEDIM
            << "D Finite Difference Hodge degree 4 convergence test:" << std::endl;
        amrex::PrintToFile("test_FDHodge_degree_4.output") << std::endl;
        for (int i = 0; i < len; ++i)
        {
            amrex::PrintToFile("test_FDHodge_degree_4.output").SetPrecision(3)
                << hodges[i] + " rate: " << rate[i] << std::endl;
        }

        if (amrex::ParallelDescriptor::MyProc() == 0)
        {
            std::rename("test_FDHodge_degree_4.output.0", "test_FDHodge_degree_4.output");
        }
    }
    amrex::Finalize();
}
