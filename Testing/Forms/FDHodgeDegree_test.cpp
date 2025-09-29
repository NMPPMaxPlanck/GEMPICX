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

#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Interpolation.H"
#include "GEMPIC_Parameters.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Forms;

namespace
{
/* Initialize the infrastructure */
ComputationalDomain get_compdom (amrex::IntVect const& nCell)
{
    std::array<amrex::Real, AMREX_SPACEDIM> const domainLo{AMREX_D_DECL(0.3, 0.6, 0.4)};
    std::array<amrex::Real, AMREX_SPACEDIM> const domainHi{AMREX_D_DECL(1.3, 1.6, 1.4)};
    amrex::IntVect const maxGridSize{AMREX_D_DECL(8, 6, 9)};
    std::array<int, AMREX_SPACEDIM> const isPeriodic{AMREX_D_DECL(1, 1, 1)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic);
}

template <int hodgeDegree, Space space>
void hodge_one_two_error (double& e1, double& e2, int const n, int maxSplineDegree = 3)
{
    // Initialize computational_domain
    amrex::IntVect const nCell{AMREX_D_DECL(n, n, n)};
    Gempic::Io::Parameters parameters{};
    ComputationalDomain infra = get_compdom(nCell);

    auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    amrex::Real const weight = 3. / 2.;
#if (AMREX_SPACEDIM == 1)
    amrex::Array<std::string, 3> const func = {
        "pi = 3.141592653589793; w * (cos(2*pi*x) + sin(2*pi*x - 0.2))",
        "pi = 3.141592653589793; w * (sin(2*pi*x) + cos(2*pi*x - 0.2))",
        "pi = 3.141592653589793; w * (cos(2*2*pi*x) + sin(2*pi*x - 0.2))",
    };
#endif
#if (AMREX_SPACEDIM == 2)
    amrex::Array<std::string, 3> const func = {
        "pi = 3.141592653589793; w * (cos(2*pi*x)*sin(2*2*pi*y) + sin(2*pi*x - 0.2)*sin(2*pi*y))",
        "pi = 3.141592653589793; w * (sin(2*pi*x)*cos(2*pi*y) + cos(2*pi*x - 0.2)*sin(2*pi*y))",
        "pi = 3.141592653589793; w * (cos(2*2*pi*x)*cos(2*pi*y) + sin(2*pi*x - 0.2)*sin(2*pi*y))",
    };
#endif
#if (AMREX_SPACEDIM == 3)
    amrex::Array<std::string, 3> const func = {
        "pi = 3.141592653589793; w * (cos(2*pi*x)*sin(2*2*pi*y)*cos(2*pi*z) + sin(2*pi*x - "
        "0.2)*sin(2*pi*y)*sin(2*2*pi*z + 1.3))",
        "pi = 3.141592653589793; w * (sin(2*pi*x)*cos(2*pi*y)*cos(2*2*pi*z) + cos(2*pi*x - "
        "0.2)*sin(2*pi*y)*sin(2*2*pi*z + 1.3))",
        "pi = 3.141592653589793; w * (cos(2*2*pi*x)*cos(2*pi*y)*sin(2*pi*z) + sin(2*pi*x - "
        "0.2)*sin(2*pi*y)*cos(2*2*pi*z + 1.3))",
    };
#endif

    // project f to a dualOneForm and a primalOneForm respectively
    amrex::Array<amrex::ParserExecutor<AMREX_SPACEDIM + 1>, 3> funcP;
    amrex::Array<amrex::Parser, 3> parser;
    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(func[i]);
        parser[i].setConstant("w", 1.);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcP[i] = parser[i].compile<AMREX_SPACEDIM + 1>();
    }

    DeRhamField<Grid::primal, space> primalForm(deRham);
    DeRhamField<Grid::dual, space> dualForm(deRham);
    DeRhamField<Grid::dual, oppositeSpace<space>> dualOppositeForm(deRham);
    DeRhamField<Grid::primal, oppositeSpace<space>> primalOppositeForm(deRham);

    deRham->projection(funcP, 0.0, dualOppositeForm);
    deRham->hodge(primalForm, dualOppositeForm, weight);

    deRham->projection(funcP, 0.0, primalOppositeForm);
    deRham->hodge(dualForm, primalOppositeForm, weight);

    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(func[i]);
        parser[i].setConstant("w", weight);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcP[i] = parser[i].compile<AMREX_SPACEDIM + 1>();
    }

    auto dr{infra.cell_size_array()};
    auto form{static_cast<int>(space)};
    e1 = 0;
    e2 = 0;
    for (int comp = 0; comp < 3; ++comp)
    {
        e1 += max_error_midpoint<hodgeDegree>(infra.m_geom, funcP[comp], primalForm.m_data[comp],
                                              dr, form, false, comp);
        e2 += max_error_midpoint<hodgeDegree>(infra.m_geom, funcP[comp], dualForm.m_data[comp], dr,
                                              form, true, comp);
    }
}

template <int hodgeDegree, Space space>
void hodge_zero_three_error (double& e1, double& e2, int const n, int maxSplineDegree = 3)
{
    // Initialize computational_domain
    amrex::IntVect const nCell{AMREX_D_DECL(n, n, n)};
    Gempic::Io::Parameters parameters{};
    ComputationalDomain infra = get_compdom(nCell);

    auto deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    amrex::Real const weight = 3. / 2.;
#if (AMREX_SPACEDIM == 1)
    std::string const func = "pi = 3.141592653589793; w * (cos(2*pi*x) + sin(2*pi*x - 0.2))";
#endif
#if (AMREX_SPACEDIM == 2)
    std::string const func =
        "pi = 3.141592653589793; w * (cos(2*pi*x)*sin(2*2*pi*y) + sin(2*pi*x - 0.2)*sin(2*pi*y))";
#endif
#if (AMREX_SPACEDIM == 3)
    std::string const func =
        "pi = 3.141592653589793; w * (cos(2*pi*x)*sin(2*2*pi*y)*cos(2*pi*z) + sin(2*pi*x - "
        "0.2)*sin(2*pi*y)*sin(2*2*pi*z + 1.3))";
#endif

    // project f to a dualOneForm and a primalOneForm respectively
    amrex::ParserExecutor<AMREX_SPACEDIM + 1> funcP;
    amrex::Parser parser;

    parser.define(func);
    parser.setConstant("w", 1.);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    funcP = parser.compile<AMREX_SPACEDIM + 1>();

    DeRhamField<Grid::primal, space> primalForm(deRham);
    DeRhamField<Grid::dual, space> dualForm(deRham);
    DeRhamField<Grid::dual, oppositeSpace<space>> dualOppositeForm(deRham);
    DeRhamField<Grid::primal, oppositeSpace<space>> primalOppositeForm(deRham);

    deRham->projection(funcP, 0.0, dualOppositeForm);
    deRham->hodge(primalForm, dualOppositeForm, weight);

    deRham->projection(funcP, 0.0, primalOppositeForm);
    deRham->hodge(dualForm, primalOppositeForm, weight);

    parser.define(func);
    parser.setConstant("w", weight);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    funcP = parser.compile<AMREX_SPACEDIM + 1>();

    auto dr{infra.cell_size_array()};
    auto form{static_cast<int>(space)};
    e1 = max_error_midpoint<hodgeDegree>(infra.m_geom, funcP, primalForm.m_data, dr, form, false);
    e2 = max_error_midpoint<hodgeDegree>(infra.m_geom, funcP, dualForm.m_data, dr, form, true);
}

/**
 * @brief Test fixture. Sets up clean environment before each test of the SplineWithPrimitive class
 */
template <typename DegreeSpace>
class FDHodgeDegreeTest : public testing::Test
{
public:
    static constexpr int s_degX{3};
    static constexpr int s_degY{3};
    static constexpr int s_degZ{3};

    inline static int const s_maxSplineDegree{
        AMREX_D_PICK(s_degX, std::max(s_degX, s_degY), std::max(std::max(s_degX, s_degY), s_degZ))};

    static constexpr int s_hodgeDegree{DegreeSpace::s_hodgeDegree};
    static constexpr Space s_space{DegreeSpace::s_space};
};

class NameGenerator
{
public:
    template <typename T>
    static std::string GetName (int)
    {
        if constexpr (T::s_space == Space::node)
        {
            return "Hodge degree " + std::to_string(T::s_hodgeDegree) + " node";
        }
        if constexpr (T::s_space == Space::edge)
        {
            return "Hodge degree " + std::to_string(T::s_hodgeDegree) + " edge";
        }
        if constexpr (T::s_space == Space::face)
        {
            return "Hodge degree " + std::to_string(T::s_hodgeDegree) + " face";
        }
        if constexpr (T::s_space == Space::cell)
        {
            return "Hodge degree " + std::to_string(T::s_hodgeDegree) + " cell";
        }
    }
};

template <int degree, Space space>
struct DegreeSpace
{
    static constexpr int s_hodgeDegree{degree};
    static constexpr Space s_space{space};
};

using ZeroThreeForms = ::testing::Types<DegreeSpace<2, Space::node>,
                                        DegreeSpace<2, Space::cell>,
                                        DegreeSpace<4, Space::node>,
                                        DegreeSpace<4, Space::cell>,
                                        DegreeSpace<6, Space::node>,
                                        DegreeSpace<6, Space::cell>>;
TYPED_TEST_SUITE(FDHodgeDegreeTest, ZeroThreeForms, NameGenerator);

// Sillily we have to define a new class for a new typed test suite
template <typename DegreeSpace>
class FDHodgeDegreeTest2 : public FDHodgeDegreeTest<DegreeSpace>
{
};

using OneTwoForms = ::testing::Types<DegreeSpace<2, Space::edge>,
                                     DegreeSpace<2, Space::face>,
                                     DegreeSpace<4, Space::edge>,
                                     DegreeSpace<4, Space::face>,
                                     DegreeSpace<6, Space::edge>,
                                     DegreeSpace<6, Space::face>>;
TYPED_TEST_SUITE(FDHodgeDegreeTest2, OneTwoForms, NameGenerator);

TYPED_TEST(FDHodgeDegreeTest, PrimalDualZeroThreeTest)
{
    int const coarse = 16, fine = 32;
    constexpr int hodgeDegree{TestFixture::s_hodgeDegree};
    constexpr Space space{TestFixture::s_space};
    amrex::Real errorPrimalFormCoarse, errorDualFormCoarse, errorPrimalFormFine, errorDualFormFine;
    amrex::Real tol = 0.21;
    amrex::Real zerotol = 1e-14;

    hodge_zero_three_error<hodgeDegree, space>(errorPrimalFormCoarse, errorDualFormCoarse, coarse);
    hodge_zero_three_error<hodgeDegree, space>(errorPrimalFormFine, errorDualFormFine, fine);

    amrex::Real rateOfConvergencePrimalForm{0};
    if (errorPrimalFormCoarse < zerotol || errorPrimalFormFine < zerotol)
    { // this happens..
        rateOfConvergencePrimalForm = hodgeDegree;
    }
    else
    {
        rateOfConvergencePrimalForm = std::log2(errorPrimalFormCoarse / errorPrimalFormFine);
    }

    amrex::Real rateOfConvergenceDualForm{0};
    if (errorDualFormCoarse < zerotol || errorDualFormFine < zerotol)
    { // this happens..
        rateOfConvergenceDualForm = hodgeDegree;
    }
    else
    {
        rateOfConvergenceDualForm = std::log2(errorDualFormCoarse / errorDualFormFine);
    }

    // test: is Hodge degree of correct order
    EXPECT_NEAR(rateOfConvergencePrimalForm, hodgeDegree, tol);
    EXPECT_NEAR(rateOfConvergenceDualForm, hodgeDegree, tol);
}

TYPED_TEST(FDHodgeDegreeTest2, PrimalDualOneTwoTest)
{
    int const coarse = 16, fine = 32;
    constexpr int hodgeDegree{TestFixture::s_hodgeDegree};
    constexpr Space space{TestFixture::s_space};
    amrex::Real errorPrimalFormCoarse, errorDualFormCoarse, errorPrimalFormFine, errorDualFormFine;
    amrex::Real tol = 0.21;

    hodge_one_two_error<hodgeDegree, space>(errorPrimalFormCoarse, errorDualFormCoarse, coarse);
    hodge_one_two_error<hodgeDegree, space>(errorPrimalFormFine, errorDualFormFine, fine);

    amrex::Real rateOfConvergencePrimalForm{std::log2(errorPrimalFormCoarse / errorPrimalFormFine)};
    amrex::Real rateOfConvergenceDualForm{std::log2(errorDualFormCoarse / errorDualFormFine)};

    // test: is Hodge degree of correct order
    EXPECT_NEAR(rateOfConvergencePrimalForm, hodgeDegree, tol);
    EXPECT_NEAR(rateOfConvergenceDualForm, hodgeDegree, tol);
}
} // namespace
