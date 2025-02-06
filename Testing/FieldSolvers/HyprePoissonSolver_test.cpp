#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_PoissonSolver.H"
#include "GEMPIC_SplineClass.H"
#include "TestUtils/GEMPIC_TestUtils.H"

#define compare_fields(...) Gempic::Test::Utils::compare_fields(__FILE__, __LINE__, __VA_ARGS__)

namespace
{
using namespace Gempic;
using namespace Forms;
using namespace FieldSolvers;

/**
 * @brief Tests the Poisson solver for an analytical rho of 1.0 + cos(x)
 *
 * @todo: Use our Hodge and compare to exact analytical function
 */
template <typename hodgeDegreeStruct>
class HyprePoissonSolverTest : public testing::Test
{
public:
    static constexpr int s_degX{3};
    static constexpr int s_degY{3};
    static constexpr int s_degZ{3};
    static constexpr int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};
    static constexpr int s_hodgeDegree{hodgeDegreeStruct::value};

    static const int s_nVar = AMREX_SPACEDIM + 1;  // x, y, z, t
    amrex::Parser m_parserRho, m_parserPhi;
    amrex::ParserExecutor<s_nVar> m_funcRho, m_funcPhi;

    static void SetUpTestSuite ()
    {
        const amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        const amrex::Vector<amrex::Real> domainHi{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(16, 16, 16)};
        const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};

        /* Initialize the infrastructure */
        amrex::ParmParse pp;  // Used in lieu of input file
        pp.addarr("ComputationalDomain.domainLo", domainLo);
        pp.addarr("ComputationalDomain.domainHi", domainHi);

        pp.addarr("ComputationalDomain.maxGridSize", maxGridSize);
        pp.addarr("ComputationalDomain.isPeriodic", isPeriodic);
    }

    // virtual void SetUp() will be called before each test is run.
    void SetUp () override
    {
        // Analytical rho and phi such that -Delta phi = rho
#if AMREX_SPACEDIM == 2
        const std::string analyticalRho = "4*sin(2*x)";
        const std::string analyticalPhi = "sin(2*x)";
#elif AMREX_SPACEDIM == 3
        const std::string analyticalRho = "4*sin(2*x)";
        const std::string analyticalPhi = "sin(2*x)";
#endif

        m_parserRho.define(analyticalRho);
        m_parserRho.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        m_funcRho = m_parserRho.compile<s_nVar>();
        m_parserPhi.define(analyticalPhi);
        m_parserPhi.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        m_funcPhi = m_parserPhi.compile<s_nVar>();
    }

    amrex::Real poisson_solve (int n)
    {
        // Initialize computational_domain
        const amrex::Vector<int> nCell{AMREX_D_DECL(n, n, n)};

        amrex::ParmParse pp;
        pp.addarr("ComputationalDomain.nCell", nCell);
        // Class that should acutally manage parameters instead of amrex::ParamParse
        // Need an instance of this to use parameters in ComputationalDomain.
        Gempic::Io::Parameters parameters;
        ComputationalDomain infra;

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, s_hodgeDegree, s_maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        auto poisson = std::make_shared<PoissonSolver>(deRham, infra);
        HypreLinearSystem<DeRhamField<Grid::dual, Space::cell>,
                          DeRhamField<Grid::primal, Space::node>, Operator::poisson, s_hodgeDegree>
            hyprePoisson(&infra, deRham, poisson);

        DeRhamField<Grid::dual, Space::cell> rho(deRham, m_funcRho);
        DeRhamField<Grid::primal, Space::node> phi(deRham);
        DeRhamField<Grid::primal, Space::node> phiAn(deRham, m_funcPhi);

        amrex::Array4<amrex::Real> const& barr = rho.m_data[0].array();
        amrex::Real temp = barr(0, 0, 0, 0);
        barr(0, 0, 0, 0) = 0.0;

        phi.m_data.setVal(0.0);
        hyprePoisson.solve(phi, rho);

        barr(0, 0, 0, 0) = temp;

        phi -= phiAn;

        return Utils::gempic_norm(phi.m_data, infra, 2);
    }
};

using MyTypes = ::testing::Types<std::integral_constant<int, 2>,
                                 std::integral_constant<int, 4>,
                                 std::integral_constant<int, 6>>;
TYPED_TEST_SUITE(HyprePoissonSolverTest, MyTypes);

TYPED_TEST(HyprePoissonSolverTest, HyprePoissonConvergence)
{
    const int coarse = 32, fine = 64;
    amrex::Real errorCoarse, errorFine;
    amrex::Real tol = 0.1;

    amrex::Real rateOfConvergence;
    constexpr int hodgeDegree = TestFixture::s_hodgeDegree;
    errorCoarse = this->poisson_solve(coarse);
    errorFine = this->poisson_solve(fine);
    rateOfConvergence = std::log2(errorCoarse / errorFine);
    amrex::Print() << "rate_of_convergence_" << hodgeDegree << ':' << rateOfConvergence << "\n";
    EXPECT_NEAR(rateOfConvergence, hodgeDegree, tol);
}
}  // namespace
