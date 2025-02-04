#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_QuasineutralSolver.H"
#include "GEMPIC_SplineClass.H"
#include "TestUtils/GEMPIC_TestUtils.H"

#define compare_fields(...) Gempic::Test::Utils::compare_fields(__FILE__, __LINE__, __VA_ARGS__)

namespace
{
using namespace Gempic;
using namespace Forms;
using namespace FieldSolvers;

/**
 * @brief Tests the CurlCurl operator of the quasineutral solver.
 * The rho field is added to avoid singularity and as a cheaper
 * alternative for the consistent rho*E deposited from particles.
 */
template <typename hodgeDegreeStruct>
class HypreCurlCurlOperatorTest : public testing::Test
{
public:
    static constexpr int s_vdim{3};
    static constexpr int s_degX{3};
    static constexpr int s_degY{3};
    static constexpr int s_degZ{3};
    static constexpr int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};
    static constexpr int s_hodgeDegree{hodgeDegreeStruct::value};

    static const int s_nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
    amrex::Parser m_parserRho;
    amrex::ParserExecutor<s_nVar> m_funcRho;

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcE;
    amrex::Array<amrex::Parser, 3> m_parserE;

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcRHS;
    amrex::Array<amrex::Parser, 3> m_parserRHS;

    static void SetUpTestSuite ()
    {
        /* Initialize the infrastructure */
        amrex::ParmParse pp;  // Used in lieu of input file

        const amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        pp.addarr("ComputationalDomain.domainLo", domainLo);

        const amrex::Vector<amrex::Real> domainHi{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
        pp.addarr("ComputationalDomain.domainHi", domainHi);

        // Grid parameters
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(8, 8, 8)};
        pp.addarr("ComputationalDomain.maxGridSize", maxGridSize);

        const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};
        pp.addarr("ComputationalDomain.isPeriodic", isPeriodic);
    }

    // virtual void SetUp() will be called before each test is run.
    void SetUp () override
    {
#if GEMPIC_SPACEDIM == 2
        const std::string analyticalRho = "10+cos(x)+cos(2*y)";
        const amrex::Array<std::string, 3> analyticalE = {"sin(y)", "sin(2*x)", "cos(x)*cos(y)"};
        const amrex::Array<std::string, 3> analyticalRHS = {"(1+10+cos(x)+cos(2*y))*sin(y)",
                                                            "(4+10+cos(x)+cos(2*y))*sin(2*x)",
                                                            "(2+10+cos(x)+cos(2*y))*cos(x)*cos(y)"};
#elif GEMPIC_SPACEDIM == 3
        const std::string analyticalRho = "10+cos(x)+cos(2*y)+sin(z)";
        const amrex::Array<std::string, 3> analyticalE = {"sin(y)", "sin(2*z)", "sin(3*x)"};
        const amrex::Array<std::string, 3> analyticalRHS = {
            "(1+10+cos(x)+cos(2*y)+sin(z))*sin(y)", "(4+10+cos(x)+cos(2*y)+sin(z))*sin(2*z)",
            "(9+10+cos(x)+cos(2*y)+sin(z))*sin(3*x)"};
#endif

        m_parserRho.define(analyticalRho);
        m_parserRho.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        m_funcRho = m_parserRho.compile<s_nVar>();

        for (int i = 0; i < 3; ++i)
        {
            m_parserE[i].define(analyticalE[i]);
            m_parserE[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcE[i] = m_parserE[i].compile<s_nVar>();

            m_parserRHS[i].define(analyticalRHS[i]);
            m_parserRHS[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcRHS[i] = m_parserRHS[i].compile<s_nVar>();
        }
    }

    amrex::Real curlcurloperator_solve (int n)
    {
        // Initialize computational_domain
        const amrex::Vector<int> nCell{AMREX_D_DECL(n, n, n)};

        Gempic::Io::Parameters parameters{};
        amrex::ParmParse pp;
        pp.addarr("ComputationalDomain.nCell", nCell);
        ComputationalDomain infra;

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, s_hodgeDegree, s_maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        HypreQuasineutralLinearSystem<DeRhamField<Grid::dual, Space::face>,
                                      DeRhamField<Grid::primal, Space::edge>, s_hodgeDegree, s_vdim,
                                      s_degX, s_degY, s_degZ>
            hypreCurlcurlPlusFieldRho(infra, deRham);

        DeRhamField<Grid::dual, Space::cell> rho(deRham, m_funcRho);
        DeRhamField<Grid::dual, Space::face> rhs(deRham, m_funcRHS);
        DeRhamField<Grid::primal, Space::edge> E(deRham, m_funcE);
        DeRhamField<Grid::primal, Space::edge> eAn(deRham, m_funcE);

        hypreCurlcurlPlusFieldRho.solve_curlcurl_plus_fieldcharge_e(rhs, E, rho);

        E -= eAn;

        amrex::GpuArray<amrex::Real, 3> dxi{GEMPIC_D_PAD_ONE(infra.geometry().InvCellSize(xDir),
                                                             infra.geometry().InvCellSize(yDir),
                                                             infra.geometry().InvCellSize(zDir))};
        return Utils::gempic_norm (E.m_data[xDir], infra, 1) * dxi[xDir] +
               Utils::gempic_norm(E.m_data[yDir], infra, 1) * dxi[yDir] +
               Utils::gempic_norm(E.m_data[zDir], infra, 1) * dxi[zDir];
    }
};

using MyTypes = ::testing::Types<std::integral_constant<int, 2>>;
TYPED_TEST_SUITE(HypreCurlCurlOperatorTest, MyTypes);

TYPED_TEST(HypreCurlCurlOperatorTest, HypreCurlCurlOperatorConvergence)
{
    const int coarse = 12, fine = 24;
    amrex::Real errorCoarse, errorFine;
    amrex::Real tol = 0.1;

    amrex::Real rateOfConvergence;
    constexpr int hodgeDegree = TestFixture::s_hodgeDegree;
    errorCoarse = this->curlcurloperator_solve(coarse);
    amrex::Print() << "errorCoarse: " << errorCoarse << "\n";
    errorFine = this->curlcurloperator_solve(fine);
    amrex::Print() << "errorFine: " << errorFine << "\n";
    rateOfConvergence = std::log2(errorCoarse / errorFine);
    amrex::Print() << "rate_of_convergence_" << hodgeDegree << ':' << rateOfConvergence << "\n";
    EXPECT_NEAR(rateOfConvergence, hodgeDegree, tol);
}
}  // namespace