#include <iostream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_PoissonSolver.H"
#include "GEMPIC_SplineClass.H"
#include "TestUtils/GEMPIC_TestUtils.H"

#define compare_fields(...) Gempic::Test::Utils::compare_fields(__FILE__, __LINE__, __VA_ARGS__)

namespace
{
using namespace Gempic;
using namespace Forms;
using namespace FieldSolvers;

//using ::testing::_;
//using ::testing::Exactly;
//using ::testing::Mock;

/**
 * @brief Tests the Poisson solver for an analytical rho of 1.0 + cos(x)
 *
 * @todo: Use our Hodge and compare to exact analytical function
 */
class PoissonSolverTest : public testing::Test
{
    //protected:
public:
    static const int s_degX{1};
    static const int s_degY{1};
    static const int s_degZ{1};
    inline static const int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

    Io::Parameters m_parameters{};

    ComputationalDomain m_infra{};  // initialized computational domain

    static const int s_nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
    amrex::Parser m_parserRho, m_parserPhi;
    amrex::ParserExecutor<s_nVar> m_funcRho, m_funcPhi;

    static void SetUpTestSuite ()
    {
        /* Initialize the infrastructure */
        const amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(-M_PI, -M_PI, -M_PI)};
        const amrex::Vector<amrex::Real> k{AMREX_D_DECL(1.0, 1.0, 1.0)};
        const amrex::Vector<int> nCell{AMREX_D_DECL(16, 16, 16)};
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(8, 8, 8)};
        const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};
        const int nGhostExtra = 0;

        amrex::ParmParse pp;
        pp.addarr("domainLo", domainLo);
        pp.addarr("k", k);
        pp.addarr("nCellVector", nCell);
        pp.addarr("maxGridSizeVector", maxGridSize);
        pp.addarr("isPeriodicVector", isPeriodic);
        pp.add("nGhostExtra", nGhostExtra);
    }

    // virtual void SetUp() will be called before each test is run.
    void SetUp () override
    {
        // Analytical rho and phi such that -Delta phi = rho
#if GEMPIC_SPACEDIM == 1
        const std::string analyticalRho = "cos(x)";
        const std::string analyticalPhi = "cos(x)";
#elif GEMPIC_SPACEDIM == 2
        const std::string analyticalRho = "10*cos(x)*cos(3*y)";
        const std::string analyticalPhi = "cos(x)*cos(3*y)";
#elif GEMPIC_SPACEDIM == 3
        const std::string analyticalRho = "3*cos(x)*cos(y)*cos(z)";
        const std::string analyticalPhi = "cos(x)*cos(y)*cos(z)";
#endif

        m_parserRho.define(analyticalRho);
        m_parserRho.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        m_funcRho = m_parserRho.compile<s_nVar>();
        m_parserPhi.define(analyticalPhi);
        m_parserPhi.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        m_funcPhi = m_parserPhi.compile<s_nVar>();
    }
};

TEST_F(PoissonSolverTest, PoissonAMReX)
{
    const int hodgeDegree{2};
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);
    // Define fields
    DeRhamField<Grid::dual, Space::cell> rho(deRham, m_funcRho);
    DeRhamField<Grid::primal, Space::node> phi(deRham);
    DeRhamField<Grid::primal, Space::node> anPhi(deRham, m_funcPhi);

    // solve Poisson using AMReX solver
    Gempic::FieldSolvers::PoissonSolver poisson(deRham, m_infra);
    poisson.solve_amrex(phi, rho);

    // Check error
    amrex::Real tol = 1e-1;
    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        compare_fields(anPhi.m_data.array(mfi), phi.m_data.array(mfi), bx, tol);
    }
}

TEST_F(PoissonSolverTest, ConjugateGradientHodge2)
{
    constexpr int hodgeDegree{2};
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);
    // Define fields
    DeRhamField<Grid::dual, Space::cell> rho(deRham);
    DeRhamField<Grid::primal, Space::node> phiIn(deRham, m_funcPhi);
    DeRhamField<Grid::primal, Space::node> phi(deRham);

    deRham->hodge(rho, phiIn);

    ConjugateGradient<DeRhamField<Grid::dual, Space::cell>, DeRhamField<Grid::primal, Space::node>,
                      Operator::hodge>
        cgHodge(deRham);

    cgHodge.solve(phi, rho);

    amrex::Real tol = 1e-12;
    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        compare_fields(phi.m_data.array(mfi), phiIn.m_data.array(mfi), bx, tol);
    }
}
}  // namespace

namespace
{
template <typename degree>
class PoissonSolverFFTTypedTest : public PoissonSolverTest
{
public:
    static constexpr int s_hodgeDegree{degree()};
};

struct NameGeneratorFFT
{
    template <typename T>
    static std::string GetName (int)
    {
        return "HodgeDegree" + std::to_string(T());
    }
};

using MyTypesFFT = ::testing::Types<std::integral_constant<int, 2>,
                                    std::integral_constant<int, 4>,
                                    std::integral_constant<int, 6>>;
TYPED_TEST_SUITE(PoissonSolverFFTTypedTest, MyTypesFFT, NameGeneratorFFT);

TYPED_TEST(PoissonSolverFFTTypedTest, AnalyticalLowTolerance)
{
    auto deRham =
        std::make_shared<FDDeRhamComplex>(this->m_infra, TestFixture::s_hodgeDegree,
                                          TestFixture::s_maxSplineDegree, HodgeScheme::FDHodge);

    // Define fields
    DeRhamField<Grid::dual, Space::cell> rho(deRham, this->m_funcRho);
    DeRhamField<Grid::primal, Space::node> phi(deRham);
    DeRhamField<Grid::primal, Space::node> anPhi(deRham, this->m_funcPhi);

    // Use FFT-based solver
    Gempic::FieldSolvers::PoissonSolver poisson(deRham, this->m_infra);
    poisson.solve_fft(phi, rho);

    // Check error
    // Degree 2 -- general size 16 for 1,2,3: error 4.203e-2
    // Degree 4 -- general size 16 for 1,2,3: error 1.28e-2
    // Degree 6 -- general size 16 for 1,2,3: error 1.28e-2
    // larger errors than lower orders only occur at boundary, probably because of fluctuation
    amrex::GpuArray<amrex::Real, 3> tols{4.203e-2, 1.28e-2, 1.28e-2};
    amrex::Real tol = tols[TestFixture::s_hodgeDegree / 2 - 1];
    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        compare_fields(anPhi.m_data.array(mfi), phi.m_data.array(mfi), bx, tol);
    }
}
}  //namespace

template <int degree, Operator op>
struct HodgeDegreePoissonMethod
{
    static constexpr int s_hodgeDegree{degree};
    static constexpr Operator s_operator{op};
};

// Supplements the gtest naming of typed tests
struct NameGenerator
{
    template <typename T>
    static std::string GetName (int)
    {
        std::string tmp = "HodgeDegree" + std::to_string(T::s_hodgeDegree);
        switch (T::s_operator)
        {
            case Operator::poisson:
                return tmp + "PoissonOperator";
            case Operator::poissonInverseHodge:
                return tmp + "PoissonInverseHodgeOperator";
            case Operator::hodge:
                return tmp + "HodgeOperator";
            default:
                return tmp + "UnknownConjugateGradientOperator";
        }
    }
};

namespace
{
template <typename degreeMethod>
class PoissonSolverTypedTest : public PoissonSolverTest
{
public:
    static constexpr int s_hodgeDegree{degreeMethod::s_hodgeDegree};
    static constexpr Operator s_operator{degreeMethod::s_operator};
};

using MyTypes = ::testing::Types<HodgeDegreePoissonMethod<2, Operator::poisson>,
                                 HodgeDegreePoissonMethod<4, Operator::poisson>,
                                 HodgeDegreePoissonMethod<6, Operator::poisson>,
                                 HodgeDegreePoissonMethod<2, Operator::poissonInverseHodge>,
                                 HodgeDegreePoissonMethod<4, Operator::poissonInverseHodge>,
                                 HodgeDegreePoissonMethod<6, Operator::poissonInverseHodge>>;
TYPED_TEST_SUITE(PoissonSolverTypedTest, MyTypes, NameGenerator);

TYPED_TEST(PoissonSolverTypedTest, AnalyticalLowTolerance)
{
    auto deRham = std::make_shared<FDDeRhamComplex>(this->m_infra, this->s_hodgeDegree,
                                                    this->s_maxSplineDegree, HodgeScheme::FDHodge);
    auto poisson = std::make_shared<Gempic::FieldSolvers::PoissonSolver>(deRham, this->m_infra);

    DeRhamField<Grid::dual, Space::cell> rho(deRham, this->m_funcRho);
    DeRhamField<Grid::primal, Space::node> phi(deRham);
    DeRhamField<Grid::primal, Space::node> anPhi(deRham, this->m_funcPhi);

    ConjugateGradient<DeRhamField<Grid::dual, Space::cell>, DeRhamField<Grid::primal, Space::node>,
                      TestFixture::s_operator>
        cgPoisson(deRham, poisson);

    cgPoisson.solve(phi, rho);

    // Check error
    amrex::Real tol = 1e-1;
    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        compare_fields(anPhi.m_data.array(mfi), phi.m_data.array(mfi), bx, tol);
    }
}
}  // namespace