#include <iostream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_PoissonSolver.H"
#include "GEMPIC_SplineClass.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;
using namespace FieldSolvers;

//using ::testing::_;
//using ::testing::Exactly;
//using ::testing::Mock;

ComputationalDomain get_compdom ()
{
    const std::array<amrex::Real, AMREX_SPACEDIM> domainLo{AMREX_D_DECL(-M_PI, -M_PI, -M_PI)};
    const std::array<amrex::Real, AMREX_SPACEDIM> domainHi{AMREX_D_DECL(M_PI, M_PI, M_PI)};
    const amrex::IntVect nCell{AMREX_D_DECL(16, 16, 16)};
    const amrex::IntVect maxGridSize{AMREX_D_DECL(8, 8, 8)};
    const std::array<int, AMREX_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic);
}

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

    ComputationalDomain m_infra;

    static const int s_nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::Parser m_parserRho, m_parserPhi;
    amrex::ParserExecutor<s_nVar> m_funcRho, m_funcPhi;

    PoissonSolverTest() : m_infra{get_compdom()}
    {
        /* Initialize the infrastructure */
        const int nGhostExtra = 0;

        m_parameters.set("nGhostExtra", nGhostExtra);

        // Analytical rho and phi such that -Delta phi = rho
#if AMREX_SPACEDIM == 1
        const std::string analyticalRho = "cos(x)";
        const std::string analyticalPhi = "cos(x)";
#elif AMREX_SPACEDIM == 2
        const std::string analyticalRho = "10*cos(x)*cos(3*y)";
        const std::string analyticalPhi = "cos(x)*cos(3*y)";
#elif AMREX_SPACEDIM == 3
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

    // solve Poisson using AMReX solver (normal applications should use
    // Gempic::FieldSolvers::make_poisson_solver(deRham, m_infra)
    // so the solver can be chosen in the input file)
    std::string solverStr{"Amrex"};
    auto poisson{
        Gempic::FieldSolvers::Impl::make_specific_poisson_solver(deRham, m_infra, solverStr)};
    poisson->solve(phi, rho);

    // Check error
    amrex::Real tol = 1e-1;
    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        COMPARE_FIELDS(anPhi.m_data.array(mfi), phi.m_data.array(mfi), bx, tol);
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

    using CGRhs = DeRhamField<Grid::dual, Space::cell>;
    using CGSol = DeRhamField<Grid::primal, Space::node>;

    deRham->hodge(rho, phiIn);

    auto cgHodge{make_conjugate_gradient<CGRhs, CGSol>(
        deRham, [=] (CGRhs &rhs, CGSol &sol) { deRham->hodge(rhs, sol); })};
    cgHodge.solve(phi, rho);

    amrex::Real tol = 1e-12;
    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        COMPARE_FIELDS(phi.m_data.array(mfi), phiIn.m_data.array(mfi), bx, tol);
    }
}
} // namespace

namespace
{
#ifdef AMREX_USE_FFT
template <typename Degree>
class PoissonSolverFFTTypedTest : public PoissonSolverTest
{
public:
    static constexpr int s_hodgeDegree{Degree()};
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

    // Use FFT-based solver (normal applications should use
    // Gempic::FieldSolvers::make_poisson_solver(deRham, m_infra)
    // so the solver can be chosen in the input file)
    std::string solverStr{"FFT"};
    auto poisson{
        Gempic::FieldSolvers::Impl::make_specific_poisson_solver(deRham, this->m_infra, solverStr)};
    poisson->solve(phi, rho);

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
        COMPARE_FIELDS(anPhi.m_data.array(mfi), phi.m_data.array(mfi), bx, tol);
    }
}
#endif
} //namespace

enum class Operator
{
    poisson,
    poissonInverseHodge
};

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
                return tmp + "ConjugateGradient";
            case Operator::poissonInverseHodge:
                return tmp + "ConjugateGradientInverseHodge";
            default:
                return tmp + "UnknownConjugateGradientOperator";
        }
    }
};

namespace
{
template <typename DegreeMethod>
class PoissonSolverTypedTest : public PoissonSolverTest
{
public:
    static constexpr int s_hodgeDegree{DegreeMethod::s_hodgeDegree};
    static constexpr Operator s_operator{DegreeMethod::s_operator};

    std::string m_solverStr;

    PoissonSolverTypedTest()
    {
        if constexpr (s_operator == Operator::poisson)
        {
            m_solverStr = "ConjugateGradient";
        }
        else //if constexpr (s_operator == Operator::poissonInverseHodge)
        {
            m_solverStr = "ConjugateGradientInverseHodge";
        }
    }
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

    auto poisson{Gempic::FieldSolvers::Impl::make_specific_poisson_solver(deRham, this->m_infra,
                                                                          this->m_solverStr)};

    DeRhamField<Grid::dual, Space::cell> rho(deRham, this->m_funcRho);
    DeRhamField<Grid::primal, Space::node> phi(deRham);
    DeRhamField<Grid::primal, Space::node> anPhi(deRham, this->m_funcPhi);

    poisson->solve(phi, rho);

    // Check error
    amrex::Real tol = 1e-1;
    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        COMPARE_FIELDS(anPhi.m_data.array(mfi), phi.m_data.array(mfi), bx, tol);
    }
}
} // namespace