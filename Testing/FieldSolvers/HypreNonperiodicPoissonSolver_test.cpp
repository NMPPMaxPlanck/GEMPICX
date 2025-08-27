#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_QuasineutralSolver.H"
#include "GEMPIC_SplineClass.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;
using namespace FieldSolvers;

/**
 * @brief Tests the non-periodic Poisson solver
 */
template <typename HodgeDegreeStruct>
class HypreNonperiodicPoissonTest : public testing::Test
{
public:
    static constexpr int s_vdim{3};
    static constexpr int s_ndata{1};
    static constexpr int s_degX{3};
    static constexpr int s_degY{3};
    static constexpr int s_degZ{3};
    static constexpr int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};
    static constexpr int s_hodgeDegree{HodgeDegreeStruct::value};

    static int const s_nVar = AMREX_SPACEDIM + 1; // x, y, z, t

    amrex::Parser m_parserRho, m_parserPhi;
    amrex::ParserExecutor<s_nVar> m_funcRho, m_funcPhi;

    HypreNonperiodicPoissonTest()
    {
        // Analytical rho and phi such that -Delta phi = rho
#if AMREX_SPACEDIM == 2
        std::string const analyticalRho = "5.0*sin(x)*sin(2*y)";
        std::string const analyticalPhi = "sin(x)*sin(2*y)";
#elif AMREX_SPACEDIM == 3
        std::string const analyticalRho = "14.0*sin(x)*sin(2*y)*sin(3*z)";
        std::string const analyticalPhi = "sin(x)*sin(2*y)*sin(3*z)";
#endif

        m_parserRho.define(analyticalRho);
        m_parserRho.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        m_funcRho = m_parserRho.compile<s_nVar>();
        m_parserPhi.define(analyticalPhi);
        m_parserPhi.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        m_funcPhi = m_parserPhi.compile<s_nVar>();
    }

    template <int n>
    amrex::Real nonperiodicpoisson_operator_solve ()
    {
        amrex::Real relTol{1.e-15};

        Gempic::Io::Parameters parameters{};
        amrex::IntVect const nCell{AMREX_D_DECL(n, n, n)};

        // Initialize computational_domain
        auto infra = Gempic::Test::Utils::get_compdom(nCell);

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, s_hodgeDegree, s_maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        QuasineutralSolver<s_hodgeDegree, s_vdim, s_ndata, s_degX, s_degY, s_degZ> hyprePoissonNonP(
            infra, deRham, relTol);

        DeRhamField<Grid::dual, Space::cell> rho(deRham, m_funcRho);
        DeRhamField<Grid::primal, Space::node> phi(deRham);
        DeRhamField<Grid::primal, Space::node> phiAn(deRham, m_funcPhi);

        hyprePoissonNonP.solve_nonperiodic_poisson_equation(rho, phi);

        phi.fill_boundary();

        if (infra.geometry().isAllPeriodic())
        {
            auto nGhost = phi.m_deRham->get_n_ghost();
            amrex::Real phiSum = phi.m_data.sum_unique(0, false, infra.geometry().periodicity());
            amrex::Real ninv =
                1.0 / GEMPIC_D_MULT(infra.m_nCell[xDir], infra.m_nCell[yDir], infra.m_nCell[zDir]);
            amrex::Real phiSumNinv = phiSum * ninv;
            phi -= phiSumNinv;
        }
        phi *= 1.0 / infra.cell_volume();

        phi -= phiAn;

        return Utils::gempic_norm(phi.m_data, infra, 2);
    }
};

using MyTypes = ::testing::Types<std::integral_constant<int, 2>>;
TYPED_TEST_SUITE(HypreNonperiodicPoissonTest, MyTypes);

TYPED_TEST(HypreNonperiodicPoissonTest, HypreNonperiodicPoissonConvergence)
{
    int const coarse = 8, fine = 16;
    amrex::Real errorCoarse, errorFine;
    amrex::Real tol = 0.2;

    amrex::Real rateOfConvergence;
    constexpr int hodgeDegree = TestFixture::s_hodgeDegree;

    errorCoarse = this->template nonperiodicpoisson_operator_solve<coarse>();
    errorFine = this->template nonperiodicpoisson_operator_solve<fine>();
    rateOfConvergence = std::log2(errorCoarse / errorFine);
    amrex::Print() << "rate_of_convergence_" << hodgeDegree << ':' << rateOfConvergence << "\n";

#if AMREX_SPACEDIM == 2
    EXPECT_NEAR(rateOfConvergence, 2.0, tol);
#elif AMREX_SPACEDIM == 3
    EXPECT_NEAR(rateOfConvergence, 4.0, tol);
#endif
}
} // namespace
