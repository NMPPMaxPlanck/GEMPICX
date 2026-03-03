/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Hypre.H"
#include "GEMPIC_PoissonSolver.H"
#include "GEMPIC_SplineClass.H"
#include "TestUtils/GEMPIC_TestUtils.H"

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
template <typename HodgeDegreeStruct>
class HyprePoissonSolverTest : public testing::Test
{
public:
    static constexpr int s_degX{3};
    static constexpr int s_degY{3};
    static constexpr int s_degZ{3};
    static constexpr int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};
    static constexpr int s_hodgeDegree{HodgeDegreeStruct::value};

    static int const s_nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::Parser m_parserRho, m_parserPhi;
    amrex::ParserExecutor<s_nVar> m_funcRho, m_funcPhi;

    HyprePoissonSolverTest()
    {
        // Analytical rho and phi such that -Delta phi = rho
#if AMREX_SPACEDIM == 2
        std::string const analyticalRho = "4*sin(2*x)";
        std::string const analyticalPhi = "sin(2*x)";
#elif AMREX_SPACEDIM == 3
        std::string const analyticalRho = "4*sin(2*x)";
        std::string const analyticalPhi = "sin(2*x)";
#endif

        m_parserRho.define(analyticalRho);
        m_parserRho.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        m_funcRho = m_parserRho.compile<s_nVar>();
        m_parserPhi.define(analyticalPhi);
        m_parserPhi.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        m_funcPhi = m_parserPhi.compile<s_nVar>();

        Gempic::Io::Parameters parameters;
        std::string solverStr{"Hypre"};
        parameters.set("PoissonSolver.solver", solverStr);
    }

    /* Initialize the infrastructure */
    amrex::Real poisson_solve (int n)
    {
        Gempic::Io::Parameters parameters;
        // Initialize computational_domain
        amrex::IntVect const nCell{AMREX_D_DECL(n, n, n)};
        auto infra = Gempic::Test::Utils::get_compdom(nCell);

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, s_hodgeDegree, s_maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        auto poisson{make_poisson_solver(deRham, infra)};

        DeRhamField<Grid::dual, Space::cell> rho(deRham, m_funcRho);
        DeRhamField<Grid::primal, Space::node> phi(deRham);
        DeRhamField<Grid::primal, Space::node> phiAn(deRham, m_funcPhi);

        amrex::Array4<amrex::Real> const& barr = rho.m_data[0].array();
        amrex::Real temp = barr(0, 0, 0, 0);
        barr(0, 0, 0, 0) = 0.0;

        phi.m_data.setVal(0.0);
        poisson->solve(phi, rho);

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
    int const coarse = 32, fine = 64;
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
} // namespace
