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
#include "GEMPIC_QuasineutralSolver.H"
#include "GEMPIC_SplineClass.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;
using namespace FieldSolvers;

/**
 * @brief Tests the Poisson solver to obtain the
 * initial A-field for the quasineutral solver.
 */
template <typename HodgeDegreeStruct>
class HypreQuasineutralAPoissonTest : public testing::Test
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

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcAmean;
    amrex::Array<amrex::Parser, 3> m_parserAmean;

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcA;
    amrex::Array<amrex::Parser, 3> m_parserA;

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcJ;
    amrex::Array<amrex::Parser, 3> m_parserJ;

    HypreQuasineutralAPoissonTest()
    {
        amrex::Array<std::string, 3> const analyticalAmean = {"0.0", "0.0", "0.0"};
#if AMREX_SPACEDIM == 2
        amrex::Array<std::string, 3> const analyticalJ = {"sin(y)", "4*sin(2*x)",
                                                          "2*cos(x)*cos(y)"};
        amrex::Array<std::string, 3> const analyticalA = {"sin(y)", "sin(2*x)", "cos(x)*cos(y)"};
#elif AMREX_SPACEDIM == 3
        amrex::Array<std::string, 3> const analyticalJ = {"sin(y)", "4*sin(2*z)", "9*sin(3*x)"};
        amrex::Array<std::string, 3> const analyticalA = {"sin(y)", "sin(2*z)", "sin(3*x)"};
#endif

        for (int i = 0; i < 3; ++i)
        {
            m_parserAmean[i].define(analyticalAmean[i]);
            m_parserAmean[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcAmean[i] = m_parserAmean[i].compile<s_nVar>();

            m_parserA[i].define(analyticalA[i]);
            m_parserA[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcA[i] = m_parserA[i].compile<s_nVar>();

            m_parserJ[i].define(analyticalJ[i]);
            m_parserJ[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcJ[i] = m_parserJ[i].compile<s_nVar>();
        }
    }

    amrex::Real apoisson_operator_solve (int n, amrex::Real& divANorm, amrex::Real& jMinusCurlHNorm)
    {
        Gempic::Io::Parameters parameters;

        // Initialize computational_domain
        amrex::IntVect const nCell{AMREX_D_DECL(n, n, n)};

        auto infra = Gempic::Test::Utils::get_compdom(nCell);

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, s_hodgeDegree, s_maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        QuasineutralSolver<s_hodgeDegree, s_vdim, s_ndata, s_degX, s_degY, s_degZ> hyprePoissonA(
            infra, deRham);

        DeRhamField<Grid::dual, Space::face> J(deRham, m_funcJ);
        DeRhamField<Grid::dual, Space::face> A(deRham);
        DeRhamField<Grid::dual, Space::face> aAn(deRham, m_funcA);

        DeRhamField<Grid::primal, Space::edge> aPrimal(deRham);
        DeRhamField<Grid::primal, Space::face> B(deRham);
        DeRhamField<Grid::dual, Space::edge> H(deRham);
        DeRhamField<Grid::dual, Space::face> curlH(deRham);
        DeRhamField<Grid::dual, Space::face> jMinusCurlH(deRham);
        DeRhamField<Grid::dual, Space::cell> divA(deRham);

        hyprePoissonA.solve_negative_poisson_equation(
            J, A); // divJ and Javg must be zero for any meaningful results

        hodge(aPrimal, A, deRham->scaling_dto_e());
        curl(B, aPrimal);
        hodge(H, B, deRham->scaling_bto_h());
        curl(curlH, H);

        div(divA, A);

        divANorm = Utils::gempic_norm(divA.m_data, infra, 2);

        linear_combination(jMinusCurlH, 1.0, J, -1.0, curlH);
        jMinusCurlHNorm = Utils::gempic_norm(jMinusCurlH.m_data[xDir], infra, 1) +
                          Utils::gempic_norm(jMinusCurlH.m_data[yDir], infra, 1) +
                          Utils::gempic_norm(jMinusCurlH.m_data[zDir], infra, 1);

        A -= aAn;

        amrex::GpuArray<amrex::Real, 3> dxi{GEMPIC_D_PAD_ONE(infra.geometry().InvCellSize(xDir),
                                                             infra.geometry().InvCellSize(yDir),
                                                             infra.geometry().InvCellSize(zDir))};

        return Utils::gempic_norm (A.m_data[xDir], infra, 1) * dxi[yDir] * dxi[zDir] +
               Utils::gempic_norm(A.m_data[yDir], infra, 1) * dxi[xDir] * dxi[zDir] +
               Utils::gempic_norm(A.m_data[zDir], infra, 1) * dxi[xDir] * dxi[yDir];
    }
};

using MyTypes = ::testing::Types<std::integral_constant<int, 2>>;
TYPED_TEST_SUITE(HypreQuasineutralAPoissonTest, MyTypes);

TYPED_TEST(HypreQuasineutralAPoissonTest, HypreQuasineutralAPoissonConvergence)
{
#if AMREX_SPACEDIM == 2
    int const coarse = 16, fine = 32;
#elif AMREX_SPACEDIM == 3
    int const coarse = 8, fine = 16;
#endif
    amrex::Real errorCoarse, errorFine;

    amrex::Real divANormCoarse, divANormFine, jMinusCurlHNormCoarse, jMinusCurlHNormFine;
    amrex::Real tol = 0.1;
    amrex::Real mtol = 1.0e-14;

    amrex::Real rateOfConvergence;
    constexpr int hodgeDegree = TestFixture::s_hodgeDegree;

    errorCoarse = this->apoisson_operator_solve(coarse, divANormCoarse, jMinusCurlHNormCoarse);
    amrex::Print() << "errorCoarse: " << errorCoarse << "\n";
    amrex::Print() << "Divergence of A (Coarse): " << divANormCoarse << "\n";
    amrex::Print() << "J - CurlH error (Coarse): " << jMinusCurlHNormCoarse << "\n";
    ASSERT_LT(divANormCoarse, mtol);
    ASSERT_LT(jMinusCurlHNormCoarse, mtol);

    errorFine = this->apoisson_operator_solve(fine, divANormFine, jMinusCurlHNormFine);

    amrex::Print() << "errorFine: " << errorFine << "\n";
    amrex::Print() << "Divergence of A (Fine): " << divANormFine << "\n";
    amrex::Print() << "J - CurlH error (Fine): " << jMinusCurlHNormFine << "\n";
    ASSERT_LT(divANormFine, mtol);
    ASSERT_LT(jMinusCurlHNormFine, mtol);

    rateOfConvergence = std::log2(errorCoarse / errorFine);

    amrex::Print() << "rate_of_convergence_" << hodgeDegree << ':' << rateOfConvergence << "\n";
    EXPECT_NEAR(rateOfConvergence, hodgeDegree, tol);
}
} // namespace
