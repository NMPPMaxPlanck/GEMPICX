/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_DKQuasineutralSolver.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_SplineClass.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;
using namespace FieldSolvers;

/**
 * @brief Calculates accuracy of Crank Nicolson
 * time-stepping for B in Faraday's equation
 */
template <typename HodgeDegreeStruct>
class HypreDKQN_Faraday_CN_Test : public testing::Test
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

    amrex::Parser m_parserRhoe;
    amrex::ParserExecutor<s_nVar> m_funcRhoe;

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcB;
    amrex::Array<amrex::Parser, 3> m_parserB;

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcRHSAn;
    amrex::Array<amrex::Parser, 3> m_parserRHSAn;

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcRHStest;
    amrex::Array<amrex::Parser, 3> m_parserRHStest;

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcBfAn;
    amrex::Array<amrex::Parser, 3> m_parserBfAn;

    amrex::Vector<amrex::Real> m_bext;

    HypreDKQN_Faraday_CN_Test()
    {
        std::string const analyticalRhoe = "-1.0";
        amrex::Array<std::string, 3> analyticalB, analyticalRHSAn, analyticalRHStest,
            analyticalBfAn;

#if AMREX_SPACEDIM == 2
        analyticalB = {"sin(y)", "sin(x)", "sin(x)*sin(y)"};
        // The 2D Gtest works for X, Y directions of background magnetic field
        int bdir = 0;
        switch (bdir)
        {
            case 0:
                // 1) Bext = Bx
                m_bext = {1.0, 0.0, 0.0};
                analyticalRHSAn = {"1.0 + sin(y) - 0.5*(-1.0)*(-cos(x)*cos(y))",
                                   "sin(x) - 0.5*(-1.0)*(-sin(y)*sin(x))",
                                   "sin(x)*sin(y) - 0.5*(-1.0)*(sin(x))"};
                analyticalRHStest = {"sin(x)*sin(y) - 0.5*(-cos(x)*cos(y))",
                                     "sin(x) - 0.5*(-sin(x)*sin(y))",
                                     "sin(x)*sin(y) - 0.5*(sin(x) + cos(x)*cos(y))"};
                analyticalBfAn = {"sin(x)*sin(y)", "sin(x)", "sin(x)*sin(y)"};
                break;

            case 1:
                // 2) Bext = By
                m_bext = {0.0, 1.0, 0.0};
                analyticalRHSAn = {"sin(y) - 0.5*(-1.0)*(sin(x)*sin(y))",
                                   "1.0 + sin(x) - 0.5*(-1.0)*(cos(y)*cos(x))",
                                   "sin(x)*sin(y) - 0.5*(-1.0)*(-sin(y))"};
                analyticalRHStest = {"sin(y) - 0.5*(sin(x)*sin(y))",
                                     "sin(x)*sin(y) - 0.5*(cos(x)*cos(y))",
                                     "sin(x)*sin(y) - 0.5*(-sin(y) - cos(y)*cos(x))"};
                analyticalBfAn = {"sin(y)", "sin(x)*sin(y)", "sin(x)*sin(y)"};
                break;
        };
#elif AMREX_SPACEDIM == 3
        analyticalB = {"sin(y)*sin(z)", "sin(z)*sin(x)", "sin(x)*sin(y)"};

        // The 3D Gtest works for all 3 directions of background magnetic field
        int bdir = 0;
        switch (bdir)
        {
            case 0:
                // 1) Bext = Bx
                m_bext = {1.0, 0.0, 0.0};
                analyticalRHSAn = {"1.0 + sin(y)*sin(z) - 0.5*(-1.0)*(cos(x)*(cos(z)-cos(y)))",
                                   "sin(z)*sin(x) - 0.5*(-1.0)*(-sin(y)*sin(x))",
                                   "sin(x)*sin(y) - 0.5*(-1.0)*(sin(z)*sin(x))"};

                analyticalRHStest = {"sin(x)*sin(y)*sin(z) - 0.5*(cos(x)*(cos(z) - cos(y)))",
                                     "sin(x)*sin(z) - 0.5*(-sin(y)*(sin(x) + cos(x)*cos(z)))",
                                     "sin(x)*sin(y) - 0.5*(sin(z)*(sin(x) + cos(x)*cos(y)))"};
                analyticalBfAn = {"sin(x)*sin(y)*sin(z)", "sin(x)*sin(z)", "sin(x)*sin(y)"};
                break;

            case 1:
                // 2) Bext = By
                m_bext = {0.0, 1.0, 0.0};
                analyticalRHSAn = {"sin(y)*sin(z) - 0.5*(-1.0)*(sin(x)*sin(y))",
                                   "1.0 + sin(z)*sin(x) - 0.5*(-1.0)*(cos(y)*(cos(x)-cos(z)))",
                                   "sin(x)*sin(y) - 0.5*(-1.0)*(-sin(z)*sin(y))"};

                analyticalRHStest = {"sin(y)*sin(z) - 0.5*(sin(x)*(sin(y) + cos(y)*cos(z)))",
                                     "sin(x)*sin(y)*sin(z) - 0.5*(cos(y)*(cos(x) - cos(z)))",
                                     "sin(x)*sin(y) - 0.5*(-sin(z)*(sin(y) + cos(y)*cos(x)))"};
                analyticalBfAn = {"sin(y)*sin(z)", "sin(x)*sin(y)*sin(z)", "sin(x)*sin(y)"};
                break;

            case 2:
                // 3) Bext = Bz
                m_bext = {0.0, 0.0, 1.0};
                analyticalRHSAn = {"sin(y)*sin(z) - 0.5*(-1.0)*(-sin(x)*sin(z))",
                                   "sin(z)*sin(x) - 0.5*(-1.0)*(sin(y)*sin(z))",
                                   "1.0 + sin(x)*sin(y) - 0.5*(-1.0)*(cos(z)*(cos(y)-cos(x)))"};

                analyticalRHStest = {"sin(y)*sin(z) - 0.5*(-sin(x)*(sin(z) + cos(z)*cos(y)))",
                                     "sin(x)*sin(z) - 0.5*(sin(y)*(sin(z) + cos(z)*cos(x)))",
                                     "sin(x)*sin(y)*sin(z) - 0.5*(cos(z)*(cos(y) - cos(x)))"};
                analyticalBfAn = {"sin(y)*sin(z)", "sin(x)*sin(z)", "sin(x)*sin(y)*sin(z)"};
                break;
        }
#endif

        m_parserRhoe.define(analyticalRhoe);
        m_parserRhoe.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        m_funcRhoe = m_parserRhoe.compile<s_nVar>();

        for (int i = 0; i < 3; ++i)
        {
            m_parserB[i].define(analyticalB[i]);
            m_parserB[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcB[i] = m_parserB[i].compile<s_nVar>();

            m_parserRHSAn[i].define(analyticalRHSAn[i]);
            m_parserRHSAn[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcRHSAn[i] = m_parserRHSAn[i].compile<s_nVar>();

            m_parserRHStest[i].define(analyticalRHStest[i]);
            m_parserRHStest[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcRHStest[i] = m_parserRHStest[i].compile<s_nVar>();

            m_parserBfAn[i].define(analyticalBfAn[i]);
            m_parserBfAn[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcBfAn[i] = m_parserBfAn[i].compile<s_nVar>();
        }

        Gempic::Io::Parameters parameters;

        amrex::Vector<std::string> const speciesNames{"electrons", "ions"};
        parameters.set("Particle.speciesNames", speciesNames);

        //
        std::string solver{"GMRES"};
        parameters.set("faraday_crank_nicolson.hypre_solver", solver);

        std::string prec{"none"};
        parameters.set("faraday_crank_nicolson.hypre_preconditioner", prec);

        int numkry{200};
        parameters.set("faraday_crank_nicolson.num_krylov", numkry);

        int maxitr{100};
        parameters.set("faraday_crank_nicolson.max_iterations", maxitr);

        int verbose{0};
        parameters.set("faraday_crank_nicolson.verbose", verbose);

        int logging{1};
        parameters.set("faraday_crank_nicolson.logging", logging);

        int euclidlevel{1};
        parameters.set("faraday_crank_nicolson.euclid_level", euclidlevel);

        int jacobi{0};
        parameters.set("faraday_crank_nicolson.euclid_use_block_jacobi", jacobi);

        // Hybrid solver parameters
        int hybridSolver{2}; // Default:1(PCG) 2(GMRES) 3(BiCGSTAB) // do no use PCG or BiCGSTAB !!
        parameters.set("faraday_crank_nicolson.hybrid_solver_type", hybridSolver);

        // TimeLoop parameters for the DKQuasineutralSolver class
        parameters.set("TimeLoop.timestepping", "onlyrk");
        parameters.set("ECalculation.chargeMatrixTypeEperp", "particle");
        parameters.set("ECalculation.chargeMatrixTypeEpar", "particle");
    }

    template <int n>
    void faraday_c_noperator_solve (amrex::Real& errorRHS, amrex::Real& errorBf)
    {
        Gempic::Io::Parameters parameters;
        // Initialize computational_domain
        amrex::IntVect const nCell{AMREX_D_DECL(n, n, n)};
        auto infra = Gempic::Test::Utils::get_compdom(nCell);

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, s_hodgeDegree, s_maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        DKQuasineutralTimestepping<s_hodgeDegree, s_vdim, s_ndata, s_degX, s_degY, s_degZ>
            hypreFaradayTimestepper(infra, deRham);

        DeRhamField<Grid::dual, Space::cell> rhoe(deRham, m_funcRhoe);
        std::vector<DeRhamField<Grid::dual, Space::cell>> rhoVec;
        rhoVec.emplace_back(deRham);

        int indexE = 0;
        copy(rhoVec[indexE], rhoe);

        DeRhamField<Grid::primal, Space::face> B(deRham, m_funcB);
        DeRhamField<Grid::dual, Space::edge> H(deRham);
        DeRhamField<Grid::dual, Space::face> curlH(deRham);
        DeRhamField<Grid::primal, Space::face> rhs(deRham);
        DeRhamField<Grid::primal, Space::face> rhsAn(deRham, m_funcRHSAn);

        DeRhamField<Grid::primal, Space::face> rhstest(deRham, m_funcRHStest);
        DeRhamField<Grid::primal, Space::face> bftest(deRham);
        DeRhamField<Grid::primal, Space::face> bftestAn(deRham, m_funcBfAn);

        auto dx{infra.cell_size_3darray()};

        auto dxi{infra.inv_cell_size_3darray()};

        amrex::Vector<amrex::Real> bmean{m_bext[xDir] * dx[yDir] * dx[zDir],
                                         m_bext[yDir] * dx[xDir] * dx[zDir],
                                         m_bext[zDir] * dx[xDir] * dx[yDir]};
        B += bmean;

        int bdir{-1};
        for (int i = 0; i < s_vdim; ++i)
        {
            if (bmean[i] != 0.0)
            {
                bdir = i;
                break;
            }
        }
        GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(bdir != -1, "BMean is 0 in all directions.");
        std::array<int, 2> pdir;
        pdir[0] = (bdir + 1) % s_vdim;
        pdir[1] = (bdir + 2) % s_vdim;

        amrex::Real dt = 1.0;

        hypreFaradayTimestepper.de_ki_calculate_rhs_faraday_crank_nicolson(
            rhs, B, H, curlH, bdir, pdir, dt, rhoVec[indexE]);
        rhs -= rhsAn;

        errorRHS = (Utils::gempic_norm(rhs.m_data[bdir], infra, 1) * dxi[pdir[0]] * dxi[pdir[1]] +
                    Utils::gempic_norm(rhs.m_data[pdir[0]], infra, 1) * dxi[pdir[1]] * dxi[bdir] +
                    Utils::gempic_norm(rhs.m_data[pdir[1]], infra, 1) * dxi[bdir] * dxi[pdir[0]]);

        hypreFaradayTimestepper.de_ki_solve_faraday_crank_nicolson_b(bftest, rhstest, bdir, dt,
                                                                     rhoVec[indexE]);
        bftest -= bftestAn;

        errorBf = (Utils::gempic_norm(bftest.m_data[bdir], infra, 1) * dxi[pdir[0]] * dxi[pdir[1]] +
                   Utils::gempic_norm(bftest.m_data[pdir[0]], infra, 1) * dxi[pdir[1]] * dxi[bdir] +
                   Utils::gempic_norm(bftest.m_data[pdir[1]], infra, 1) * dxi[bdir] * dxi[pdir[0]]);
    }
};

using MyTypes = ::testing::Types<std::integral_constant<int, 2>>;
TYPED_TEST_SUITE(HypreDKQN_Faraday_CN_Test, MyTypes);

TYPED_TEST(HypreDKQN_Faraday_CN_Test, HypreDKQN_Faraday_CrankNicolson_Convergence)
{
#if AMREX_SPACEDIM == 2
    int const coarse = 24, fine = 48;
#elif AMREX_SPACEDIM == 3
    int const coarse = 12, fine = 24;
#endif
    amrex::Real errorCoarseRHS, errorFineRHS, errorCoarseBf, errorFineBf;
    amrex::Real tol = 0.1;
    amrex::Real rateOfConvergenceRHS, rateOfConvergenceBf;

    constexpr int hodgeDegree = TestFixture::s_hodgeDegree;

    this->template faraday_c_noperator_solve<coarse>(errorCoarseRHS, errorCoarseBf);
    this->template faraday_c_noperator_solve<fine>(errorFineRHS, errorFineBf);

    rateOfConvergenceRHS = std::log2(errorCoarseRHS / errorFineRHS);
    rateOfConvergenceBf = std::log2(errorCoarseBf / errorFineBf);

    amrex::Print() << "errorCoarseRHS: " << errorCoarseRHS << "  errorFineRHS: " << errorFineRHS
                   << "  rate_of_convergence_RHS_" << hodgeDegree << ":" << rateOfConvergenceRHS
                   << "\n";
    EXPECT_NEAR(rateOfConvergenceRHS, 2, tol);

    amrex::Print() << "errorCoarseBf: " << errorCoarseBf << "  errorFineBf: " << errorFineBf
                   << "  rate_of_convergence_Bf_" << hodgeDegree << ":" << rateOfConvergenceBf
                   << "\n";
    EXPECT_NEAR(rateOfConvergenceBf, 2, tol);
}
} // namespace
