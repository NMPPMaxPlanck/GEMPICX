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
using namespace Particle;

/**
 * @brief Calculates Epar and Eperp for the
 * DK QN model, for Bext in any direction.
 * RHS is simply an analytical function.
 */
template <typename SplineDegreeStruct>
class HypreDKQN_E_Test : public testing::Test
{
public:
    static constexpr int s_vdim{3};
    static constexpr int s_ndata{1};

    static constexpr int s_degX{std::tuple_element_t<0, SplineDegreeStruct>::value};
    static constexpr int s_degY{std::tuple_element_t<1, SplineDegreeStruct>::value};
    static constexpr int s_degZ{std::tuple_element_t<2, SplineDegreeStruct>::value};

    static constexpr int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};
    static constexpr int s_hodgeDegree{2};

    static int const s_nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcE;
    amrex::Array<amrex::Parser, 3> m_parserE;

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcB;
    amrex::Array<amrex::Parser, 3> m_parserB;

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcRHS;
    amrex::Array<amrex::Parser, 3> m_parserRHS;

    amrex::Vector<amrex::Real> m_bext;
    std::vector<std::string> m_speciesNames{{"electrons"}};

    Gempic::Io::Parameters m_parameters;

    HypreDKQN_E_Test()
    {
        amrex::Array<std::string, 3> analyticalE, analyticalB, analyticalRHS;

        // The Gtest works for all three directions of background magnetic field
        int bdir = 0;
        switch (bdir)
        {
            case 0:
                // Bext = Bx
                m_bext = {1.0, 0.0, 0.0};
#if AMREX_SPACEDIM == 2
                analyticalE = {"1.0*sin(x)*cos(y)", "-(-2.0*sin(x)*cos(y))*(1.0+0.5*sin(y))",
                               "(-0.5-1.0*cos(x)*sin(y))*(1.0+0.5*sin(y))"};
                analyticalB = {"0.5*sin(y)", "0.0", "0.0"};
                analyticalRHS = {"2.0*sin(x)*cos(y)", "0.5+1.0*cos(x)*sin(y)", "2.0*sin(x)*cos(y)"};
#elif AMREX_SPACEDIM == 3
                analyticalE = {"1.0*sin(y)*cos(z)", "-(-2.0*sin(y)*cos(z))*(1.0+0.5*sin(z))",
                               "(-0.5-1.0*cos(y)*sin(z))*(1.0+0.5*sin(z))"};
                analyticalB = {"0.5*sin(z)", "0.0", "0.0"};
                analyticalRHS = {"3.0*sin(y)*cos(z)", "0.5+1.0*cos(y)*sin(z)", "2.0*sin(y)*cos(z)"};
#endif
                break;

            case 1:
                // Bext = By
                m_bext = {0.0, 1.0, 0.0};
#if AMREX_SPACEDIM == 2
                analyticalE = {"(-0.5-1.0*cos(x)*sin(y))*(1.0+0.5*sin(y))", "1.0*sin(x)*cos(y)",
                               "-(-2.0*sin(x)*cos(y))*(1.0+0.5*sin(y))"};
                analyticalB = {"0.0", "0.5*sin(y)", "0.0"};
                analyticalRHS = {"2.0*sin(x)*cos(y)", "2.0*sin(x)*cos(y)", "0.5+1.0*cos(x)*sin(y)"};
#elif AMREX_SPACEDIM == 3
                analyticalE = {"(-0.5-1.0*cos(z)*sin(x))*(1.0+0.5*sin(x))", "1.0*sin(x)*cos(z)",
                               "-(-2.0*sin(z)*cos(x))*(1.0+0.5*sin(x))"};
                analyticalB = {"0.0", "0.5*sin(x)", "0.0"};
                analyticalRHS = {"2.0*sin(z)*cos(x)", "3.0*sin(x)*cos(z)", "0.5+1.0*cos(z)*sin(x)"};
#endif
                break;

            case 2:
                // Bext = Bz
                m_bext = {0.0, 0.0, 1.0};
#if AMREX_SPACEDIM == 2
                analyticalE = {"-(-2.0*sin(x)*cos(y))*(1.0+0.5*sin(y))",
                               "(-0.5-1.0*cos(x)*sin(y))*(1.0+0.5*sin(y))", "1.0*sin(x)*cos(y)"};
                analyticalB = {"0.0", "0.0", "0.5*sin(y)"};
                analyticalRHS = {"0.5+1.0*cos(x)*sin(y)", "2.0*sin(x)*cos(y)", "3.0*sin(x)*cos(y)"};
#elif AMREX_SPACEDIM == 3
                analyticalE = {"-(-2.0*sin(x)*cos(y))*(1.0+0.5*sin(y))",
                               "(-0.5-1.0*cos(x)*sin(y))*(1.0+0.5*sin(y))", "1.0*sin(x)*cos(y)"};
                analyticalB = {"0.0", "0.0", "0.5*sin(y)"};
                analyticalRHS = {"0.5+1.0*cos(x)*sin(y)", "2.0*sin(x)*cos(y)", "3.0*sin(x)*cos(y)"};
#endif
                break;
        }

        for (int i = 0; i < 3; ++i)
        {
            m_parserE[i].define(analyticalE[i]);
            m_parserE[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcE[i] = m_parserE[i].compile<s_nVar>();

            m_parserB[i].define(analyticalB[i]);
            m_parserB[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcB[i] = m_parserB[i].compile<s_nVar>();

            m_parserRHS[i].define(analyticalRHS[i]);
            m_parserRHS[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcRHS[i] = m_parserRHS[i].compile<s_nVar>();
        }

        // SINGLE SPECIES
        m_parameters.set("Particle.speciesNames", m_speciesNames);

        amrex::Real charge{-1.0};
        m_parameters.set("Particle.electrons.charge", charge);

        amrex::Real mass{1.0};
        m_parameters.set("Particle.electrons.mass", mass);

        std::string solver{"GMRES"}; // do not use BiCGSTAB !!
        m_parameters.set("Eperp_particle_asymm.hypre_solver", solver);
        m_parameters.set("Epar_particle.hypre_solver", solver);

        std::string prec{"none"};
        m_parameters.set("Eperp_particle_asymm.hypre_preconditioner", prec);
        m_parameters.set("Epar_particle.hypre_preconditioner", prec);

        int numkry{200};
        m_parameters.set("Eperp_particle_asymm.num_krylov", numkry);
        m_parameters.set("Epar_particle.num_krylov", numkry);

        int maxitr{100};
        m_parameters.set("Eperp_particle_asymm.max_iterations", maxitr);
        m_parameters.set("Epar_particle.max_iterations", maxitr);

        int verbose{0};
        m_parameters.set("Eperp_particle_asymm.verbose", verbose);
        m_parameters.set("Epar_particle.verbose", verbose);

        int logging{1};
        m_parameters.set("Eperp_particle_asymm.logging", logging);
        m_parameters.set("Epar_particle.logging", logging);

        int euclidlevel{1};
        m_parameters.set("Eperp_particle_asymm.euclid_level", euclidlevel);
        m_parameters.set("Epar_particle.euclid_level", euclidlevel);

        int jacobi{0};
        m_parameters.set("Eperp_particle_asymm.euclid_use_block_jacobi", jacobi);
        m_parameters.set("Epar_particle.euclid_use_block_jacobi", jacobi);

        // Hybrid solver parameters
        int hybridSolver{2}; // Default:1(PCG) 2(GMRES) 3(BiCGSTAB) // do no use PCG or BiCGSTAB !!
        m_parameters.set("Eperp_particle_asymm.hybrid_solver_type", hybridSolver);
        m_parameters.set("Epar_particle.hybrid_solver_type", hybridSolver);

        // TimeLoop parameters for the DKQuasineutralSolver class
        m_parameters.set("TimeLoop.timestepping", "onlyrk");
        m_parameters.set("ECalculation.chargeMatrixTypeEperp", "particle");
        m_parameters.set("ECalculation.chargeMatrixTypeEpar", "particle");
    }

    template <int n>
    void exboperator_solve (amrex::Real& errorPdir0,
                            amrex::Real& errorPdir1,
                            amrex::Real& errorBdir)
    {
        // Initialize computational_domain
        amrex::IntVect const nCell{AMREX_D_DECL(n, n, n)};
        auto infra = Gempic::Test::Utils::get_compdom(nCell);

        // Initialize particle groups
        std::vector<std::shared_ptr<ParticleSpecies<s_vdim>>>
            ions; // Use 'init_particles(infra, ions);' if adding large number of particles randomly

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, s_hodgeDegree, s_maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        int numspec = 1;
        ions.resize(numspec);
        for (int spec{0}; spec < numspec; spec++)
        {
            ions[spec] = std::make_shared<ParticleSpecies<s_vdim>>(m_speciesNames[spec], infra);
        }

#if AMREX_SPACEDIM == 2
        int const percelldir{3};
#elif AMREX_SPACEDIM == 3
        int const percelldir{2};
#endif

        auto dx{infra.cell_size_3darray()};

        Gempic::Test::Utils::hypre_dk_add_particles<n, percelldir>(ions, infra);

        amrex::Real relTol{1.e-15};
        amrex::Real accTol{1.e-8};
        DKQuasineutralCalculationE<s_hodgeDegree, s_vdim, s_ndata, s_degX, s_degY, s_degZ> hypreE(
            infra, deRham, relTol, accTol);

        DeRhamField<Grid::dual, Space::face> rhs(deRham, m_funcRHS);
        DeRhamField<Grid::primal, Space::edge> E(deRham);
        DeRhamField<Grid::primal, Space::face> B(deRham, m_funcB);
        DeRhamField<Grid::primal, Space::edge> eAn(deRham, m_funcE);

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

        hypreE.solve_de_fi_model_eperp_particle_combo(E, rhs, B, ions, bdir, pdir);
        hypreE.solve_de_fi_model_epar_particle(E, rhs, ions, bdir);

        E -= eAn;

        auto dxi{infra.inv_cell_size_3darray()};

        errorPdir0 = Utils::gempic_norm(E.m_data[pdir[0]], infra, 1) * dxi[pdir[0]];
        errorPdir1 = Utils::gempic_norm(E.m_data[pdir[1]], infra, 1) * dxi[pdir[1]];
        errorBdir = Utils::gempic_norm(E.m_data[bdir], infra, 1) * dxi[bdir];
    }
};

using MyTypes = ::testing::Types<std::tuple<std::integral_constant<int, 1>,
                                            std::integral_constant<int, 1>,
                                            std::integral_constant<int, 1>>,
                                 std::tuple<std::integral_constant<int, 3>,
                                            std::integral_constant<int, 3>,
                                            std::integral_constant<int, 3>>,
                                 std::tuple<std::integral_constant<int, 1>,
                                            std::integral_constant<int, 2>,
                                            std::integral_constant<int, 3>>>;

TYPED_TEST_SUITE(HypreDKQN_E_Test, MyTypes);

TYPED_TEST(HypreDKQN_E_Test, HypreDKQN_E_Convergence)
{
#if AMREX_SPACEDIM == 2
    int const coarse = 24, fine = 48;
#elif AMREX_SPACEDIM == 3
    int const coarse = 10, fine = 20;
#endif
    amrex::Real error0Coarse, error1Coarse, errorparCoarse;
    amrex::Real error0Fine, error1Fine, errorparFine;
    amrex::Real tol = 0.2;

    amrex::Real rateOfConvergence0, rateOfConvergence1, rateOfConvergencepar;
    constexpr int splineDegreeX = TestFixture::s_degX;
    constexpr int splineDegreeY = TestFixture::s_degY;
    constexpr int splineDegreeZ = TestFixture::s_degZ;

    this->template exboperator_solve<coarse>(error0Coarse, error1Coarse, errorparCoarse);
    this->template exboperator_solve<fine>(error0Fine, error1Fine, errorparFine);

    rateOfConvergence0 = std::log2(error0Coarse / error0Fine);
    rateOfConvergence1 = std::log2(error1Coarse / error1Fine);
    rateOfConvergencepar = std::log2(errorparCoarse / errorparFine);

    amrex::Print() << "error0Coarse: " << error0Coarse << "  error0Fine: " << error0Fine
                   << "  rate_of_convergence0_<" << splineDegreeX << "," << splineDegreeY << ","
                   << splineDegreeZ << ">:" << rateOfConvergence0 << "\n";
    amrex::Print() << "error1Coarse: " << error1Coarse << "  error1Fine: " << error1Fine
                   << "  rate_of_convergence1_<" << splineDegreeX << "," << splineDegreeY << ","
                   << splineDegreeZ << ">:" << rateOfConvergence1 << "\n";
    amrex::Print() << "errorparCoarse: " << errorparCoarse << "  errorparFine: " << errorparFine
                   << "  rate_of_convergencepar_<" << splineDegreeX << "," << splineDegreeY << ","
                   << splineDegreeZ << ">:" << rateOfConvergencepar << "\n";

    EXPECT_NEAR(rateOfConvergence0, 2, tol);
    EXPECT_NEAR(rateOfConvergence1, 2, tol);
    EXPECT_NEAR(rateOfConvergencepar, 2, tol);
}
} // namespace
