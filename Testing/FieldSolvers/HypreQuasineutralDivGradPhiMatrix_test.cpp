/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_Particles.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_QuasineutralSolver.H"
#include "GEMPIC_Sampler.H"
#include "GEMPIC_SplineClass.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace ParticleMeshCoupling;
using namespace FieldSolvers;

/**
 * @brief Tests the DivGradV matrix of the quasineutral solver.
 * The rho*gradV is deposited from particles.
 */
template <typename SplineDegreeStruct>
class HypreQuasineutralDivGradPhiMatrixTest : public testing::Test
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

    amrex::Parser m_parserPhi;
    amrex::ParserExecutor<s_nVar> m_funcPhi;

    HypreQuasineutralDivGradPhiMatrixTest()
    {
#if AMREX_SPACEDIM == 2
        /*/ SINGLE SPECIES
        const std::string analyticalPhi = "sqrt(3.0)*2.0*(cos(x) + cos(y))";//*/
        // DOUBLE SPECIES
        std::string const analyticalPhi =
            "((1.0 - sqrt(2.0))/(0.5 + 2.0/sqrt(5.0)))*2.0*(cos(x) + cos(y))"; //*/
#elif AMREX_SPACEDIM == 3
        /*/ SINGLE SPECIES
        const std::string analyticalPhi = "sqrt(3.0)*2.0*(cos(x) + cos(y) + cos(z))";//*/
        // DOUBLE SPECIES
        std::string const analyticalPhi =
            "((1.0 - sqrt(2.0))/(0.5 + 2.0/sqrt(5.0)))*2.0*(cos(x) + cos(y) + cos(z))"; //*/
#endif

        m_parserPhi.define(analyticalPhi);
        m_parserPhi.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        m_funcPhi = m_parserPhi.compile<s_nVar>();

        Gempic::Io::Parameters parameters;

        /*/ SINGLE SPECIES
        std::string speciesNames{"species0"};
        parameters.set("Particle.speciesNames", speciesNames);

        amrex::Real charge{-sqrt(3.0)};
        parameters.set("Particle.species0.charge", charge);

        amrex::Real mass{3.0};
        parameters.set("Particle.species0.mass", mass);//*/

        // DOUBLE SPECIES
        amrex::Vector<std::string> const speciesNames{"species0", "species1"};
        parameters.set("Particle.speciesNames", speciesNames);

        amrex::Real charge0{-1.0};
        parameters.set("Particle.species0.charge", charge0);

        amrex::Real mass0{2.0};
        parameters.set("Particle.species0.mass", mass0);

        amrex::Real charge1{sqrt(2.0)};
        parameters.set("Particle.species1.charge", charge1);

        amrex::Real mass1{sqrt(5.0)};
        parameters.set("Particle.species1.mass", mass1); //*/
    }

    template <int n>
    amrex::Real divgradphimatrix_solve (amrex::Real& divJFinalNorm)
    {
        // For studies other than convergence, this should be in SetUpTestSuite under Grid
        // parameters
        Gempic::Io::Parameters parameters{};
        amrex::IntVect const nCell{AMREX_D_DECL(n, n, n)};

        // Initialize computational_domain
        auto infra = Gempic::Test::Utils::get_compdom(nCell);

        amrex::Real dt = 0.05;

        // Initialize particles
        std::vector<std::shared_ptr<ParticleSpecies<s_vdim>>>
            ions; // Use 'init_particles(infra, ions);' if adding large number of particles
                  // randomly

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, s_hodgeDegree, s_maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        // Computed fields
        DeRhamField<Grid::dual, Space::face> J(deRham);
        DeRhamField<Grid::dual, Space::cell> divJ(deRham);
        DeRhamField<Grid::primal, Space::node> phiCorr(deRham);
        DeRhamField<Grid::primal, Space::edge> eCorr(deRham);

        // Analytical fields
        DeRhamField<Grid::primal, Space::node> phiCorrAn(deRham, m_funcPhi);

        // Adding AMREX_SPACEDIM individual particles starts here
        /*/ SINGLE SPECIES
        int numspec = 1;//*/
        // DOUBLE SPECIES
        int numspec = 2; //*/

        ions.resize(numspec);
        for (int spec{0}; spec < numspec; spec++)
        {
            ions[spec] = std::make_shared<ParticleSpecies<s_vdim>>(spec, infra);
        }

        int const percelldir{3}; // GTest works for any value of percelldir

        int const percell{GEMPIC_D_MULT(percelldir, percelldir, percelldir)};
        int const numParticles{percell * GEMPIC_D_MULT(n, n, n)};

        amrex::Vector<amrex::Real> oss[percell];

        // Any value of 'off' in [0,1] works for deg 1,2,3
        amrex::Real off(0.5);

        GEMPIC_D_LOOP_BEGIN(for (int i = 0; i < percelldir; i++),
                            for (int j = 0; j < percelldir; j++),
                            for (int k = 0; k < percelldir; k++))
            oss[AMREX_D_TERM(i, +j * percelldir, +k * percelldir * percelldir)] = {AMREX_D_DECL(
                (i + off) / percelldir, (j + off) / percelldir, (k + off) / percelldir)};
        GEMPIC_D_LOOP_END

        amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions;
        amrex::Array<amrex::GpuArray<amrex::Real, s_vdim>, numParticles> velocities;
        amrex::Array<amrex::Real, numParticles> weights;

        int i{0}, j{0}, k{0};
        auto const& dx{infra.cell_size_array()};

        GEMPIC_D_LOOP_BEGIN(for (i = 0; i < n; i++), for (j = 0; j < n; j++),
                            for (k = 0; k < n; k++))

            for (int p = 0; p < percell; p++)
            {
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> loc = {
                    AMREX_D_DECL(infra.geometry().ProbLo(xDir) + (i + oss[p][0]) * dx[xDir],
                                 infra.geometry().ProbLo(yDir) + (j + oss[p][1]) * dx[yDir],
                                 infra.geometry().ProbLo(zDir) + (k + oss[p][2]) * dx[zDir])};

                positions[p * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] = {
                    AMREX_D_DECL(loc[xDir], loc[yDir], loc[zDir])};
#if AMREX_SPACEDIM == 2 // analyticalDivJ = "-sqrt(3.0)*0.1*(cos(x) + cos(y))"
                velocities[p * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] = {
                    0.1 * sin(loc[xDir]), 0.1 * sin(loc[yDir]), 0.0};
#elif AMREX_SPACEDIM == 3 //analyticalDivJ = "-sqrt(3.0)*0.1*(cos(x) + cos(y) + cos(z))"
                velocities[p * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] = {
                    0.1 * sin(loc[xDir]), 0.1 * sin(loc[yDir]), 0.1 * sin(loc[zDir])};
#endif
                weights[p * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] =
                    (1.0 / percell) * infra.cell_volume();
            }
        GEMPIC_D_LOOP_END

        for (int spec{0}; spec < numspec; spec++)
        {
            Gempic::Test::Utils::add_single_particles(ions[spec].get(), infra, weights, positions,
                                                      velocities);
            ions[spec]->Redistribute();
        }
        // Adding AMREX_SPACEDIM individual particles ends here

        QuasineutralSolver<s_hodgeDegree, s_vdim, s_ndata, s_degX, s_degY, s_degZ>
            hypreParticleDivGradV(infra, deRham);

        hypreParticleDivGradV.push_particles_and_correct_div_j(J, phiCorr, ions, dt);

        div(divJ, J);
        divJFinalNorm = Utils::gempic_norm(divJ.m_data, infra, 2);

        phiCorr -= phiCorrAn;
        return Utils::gempic_norm(phiCorr.m_data, infra, 2);
    }
};

// All 27 permutations of 1,2,3 work for spline degrees
using MyTypes = ::testing::Types<std::tuple<std::integral_constant<int, 1>,
                                            std::integral_constant<int, 1>,
                                            std::integral_constant<int, 1>>,
                                 std::tuple<std::integral_constant<int, 2>,
                                            std::integral_constant<int, 2>,
                                            std::integral_constant<int, 2>>,
                                 std::tuple<std::integral_constant<int, 3>,
                                            std::integral_constant<int, 3>,
                                            std::integral_constant<int, 3>>,
                                 std::tuple<std::integral_constant<int, 1>,
                                            std::integral_constant<int, 2>,
                                            std::integral_constant<int, 3>>>;

TYPED_TEST_SUITE(HypreQuasineutralDivGradPhiMatrixTest, MyTypes);

TYPED_TEST(HypreQuasineutralDivGradPhiMatrixTest, HypreQuasineutralDivGradPhiMatrix)
{
    constexpr int coarse = 8;
    constexpr int fine = 16;
    amrex::Real errorCoarse, errorFine;
    amrex::Real tol = 0.15;

    amrex::Real divJFinalNormCoarse, divJFinalNormFine;
    amrex::Real divJTol = 1.0e-14;

    amrex::Real rateOfConvergence;
    constexpr int splineDegreeX = TestFixture::s_degX;
    constexpr int splineDegreeY = TestFixture::s_degY;
    constexpr int splineDegreeZ = TestFixture::s_degZ;

    errorCoarse = this->template divgradphimatrix_solve<coarse>(divJFinalNormCoarse);
    ASSERT_LT(divJFinalNormCoarse, divJTol);
    amrex::Print() << "errorCoarse: " << errorCoarse
                   << ". Divergence of J after particle push: " << divJFinalNormCoarse << "\n";

    errorFine = this->template divgradphimatrix_solve<fine>(divJFinalNormFine);
    ASSERT_LT(divJFinalNormFine, divJTol);
    amrex::Print() << "errorFine: " << errorFine
                   << ". Divergence of J after particle push: " << divJFinalNormFine << "\n";

    rateOfConvergence = std::log2(errorCoarse / errorFine);
    amrex::Print() << "rate_of_convergence_<" << splineDegreeX << "," << splineDegreeY << ","
                   << splineDegreeZ << ">:" << rateOfConvergence << "\n";
    EXPECT_NEAR(rateOfConvergence, 2, tol);
}

} // namespace
