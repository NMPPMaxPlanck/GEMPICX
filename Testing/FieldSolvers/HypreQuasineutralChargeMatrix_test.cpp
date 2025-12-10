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
 * @brief Tests the Charge matrix of the quasineutral solver.
 * The rho*E is deposited from particles.
 */
template <typename SplineDegreeStruct>
class HypreQuasineutralChargeMatrixTest : public testing::Test
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

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcRhoE;
    amrex::Array<amrex::Parser, 3> m_parserRhoE;

    HypreQuasineutralChargeMatrixTest()
    {
#if AMREX_SPACEDIM == 2
        amrex::Array<std::string, 3> const analyticalE = {"sin(y)", "sin(x)", "sin(x+y)"};
        /*/ SINGLE SPECIES
        amrex::Array<std::string, 3> const analyticalRhoE = {"sin(y)", "sin(x)", "sin(x+y)"};//*/
        // DOUBLE SPECIES
        amrex::Array<std::string, 3> const analyticalRhoE = {"(0.5+2.0/sqrt(5.0))*sin(y)",
                                                             "(0.5+2.0/sqrt(5.0))*sin(x)",
                                                             "(0.5+2.0/sqrt(5.0))*sin(x+y)"}; //*/
#elif AMREX_SPACEDIM == 3
        amrex::Array<std::string, 3> const analyticalE = {"sin(y)", "sin(z)", "sin(x)"};
        /*/ SINGLE SPECIES
        amrex::Array<std::string, 3> const analyticalRhoE = {"sin(y)", "sin(z)", "sin(x)"};//*/
        // DOUBLE SPECIES
        amrex::Array<std::string, 3> const analyticalRhoE = {"(0.5+2.0/sqrt(5.0))*sin(y)",
                                                             "(0.5+2.0/sqrt(5.0))*sin(z)",
                                                             "(0.5+2.0/sqrt(5.0))*sin(x)"}; //*/
#endif

        for (int i = 0; i < 3; ++i)
        {
            m_parserE[i].define(analyticalE[i]);
            m_parserE[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcE[i] = m_parserE[i].compile<s_nVar>();

            m_parserRhoE[i].define(analyticalRhoE[i]);
            m_parserRhoE[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            m_funcRhoE[i] = m_parserRhoE[i].compile<s_nVar>();
        }

        Gempic::Io::Parameters parameters;

        /*/ SINGLE SPECIES
        std::string speciesNames{"species0"};
        parameters.set("Particle.speciesNames", speciesNames);

        amrex::Real charge{-sqrt(2.0)};
        parameters.set("Particle.species0.charge", charge);

        amrex::Real mass{2.0};
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
    amrex::Real chargematrix_solve ()
    {
        Gempic::Io::Parameters parameters;
        amrex::IntVect const nCell{AMREX_D_DECL(n, n, n)};

        // Initialize computational_domain
        auto infra = Gempic::Test::Utils::get_compdom(nCell);

        // Initialize particles
        std::vector<std::shared_ptr<ParticleSpecies<s_vdim>>>
            ions; // Use 'init_particles(infra, ions);' if adding large number of particles
                  // randomly

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, s_hodgeDegree, s_maxSplineDegree,
                                                        HodgeScheme::FDHodge);

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

        int const numParticles{AMREX_SPACEDIM * GEMPIC_D_MULT(n, n, n)};

        amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions;
        amrex::Array<amrex::GpuArray<amrex::Real, s_vdim>, numParticles> velocities;
        amrex::Array<amrex::Real, numParticles> weights;

        // particles on edges or faces both works, although below implementation is with particles
        // on edges
        amrex::Vector<amrex::Real> offset[AMREX_SPACEDIM] = {
            AMREX_D_DECL({AMREX_D_DECL(0.5, 0.0, 0.0)}, {AMREX_D_DECL(0.0, 0.5, 0.0)},
                         {AMREX_D_DECL(0.0, 0.0, 0.5)})};

        int i{0}, j{0}, k{0};
        auto const dx{infra.cell_size_array()};

        for (int ndir = 0; ndir < AMREX_SPACEDIM; ++ndir)
        {
            GEMPIC_D_LOOP_BEGIN(for (i = 0; i < n; i++), for (j = 0; j < n; j++),
                                for (k = 0; k < n; k++))

                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> loc = {AMREX_D_DECL(
                    infra.geometry().ProbLo(xDir) + (i + offset[ndir][xDir]) * dx[xDir],
                    infra.geometry().ProbLo(yDir) + (j + offset[ndir][yDir]) * dx[yDir],
                    infra.geometry().ProbLo(zDir) + (k + offset[ndir][zDir]) * dx[zDir])};

                positions[ndir * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] = {
                    AMREX_D_DECL(loc[xDir], loc[yDir], loc[zDir])};
                velocities[ndir * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] = {0.0, 0.0, 0.0};
                weights[ndir * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] =
                    (1.0 / AMREX_SPACEDIM) * infra.cell_volume();
            GEMPIC_D_LOOP_END
        }

        for (int spec{0}; spec < numspec; spec++)
        {
            Gempic::Test::Utils::add_single_particles(ions[spec].get(), infra, weights, positions,
                                                      velocities);
            ions[spec]->Redistribute();
        }
        // Adding AMREX_SPACEDIM individual particles ends here

        // Computed fields
        DeRhamField<Grid::primal, Space::edge> E(deRham);

        // Analytical fields
        DeRhamField<Grid::primal, Space::edge> eAn(deRham, m_funcE);
        DeRhamField<Grid::dual, Space::face> rhoEAn(deRham, m_funcRhoE);

        QuasineutralSolver<s_hodgeDegree, s_vdim, s_ndata, s_degX, s_degY, s_degZ> hypreParticleRho(
            infra, deRham);

        hypreParticleRho.solve_particle_charge_e(rhoEAn, E, ions);

        E -= eAn;

        amrex::GpuArray<amrex::Real, 3> dxi{GEMPIC_D_PAD_ONE(infra.geometry().InvCellSize(xDir),
                                                             infra.geometry().InvCellSize(yDir),
                                                             infra.geometry().InvCellSize(zDir))};
        return Utils::gempic_norm (E.m_data[xDir], infra, 1) * dxi[xDir] +
               Utils::gempic_norm(E.m_data[yDir], infra, 1) * dxi[yDir] +
               Utils::gempic_norm(E.m_data[zDir], infra, 1) * dxi[zDir];
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

TYPED_TEST_SUITE(HypreQuasineutralChargeMatrixTest, MyTypes);

TYPED_TEST(HypreQuasineutralChargeMatrixTest, HypreQuasineutralChargeMatrix)
{
    constexpr int coarse = 8;
    constexpr int fine = 16;
    amrex::Real errorCoarse, errorFine;
    amrex::Real tol = 0.15;

    amrex::Real rateOfConvergence;
    constexpr int splineDegreeX = TestFixture::s_degX;
    constexpr int splineDegreeY = TestFixture::s_degY;
    constexpr int splineDegreeZ = TestFixture::s_degZ;
    errorCoarse = this->template chargematrix_solve<coarse>();
    amrex::Print() << "errorCoarse: " << errorCoarse << "\n";
    errorFine = this->template chargematrix_solve<fine>();
    amrex::Print() << "errorFine: " << errorFine << "\n";
    rateOfConvergence = std::log2(errorCoarse / errorFine);
    amrex::Print() << "rate_of_convergence_<" << splineDegreeX << "," << splineDegreeY << ","
                   << splineDegreeZ << ">:" << rateOfConvergence << "\n";
    EXPECT_NEAR(rateOfConvergence, 2, tol);
}

} // namespace
