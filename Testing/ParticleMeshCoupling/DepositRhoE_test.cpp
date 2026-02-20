#include <type_traits>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_Particles.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_Sampler.H"
#include "GEMPIC_SplineClass.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace ParticleMeshCoupling;

/**
 * @brief Tests the deposit_rho_weighted_electric_field function using analytical function fields
 */
template <typename SplineDegreeStruct>
class DepositRhoETest : public testing::Test
{
public:
    static constexpr int s_vdim{3};
    static constexpr int s_ndata{1};

    static constexpr int s_degX{SplineDegreeStruct::value};
    static constexpr int s_degY{SplineDegreeStruct::value};
    static constexpr int s_degZ{SplineDegreeStruct::value};
    static constexpr int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};
    static constexpr int s_hodgeDegree{2};

    static int const s_nVar = AMREX_SPACEDIM + 1; // x, y, z, t

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcE;
    amrex::Array<amrex::Parser, 3> m_parserE;

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcRhoE;
    amrex::Array<amrex::Parser, 3> m_parserRhoE;

    DepositRhoETest()
    {
        if constexpr (AMREX_SPACEDIM != 1)
        {
            amrex::Array<std::string, 3> analyticalE;
            amrex::Array<std::string, 3> analyticalRhoE;
            if constexpr (AMREX_SPACEDIM == 2)
            {
                analyticalE = {"sin(x)", "sin(y)", "sin(x+y)"};
                analyticalRhoE = {"cos(x+y)*sin(x)", "cos(x+y)*sin(y)", "cos(x+y)*sin(x+y)"};
            }
            else if constexpr (AMREX_SPACEDIM == 3)
            {
                analyticalE = {"sin(x)", "sin(y)", "sin(z)"};
                analyticalRhoE = {"cos(x+y+z)*sin(x)", "cos(x+y+z)*sin(y)", "cos(x+y+z)*sin(z)"};
            }

            for (int i = 0; i < 3; ++i)
            {
                m_parserE[i].define(analyticalE[i]);
                m_parserE[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
                m_funcE[i] = m_parserE[i].compile<s_nVar>();

                m_parserRhoE[i].define(analyticalRhoE[i]);
                m_parserRhoE[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
                m_funcRhoE[i] = m_parserRhoE[i].compile<s_nVar>();
            }
        }

        Gempic::Io::Parameters parameters;
        // Particle parameters (data read by particle_groups constructor)
        std::string speciesNames{"ions"};
        parameters.set("Particle.speciesNames", speciesNames);

        amrex::Real charge{1.0};
        parameters.set("Particle.ions.charge", charge);

        amrex::Real mass{1.0};
        parameters.set("Particle.ions.mass", mass);

        // Gaussian parameters
        amrex::Vector<amrex::Real> vMean{{0.0, 0.0, 0.0}}; // for nodesplinederiv testing
        parameters.set("Particle.ions.G0.vMean", vMean);
    }

    // virtual void SetUp() will be called before each test is run.
    void SetUp () override
    {
        if constexpr (AMREX_SPACEDIM == 1)
        {
            GTEST_SKIP() << "This function works in 2D and 3D.";
        }
    }

    template <int n>
    amrex::Real rhoe_solve ()
    {
        // For studies other than convergence, this should be in SetUpTestSuite under Grid
        // parameters
        Gempic::Io::Parameters parameters{};

        // Initialize computational_domain
        amrex::IntVect const nCell{AMREX_D_DECL(n, n, n)};
        ComputationalDomain infra = Gempic::Test::Utils::get_compdom(nCell);

        // Initialize particles
        std::vector<std::shared_ptr<ParticleSpecies<s_vdim>>>
            ions; // Use 'init_particles(infra, ions);' if adding large number of particles
                  // randomly

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, s_hodgeDegree, s_maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        // particles
        ions.resize(1);
        for (int spec{0}; spec < 1; spec++)
        {
            ions[0] = std::make_shared<ParticleSpecies<s_vdim>>(0, infra);
        }

        int const percelldir{1};

        int const percell{GEMPIC_D_MULT(percelldir, percelldir, percelldir)};
        int const numParticles{percell * GEMPIC_D_MULT(n, n, n)};

        std::vector<std::vector<amrex::Real>> oss(percell);

        // Any value of 'off' in [0,1] works for deg 2,3,4,5
        amrex::Real off{0.0};

        GEMPIC_D_LOOP_BEGIN(for (int i = 0; i < percelldir; i++),
                            for (int j = 0; j < percelldir; j++),
                            for (int k = 0; k < percelldir; k++))
            oss[GEMPIC_D_ADD(i, j * percelldir, k * percelldir * percelldir)] = {AMREX_D_DECL(
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

#if AMREX_SPACEDIM == 2
                velocities[p * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] = {0.0, 0.0, 0.0};
                weights[p * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] =
                    (1.0 / percell) * infra.cell_volume() * cos(loc[xDir] + loc[yDir]);
#elif AMREX_SPACEDIM == 3
                velocities[p * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] = {0.0, 0.0, 0.0};
                weights[p * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] =
                    (1.0 / percell) * infra.cell_volume() * cos(loc[xDir] + loc[yDir] + loc[zDir]);
#endif
            }
        GEMPIC_D_LOOP_END

        Gempic::Test::Utils::add_single_particles(ions[0].get(), infra, weights, positions,
                                                  velocities);

        ions[0]->Redistribute();

        // Computed fields
        DeRhamField<Grid::primal, Space::edge> E(deRham, m_funcE);
        DeRhamField<Grid::dual, Space::face> rhoE(deRham);

        // Analytical fields
        DeRhamField<Grid::dual, Space::face> rhoEAn(deRham, m_funcRhoE);

        deposit_rho_weighted_electric_field<s_degX, s_degY, s_degZ, s_vdim, s_ndata>(rhoE, E, ions,
                                                                                     infra);

        rhoE -= rhoEAn;

        amrex::GpuArray<amrex::Real, 3> dxi3d{infra.inv_cell_size_3darray()};

        return (Utils::gempic_norm(rhoE.m_data[xDir], infra, 1) * dxi3d[yDir] * dxi3d[zDir] +
                Utils::gempic_norm(rhoE.m_data[yDir], infra, 1) * dxi3d[zDir] * dxi3d[xDir] +
                Utils::gempic_norm(rhoE.m_data[zDir], infra, 1) * dxi3d[xDir] * dxi3d[yDir]);
    }
};

using MyTypes = ::testing::Types<std::integral_constant<int, 2>,
                                 std::integral_constant<int, 3>,
                                 std::integral_constant<int, 4>,
                                 std::integral_constant<int, 5>>;
TYPED_TEST_SUITE(DepositRhoETest, MyTypes);

TYPED_TEST(DepositRhoETest, SingleParticlePerCellRhoE)
{
    constexpr int coarse = 20;
    constexpr int fine = 40;
    amrex::Real errorCoarse, errorFine;
    amrex::Real tol = 0.15;

    amrex::Real rateOfConvergence;
    errorCoarse = this->template rhoe_solve<coarse>();
    errorFine = this->template rhoe_solve<fine>();
    rateOfConvergence = std::log2(errorCoarse / errorFine);
    EXPECT_NEAR(rateOfConvergence, 2.0, tol);
}
} // namespace
