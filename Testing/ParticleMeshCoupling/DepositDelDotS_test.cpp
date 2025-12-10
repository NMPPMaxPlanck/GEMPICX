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
 * @brief Tests the deposit_deldotS function using analytical function fields
 */
template <typename SplineDegreeStruct>
class DepositDelDotSTest : public testing::Test
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

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcJ;
    amrex::Array<amrex::Parser, 3> m_parserJ;

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcDelDotS;
    amrex::Array<amrex::Parser, 3> m_parserDelDotS;

    DepositDelDotSTest()
    {
        if constexpr (AMREX_SPACEDIM != 1)
        {
            amrex::Array<std::string, 3> analyticalDelDotS;
            if constexpr (AMREX_SPACEDIM == 2)
            {
                analyticalDelDotS = {"sin(x)*(2.0*cos(x) + cos(y))", "sin(y)*(cos(x) + 2.0*cos(y))",
                                     "sin(2.0*x+y) + sin(2.0*y+x)"};
            }
            else if constexpr (AMREX_SPACEDIM == 3)
            {
                analyticalDelDotS = {"sin(x)*(2.0*cos(x) + cos(y) + cos(z))",
                                     "sin(y)*(cos(x) + 2.0*cos(y) + cos(z))",
                                     "sin(z)*(cos(x) + cos(y) + 2.0*cos(z))"};
            }

            for (int i = 0; i < 3; ++i)
            {
                m_parserDelDotS[i].define(analyticalDelDotS[i]);
                m_parserDelDotS[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
                m_funcDelDotS[i] = m_parserDelDotS[i].compile<s_nVar>();
            }
        }

        Gempic::Io::Parameters parameters;
        // Particle parameters (data read by particle_groups constructor)
        std::string speciesNames{"ions"};
        parameters.set("Particle.speciesNames", speciesNames);

        std::string samplerName{"Sobol"};
        parameters.set("Particle.sampler", samplerName);

        int nPartPerCell{20000};
        parameters.set("Particle.ions.nPartPerCell", nPartPerCell);

        amrex::Real charge{1.0};
        parameters.set("Particle.ions.charge", charge);

        amrex::Real mass{1.0};
        parameters.set("Particle.ions.mass", mass);

        std::string density{"1.0"};
        parameters.set("Particle.ions.density", density);

        int numGaussians{1};
        parameters.set("Particle.ions.numGaussians", numGaussians);

        // Gaussian parameters
        amrex::Vector<amrex::Real> vMean{{0.0, 0.0, 0.0}};
        parameters.set("Particle.ions.G0.vMean", vMean);

        amrex::Vector<amrex::Real> vThermal{{0.0, 0.0, 0.0}};
        parameters.set("Particle.ions.G0.vThermal", vThermal);

        amrex::Real vWeightG0{1.0};
        parameters.set("Particle.ions.G0.vWeight", vWeightG0);
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
    amrex::Real del_dot_s_solve ()
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

        int const percelldir{1}; // GTest works for any value of percelldir

        int const percell{GEMPIC_D_MULT(percelldir, percelldir, percelldir)};
        int const numParticles{percell * GEMPIC_D_MULT(n, n, n)};

        std::vector<std::vector<amrex::Real>> oss(percell);

        // Any value of 'off' in [0,1] works for deg 3,4,5
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
                velocities[p * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] = {
                    sin(loc[xDir]), sin(loc[yDir]), sin(loc[xDir] + loc[yDir])};
#elif AMREX_SPACEDIM == 3
                velocities[p * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] = {
                    sin(loc[xDir]), sin(loc[yDir]), sin(loc[zDir])};
#endif
                weights[p * GEMPIC_D_MULT(n, n, n) + i + j * n + k * n * n] =
                    (1.0 / percell) * infra.cell_volume();
            }
        GEMPIC_D_LOOP_END

        Gempic::Test::Utils::add_single_particles(ions[0].get(), infra, weights, positions,
                                                  velocities);

        ions[0]->Redistribute();

        // Computed fields
        DeRhamField<Grid::dual, Space::face> delDotS(deRham);

        // Analytical fields
        DeRhamField<Grid::dual, Space::face> delDotSAn(deRham, m_funcDelDotS);

        // Deposit initial charge
        for (auto& particleSpecies : ions)
        {
            amrex::Real const mass = particleSpecies->get_mass();

            for (auto& pti : *particleSpecies)
            {
                long const np = pti.numParticles();
                auto const ptd = pti.GetParticleTile().getParticleTileData();
                auto const ii = particleSpecies->get_data_indices();

                amrex::GpuArray<amrex::Array4<amrex::Real>, s_vdim> deldotsA;

                for (int cc = 0; cc < s_vdim; cc++)
                {
                    deldotsA[cc] = (delDotS.m_data[cc])[pti].array();
                }
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{infra.geometry().ProbLoArray()};

                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                        AMREX_D_EXPR(positionParticle[xDir] = ptd.rdata(ii.m_iposx)[pp],
                                     positionParticle[yDir] = ptd.rdata(ii.m_iposy)[pp],
                                     positionParticle[zDir] = ptd.rdata(ii.m_iposz)[pp]);

                        SplineWithFirstDerivative<s_degX, s_degY, s_degZ> splineDeriv(
                            positionParticle, plo, infra.inv_cell_size_array(), dx);

                        amrex::GpuArray<amrex::Real, s_vdim> vel{ptd.rdata(ii.m_ivelx)[pp],
                                                                 ptd.rdata(ii.m_ively)[pp],
                                                                 ptd.rdata(ii.m_ivelz)[pp]};
                        deposit_deldot_s(deldotsA, splineDeriv, vel,
                                         mass * ptd.rdata(ii.m_iweight)[pp]);
                    });
            }
        }
        delDotS.post_particle_loop_sync();

        delDotS -= delDotSAn;

        amrex::GpuArray<amrex::Real, 3> dxi3d{infra.inv_cell_size_3darray()};

        return (Utils::gempic_norm(delDotS.m_data[xDir], infra, 1) * dxi3d[yDir] * dxi3d[zDir] +
                Utils::gempic_norm(delDotS.m_data[yDir], infra, 1) * dxi3d[zDir] * dxi3d[xDir] +
                Utils::gempic_norm(delDotS.m_data[zDir], infra, 1) * dxi3d[xDir] * dxi3d[yDir]);
    }
};

using MyTypes = ::testing::Types<std::integral_constant<int, 2>,
                                 std::integral_constant<int, 3>,
                                 std::integral_constant<int, 4>,
                                 std::integral_constant<int, 5>>;
TYPED_TEST_SUITE(DepositDelDotSTest, MyTypes);

TYPED_TEST(DepositDelDotSTest, SingleParticlePerCellDelDotS)
{
    constexpr int coarse = 20;
    constexpr int fine = 40;
    amrex::Real errorCoarse, errorFine;
    amrex::Real tol = 0.1;

    amrex::Real rateOfConvergence;
    errorCoarse = this->template del_dot_s_solve<coarse>();
    errorFine = this->template del_dot_s_solve<fine>();
    rateOfConvergence = std::log2(errorCoarse / errorFine);
    EXPECT_NEAR(rateOfConvergence, 2.0, tol);
}
} // namespace
