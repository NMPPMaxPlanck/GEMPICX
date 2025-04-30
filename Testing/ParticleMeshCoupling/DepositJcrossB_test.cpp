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
#include "GEMPIC_ParticleGroups.H"
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
 * @brief Tests the deposit_jcrossb function using analytical function fields
 */
template <typename SplineDegreeStruct>
class DepositJcrossBTest : public testing::Test
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

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcB;
    amrex::Array<amrex::Parser, 3> m_parserB;

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> m_funcJcrossB;
    amrex::Array<amrex::Parser, 3> m_parserJcrossB;

    DepositJcrossBTest()
    {
        if constexpr (AMREX_SPACEDIM != 1)
        {
            amrex::Array<std::string, 3> analyticalB;
            amrex::Array<std::string, 3> analyticalJcrossB;
            if constexpr (AMREX_SPACEDIM == 2)
            {
                analyticalB = {"sin(x)", "sin(y)", "sin(x+y)"};
                analyticalJcrossB = {"sin(x)", "-sin(y)", "sin(y-x)"};
            }
            else if constexpr (AMREX_SPACEDIM == 3)
            {
                analyticalB = {"sin(x)", "sin(y)", "sin(z)"};
                analyticalJcrossB = {"sin(z-y)", "sin(x-z)", "sin(y-x)"};
            }

            for (int i = 0; i < 3; ++i)
            {
                m_parserB[i].define(analyticalB[i]);
                m_parserB[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
                m_funcB[i] = m_parserB[i].compile<s_nVar>();

                m_parserJcrossB[i].define(analyticalJcrossB[i]);
                m_parserJcrossB[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
                m_funcJcrossB[i] = m_parserJcrossB[i].compile<s_nVar>();
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
            GTEST_SKIP() << "This function only works in 2D and 3D.";
        }
    }

    template <int n>
    amrex::Real jcross_b_solve ()
    {
        Gempic::Io::Parameters parameters{};

        // Initialize computational_domain
        amrex::IntVect const nCell{AMREX_D_DECL(n, n, n)};
        ComputationalDomain infra = Gempic::Test::Utils::get_compdom(nCell);

        // Initialize particle groups
        std::vector<std::shared_ptr<ParticleGroups<s_vdim>>>
            ions; // Use 'init_particles(infra, ions);' if adding large number of particles
                  // randomly

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(infra, s_hodgeDegree, s_maxSplineDegree,
                                                        HodgeScheme::FDHodge);

        // particles
        ions.resize(1);
        for (int spec{0}; spec < 1; spec++)
        {
            ions[0] = std::make_shared<ParticleGroups<s_vdim>>(0, infra);
        }

        int const numParticles{GEMPIC_D_MULT(n, n, n)};

        amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions;
        amrex::Array<amrex::GpuArray<amrex::Real, s_vdim>, numParticles> velocities;
        amrex::Array<amrex::Real, numParticles> weights;

        int i{0}, j{0}, k{0};
        auto const& dx{infra.cell_size_array()};
        GEMPIC_D_LOOP_BEGIN(for (i = 0; i < n; i++), for (j = 0; j < n; j++),
                            for (k = 0; k < n; k++))

            amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> loc = {
                AMREX_D_DECL(infra.geometry().ProbLo(xDir) + (i + 0.5) * dx[xDir],
                             infra.geometry().ProbLo(yDir) + (j + 0.5) * dx[yDir],
                             infra.geometry().ProbLo(zDir) + (k + 0.5) * dx[zDir])};

            positions[i + j * n + k * n * n] = {AMREX_D_DECL(loc[xDir], loc[yDir], loc[zDir])};
#if AMREX_SPACEDIM == 2
            velocities[i + j * n + k * n * n] = {cos(loc[xDir]), cos(loc[yDir]),
                                                 cos(loc[xDir] + loc[yDir])};
#elif AMREX_SPACEDIM == 3
            velocities[i + j * n + k * n * n] = {cos(loc[xDir]), cos(loc[yDir]), cos(loc[zDir])};
#endif
            weights[i + j * n + k * n * n] = infra.cell_volume();
        GEMPIC_D_LOOP_END

        Gempic::Test::Utils::add_single_particles(ions[0].get(), infra, weights, positions,
                                                  velocities);

        ions[0]->Redistribute();

        // Computed fields
        DeRhamField<Grid::primal, Space::face> B(deRham, m_funcB);
        DeRhamField<Grid::dual, Space::face> jcrossB(deRham);

        // Analytical fields
        DeRhamField<Grid::dual, Space::face> jcrossBAn(deRham, m_funcJcrossB);

        // Deposit initial charge
        for (auto& particleSpecies : ions)
        {
            amrex::Real charge = particleSpecies->get_charge();

            for (auto& particleGrid : *particleSpecies)
            {
                long const np = particleGrid.numParticles();
                auto* const particles = particleGrid.GetArrayOfStructs()().data();
                auto* const weight = particleGrid.GetStructOfArrays().GetRealData(s_vdim).data();

                auto* const velx = particleGrid.GetStructOfArrays().GetRealData(0).data();
                auto* const vely = particleGrid.GetStructOfArrays().GetRealData(1).data();
                auto* const velz = particleGrid.GetStructOfArrays().GetRealData(2).data();

                amrex::GpuArray<amrex::Array4<amrex::Real>, s_vdim> bA;
                amrex::GpuArray<amrex::Array4<amrex::Real>, s_vdim> jcrossbA;

                for (int cc = 0; cc < s_vdim; cc++)
                {
                    bA[cc] = (B.m_data[cc])[particleGrid].array();
                    jcrossbA[cc] = (jcrossB.m_data[cc])[particleGrid].array();
                }
                amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{infra.geometry().ProbLoArray()};

                amrex::ParallelFor(
                    np,
                    [=] AMREX_GPU_DEVICE(long pp)
                    {
                        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionParticle;
                        for (unsigned int d = 0; d < AMREX_SPACEDIM; ++d)
                        {
                            positionParticle[d] = particles[pp].pos(d);
                        }

                        SplineBase<s_degX, s_degY, s_degZ> spline(positionParticle, plo,
                                                                  infra.inv_cell_size_array());

                        amrex::GpuArray<amrex::Real, s_vdim> vel{velx[pp], vely[pp], velz[pp]};
                        deposit_jcrossb(jcrossbA, spline, vel, bA, charge * weight[pp]);
                    });
            }
        }
        jcrossB.post_particle_loop_sync();

        jcrossB -= jcrossBAn;

        amrex::GpuArray<amrex::Real, 3> dxi3d{GEMPIC_D_PAD_ONE(infra.geometry().InvCellSize(xDir),
                                                               infra.geometry().InvCellSize(yDir),
                                                               infra.geometry().InvCellSize(zDir))};
        return (Utils::gempic_norm(jcrossB.m_data[xDir], infra, 1) * dxi3d[yDir] * dxi3d[zDir] +
                Utils::gempic_norm(jcrossB.m_data[yDir], infra, 1) * dxi3d[zDir] * dxi3d[xDir] +
                Utils::gempic_norm(jcrossB.m_data[zDir], infra, 1) * dxi3d[xDir] * dxi3d[yDir]);
    }
};

using MyTypes = ::testing::Types<std::integral_constant<int, 3>,
                                 std::integral_constant<int, 4>,
                                 std::integral_constant<int, 5>>;
TYPED_TEST_SUITE(DepositJcrossBTest, MyTypes);

TYPED_TEST(DepositJcrossBTest, SingleParticlePerCellJcrossB)
{
    constexpr int coarse = 12;
    constexpr int fine = 24;
    amrex::Real errorCoarse, errorFine;
    amrex::Real tol = 0.15;

    amrex::Real rateOfConvergence;
    errorCoarse = this->template jcross_b_solve<coarse>();
    errorFine = this->template jcross_b_solve<fine>();
    rateOfConvergence = std::log2(errorCoarse / errorFine);
    EXPECT_NEAR(rateOfConvergence, 2.0, tol);
}
} // namespace
