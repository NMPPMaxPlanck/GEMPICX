#include <gtest/gtest.h>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_Particle.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Particle;

namespace
{
// Test fixture. Sets up clean environment before each test.
class ParticleIteratorTest : public testing::Test
{
protected:
    // Number of species
    static int const s_vDim{0};

    amrex::Real m_charge{0.5};
    amrex::Real m_mass{1};

    ComputationalDomain m_infra;
    std::unique_ptr<ParticleSpecies<s_vDim>> m_particles;

    ParticleIteratorTest() : m_infra{Gempic::Test::Utils::get_default_compdom()}
    {
        // particles
        m_particles = std::make_unique<ParticleSpecies<s_vDim>>(m_charge, m_mass, m_infra);
    }
};

// Adds no particles. Checks that the iterator is valid
TEST_F(ParticleIteratorTest, NullTest)
{
    {
        EXPECT_EQ(0, m_particles->TotalNumberOfParticles());
        EXPECT_EQ(m_particles->begin(), m_particles->end());
    }

    for (auto& particleGrid : *m_particles)
    {
        FAIL() << "The particle species is empty, so this should never be reached.";
        EXPECT_EQ(particleGrid.numParticles(), 0);
    }
}

// Example of actual use. requires the particle iterator, which means actual particles must be added
TEST_F(ParticleIteratorTest, NotNullTest)
{
    constexpr int numParticles{2};
    auto dx = m_infra.geometry().CellSizeArray();
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {m_infra.geometry().ProbLoArray(),
         {AMREX_D_DECL(m_infra.geometry().ProbHi(xDir) - 0.5 * dx[xDir],
                       m_infra.geometry().ProbHi(yDir) - 0.5 * dx[yDir],
                       m_infra.geometry().ProbHi(zDir) - 0.5 * dx[zDir])}}};

    amrex::Array<amrex::Real, numParticles> weights{1, 3};

    Gempic::Test::Utils::add_single_particles(m_particles.get(), m_infra, weights, positions);
    m_particles->Redistribute(); // assign particles to the grid they are in

    bool particleLoopRun{false};
    int numberOfParticles{0};
    int numberOfBoxes{0};
    for (auto& particleGrid : *m_particles)
    {
        particleLoopRun = true;
        numberOfParticles += particleGrid.numParticles();
        auto* const weight = particleGrid.GetStructOfArrays().GetRealData(AMREX_SPACEDIM).data();
        EXPECT_TRUE(weight[0] == weights[0] || weight[0] == weights[1]);
        numberOfBoxes++;
    }
    amrex::ParallelDescriptor::ReduceIntSum(numberOfParticles);
    amrex::ParallelDescriptor::ReduceIntSum(numberOfBoxes);
    ASSERT_TRUE(particleLoopRun);
    EXPECT_EQ(numParticles, numberOfParticles); // All particles reached
    EXPECT_EQ(numParticles, numberOfBoxes);     // Only grids with particles reached
}
} // namespace
