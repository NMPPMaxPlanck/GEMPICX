#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_Particle.H"
#include "TestUtils/GEMPIC_AmrexTestEnv.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Particle;

TEST(ParticleIOTest, WriteAndReadParticles)
{
    int constexpr vDim{0};
    int constexpr nData{1};

    amrex::Real charge{0.5};
    amrex::Real mass{1};

    ComputationalDomain infra{Gempic::Test::Utils::get_default_compdom()};
    auto particles = std::make_unique<ParticleSpecies<vDim, nData>>(charge, mass, infra);

    int constexpr numParticles{37};
    double constexpr baseFraction = 1.0 / numParticles;
    // To spread particles across the entire domain, use
    auto p0 = infra.prob_low_3darray();
    std::array<amrex::Real, AMREX_SPACEDIM> pL = {AMREX_D_DECL(infra.geometry().ProbLength(xDir),
                                                               infra.geometry().ProbLength(yDir),
                                                               infra.geometry().ProbLength(zDir))};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions;
    // weights are used later to identify individual particles
    amrex::Array<amrex::Real, numParticles> weights;
    // using 3, 4 and 5 because they have no common divisors,
    // and I want to cover the whole domain
    for (auto pp = 0; pp < numParticles; pp++)
    {
        AMREX_D_TERM(
            positions[pp][xDir] = p0[xDir] + ((3 * pp) % numParticles) * baseFraction * pL[xDir];
            , positions[pp][yDir] = p0[yDir] + ((4 * pp) % numParticles) * baseFraction * pL[yDir];
            , positions[pp][zDir] = p0[zDir] + ((5 * pp) % numParticles) * baseFraction * pL[zDir];)
        weights[pp] = pp + 1.1;
    }

    Gempic::Test::Utils::add_single_particles(particles.get(), infra, weights, positions);
    particles->Redistribute(); // assign particles to the grid they are in

    // write particles
    particles->Checkpoint("checkpoint_data", "particles0");
    amrex::AsyncOut::Finish();
    amrex::ParallelDescriptor::Barrier();

    // read particles
    auto particles2 = std::make_unique<ParticleSpecies<vDim, nData>>(charge, mass, infra);
    particles2->restart("checkpoint_data", "particles0");

    // compare
    EXPECT_EQ(numParticles, particles2.get()->TotalNumberOfParticles());
    int const np = particles2->NumberOfParticlesAtLevel(0);
    EXPECT_EQ(np, numParticles);

    auto const idxw = 0;
    for (auto& pti : *particles2)
    {
        auto const ptd = pti.GetParticleTile().getParticleTileData();
        auto* const ww = ptd.rdata(idxw + AMREX_SPACEDIM);
        double const tol = 1e-10;
        long const npt = pti.numParticles();

        for (long pp = 0; pp < npt; pp++)
        {
            int const idx = int(ww[pp]) - 1;
            // **Note**: `idx` will usually be different from
            // `pp` when using more than one MPI process or
            // more than one grid.
            // The `add_single_particles` function does not
            // guarantee that particles are added in order of
            // argument arrays.
            EXPECT_NEAR(positions[idx][0], ptd[pp].pos(xDir), tol);
            if (AMREX_SPACEDIM > 1)
            {
                EXPECT_NEAR(positions[idx][1], ptd[pp].pos(yDir), tol);
            }
            if (AMREX_SPACEDIM > 2)
            {
                EXPECT_NEAR(positions[idx][2], ptd[pp].pos(zDir), tol);
            }
            EXPECT_NEAR(weights[idx], ww[pp], tol);
        }
    }
}

template <unsigned int vDim, unsigned int nData>
void kill_single_particle (std::unique_ptr<ParticleSpecies<vDim, nData>>& particles)
{
    for (auto& pti : *particles)
    {
        if (pti.index() == 0)
        {
            auto const ptd = pti.GetParticleTile().getParticleTileData();
            long const npt = pti.numParticles();

            amrex::ParallelFor(npt,
                               [=] AMREX_GPU_DEVICE(long pp)
                               {
                                   if (pp == 0)
                                   {
                                       // Make invalid (happens automatically if e.g. particle exits
                                       // non-periodic boundaries)
                                       ptd.id(pp) = -1;
                                   }
                               });
        }
    }
    particles->Redistribute(); // assign particles to the grid they are in, lose dead particle
}

TEST(ParticleIOTest, WriteAndReadIntoNonEmptyContainer)
{
    int constexpr vDim{0};
    int constexpr nData{1};

    amrex::Real charge{0.5};
    amrex::Real mass{1};

    ComputationalDomain infra{Gempic::Test::Utils::get_default_compdom()};
    auto particles = std::make_unique<ParticleSpecies<vDim, nData>>(charge, mass, infra);

    int constexpr numParticles{2};

    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions = {
        infra.geometry().ProbLoArray(), infra.geometry().ProbLoArray()};
    amrex::Array<amrex::Real, numParticles> weights = {0.1, 0.5};

    Gempic::Test::Utils::add_single_particles(particles.get(), infra, weights, positions);

    // remove exactly one particle from the container
    kill_single_particle(particles);
    ASSERT_EQ(numParticles - 1, particles.get()->TotalNumberOfParticles());

    // write particles
    particles->Checkpoint("checkpoint_data", "particles1");
    amrex::AsyncOut::Finish();
    amrex::ParallelDescriptor::Barrier();

    // read particles
    auto particles2 = std::make_unique<ParticleSpecies<vDim, nData>>(charge, mass, infra);
    Gempic::Test::Utils::add_single_particles(particles2.get(), infra, weights, positions);
    ASSERT_EQ(numParticles, particles2.get()->TotalNumberOfParticles());
    particles2->restart("checkpoint_data", "particles1");

    // compare
    EXPECT_EQ(numParticles - 1, particles2.get()->TotalNumberOfParticles());
}
