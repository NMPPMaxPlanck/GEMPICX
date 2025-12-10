#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_Config.H"
#include "GEMPIC_Particle.H"
#include "TestUtils/GEMPIC_AmrexTestEnv.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Particle;

//! Global instance of the environment (for access in tests)
GempicTests::AmrexTestEnv* utestEnv = nullptr;

int main (int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    // gtest takes ownership of the TestEnvironment ptr - we don't delete it.
    auto* utestEnv = new GempicTests::AmrexTestEnv(argc, argv);
    ::testing::AddGlobalTestEnvironment(utestEnv);
    return RUN_ALL_TESTS();
}

TEST(ParticleAddTest, AddAndWriteParticles)
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
    // FIXME: write p0 and pL to a file such that they can subsequently
    //        be read in Python
    //std::cout << "p0 x = " << p0[xDir]
    //          << ", y = "  << p0[yDir]
    //          << ", z = "  << p0[zDir]
    //          << std::endl;
    //std::cout << "pL x = " << pL[xDir]
    //          << ", y = "  << pL[yDir]
    //          << ", z = "  << pL[zDir]
    //          << std::endl;
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
    EXPECT_EQ(numParticles, particles.get()->TotalNumberOfParticles());
    particles->Redistribute(); // assign particles to the grid they are in
    int const np = particles->NumberOfParticlesAtLevel(0);
    EXPECT_EQ(np, numParticles);

    auto const idxw = 0;
    for (auto& pti : *particles)
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

    // `writeRealComp(vDim + nData, 1)` is used in FlushFormatPlotfile
    amrex::Vector<int> const writeRealComp(1, 1);
    amrex::Vector<int> const& writeIntComp = {};
    amrex::Vector<std::string> const& intCompNames = {};
    amrex::Vector<std::string> const& realCompNames = {"weight"};
    particles->WritePlotFile("ParticleAddOutput", "particles", writeRealComp, writeIntComp,
                             realCompNames, intCompNames);
}
