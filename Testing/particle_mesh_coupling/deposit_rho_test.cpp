/** Testing for deposit_rho function
 *  \todo: Consider mocking particles.
*/

/*
#include <GEMPIC_parameters.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_amrex_init.H>
*/
#include <AMReX.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include "gtest/gtest.h"
#include "test_utils/GEMPIC_test_utils.H"

using namespace Particles;

#define checkField(...) GEMPIC_TestUtils::checkField(__FILE__, __LINE__, __VA_ARGS__)

namespace {
        // When using amrex::ParallelFor you have to create a standalone helper function that does the execution on GPU and call that function from the unit test because of how GTest creates tests within a TEST_F fixture.
        template <int vDim, int degX, int degY, int degZ>
        void updateRhoParallelFor(amrex::ParIter<0, 0, vDim + 1, 0>& pti,
                                  computational_domain& infra, 
                                  amrex::MultiFab& rho,
                                  amrex::Real charge) {
            const long np{pti.numParticles()};
            const auto& particles{pti.GetArrayOfStructs()};
            const auto partData{particles().data()};
            const auto weight{pti.GetStructOfArrays().GetRealData(vDim).data()};
            amrex::Array4<amrex::Real> const& rhoarr{rho[pti].array()};

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
            {
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
                for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
                    position[d] = partData[pp].pos(d);
                Spline::SplineBase<degX, degY, degZ> spline(position, infra.plo, infra.dxi);
                // Needs at least max(degX, degY, degZ) ghost cells
                gempic_deposit_rho(spline, charge * weight[pp], rhoarr);
            });
        }

    // Test fixture. Sets up clean environment before each test.
    class DepositRhoTest : public testing::Test {
        protected:

        // Degree of splines in each direction
        static const int degX{1};
        //static const int degY{AMREX_D_PICK(0, 1, 1)};
        //static const int degZ{AMREX_D_PICK(0, 0, 1)};
        static const int degY{1};
        static const int degZ{1};

        // Number of species (second species only used for DoubleParticleMultipleSpecies)
        static const int numSpec{2};
        // Number of velocity dimensions.
        static const int vDim{0};
        // Number of ghost cells in mesh
        const int Nghost{GEMPIC_TestUtils::initNGhost(degX, degY, degZ)};
        const amrex::IntVect Nghosts{AMREX_D_DECL(Nghost, Nghost, Nghost)};
        const amrex::IntVect dstNGhosts{AMREX_D_DECL(0, 0, 0)};


        amrex::Array<amrex::Real, numSpec> charge{1, -1};
        amrex::Array<amrex::Real, numSpec> mass{1, 0.1};

        computational_domain infra;
        amrex::GpuArray<std::unique_ptr<particle_groups<vDim>>, numSpec> particleGroup;
        amrex::MultiFab rho;

        // virtual void SetUp() will be called before each test is run.
        void SetUp() override {
            /* Initialize the infrastructure */
            const amrex::RealBox realBox({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                         {AMREX_D_DECL(10.0, 10.0, 10.0)});
            const amrex::IntVect nCell{AMREX_D_DECL(10, 10, 10)};
            const amrex::IntVect maxGridSize{AMREX_D_DECL(10, 10, 10)};
            const amrex::IntVect isPeriodic{AMREX_D_DECL(1, 1, 1)};

            infra.initialize_computational_domain(nCell, maxGridSize, isPeriodic, realBox);

            // Setup rho. This is  the special part of this text fixture.
            // node centered BA:
            const amrex::BoxArray &nba{amrex::convert(infra.grid, amrex::IntVect::TheNodeVector())};
            int Ncomp{1};            

            rho.define(nba, infra.distriMap, Ncomp, Nghost);
            rho.setVal(0.0);
            // Ensure rho exists and is 0 everywhere
            ASSERT_EQ(0, rho.norm2(0, infra.geom.periodicity()));

            // particle groups
            for (int spec{0}; spec < numSpec; spec++)
            {
                particleGroup[spec] =
                    std::make_unique<particle_groups<vDim>>(charge[spec], mass[spec], infra);
            }
        }
    };

    /** Single particle tests. The only reason most of these maneuvres are necessary is because of
     *  amrex::Array4<amrex::Real> const& rhoarr{rho[pti].array()};
     *  which is required for the connection between MultiFab rho and deposit_rho function. This in
     *  turn requires the pti iterator, which means actual particles must be added, instead of
     *  simply supplying positions directly.
     */

    // Adds a particle with 0 weight. Checks that rho is unchanged.
    TEST_F(DepositRhoTest, NullTest) {
        // Adding particle to one cell
        const int numParticles{1};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{*infra.geom.ProbLo()};
        
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles(particleGroup, infra, weights, positions);

        // (default) charge correctly transferred from GEMPIC_TestUtils::addSingleParticles
        EXPECT_EQ(1, particleGroup[0]->getCharge()); 
        
        // rho unchanged by GEMPIC_TestUtils::addSingleParticles
        EXPECT_EQ(0, rho.norm2(0, infra.geom.periodicity()));

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle.
        bool particleLoopRun{false};
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            particleLoopRun = true;

            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np); // Only one particle added by GEMPIC_TestUtils::addSingleParticles

            const auto& particles{pti.GetArrayOfStructs()};
            const auto partData{particles().data()};
            const auto weight{pti.GetStructOfArrays().GetRealData(vDim).data()};

            EXPECT_EQ(1, weight[0]); // weight correctly transferred from GEMPIC_TestUtils::addSingleParticles

            amrex::Array4<amrex::Real> const& rhoarr{rho[pti].array()};
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
            for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
                position[d] = partData[0].pos(d);

            Spline::SplineBase<degX, degY, degZ> spline(position, infra.plo, infra.dxi);

            gempic_deposit_rho(spline, 0, rhoarr);
        }
        ASSERT_TRUE(particleLoopRun);

        EXPECT_EQ(0, rho.norm2(0, infra.geom.periodicity()));
    }    
    
    // Adds one particle exactly between two nodes
    TEST_F(DepositRhoTest, SingleParticleMiddle) {
        ASSERT_EQ(0, rho.norm2(0, infra.geom.periodicity()));
        const int numParticles{1};

        // Add particle in the middle of final cell to check periodic boundary conditions
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{AMREX_D_DECL(infra.geom.ProbHi(xDir) - 0.5*infra.dx[xDir],
                      infra.geom.ProbHi(yDir) - 0.5*infra.dx[yDir],
                      infra.geom.ProbHi(zDir) - 0.5*infra.dx[zDir])};
        amrex::Array<amrex::Real, numParticles> weights{3};
        // Expect the 2^GEMPIC_SPACEDIM nearest nodes of rhoarr (9/10, 9/10, 9/10) to be non-zero and receiving 1/2^GEMPIC_SPACEDIM the weight of the particle (3)
        const auto charge{particleGroup[0]->getCharge()};
        amrex::Real expectedVal{charge * infra.dxi[GEMPIC_SPACEDIM] * weights[0] * pow(0.5, GEMPIC_SPACEDIM)};

        GEMPIC_TestUtils::addSingleParticles(particleGroup, infra, weights, positions);
        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle.
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np); // Only one particle added

            updateRhoParallelFor<vDim, degX, degY, degZ>(pti, infra, rho, particleGroup[0]->getCharge());

            // Expect the eight nearest nodes of rhoarr (9/10, 9/10, 9/10) to be non-zero and receiving 1/8 the weight of the particle (3)
            checkField(rho[pti].array(), infra.n_cell.dim3(),
                    // Expect the eight nearest nodes of rhoarr (9/10, 9/10, 9/10) to be non-zero 
                    {[] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(a >= 9,
                                                                              && b >= 9,
                                                                              && c >= 9);}},
                    // and receiving 1/8 the weight of the particle (3)
                    {expectedVal},
                    // with the remaining entries being 0
                    0);
        }
        rho.SumBoundary(0, 1, Nghosts, dstNGhosts, infra.geom.periodicity());
        rho.FillBoundary(infra.geom.periodicity());
        
        // Maximum occurs evenly split between 2^GEMPIC_SPACEDIM nodes. The sum is still 1.
        EXPECT_EQ(expectedVal, rho.norm0());
        EXPECT_EQ(weights[0], rho.norm1(0, infra.geom.periodicity()));
    }

    // Adds one particle closer to on node than the other
    TEST_F(DepositRhoTest, SingleParticleUnevenNodeSplit) { 
        ASSERT_EQ(0, rho.norm2(0, infra.geom.periodicity()));
        const int numParticles{1};

        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{AMREX_D_DECL(infra.geom.ProbLo(xDir) + 0.25*infra.dx[xDir],
                      infra.geom.ProbLo(yDir) + 0.25*infra.dx[yDir],
                      infra.geom.ProbLo(zDir) + 0.25*infra.dx[zDir])};
        amrex::Array<amrex::Real, numParticles> weights{1};

        GEMPIC_TestUtils::addSingleParticles(particleGroup, infra, weights, positions);
        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle.
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np); // Only one particle added

            updateRhoParallelFor<vDim, degX, degY, degZ>(pti, infra, rho, particleGroup[0]->getCharge());

            // Expect the 2^GEMPIC_SPACEDIM nearest nodes of rhoarr (0/1, 0/1, 0/1) to be non-zero and  0 nodes receiving (3/4) and 1 nodes receiving (1/4) the weight of the particle (1)
            checkField(rho[pti].array(), infra.n_cell.dim3(),
                // Expect the 2^SPACEDIM nearest nodes of rhoarr (0/1, 0/1, 0/1) to be non-zero 
                {[] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(a == 0,
                                                                          && b == 0,
                                                                          && c == 0);},
                [] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(a == 1,
                                                                         && b == 1,
                                                                         && c == 1);},
                [] (AMREX_D_DECL(int a, int b, int c)) {return GEMPIC_D_ADD(a, b, c) == 1;},
                [](AMREX_D_DECL(int a, int b, int c)){return (GEMPIC_D_ADD(a*a, b*b, c*c) == 2);}},
                // and  0 nodes receiving (3/4) and 1 nodes receiving (1/4) the weight of the particle (1)
                {pow(0.75, GEMPIC_SPACEDIM),
                pow(0.25, GEMPIC_SPACEDIM),
                0.25*pow(0.75, GEMPIC_SPACEDIM - 1),
                pow(0.25, GEMPIC_SPACEDIM - 1)*0.75},
                // with the remaining entries being 0
                0);
        }
        rho.SumBoundary(0, 1, Nghosts, dstNGhosts, infra.geom.periodicity());
        rho.FillBoundary(infra.geom.periodicity());
        
        // Maximum occurs on node (0, 0, 0) with value (3/4)^GEMPIC_SPACEDIM. The sum is still 1.
        EXPECT_EQ(pow(0.75,GEMPIC_SPACEDIM), rho.norm0());
        EXPECT_EQ(1, rho.norm1(0, infra.geom.periodicity()));
    }    
    
    // Adds two particles in different cells to check that they don't interfere with each other
    TEST_F(DepositRhoTest, DoubleParticleSeparate) {
        const int numParticles{2};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{{{
            AMREX_D_DECL(0, 0, 0)},
            {AMREX_D_DECL(infra.geom.ProbLo(xDir) + 5.5*infra.dx[xDir],
                          infra.geom.ProbLo(yDir) + 5.5*infra.dx[yDir],
                          infra.geom.ProbLo(zDir) + 5.5*infra.dx[zDir])}}};
        
        amrex::Array<amrex::Real, numParticles> weights{1, 3};
        amrex::Real expectedValA{1}, expectedValB{3*pow(0.5, GEMPIC_SPACEDIM)};

        GEMPIC_TestUtils::addSingleParticles(particleGroup, infra, weights, positions);
        
        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over two distant particles.
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np); // Two particles added

            updateRhoParallelFor<vDim, degX, degY, degZ>(pti, infra, rho, particleGroup[0]->getCharge());

            // See SingleParticle test for explanation of expectations
            checkField(rho[pti].array(), infra.n_cell.dim3(),
                {[](AMREX_D_DECL(int a, int b, int c)){return AMREX_D_TERM(a == 0,
                                                                        && b == 0,
                                                                        && c == 0);},
                [](AMREX_D_DECL(int a, int b, int c)){return AMREX_D_TERM((a == 5 || a == 6),
                                                                       && (b == 5 || b == 6),
                                                                       && (c == 5 || c == 6));}},
                {expectedValA, expectedValB},
                // with the remaining entries being 0
                0);
        }
        rho.SumBoundary(0, 1, Nghosts, dstNGhosts, infra.geom.periodicity());
        rho.FillBoundary(infra.geom.periodicity());
        
        // The maximum expectedVal depends on the dimension on the problem
        EXPECT_EQ(std::max(expectedValA, expectedValB), rho.norm0());
        // Total charge added is the sum of each weight*charge, here 1 + 3
        EXPECT_EQ(4, rho.norm1(0, infra.geom.periodicity()));
    }

    // Adds particles in the same cell to check that they add up correctly
    TEST_F(DepositRhoTest, DoubleParticleOverlap) {
        const int numParticles{2};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{{{
            AMREX_D_DECL(0, 0, 0)},
            {AMREX_D_DECL(
            infra.geom.ProbLo(xDir) + 0.5*infra.dx[xDir],
            infra.geom.ProbLo(yDir) + 0.5*infra.dx[yDir],
            infra.geom.ProbLo(zDir) + 0.5*infra.dx[zDir])}}};
        amrex::Array<amrex::Real, numParticles> weights{1, 3};
        
        amrex::Real expectedValA{1 + 3*pow(0.5, GEMPIC_SPACEDIM)};
        amrex::Real expectedValB{3*pow(0.5, GEMPIC_SPACEDIM)};

        GEMPIC_TestUtils::addSingleParticles(particleGroup, infra, weights, positions);
        
        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over two close particles.
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np); // Two particles added

            updateRhoParallelFor<vDim, degX, degY, degZ>(pti, infra, rho, particleGroup[0]->getCharge());

            // See SingleParticle test for explanation of expectations
            checkField(rho[pti].array(), infra.n_cell.dim3(),
                    {[] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(a == 0,
                                                                              && b == 0,
                                                                              && c == 0);},
                    [] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(a <= 1,
                                                                             && b <= 1,
                                                                             && c <= 1);}},
                    {expectedValA, expectedValB},
                    0);
        }
        rho.SumBoundary(0, 1, Nghosts, dstNGhosts, infra.geom.periodicity());
        rho.FillBoundary(infra.geom.periodicity());
        
        EXPECT_EQ(expectedValA, rho.norm0());
        EXPECT_EQ(4, rho.norm1(0, infra.geom.periodicity()));
    }

    // Adds particles of different species in the same cell
    TEST_F(DepositRhoTest, DoubleParticleMultipleSpecies) {
        const int numParticles{1};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> pPos{{
            AMREX_D_DECL(0, 0, 0)}};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> ePos{{
            AMREX_D_DECL(infra.geom.ProbLo(xDir) + 0.5*infra.dx[xDir],
                         infra.geom.ProbLo(yDir) + 0.5*infra.dx[yDir],
                         infra.geom.ProbLo(zDir) + 0.5*infra.dx[zDir])}};
        amrex::Array<amrex::Real, numParticles> pWeights{1};
        amrex::Array<amrex::Real, numParticles> eWeights{3};
        int pSpec{0}, eSpec{1};
        
        GEMPIC_TestUtils::addSingleParticles(particleGroup, infra, pWeights, pPos, pSpec);
        GEMPIC_TestUtils::addSingleParticles(particleGroup, infra, eWeights, ePos, eSpec);

        const auto pCharge{particleGroup[pSpec]->getCharge()};
        const auto eCharge{particleGroup[eSpec]->getCharge()};

        amrex::Real expectedValA{pCharge + eCharge*3*pow(0.5, GEMPIC_SPACEDIM)};
        amrex::Real expectedValB{eCharge*3*pow(0.5, GEMPIC_SPACEDIM)};
        
        for (int spec{0}; spec < numSpec; spec++)
        {
            particleGroup[spec]->Redistribute();  // assign particles to the tile they are in
            const auto charge{particleGroup[spec]->getCharge()};
            // Particle iteration
            for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[spec], 0); pti.isValid(); ++pti)
            {
                const long np{pti.numParticles()};
                EXPECT_EQ(numParticles, np); // Two particles added

                updateRhoParallelFor<vDim, degX, degY, degZ>(pti, infra, rho, charge);

                if (spec == numSpec - 1) {
                    // See SingleParticle test for explanation of expectations
                    checkField(rho[pti].array(), infra.n_cell.dim3(),
                            {[] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(a == 0,
                                                                                      && b == 0,
                                                                                      && c == 0);},
                            [] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(a <= 1,
                                                                                     && b <= 1,
                                                                                     && c <= 1);}},
                    {expectedValA, expectedValB},
                    0);
                }
            }
        }
            rho.SumBoundary(0, 1, Nghosts, dstNGhosts, infra.geom.periodicity());
            rho.FillBoundary(infra.geom.periodicity());
            
            // The maximum expectedVal depends on the dimension on the problem
            EXPECT_EQ(std::max(std::abs(expectedValA), std::abs(expectedValB)), rho.norm0());
            
            // Probably not GPU safe. Second argument of sum_unique is bool local, which decides if parallel reduction is done
            EXPECT_EQ(pCharge*pWeights[0] + eCharge*eWeights[0], rho.sum_unique(0, 0, infra.geom.periodicity()));
    }
}