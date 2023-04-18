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
#include <GEMPIC_gempic_norm.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling_C2.H>
#include "gtest/gtest.h"
#include "test_utils/GEMPIC_test_utils.H"

using namespace Particles;

//#define checkRho(rhoarr, top, condVec, checks, defCheck) checkRhoLine(__LINE__, rhoarr, top, condVec, checks, defCheck)

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
                splines_at_particles<degX, degY, degZ> spline;
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
                for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
                    position[d] = partData[pp].pos(d);
                spline.init_particles(position, infra.plo, infra.dxi);
                // Needs at least max(degX, degY, degZ) ghost cells
                gempic_deposit_rho_C3<degX, degY, degZ>(
                    spline, charge * infra.dxi[GEMPIC_SPACEDIM] * weight[pp],
                    rhoarr);
            });
        }

        /* Helper function to check entries of rho given a series of conditions and a default
         * value. Check order is prioritized, so a set of indices only fulfill the first succesful
         * condition.
         * 
         * Parameters:
         * ----------
         * @param line: int, the line from which the function was called
         * @param rhoarr: amrex::Array4, array containing rho values in an easily reached accessor
         * @param top: Dim3, top boundaries of box for rhoarray
         * @param condVec: vector<condLambda>, Vector of lambdas that check if the {SPACEDIM} indices fulfill a given condition.
         * @param checks: vector<amrex::Real>, Vector of values to compare to if indices fulfill the corresponding condVec condition.
         * @param defCheck: amrex::Real, Default value for all indices not fulfilling any of the given conditions.
         */
        using condLambda = bool(*)(AMREX_D_DECL(int, int, int));
        void checkRho(int line,
                      amrex::Array4<amrex::Real> const& rhoarr,
                      amrex::Dim3 const&& top,
                      std::vector<condLambda>&& condVec,
                      std::vector<amrex::Real>&& checks,
                      amrex::Real defCheck) {
            // Expect only one node of rhoarr (0, 0, 0) to be non-zero and receiving full weight of particle (1)
            for (int i{0}; i <= top.x; i++) { 
                for (int j{0}; j <= top.y; j++) {
                    for (int k{0}; k <= top.z; k++) {
                        int condNum{0};
                        amrex::Array<int, GEMPIC_SPACEDIM> idx{AMREX_D_DECL(i, j, k)};
                        for (auto cond : condVec) {
                            if (cond(AMREX_D_DECL(i, j, k))) {
                                EXPECT_EQ(checks[condNum], *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0)) <<
                                   "LINE:" << line << ": Failed condition " << condNum <<
                                   ".\nIndices: " << GEMPIC_TestUtils::stringArray(idx, GEMPIC_SPACEDIM);
                                   break;
                            }
                            condNum++;
                        }
                        if (condNum == condVec.size()) {
                            EXPECT_EQ(defCheck, *rhoarr.ptr(AMREX_D_DECL(i, j, k), 0)) <<
                                "LINE:" << line << ": Failed default value check:" << defCheck <<
                                ".\nIndices: " << GEMPIC_TestUtils::stringArray(idx, GEMPIC_SPACEDIM);
                        }
                    }
                }
            }
        }

    // Test fixture. Sets up clean environment before each test.
    class DepositRhoTest : public testing::Test {
        protected:

        // Degree of splines in each direction
        static const int degX{1};
        static const int degY{1};
        static const int degZ{1};

        // Number of species (second species only used for DoubleParticleMultipleSpecies)
        static const int numSpec{2};
        // Number of velocity dimensions.
        static const int vDim{0};
        // Number of ghost cells in mesh
        const int Nghost{GEMPIC_TestUtils::initNGhost(degX, degY, degZ)};

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
            ASSERT_EQ(0,Gempic::Utils::gempic_norm(rho, infra, 2));
            
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
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        // (default) charge correctly transferred from GEMPIC_TestUtils::addSingleParticles
        EXPECT_EQ(1, particleGroup[0]->getCharge()); 
        
        // rho unchanged by GEMPIC_TestUtils::addSingleParticles
        EXPECT_EQ(0,Gempic::Utils::gempic_norm(rho, infra, 2));

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
            splines_at_particles<degX, degY, degZ> spline;
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
            for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
                position[d] = partData[0].pos(d);
            spline.init_particles(position, infra.plo, infra.dxi);
            // Needs at least max(degX, degY, degZ) ghost cells
            gempic_deposit_rho_C3<degX, degY, degZ>(spline, 0, rhoarr);
        }
        ASSERT_TRUE(particleLoopRun);
        
        EXPECT_EQ(0,Gempic::Utils::gempic_norm(rho, infra, 2));
    }

    // Adds one particle exactly on a node
    TEST_F(DepositRhoTest, SingleParticleOnNode) {
        // Adding particle to one cell
        const int numParticles{1};

        // Particle at position (0,0,0) in box (0,0,0)
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{*infra.geom.ProbLo()};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle.
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np); // Only one particle added

            updateRhoParallelFor<vDim, degX, degY, degZ>(pti, infra, rho, particleGroup[0]->getCharge());

            // Expect only one node of rhoarr (0, 0, 0) to be non-zero and receiving full weight of particle (1)
            checkRho(__LINE__, rho[pti].array(), infra.n_cell.dim3(),
                    // Expect only one node of rhoarr (0, 0, 0) to be non-zero
                    {[] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(a == 0,
                                                                              && b == 0,
                                                                              && c == 0);}},
                    // and receiving full weight of particle (1)
                    {1},
                    // with the remaining entries being 0
                    0);
        }
        rho.SumBoundary(0, 1, {Nghost, Nghost, Nghost}, {0, 0, 0}, infra.geom.periodicity());
        rho.FillBoundary(infra.geom.periodicity());

        // Maximum occurs on node (0,0,0) and contains all of the 1 charge 1 particle of weight 1
        EXPECT_EQ(1, rho.norm0());
        EXPECT_EQ(1, rho.norm1(0, infra.geom.periodicity()));
    }

    // Adds one particle exactly between two nodes
    TEST_F(DepositRhoTest, SingleParticleMiddle) {
        const int numParticles{1};

        // Add particle in the middle of final cell to check periodic boundary conditions
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{AMREX_D_DECL(infra.geom.ProbHi(0) - 0.5*infra.dx[0],
                      infra.geom.ProbHi(1) - 0.5*infra.dx[1],
                      infra.geom.ProbHi(2) - 0.5*infra.dx[2])};
        amrex::Array<amrex::Real, numParticles> weights{3};
        // Expect the 2^GEMPIC_SPACEDIM nearest nodes of rhoarr (9/10, 9/10, 9/10) to be non-zero and receiving 1/2^GEMPIC_SPACEDIM the weight of the particle (3)
        const auto charge{particleGroup[0]->getCharge()};
        amrex::Real expectedVal{charge * infra.dxi[GEMPIC_SPACEDIM] * weights[0] * pow(0.5, GEMPIC_SPACEDIM)};

        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);
        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle.
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np); // Only one particle added

            updateRhoParallelFor<vDim, degX, degY, degZ>(pti, infra, rho, particleGroup[0]->getCharge());

            // Expect the eight nearest nodes of rhoarr (9/10, 9/10, 9/10) to be non-zero and receiving 1/8 the weight of the particle (3)
            checkRho(__LINE__, rho[pti].array(), infra.n_cell.dim3(),
                    // Expect the eight nearest nodes of rhoarr (9/10, 9/10, 9/10) to be non-zero 
                    {[] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(a >= 9,
                                                                              && b >= 9,
                                                                              && c >= 9);}},
                    // and receiving 1/8 the weight of the particle (3)
                    {expectedVal},
                    // with the remaining entries being 0
                    0);
        }
        rho.SumBoundary(0, 1, {Nghost, Nghost, Nghost}, {0, 0, 0}, infra.geom.periodicity());
        rho.FillBoundary(infra.geom.periodicity());
        
        // Maximum occurs evenly split between 2^GEMPIC_SPACEDIM nodes. The sum is still 1.
        EXPECT_EQ(expectedVal, rho.norm0());
        EXPECT_EQ(weights[0], rho.norm1(0, infra.geom.periodicity()));
    }

    // Adds one particle closer to on node than the other
    TEST_F(DepositRhoTest, SingleParticleUnevenNodeSplit) { 
        const int numParticles{1};

        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{AMREX_D_DECL(infra.geom.ProbLo(0) + 0.25*infra.dx[0],
                      infra.geom.ProbLo(1) + 0.25*infra.dx[1],
                      infra.geom.ProbLo(2) + 0.25*infra.dx[2])};
        amrex::Array<amrex::Real, numParticles> weights{1};

        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);
        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle.
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np); // Only one particle added

            updateRhoParallelFor<vDim, degX, degY, degZ>(pti, infra, rho, particleGroup[0]->getCharge());

            // Expect the 2^GEMPIC_SPACEDIM nearest nodes of rhoarr (0/1, 0/1, 0/1) to be non-zero and  0 nodes receiving (3/4) and 1 nodes receiving (1/4) the weight of the particle (1)
            checkRho(__LINE__, rho[pti].array(), infra.n_cell.dim3(),
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
        rho.SumBoundary(0, 1, {Nghost, Nghost, Nghost}, {0, 0, 0}, infra.geom.periodicity());
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
            {AMREX_D_DECL(infra.geom.ProbLo(0) + 5.5*infra.dx[0],
                          infra.geom.ProbLo(1) + 5.5*infra.dx[1],
                          infra.geom.ProbLo(2) + 5.5*infra.dx[2])}}};
        
        amrex::Array<amrex::Real, numParticles> weights{1, 3};
        amrex::Real expectedValA{1}, expectedValB{3*pow(0.5, GEMPIC_SPACEDIM)};

        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);
        
        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over two distant particles.
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np); // Two particles added

            updateRhoParallelFor<vDim, degX, degY, degZ>(pti, infra, rho, particleGroup[0]->getCharge());

            // See SingleParticle test for explanation of expectations
            checkRho(__LINE__, rho[pti].array(), infra.n_cell.dim3(),
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
        rho.SumBoundary(0, 1, {Nghost, Nghost, Nghost}, {0, 0, 0}, infra.geom.periodicity());
        rho.FillBoundary(infra.geom.periodicity());
        
        EXPECT_EQ(expectedValA, rho.norm0());
        // Total charge added is the sum of each weight*charge, here 1 + 3
        EXPECT_EQ(4, rho.norm1(0, infra.geom.periodicity()));
    }

    // Adds particles in the same cell to check that they add up correctly
    TEST_F(DepositRhoTest, DoubleParticleOverlap) {
        const int numParticles{2};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{{{
            AMREX_D_DECL(0, 0, 0)},
            {AMREX_D_DECL(
            infra.geom.ProbLo(0) + 0.5*infra.dx[0],
            infra.geom.ProbLo(1) + 0.5*infra.dx[1],
            infra.geom.ProbLo(2) + 0.5*infra.dx[2])}}};
        amrex::Array<amrex::Real, numParticles> weights{1, 3};
        
        amrex::Real expectedValA{1 + 3*pow(0.5, GEMPIC_SPACEDIM)};
        amrex::Real expectedValB{3*pow(0.5, GEMPIC_SPACEDIM)};

        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);
        
        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over two close particles.
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np); // Two particles added

            updateRhoParallelFor<vDim, degX, degY, degZ>(pti, infra, rho, particleGroup[0]->getCharge());

            // See SingleParticle test for explanation of expectations
            checkRho(__LINE__, rho[pti].array(), infra.n_cell.dim3(),
                    {[] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(a == 0,
                                                                              && b == 0,
                                                                              && c == 0);},
                    [] (AMREX_D_DECL(int a, int b, int c)) {return AMREX_D_TERM(a <= 1,
                                                                             && b <= 1,
                                                                             && c <= 1);}},
                    {expectedValA, expectedValB},
                    0);
        }
        rho.SumBoundary(0, 1, {Nghost, Nghost, Nghost}, {0, 0, 0}, infra.geom.periodicity());
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
            AMREX_D_DECL(infra.geom.ProbLo(0) + 0.5*infra.dx[0],
                         infra.geom.ProbLo(1) + 0.5*infra.dx[1],
                         infra.geom.ProbLo(2) + 0.5*infra.dx[2])}};
        amrex::Array<amrex::Real, numParticles> pWeights{1};
        amrex::Array<amrex::Real, numParticles> eWeights{3};
        int pSpec{0}, eSpec{1};
        
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, pWeights, pPos, pSpec);
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, eWeights, ePos, eSpec);

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
                    checkRho(__LINE__, rho[pti].array(), infra.n_cell.dim3(),
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
            rho.SumBoundary(0, 1, {Nghost, Nghost, Nghost}, {0, 0, 0}, infra.geom.periodicity());
            rho.FillBoundary(infra.geom.periodicity());
            
            EXPECT_EQ(expectedValA, rho.norm0());
            
            // Probably not GPU safe. Second argument of sum_unique is bool local, which decides if parallel reduction is done
            EXPECT_EQ(pCharge*pWeights[0] + eCharge*eWeights[0], rho.sum_unique(0, 0, infra.geom.periodicity()));
    }
}