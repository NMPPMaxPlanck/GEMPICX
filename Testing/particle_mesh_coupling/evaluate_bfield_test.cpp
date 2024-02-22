/** Testing for evaluate_bfield function 
*/

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_particle_groups.H>
#include "gtest/gtest.h"
#include "test_utils/GEMPIC_test_utils.H"
#include "GEMPIC_Spline_Class.H"

using namespace Gempic;
using namespace CompDom;
using namespace Particles;
using namespace GEMPIC_FDDeRhamComplex;
using namespace GEMPIC_Fields;

//Basics first
namespace {
    // When using amrex::ParallelFor you have to create a standalone helper function that does the execution on GPU and call that function from the unit test because of how GTest creates tests within a TEST_F fixture.
    template <int vDim, int degX, int degY, int degZ>
    amrex::GpuArray<amrex::Real, vDim>* updateBFieldParallelFor(amrex::ParIter<0, 0, vDim + 1, 0>& pti,
                                 DeRhamField<Grid::primal, Space::edge>& B,
                                 computational_domain& infra) {
        const long np{pti.numParticles()};
        const auto& particles{pti.GetArrayOfStructs()};
        const auto partData{particles().data()};
        amrex::AsyncArray<amrex::GpuArray<amrex::Real, vDim>> bfieldsArr(2);
        amrex::GpuArray<amrex::Real, vDim>* bfields = bfieldsArr.data();

        amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> bArray;
        for (int cc{0}; cc < vDim; cc++) bArray[cc] = (B.data[cc])[pti].array();

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
        {
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
            for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
                position[d] = partData[0].pos(d);
            Spline::SplineBase<degX, degY, degZ> spline(position, infra.plo, infra.dxi);

            bfields[pp] = spline.template evalSplineField<Field::PrimalTwoForm>(bArray);
        });

        return bfields;
    }

    // Test fixture
    class EvaluateBFieldTest : public testing::Test {
        protected:

        static const int degX{1};
        static const int degY{1};
        static const int degZ{1};
        inline static const int maxSplineDegree{std::max(std::max(degX, degY), degZ)};

        inline static const int hodgeDegree{2};
        static const int numSpec{1};
        static const int vDim{3};
        static const int spec{0};
        Parameters parameters{};

        computational_domain infra{false}; // "uninitialized" computational domain
        amrex::GpuArray<std::unique_ptr<particle_groups<vDim>>, numSpec> particleGroup;
        std::shared_ptr<GEMPIC_FDDeRhamComplex::FDDeRhamComplex> deRham;

        static void SetUpTestSuite()
        {
            /* Initialize the infrastructure */
            //const amrex::RealBox realBox({AMREX_D_DECL(0.0, 0.0, 0.0)},
            //                             {AMREX_D_DECL(10.0, 10.0, 10.0)});
            amrex::Vector<amrex::Real> domain_lo{AMREX_D_DECL(0.0, 0.0, 0.0)};
            // 
            amrex::Vector<amrex::Real> k{AMREX_D_DECL(0.2*M_PI, 0.2*M_PI, 0.2*M_PI)};
            const amrex::Vector<int> nCell{AMREX_D_DECL(10, 10, 10)};
            const amrex::Vector<int> maxGridSize{AMREX_D_DECL(10, 10, 10)};
            const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};


            amrex::ParmParse pp;
            pp.addarr("domain_lo", domain_lo);
            pp.addarr("k", k);
            pp.addarr("n_cell_vector", nCell);
            pp.addarr("max_grid_size_vector", maxGridSize);
            pp.addarr("is_periodic_vector", isPeriodic);

            // particle settings
            double charge{1};
            double mass{1};

            pp.add("particle.species0.charge", charge);
            pp.add("particle.species0.mass", mass);
        }

        // virtual void SetUp() will be called before each test is run.
        void SetUp() override {
            /* Initialize the infrastructure */
            infra = computational_domain{};

            // Initialize the De Rham Complex
            deRham = std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, maxSplineDegree, HodgeScheme::FDHodge);

            // particles
            for (int spec{0}; spec < numSpec; spec++)
            {
                particleGroup[spec] =
                    std::make_unique<particle_groups<vDim>>(spec, infra);
            }

        }
    };

    TEST_F(EvaluateBFieldTest, NullTest) {
        // Adding particle to one cell
        const int numParticles{1};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{{{*infra.geom.ProbLo()}}};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles(particleGroup, infra, weights, positions);

        // (default) charge correctly transferred from addSingleParticles
        EXPECT_EQ(1, particleGroup[0]->getCharge());

        // Parse analytical fields and initialize parserEval. Has to be the same as Bx,By,Bz
        const amrex::Array<std::string, 3> analyticalFuncB = {"0.0", 
                                                              "0.0",
                                                              "0.0"};

        const int nVar{GEMPIC_SPACEDIM + 1};  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
        amrex::Array<amrex::Parser, 3> parser;

        for (int i{0}; i<3; ++i)
        {
            parser[i].define(analyticalFuncB[i]);
            parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            funcB[i] = parser[i].compile<nVar>();
        }

        DeRhamField<Grid::primal, Space::edge> B(deRham, funcB);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle. Hopefully.

        bool particleLoopRun{false};
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[spec], 0); pti.isValid(); ++pti)
        {
            particleLoopRun = true;

            const long np{pti.numParticles()};
            EXPECT_EQ(1, np); // Only one particle added by addSingleParticles

            const auto& particles{pti.GetArrayOfStructs()};
            const auto partData{particles().data()};

            amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> bArray;
            for (int cc{0}; cc < vDim; cc++) bArray[cc] = (B.data[cc])[pti].array();

            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
            for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
                position[d] = partData[0].pos(d);
            Spline::SplineBase<degX, degY, degZ> spline(position, infra.plo, infra.dxi);

            
            amrex::GpuArray<amrex::Real, vDim> bfield =
                spline.template evalSplineField<Field::PrimalTwoForm>(bArray);

            EXPECT_EQ(bfield[xDir], 0);
            EXPECT_EQ(bfield[yDir], 0);
            EXPECT_EQ(bfield[zDir], 0);
        }
        ASSERT_TRUE(particleLoopRun);
    }

    TEST_F(EvaluateBFieldTest, SingleParticleNode) {
        // Adding particle to one cell
        const int numParticles{1};
        // Particle at position (0,0,0) in box (0,0,0)
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{{{*infra.geom.ProbLo()}}};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles(particleGroup, infra, weights, positions);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in

        // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
        // Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncB{"1.0", "1.0", "1.0"};

        const int nVar{GEMPIC_SPACEDIM + 1};  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
        amrex::Array<amrex::Parser, 3> parser;

        for (int i{0}; i < 3; ++i)
        {
            parser[i].define(analyticalFuncB[i]);
            parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            funcB[i] = parser[i].compile<nVar>();
        }

        DeRhamField<Grid::primal, Space::edge> B(deRham, funcB);

        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np);

            amrex::GpuArray<amrex::Real, vDim>* bfields = updateBFieldParallelFor<vDim, degX, degY, degZ>(pti, B, infra);
                        
            EXPECT_EQ(bfields[0][xDir], 1.0);
            EXPECT_EQ(bfields[0][yDir], 1.0);
            EXPECT_EQ(bfields[0][zDir], 1.0);
        }
    }

    TEST_F(EvaluateBFieldTest, SingleParticleMiddle) {
        // Adding particle to one cell
        const int numParticles{1};
        // Add particle in the middle of final cell to check periodic boundary conditions
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{{{AMREX_D_DECL(infra.geom.ProbHi(xDir) - 1.5*infra.dx[xDir],
                        infra.geom.ProbHi(yDir) - 1.5*infra.dx[yDir],
                        infra.geom.ProbHi(zDir) - 1.5*infra.dx[zDir])}}};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles(particleGroup, infra, weights, positions);
        
        particleGroup[0]->Redistribute();  // assign particles to the tile they are in

        // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
        // Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncB{"1.0", "1.0", "1.0"};

        const int nVar{GEMPIC_SPACEDIM + 1};  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
        amrex::Array<amrex::Parser, 3> parser;

        for (int i{0}; i < 3; ++i)
        {
            parser[i].define(analyticalFuncB[i]);
            parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            funcB[i] = parser[i].compile<nVar>();
        }

        DeRhamField<Grid::primal, Space::edge> B(deRham, funcB);

        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np);
            
            amrex::GpuArray<amrex::Real, vDim>* bfields = updateBFieldParallelFor<vDim, degX, degY, degZ>(pti, B, infra);
                        
            EXPECT_EQ(bfields[0][xDir], 1.0);
            EXPECT_EQ(bfields[0][yDir], 1.0);
            EXPECT_EQ(bfields[0][zDir], 1.0);
        }
    }

    TEST_F(EvaluateBFieldTest, SingleParticleUnevenNodeSplit) {
        // Adding particle to one cell
        const int numParticles{1};
        // Add particle in the middle of final cell to check periodic boundary conditions
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{{{AMREX_D_DECL(infra.geom.ProbHi(xDir) - 1.25*infra.dx[xDir],
                        infra.geom.ProbHi(yDir) - 1.25*infra.dx[yDir],
                        infra.geom.ProbHi(zDir) - 1.25*infra.dx[zDir])}}};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles(particleGroup, infra, weights, positions);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in

        // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
        // Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncB{"1.0", "1.0", "1.0"};

        const int nVar{GEMPIC_SPACEDIM + 1};  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
        amrex::Array<amrex::Parser, 3> parser;

        for (int i{0}; i < 3; ++i)
        {
            parser[i].define(analyticalFuncB[i]);
            parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            funcB[i] = parser[i].compile<nVar>();
        }

        DeRhamField<Grid::primal, Space::edge> B(deRham, funcB);

        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np);
            
            amrex::GpuArray<amrex::Real, vDim>* bfields = updateBFieldParallelFor<vDim, degX, degY, degZ>(pti, B, infra);
                        
            EXPECT_EQ(bfields[0][xDir], 1.0);
            EXPECT_EQ(bfields[0][yDir], 1.0);
            EXPECT_EQ(bfields[0][zDir], 1.0);
        }
    }

    TEST_F(EvaluateBFieldTest, DoubleParticleSeparate) {
        const int numParticles{2};
        // Particles in different cells to check that they don't interfere with each other
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{{{
            AMREX_D_DECL(0, 0, 0)},
            {AMREX_D_DECL(
            infra.geom.ProbLo(xDir) + 5.5*infra.dx[xDir],
            infra.geom.ProbLo(yDir) + 5.5*infra.dx[yDir],
            infra.geom.ProbLo(zDir) + 5.5*infra.dx[zDir])}}};
        amrex::Array<amrex::Real, numParticles> weights{1, 1};
        GEMPIC_TestUtils::addSingleParticles(particleGroup, infra, weights, positions);
        
        particleGroup[0]->Redistribute();  // assign particles to the tile they are in

        // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
        // Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncB{"1.0", "1.0", "1.0"};

        const int nVar{GEMPIC_SPACEDIM + 1};  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
        amrex::Array<amrex::Parser, 3> parser;

        for (int i{0}; i < 3; ++i)
        {
            parser[i].define(analyticalFuncB[i]);
            parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            funcB[i] = parser[i].compile<nVar>();
        }

        DeRhamField<Grid::primal, Space::edge> B(deRham, funcB);

        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np);

            amrex::GpuArray<amrex::Real, vDim>* bfields = updateBFieldParallelFor<vDim, degX, degY, degZ>(pti, B, infra);
                        
            EXPECT_EQ(bfields[0][xDir], 1.0);
            EXPECT_EQ(bfields[0][yDir], 1.0);
            EXPECT_EQ(bfields[0][zDir], 1.0);
                        
            EXPECT_EQ(bfields[1][xDir], 1.0);
            EXPECT_EQ(bfields[1][yDir], 1.0);
            EXPECT_EQ(bfields[1][zDir], 1.0);
        }
    }

    TEST_F(EvaluateBFieldTest, DoubleParticleOverlap) {
        const int numParticles{2};
        // Particles in different cells to check that they don't interfere with each other
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{{{
            AMREX_D_DECL(0, 0, 0)},
            {AMREX_D_DECL(
            infra.geom.ProbLo(xDir) + 0.5*infra.dx[xDir],
            infra.geom.ProbLo(yDir) + 0.5*infra.dx[yDir],
            infra.geom.ProbLo(zDir) + 0.5*infra.dx[zDir])}}};
        amrex::Array<amrex::Real, numParticles> weights{1, 1};
        GEMPIC_TestUtils::addSingleParticles(particleGroup, infra, weights, positions);
        
        particleGroup[0]->Redistribute();  // assign particles to the tile they are in

        // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
        // Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncB{"1.0", "1.0", "1.0"};

        const int nVar{GEMPIC_SPACEDIM + 1};  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
        amrex::Array<amrex::Parser, 3> parser;

        for (int i{0}; i < 3; ++i)
        {
            parser[i].define(analyticalFuncB[i]);
            parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            funcB[i] = parser[i].compile<nVar>();
        }

        DeRhamField<Grid::primal, Space::edge> B(deRham, funcB);

        amrex::MFItInfo mfii{};
        mfii.do_tiling = amrex::TilingIfNotGPU();

        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0, mfii); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np);

            amrex::GpuArray<amrex::Real, vDim>* bfields = updateBFieldParallelFor<vDim, degX, degY, degZ>(pti, B, infra);
                        
            EXPECT_EQ(bfields[0][xDir], 1.0);
            EXPECT_EQ(bfields[0][yDir], 1.0);
            EXPECT_EQ(bfields[0][zDir], 1.0);
                        
            EXPECT_EQ(bfields[1][xDir], 1.0);
            EXPECT_EQ(bfields[1][yDir], 1.0);
            EXPECT_EQ(bfields[1][zDir], 1.0);
        }
    }

    TEST_F(EvaluateBFieldTest, TestingForLiterallyAnythingOtherThanUnity) {
        GTEST_SKIP() << "Such advanced tests have not yet been implemented!";
    }
}