/** Testing for evaluate_efield function 
*/

#include <AMReX.H>
#include <GEMPIC_Config.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling_C2.H>
#include "gtest/gtest.h"
#include "test_utils/GEMPIC_test_utils.H"

using namespace Particles;
using namespace GEMPIC_FDDeRhamComplex;
using namespace GEMPIC_Fields;

//Basics first
namespace {
    // When using amrex::ParallelFor you have to create a standalone helper function that does the execution on GPU and call that function from the unit test because of how GTest creates tests within a TEST_F fixture.
    template <int vDim, int degX, int degY, int degZ>
    void updateEFieldParallelFor(amrex::ParIter<0, 0, vDim + 1, 0>& pti,
                                 DeRhamField<Grid::primal, Space::edge>& E,
                                 computational_domain& infra) {
        const long np{pti.numParticles()};
        const auto& particles{pti.GetArrayOfStructs()};
        const auto partData{particles().data()};
        amrex::GpuArray<amrex::GpuArray<amrex::Real, vDim>, 2> efields;

        amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> eArray;
        for (int cc{0}; cc < vDim; cc++) eArray[cc] = (E.data[cc])[pti].array();

        amrex::ParallelFor(np, [=, &efields] AMREX_GPU_DEVICE(long pp)
        {
            splines_at_particles<degX, degY, degZ> spline;
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
            for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
            {
                position[d] = partData[pp].pos(d);
            }
            spline.init_particles(position, infra.plo, infra.dxi);

            amrex::GpuArray<amrex::Real, vDim> efield = evaluate_efield<vDim, degX, degY, degZ>(spline, eArray);
            efields[pp] = efield;
        });
                    
        EXPECT_EQ(efields[0][0], 1.0);
        EXPECT_EQ(efields[0][1], 1.0);
        EXPECT_EQ(efields[0][2], 1.0);
                    
        if(np == 2) {
            EXPECT_EQ(efields[1][0], 1.0);
            EXPECT_EQ(efields[1][1], 1.0);
            EXPECT_EQ(efields[1][2], 1.0);
        }
    }

    // Test fixture
    class EvaluateEFieldTest : public testing::Test {
        protected:

        static const int degX{1};
        static const int degY{1};
        static const int degZ{1};

        static const int numSpec{1};
        static const int vDim{3};
        static const int spec{0};
        const int Nghost{GEMPIC_TestUtils::initNGhost(degX, degY, degZ)};
        Parameters params;

        double charge{1};
        double mass{1};

        computational_domain infra;
        amrex::GpuArray<std::unique_ptr<particle_groups<vDim>>, numSpec> particleGroup;

        // virtual void SetUp() will be called before each test is run.
        void SetUp() override {
            /* Initialize the infrastructure */
            const amrex::RealBox realBox({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                            {AMREX_D_DECL(10.0, 10.0, 10.0)});
            const amrex::IntVect nCell{AMREX_D_DECL(10, 10, 10)};
            const amrex::IntVect maxGridSize{AMREX_D_DECL(10, 10, 10)};
            const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{1, 1, 1};
            const int hodgeDegree{2};


            // This class does the same as GEMPIC_parameters.H and needs to be urgently redesigned.
            // {1, 1, 1} represents periodicity, has different types than Params and gempic_parameters.
            infra.initialize_computational_domain(nCell, maxGridSize, {1, 1, 1}, realBox);

            params = Parameters(realBox, nCell, maxGridSize, isPeriodic, hodgeDegree);
            
            // particles
            for (int spec{0}; spec < numSpec; spec++)
            {
                particleGroup[spec] =
                    std::make_unique<particle_groups<vDim>>(charge, mass, infra);
            }

        }
    };

    TEST_F(EvaluateEFieldTest, NullTest) {
        // Adding particle to one cell
        const int numParticles{1};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{*infra.geom.ProbLo()};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        // (default) charge correctly transferred from addSinglePparticles
        EXPECT_EQ(1, particleGroup[0]->getCharge()); 

        // Parse analytical fields and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
        // Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncE{"0.0", "0.0", "0.0"};

        const int nVar{4};  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcE;
        amrex::Parser parser;

        for (int i{0}; i < 3; ++i)
        {
            parser.define(analyticalFuncE[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcE[i] = parser.compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham{std::make_shared<FDDeRhamComplex>(params)};

        DeRhamField<Grid::primal, Space::edge> E(deRham);
        deRham->projection(funcE, 0.0, E);

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

            amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> eArray;
            for (int cc{0}; cc < vDim; cc++) eArray[cc] = (E.data[cc])[pti].array();

            splines_at_particles<degX, degY, degZ> spline;
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
            for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
                position[d] = partData[0].pos(d);
            spline.init_particles(position, infra.plo, infra.dxi);

            amrex::GpuArray<amrex::Real, vDim> efield =
                evaluate_efield<vDim, degX, degY, degZ>(spline, eArray);

            EXPECT_EQ(efield[0], 0);
            EXPECT_EQ(efield[1], 0);
            EXPECT_EQ(efield[2], 0);
        }
        ASSERT_TRUE(particleLoopRun);
    }

    TEST_F(EvaluateEFieldTest, SingleParticleNode) {
        // Adding particle to one cell
        const int numParticles{1};
        // Particle at position (0,0,0) in box (0,0,0)
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{*infra.geom.ProbLo()};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in

        // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
        // Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncE{"1.0", "1.0", "1.0"};

        const int nVar{4};  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcE;
        amrex::Parser parser;

        for (int i{0}; i < 3; ++i)
        {
            parser.define(analyticalFuncE[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcE[i] = parser.compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham{std::make_shared<FDDeRhamComplex>(params)};

        DeRhamField<Grid::primal, Space::edge> E(deRham);
        deRham->projection(funcE, 0.0, E);

        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np);

            updateEFieldParallelFor<vDim, degX, degY, degZ>(pti, E, infra);
        }
    }

    TEST_F(EvaluateEFieldTest, SingleParticleMiddle) {
        // Adding particle to one cell
        const int numParticles{1};
        // Add particle in the middle of final cell to check periodic boundary conditions
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{AMREX_D_DECL(infra.geom.ProbHi()[0] - 1.5*infra.dx[0],
                      infra.geom.ProbHi()[1] - 1.5*infra.dx[1],
                      infra.geom.ProbHi()[2] - 1.5*infra.dx[2])};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);
        
        particleGroup[0]->Redistribute();  // assign particles to the tile they are in

        // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
        // Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncE{"1.0", "1.0", "1.0"};

        const int nVar{4};  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcE;
        amrex::Parser parser;

        for (int i{0}; i < 3; ++i)
        {
            parser.define(analyticalFuncE[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcE[i] = parser.compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham{std::make_shared<FDDeRhamComplex>(params)};

        DeRhamField<Grid::primal, Space::edge> E(deRham);
        deRham->projection(funcE, 0.0, E);

        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np);

            updateEFieldParallelFor<vDim, degX, degY, degZ>(pti, E, infra);
        }
    }

    TEST_F(EvaluateEFieldTest, SingleParticleUnevenNodeSplit) {
        // Adding particle to one cell
        const int numParticles{1};
        // Add particle in the middle of final cell to check periodic boundary conditions
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{AMREX_D_DECL(infra.geom.ProbHi()[0] - 1.25*infra.dx[0],
                      infra.geom.ProbHi()[1] - 1.25*infra.dx[1],
                      infra.geom.ProbHi()[2] - 1.25*infra.dx[2])};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in

        // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
        // Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncE{"1.0", "1.0", "1.0"};

        const int nVar{4};  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcE;
        amrex::Parser parser;

        for (int i{0}; i < 3; ++i)
        {
            parser.define(analyticalFuncE[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcE[i] = parser.compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham{std::make_shared<FDDeRhamComplex>(params)};

        DeRhamField<Grid::primal, Space::edge> E(deRham);
        deRham->projection(funcE, 0.0, E);

        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np);

            updateEFieldParallelFor<vDim, degX, degY, degZ>(pti, E, infra);
        }
    }

    TEST_F(EvaluateEFieldTest, DoubleParticleSeparate) {
        const int numParticles{2};
        // Particles in different cells to check that they don't interfere with each other
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{{{
            AMREX_D_DECL(0, 0, 0)},
            {AMREX_D_DECL(
            infra.geom.ProbLo(0) + 5.5*infra.dx[0],
            infra.geom.ProbLo(1) + 5.5*infra.dx[1],
            infra.geom.ProbLo(2) + 5.5*infra.dx[2])}}};
        amrex::Array<amrex::Real, numParticles> weights{1, 1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);
        
        particleGroup[0]->Redistribute();  // assign particles to the tile they are in

        // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
        // Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncE{"1.0", "1.0", "1.0"};

        const int nVar{4};  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcE;
        amrex::Parser parser;

        for (int i{0}; i < 3; ++i)
        {
            parser.define(analyticalFuncE[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcE[i] = parser.compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham{std::make_shared<FDDeRhamComplex>(params)};

        DeRhamField<Grid::primal, Space::edge> E(deRham);
        deRham->projection(funcE, 0.0, E);

        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np);

            updateEFieldParallelFor<vDim, degX, degY, degZ>(pti, E, infra);
        }
    }

    TEST_F(EvaluateEFieldTest, DoubleParticleOverlap) {
        const int numParticles{2};
        // Particles in different cells to check that they don't interfere with each other
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{{{
            AMREX_D_DECL(0, 0, 0)},
            {AMREX_D_DECL(
            infra.geom.ProbLo(0) + 0.5*infra.dx[0],
            infra.geom.ProbLo(1) + 0.5*infra.dx[1],
            infra.geom.ProbLo(2) + 0.5*infra.dx[2])}}};
        amrex::Array<amrex::Real, numParticles> weights{1, 1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);
        
        particleGroup[0]->Redistribute();  // assign particles to the tile they are in

        // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
        // Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncE{"1.0", "1.0", "1.0"};

        const int nVar{4};  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcE;
        amrex::Parser parser;

        for (int i{0}; i < 3; ++i)
        {
            parser.define(analyticalFuncE[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcE[i] = parser.compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham{std::make_shared<FDDeRhamComplex>(params)};

        DeRhamField<Grid::primal, Space::edge> E(deRham);
        deRham->projection(funcE, 0.0, E);

        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[0], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np);

            updateEFieldParallelFor<vDim, degX, degY, degZ>(pti, E, infra);
        }
    }
}
