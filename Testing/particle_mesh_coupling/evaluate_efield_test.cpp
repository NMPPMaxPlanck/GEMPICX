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

/*
#include <GEMPIC_amrex_init.H>
#include <GEMPIC_parameters.H>
#include <GEMPIC_computational_domain.H>
#include <GEMPIC_gempic_norm.H>
#include <gtest/gtest.h>
#include <algorithm> // std::all_of, for add_particles function
#include <stdexcept>
*/

using namespace Particles;
using namespace GEMPIC_FDDeRhamComplex;
using namespace GEMPIC_Fields;

//Basics first
namespace {
    // Test fixture
    class EvaluateEFieldTest : public testing::Test {
        protected:

        static const int degx{1};
        static const int degy{1};
        static const int degz{1};

        static const int numspec = 1;
        // Number of velocity dimensions. Really ought to be 0, but then GEMPIC_TestUtils::addSingleParticles doesn't work.
        static const int vdim = 3;
        static const int ndata = 1;
        static const int spec = 0;
        const int Nghost = GEMPIC_TestUtils::initNGhost(degx, degy, degz);
        Parameters params;

        double charge = 1;
        double mass = 1;

        computational_domain infra;
        amrex::GpuArray<std::unique_ptr<particle_groups<vdim>>, numspec> part_gr;

        // virtual void SetUp() will be called before each test is run.
        void SetUp() override {
            /* Initialize the infrastructure */
            const amrex::RealBox realBox({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                            {AMREX_D_DECL(10.0, 10.0, 10.0)});
            const amrex::IntVect nCell = {AMREX_D_DECL(10, 10, 10)};
            const amrex::IntVect maxGridSize = {AMREX_D_DECL(10, 10, 10)};
            const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic = {1, 1, 1};
            const int hodgeDegree = 2;


            // This class does the same as GEMPIC_parameters.H and needs to be urgently redesigned.
            // {1, 1, 1} represents periodicity, has different types than Params and gempic_parameters.
            infra.initialize_computational_domain(nCell, maxGridSize, {1, 1, 1}, realBox);

            params = Parameters(realBox, nCell, maxGridSize, isPeriodic, hodgeDegree);
            
            // particles
            for (int spec = 0; spec < numspec; spec++)
            {
                part_gr[spec] =
                    std::make_unique<particle_groups<vdim>>(charge, mass, infra);
            }

        }
    };

    TEST_F(EvaluateEFieldTest, NullTest) {
        // Adding particle to one cell
        const int numparticles{1};
        amrex::Array<amrex::Array<amrex::Real, GEMPIC_SPACEDIM>, numparticles> positions{*infra.geom.ProbLo()};

        amrex::Array<amrex::Real, numparticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vdim, numspec, numparticles>(part_gr, infra, weights, positions);

        // (default) charge correctly transferred from addSinglePparticles
        EXPECT_EQ(1, part_gr[0]->getCharge()); 

        // Parse analytical fields and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
        // Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncE = {"0.0", "0.0", "0.0"};

        const int nVar = 4;  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcE;
        amrex::Parser parser;

        for (int i = 0; i < 3; ++i)
        {
            parser.define(analyticalFuncE[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcE[i] = parser.compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(params);

        DeRhamField<Grid::primal, Space::edge> E(deRham);
        deRham->projection(funcE, 0.0, E);

        part_gr[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle. Hopefully.
        bool particle_loop_run=false;

        for (amrex::ParIter<0, 0, vdim + ndata, 0> pti(*part_gr[spec], 0); pti.isValid(); ++pti)
        {
            particle_loop_run=true;

            const long np = pti.numParticles();

            EXPECT_EQ(1, np); // Only one particle added by addSingleParticles

            const auto& particles = pti.GetArrayOfStructs();
            const auto partData = particles().data();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
            for (int cc = 0; cc < vdim; cc++) eA[cc] = (E.data[cc])[pti].array();

            splines_at_particles<degx, degy, degz> spline;
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
            for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                position[d] = partData[0].pos(d);
            spline.init_particles(position, infra.plo, infra.dxi);

            amrex::GpuArray<amrex::Real, vdim> efield =
                evaluate_efield<vdim, degx, degy, degz>(spline, eA);

            EXPECT_EQ(efield[0], 0);
            EXPECT_EQ(efield[1], 0);
            EXPECT_EQ(efield[2], 0);
        }
        ASSERT_TRUE(particle_loop_run);
    }

    TEST_F(EvaluateEFieldTest, SingleParticleNode) {
        // Adding particle to one cell
        const int numparticles{1};

        amrex::Array<amrex::Real, numparticles> weights{1};
        // Particle at position (0,0,0) in box (0,0,0)
        amrex::Array<amrex::Array<amrex::Real, GEMPIC_SPACEDIM>, numparticles> positions{*infra.geom.ProbLo()};

        GEMPIC_TestUtils::addSingleParticles<vdim, numspec, numparticles>(part_gr, infra, weights, positions);
        part_gr[0]->Redistribute();  // assign particles to the tile they are in

        // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
        // Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncE = {"1.0", "1.0", "1.0"};

        const int nVar = 4;  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcE;
        amrex::Parser parser;

        for (int i = 0; i < 3; ++i)
        {
            parser.define(analyticalFuncE[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcE[i] = parser.compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(params);

        DeRhamField<Grid::primal, Space::edge> E(deRham);
        deRham->projection(funcE, 0.0, E);

        for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*part_gr[0], 0); pti.isValid(); ++pti)
        {
            const long np = pti.numParticles();
            EXPECT_EQ(1, np); // Only one particle added

            const auto& particles = pti.GetArrayOfStructs();
            const auto partData = particles().data();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
            for (int cc = 0; cc < vdim; cc++) eA[cc] = (E.data[cc])[pti].array();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
            {
                splines_at_particles<degx, degy, degz> spline;
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                {
                    position[d] = partData[0].pos(d);
                    EXPECT_EQ(position[d], 0.0);
                }
                spline.init_particles(position, infra.plo, infra.dxi);

                EXPECT_EQ(spline.spline_cell[0][0], 1.0);
                EXPECT_EQ(spline.spline_cell[1][0], 1.0);
                EXPECT_EQ(spline.spline_cell[2][0], 1.0);

                EXPECT_EQ(spline.spline_node[0][0], 1.0);
                EXPECT_EQ(spline.spline_node[1][0], 1.0);
                EXPECT_EQ(spline.spline_node[2][0], 1.0);

                amrex::GpuArray<amrex::Real, vdim> efield =
                    evaluate_efield<vdim, degx, degy, degz>(spline, eA);
                    
                EXPECT_EQ(efield[0], 1.0);
                EXPECT_EQ(efield[1], 1.0);
                EXPECT_EQ(efield[2], 1.0);
            });
        }
    }

    TEST_F(EvaluateEFieldTest, SingleParticleMiddle) {
        // Adding particle to one cell
        const int numparticles{1};

        amrex::Array<amrex::Real, numparticles> weights{1};
        // Add particle in the middle of final cell to check periodic boundary conditions
        amrex::Array<amrex::Array<amrex::Real, GEMPIC_SPACEDIM>, numparticles> positions{AMREX_D_DECL(infra.geom.ProbHi()[0] - 1.5*infra.dx[0],
                      infra.geom.ProbHi()[1] - 1.5*infra.dx[1],
                      infra.geom.ProbHi()[2] - 1.5*infra.dx[2])};

        GEMPIC_TestUtils::addSingleParticles<vdim, numspec, numparticles>(part_gr, infra, weights, positions);
        part_gr[0]->Redistribute();  // assign particles to the tile they are in

        // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
        // Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncE = {"1.0", "1.0", "1.0"};

        const int nVar = 4;  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcE;
        amrex::Parser parser;

        for (int i = 0; i < 3; ++i)
        {
            parser.define(analyticalFuncE[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcE[i] = parser.compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(params);

        DeRhamField<Grid::primal, Space::edge> E(deRham);
        deRham->projection(funcE, 0.0, E);

        for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*part_gr[0], 0); pti.isValid(); ++pti)
        {
            const long np = pti.numParticles();
            EXPECT_EQ(1, np); // Only one particle added

            const auto& particles = pti.GetArrayOfStructs();
            const auto partData = particles().data();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
            for (int cc = 0; cc < vdim; cc++) eA[cc] = (E.data[cc])[pti].array();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
            {
                splines_at_particles<degx, degy, degz> spline;
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                {
                    position[d] = partData[0].pos(d);
                    EXPECT_EQ(position[d], 8.5);
                }
                spline.init_particles(position, infra.plo, infra.dxi);

                EXPECT_EQ(spline.spline_cell[0][0], 1.0);
                EXPECT_EQ(spline.spline_cell[1][0], 1.0);
                EXPECT_EQ(spline.spline_cell[2][0], 1.0);

                EXPECT_EQ(spline.spline_node[0][0], 0.5);
                EXPECT_EQ(spline.spline_node[1][0], 0.5);
                EXPECT_EQ(spline.spline_node[2][0], 0.5);

                amrex::GpuArray<amrex::Real, vdim> efield =
                    evaluate_efield<vdim, degx, degy, degz>(spline, eA);
                    
                EXPECT_EQ(efield[0], 1.0);
                EXPECT_EQ(efield[1], 1.0);
                EXPECT_EQ(efield[2], 1.0);
            });
        }
    }

    TEST_F(EvaluateEFieldTest, SingleParticleUnevenNodeSplit) {
        // Adding particle to one cell
        const int numparticles{1};

        amrex::Array<amrex::Real, numparticles> weights{1};
        // Add particle in the middle of final cell to check periodic boundary conditions
        amrex::Array<amrex::Array<amrex::Real, GEMPIC_SPACEDIM>, numparticles> positions{AMREX_D_DECL(infra.geom.ProbHi()[0] - 1.25*infra.dx[0],
                      infra.geom.ProbHi()[1] - 1.25*infra.dx[1],
                      infra.geom.ProbHi()[2] - 1.25*infra.dx[2])};

        GEMPIC_TestUtils::addSingleParticles<vdim, numspec, numparticles>(part_gr, infra, weights, positions);
        part_gr[0]->Redistribute();  // assign particles to the tile they are in

        // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
        // Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncE = {"1.0", "1.0", "1.0"};

        const int nVar = 4;  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcE;
        amrex::Parser parser;

        for (int i = 0; i < 3; ++i)
        {
            parser.define(analyticalFuncE[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcE[i] = parser.compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(params);

        DeRhamField<Grid::primal, Space::edge> E(deRham);
        deRham->projection(funcE, 0.0, E);

        for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*part_gr[0], 0); pti.isValid(); ++pti)
        {
            const long np = pti.numParticles();
            EXPECT_EQ(1, np); // Only one particle added

            const auto& particles = pti.GetArrayOfStructs();
            const auto partData = particles().data();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
            for (int cc = 0; cc < vdim; cc++) eA[cc] = (E.data[cc])[pti].array();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
            {
                splines_at_particles<degx, degy, degz> spline;
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                {
                    position[d] = partData[0].pos(d);
                    EXPECT_EQ(position[d], 8.75);
                }
                spline.init_particles(position, infra.plo, infra.dxi);

                EXPECT_EQ(spline.spline_cell[0][0], 1.0);
                EXPECT_EQ(spline.spline_cell[1][0], 1.0);
                EXPECT_EQ(spline.spline_cell[2][0], 1.0);

                EXPECT_EQ(spline.spline_node[0][0], 0.25);
                EXPECT_EQ(spline.spline_node[1][0], 0.25);
                EXPECT_EQ(spline.spline_node[2][0], 0.25);

                amrex::GpuArray<amrex::Real, vdim> efield =
                    evaluate_efield<vdim, degx, degy, degz>(spline, eA);
                    
                EXPECT_EQ(efield[0], 1.0);
                EXPECT_EQ(efield[1], 1.0);
                EXPECT_EQ(efield[2], 1.0);
            });
        }
    }

    TEST_F(EvaluateEFieldTest, DoubleParticleSeparate) {
        const int numparticles{2};
        // Particles in different cells to check that they don't interfere with each other
        amrex::Array<amrex::Array<amrex::Real, GEMPIC_SPACEDIM>, numparticles> positions{{{
            AMREX_D_DECL(0, 0, 0)},
            {AMREX_D_DECL(
            infra.geom.ProbLo(0) + 5.5*infra.dx[0],
            infra.geom.ProbLo(1) + 5.5*infra.dx[1],
            infra.geom.ProbLo(2) + 5.5*infra.dx[2])}}};

        amrex::Array<amrex::Real, numparticles> weights{1, 1};

        GEMPIC_TestUtils::addSingleParticles<vdim, numspec, numparticles>(part_gr, infra, weights, positions);
        
        part_gr[0]->Redistribute();  // assign particles to the tile they are in

        // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
        // Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncE = {"1.0", "1.0", "1.0"};

        const int nVar = 4;  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcE;
        amrex::Parser parser;

        for (int i = 0; i < 3; ++i)
        {
            parser.define(analyticalFuncE[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcE[i] = parser.compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(params);

        DeRhamField<Grid::primal, Space::edge> E(deRham);
        deRham->projection(funcE, 0.0, E);

        // Particle iteration ... over two distant particles.

        for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*part_gr[0], 0); pti.isValid(); ++pti)
        {
            const long np = pti.numParticles();
            EXPECT_EQ(2, np); // Only one particle added

            const auto& particles = pti.GetArrayOfStructs();
            const auto partData = particles().data();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
            for (int cc = 0; cc < vdim; cc++) eA[cc] = (E.data[cc])[pti].array();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
            {
                splines_at_particles<degx, degy, degz> spline;
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                {
                    position[d] = partData[0].pos(d);
                    EXPECT_TRUE((position[d] == 5.5) || (position[d] == 0.0));
                }
                spline.init_particles(position, infra.plo, infra.dxi);

                amrex::GpuArray<amrex::Real, vdim> efield =
                    evaluate_efield<vdim, degx, degy, degz>(spline, eA);
                    
                EXPECT_EQ(efield[0], 1.0);
                EXPECT_EQ(efield[1], 1.0);
                EXPECT_EQ(efield[2], 1.0);
            });
        }
    }

    TEST_F(EvaluateEFieldTest, DoubleParticleOverlap) {
        const int numparticles{2};
        // Particles in different cells to check that they don't interfere with each other
        amrex::Array<amrex::Array<amrex::Real, GEMPIC_SPACEDIM>, numparticles> positions{{{
            AMREX_D_DECL(0, 0, 0)},
            {AMREX_D_DECL(
            infra.geom.ProbLo(0) + 0.5*infra.dx[0],
            infra.geom.ProbLo(1) + 0.5*infra.dx[1],
            infra.geom.ProbLo(2) + 0.5*infra.dx[2])}}};

        amrex::Array<amrex::Real, numparticles> weights{1, 1};

        GEMPIC_TestUtils::addSingleParticles<vdim, numspec, numparticles>(part_gr, infra, weights, positions);
        
        part_gr[0]->Redistribute();  // assign particles to the tile they are in

        // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
        // Ey, Ez
        const amrex::Array<std::string, 3> analyticalFuncE = {"1.0", "1.0", "1.0"};

        const int nVar = 4;  // x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcE;
        amrex::Parser parser;

        for (int i = 0; i < 3; ++i)
        {
            parser.define(analyticalFuncE[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcE[i] = parser.compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham = std::make_shared<FDDeRhamComplex>(params);

        DeRhamField<Grid::primal, Space::edge> E(deRham);
        deRham->projection(funcE, 0.0, E);

        // Particle iteration ... over two distant particles.

        for (amrex::ParIter<0, 0, vdim + 1, 0> pti(*part_gr[0], 0); pti.isValid(); ++pti)
        {
            const long np = pti.numParticles();
            EXPECT_EQ(2, np); // Only one particle added

            const auto& particles = pti.GetArrayOfStructs();
            const auto partData = particles().data();

            amrex::GpuArray<amrex::Array4<amrex::Real>, vdim> eA;
            for (int cc = 0; cc < vdim; cc++) eA[cc] = (E.data[cc])[pti].array();

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
            {
                splines_at_particles<degx, degy, degz> spline;
                amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
                for (unsigned int d = 0; d < GEMPIC_SPACEDIM; ++d)
                {
                    position[d] = partData[0].pos(d);
                    EXPECT_TRUE((position[d] == 0.5) || (position[d] == 0.0));
                }
                spline.init_particles(position, infra.plo, infra.dxi);

                amrex::GpuArray<amrex::Real, vdim> efield =
                    evaluate_efield<vdim, degx, degy, degz>(spline, eA);
                    
                EXPECT_EQ(efield[0], 1.0);
                EXPECT_EQ(efield[1], 1.0);
                EXPECT_EQ(efield[2], 1.0);
            });
        }
    }
}