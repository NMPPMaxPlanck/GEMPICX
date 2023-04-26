#include <AMReX.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_particle_mesh_coupling_C2.H>
#include "gtest/gtest.h"
#include "test_utils/GEMPIC_test_utils.H"

using namespace Particles;
using namespace GEMPIC_FDDeRhamComplex;
using namespace GEMPIC_Fields;

namespace {
    // When using amrex::ParallelFor you have to create a standalone helper function that does the execution on GPU and call that function from the unit test because of how GTest creates tests within a TEST_F fixture.
    template <int vDim, int degX, int degY, int degZ, int degP, int degP1, int degP2, int pDim, int pLength>
    void accumulateJUpdateVC2ParallelFor(amrex::ParIter<0, 0, vDim + 1, 0>& pti,
                                 DeRhamField<Grid::primal, Space::face>& B,
                                 DeRhamField<Grid::dual, Space::face>& J,
                                 computational_domain& infra,
                                 amrex::Real weight,
                                 amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx,
                                 amrex::GpuArray<amrex::Real, 2>& bfields) {

        const long np{pti.numParticles()};

        const auto& particles{pti.GetArrayOfStructs()};
        const auto partData{particles().data()};

        amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> jA;
        for (int cc = 0; cc < vDim; cc++) jA[cc] = (J.data[cc])[pti].array();

        amrex::AsyncArray<amrex::GpuArray<amrex::Real, 2>> bFieldsPtr(1);
        amrex::GpuArray<amrex::Real, 2>* bFields = bFieldsPtr.data();

        amrex::GpuArray<amrex::Array4<amrex::Real>, int(vDim / 2.5) * 2 + 1> bA;
        for (int cc = 0; cc < (int(vDim / 2.5) * 2 + 1); cc++) bA[cc] = (B.data[cc])[pti].array();

        amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE(long pp)
        {


            amrex::GpuArray<amrex::Real, std::max(degX, std::max(degY, degZ)) + 4> primitive;
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
            for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
                position[d] = partData[0].pos(d);

            amrex::Real x_new = 0;

            splines_at_particles<degX, degY, degZ> spline;
            spline1d_at_particles<degP1> spline_new;
            spline1d_at_particles<degP2> spline_old;

            spline_new.init_position(x_new, infra.plo[0], infra.dxi[0]);
            spline_old.init_position(position[0], infra.plo[0], infra.dxi[0]);
            spline.init_particles(position, infra.plo, infra.dxi);

            accumulate_j_update_v_C2<splines_at_particles<1, 1, 1>, vDim, degP, degP1, degP2, pDim, pLength>(spline, spline_new, spline_old, weight, dx, bA, jA, bFields[0], primitive);
        });

        bfields = bFields[0];
    }

    class AccumulateJUpdateVC2Test : public testing::Test {
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

        amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> jA;
        amrex::GpuArray<amrex::Array4<amrex::Real>, int(vDim / 2.5) * 2 + 1> bA;

        computational_domain infra;
        amrex::GpuArray<std::unique_ptr<particle_groups<vDim>>, numSpec> particleGroup;

        static const int degP{1};
        static const int degP1{1};
        static const int degP2{1};

        static const int numParticles{1};

        static const int pDim{1};
        static const int pLength{5};
        
        amrex::Real weight = 1.0;
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx = {1.0, 1.0, 1.0};

        splines_at_particles<degX, degY, degZ> spline;
        spline1d_at_particles<degP1> spline_new;
        spline1d_at_particles<degP2> spline_old;

        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};
        amrex::GpuArray<amrex::Real, std::max(degX, std::max(degY, degZ)) + 4> primitive;

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
            for (int species{0}; species < numSpec; species++)
            {
                particleGroup[species] =
                    std::make_unique<particle_groups<vDim>>(charge, mass, infra);
            }
        }
    };

    TEST_F(AccumulateJUpdateVC2Test, NullTest) {
        // Adding particle to one cell
        const int numParticles{1};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{*infra.geom.ProbLo()};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        // (default) charge correctly transferred from addSinglePparticles
        EXPECT_EQ(1, particleGroup[0]->getCharge()); 

        const amrex::Array<std::string, 3> analyticalFuncB = {"0.0", 
                                                              "0.0",
                                                              "0.0"};

        const amrex::Array<std::string, 3> analyticalFuncJ = {"0.0", 
                                                              "0.0",
                                                              "0.0"};

        const int nVar{4}; //x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcB;
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcJ; 
        amrex::Parser parser;

        for (int i{0}; i<3; ++i)
        {
            parser.define(analyticalFuncB[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcB[i] = parser.compile<4>();
        }

        for (int i{0}; i<3; ++i)
        {
            parser.define(analyticalFuncJ[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcJ[i] = parser.compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham{std::make_shared<FDDeRhamComplex>(params)};

        DeRhamField<Grid::primal, Space::face> B(deRham);
        deRham -> projection(funcB, 0.0, B);

        DeRhamField<Grid::dual, Space::face> J(deRham);
        deRham -> projection(funcJ, 0.0, J);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle. Hopefully.

        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[spec], 0); pti.isValid(); ++pti)
        {

            const long np{pti.numParticles()};
            EXPECT_EQ(1, np); // Only one particle added by addSingleParticles

            amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

            accumulateJUpdateVC2ParallelFor<vDim, degX, degY, degZ, degP, degP1, degP2, pDim, pLength>(pti, B, J, infra, weight, dx, bfields);

            EXPECT_EQ(bfields[0], 0);
            EXPECT_EQ(bfields[1], 0);
        }
    }

    TEST_F(AccumulateJUpdateVC2Test, SingleParticleMiddle) {
        // Adding particle to one cell
        const int numParticles{1};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{AMREX_D_DECL(infra.geom.ProbHi()[0] - 5.5*infra.dx[0],
                      infra.geom.ProbHi()[1] - 5.5*infra.dx[1],
                      infra.geom.ProbHi()[2] - 5.5*infra.dx[2])};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        // (default) charge correctly transferred from addSingleParticles
        EXPECT_EQ(1, particleGroup[0]->getCharge()); 

        const amrex::Array<std::string, 3> analyticalFuncB = {"1.0", 
                                                              "1.0",
                                                              "1.0"};

        const amrex::Array<std::string, 3> analyticalFuncJ = {"1.0", 
                                                              "1.0",
                                                              "1.0"};

        const int nVar{4}; //x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcB;
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcJ; 
        amrex::Parser parser;

        for (int i{0}; i<3; ++i)
        {
            parser.define(analyticalFuncB[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcB[i] = parser.compile<4>();
        }

        for (int i{0}; i<3; ++i)
        {
            parser.define(analyticalFuncJ[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcJ[i] = parser.compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham{std::make_shared<FDDeRhamComplex>(params)};

        DeRhamField<Grid::primal, Space::face> B(deRham);
        deRham -> projection(funcB, 0.0, B);

        DeRhamField<Grid::dual, Space::face> J(deRham);
        deRham -> projection(funcJ, 0.0, J);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle. Hopefully.

        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[spec], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(1, np); // Only one particle added by addSingleParticles

            amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

            accumulateJUpdateVC2ParallelFor<vDim, degX, degY, degZ, degP, degP1, degP2, pDim, pLength>(pti, B, J, infra, weight, dx, bfields);

            EXPECT_EQ(bfields[0], -4.5);
            EXPECT_EQ(bfields[1], -4.5);
        }
    }


    TEST_F(AccumulateJUpdateVC2Test, SingleParticleUnevenNodeSplit) {
        // Adding particle to one cell
        const int numParticles{1};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{AMREX_D_DECL(infra.geom.ProbHi()[0] - 5.25*infra.dx[0],
                      infra.geom.ProbHi()[1] - 5.25*infra.dx[1],
                      infra.geom.ProbHi()[2] - 5.25*infra.dx[2])};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        // (default) charge correctly transferred from addSingleParticles
        EXPECT_EQ(1, particleGroup[0]->getCharge()); 

        const amrex::Array<std::string, 3> analyticalFuncB = {"1.0", 
                                                              "1.0",
                                                              "1.0"};

        const amrex::Array<std::string, 3> analyticalFuncJ = {"1.0", 
                                                              "1.0",
                                                              "1.0"};

        const int nVar{4}; //x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcB;
        amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcJ; 
        amrex::Parser parser;

        for (int i{0}; i<3; ++i)
        {
            parser.define(analyticalFuncB[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcB[i] = parser.compile<4>();
        }

        for (int i{0}; i<3; ++i)
        {
            parser.define(analyticalFuncJ[i]);
            parser.registerVariables({"x", "y", "z", "t"});
            funcJ[i] = parser.compile<4>();
        }

        // Initialize the De Rham Complex
        auto deRham{std::make_shared<FDDeRhamComplex>(params)};

        DeRhamField<Grid::primal, Space::face> B(deRham);
        deRham -> projection(funcB, 0.0, B);

        DeRhamField<Grid::dual, Space::face> J(deRham);
        deRham -> projection(funcJ, 0.0, J);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle. Hopefully.

        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[spec], 0); pti.isValid(); ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(1, np); // Only one particle added by addSingleParticles

            amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

            accumulateJUpdateVC2ParallelFor<vDim, degX, degY, degZ, degP, degP1, degP2, pDim, pLength>(pti, B, J, infra, weight, dx, bfields);

            EXPECT_EQ(bfields[0], -4.75);
            EXPECT_EQ(bfields[1], -4.75);
        }
    }

}