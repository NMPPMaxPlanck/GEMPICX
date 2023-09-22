#include <AMReX.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_particle_groups.H>
#include <GEMPIC_particle_mesh_coupling.H>
#include <GEMPIC_hs_zigzag.H>
#include <GEMPIC_Spline_Class.H>
#include "gtest/gtest.h"
#include "test_utils/GEMPIC_test_utils.H"
#include "gmock/gmock.h"

#include "GEMPIC_Splitting.H"

using namespace Particles;
using namespace GEMPIC_FDDeRhamComplex;
using namespace GEMPIC_Fields;
using namespace Time_Loop;

using ::testing::Mock;
using ::testing::Exactly;
using ::testing::_;

namespace Gempic {

namespace Particles {

    template<>
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, 4> Gempic::Particles::push_v_efield<4>(
        amrex::GpuArray<amrex::Real, 4> vel,
        amrex::Real dt,
        amrex::Real chargemass,  // charge/mass
        amrex::GpuArray<amrex::Real, 4> &Ep)
    {
        amrex::GpuArray<amrex::Real, 4> newPos;
        for (int i = 0; i < 4; i++) {
            newPos[i] = 1;
        }
        return newPos;
    }
}

}

namespace {
    template<int vDim, int numspec, int degX, int degY, int degZ, int degmw, int ndata>
    class MockHSZigZagC2 : public Time_Loop::HSZigZagC2<vDim, numspec, degX, degY, degZ, degmw, ndata> {
        public:
    };

    template<int degX, int degY, int degZ>
    class MockSpline : public Spline::SplineWithPrimitive<degX, degY, degZ> {
        public:
        MockSpline(amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const &position,
                   amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const &plo,
                   amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> const &dxInverse) :
                   Spline::SplineWithPrimitive<degX, degY, degZ>(position, plo, dxInverse) {}

        template<int vDim, int form>
        AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, vDim> evalSplineField (const amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> fieldArray) const
        {
            amrex::GpuArray<amrex::Real, vDim> fields;
            for (int comp = 0; comp < vDim; comp++)
            {
                fields[comp] = 0.;
            }

            return fields;
        }
    };

    class OperatorHamiltonTest : public testing::Test {
        protected:

        static const int degX{1};
        static const int degY{1};
        static const int degZ{1};
        static const int degmw{2};

        static const int numSpec{1};
        static const int vDim{3};
        static const int spec{0};
        static const int nData{1};
        const int Nghost{GEMPIC_TestUtils::initNGhost(degX, degY, degZ)};
        Parameters params;

        double charge{1};
        double mass{1};

        computational_domain infra;
        amrex::GpuArray<std::unique_ptr<particle_groups<vDim>>, numSpec> particleGroup;

        // virtual void SetUp() will be called before each test is run.
        void SetUp() override {
            /* Initialize the infrastructure */
            const amrex::RealBox realBox({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(10.0, 10.0, 10.0)});
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
}

namespace Gempic::Particles{
    // You cannot do partial template specialization for functions, so here is an explicit specialization for a special case
    template <>
    AMREX_GPU_HOST_DEVICE void accumulate_J_integrate_B<MockSpline<1, 1, 1>,4,0>(
        MockSpline<1, 1, 1> &spline,
        amrex::Real weight,
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx,
        amrex::GpuArray<amrex::Array4<amrex::Real>, int(4 / 2.5) * 2 + 1> const &bArray,
        amrex::GpuArray<amrex::Array4<amrex::Real>, 4> const &jArray,
        amrex::GpuArray<amrex::Real, 2> &fields)
    {
        
    }
}

namespace {
    TEST_F(OperatorHamiltonTest, ApplyHEParticleTest) {
        // Adding particle to one cell
        const int numParticles{1};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{*infra.geom.ProbLo()};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        // (default) charge correctly transferred from addSingleParticles
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

        DeRhamField<Grid::primal, Space::edge> E(deRham, funcE);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle. Hopefully.

        bool particleLoopRun{false};
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[spec], 0); pti.isValid(); ++pti)
        {
            particleLoopRun = true;

            MockSpline<degX, degY, degZ> mockSpline({1., 1., 1.}, {1., 1., 1.}, {1., 1., 1., 1.});
            // EXPECT_CALL(mockHSZigZagC2, push_v_efield).Times(Exactly(1));

            const long np{pti.numParticles()};
            EXPECT_EQ(1, np); // Only one particle added by addSingleParticles

            amrex::Particle<0, 0>* AMREX_RESTRICT particles =
                &(pti.GetArrayOfStructs()[0]);

            amrex::GpuArray<amrex::Array4<amrex::Real>, 4> eArray;
            for (int cc{0}; cc < vDim; cc++) eArray[cc] = (E.data[cc])[pti].array();

            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
            for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
                position[d] = particles[0].pos(d);

            auto particle_attributes = &pti.GetStructOfArrays();
            amrex::ParticleReal* const AMREX_RESTRICT velx =
                particle_attributes->GetRealData(0).data();
            amrex::ParticleReal* const AMREX_RESTRICT vely =
                particle_attributes->GetRealData(1).data();
            amrex::ParticleReal* const AMREX_RESTRICT velz =
                particle_attributes->GetRealData(2).data();
            amrex::GpuArray<amrex::Real, 4> vel{0, 0, 0, 0};
            
            OperatorHamilton<4, degX, degY, degZ, degmw> operatorHamilton;

            operatorHamilton.template apply_H_e_particle<MockSpline<degX, degY, degZ>>(
                eArray,
                mockSpline,
                vel,
                velx,
                vely,
                velz,
                1,
                1,
                0
            );

            amrex::GpuArray<amrex::Real, 3> efield({1, 1, 1});

            EXPECT_EQ(1, velx[0]);
            EXPECT_EQ(1, vely[0]);
            EXPECT_EQ(1, velz[0]);
        }
        ASSERT_TRUE(particleLoopRun);
    }

    TEST_F(OperatorHamiltonTest, ApplyHpiTest) {
        // Adding particle to one cell
        const int numParticles{1};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{*infra.geom.ProbLo()};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        // (default) charge correctly transferred from addSingleParticles
        EXPECT_EQ(1, particleGroup[0]->getCharge());

        // Initialize the De Rham Complex
        auto deRham{std::make_shared<FDDeRhamComplex>(params)};

        DeRhamField<Grid::dual, Space::face> J(deRham);
        DeRhamField<Grid::primal, Space::face> B(deRham);

        const amrex::Array<std::string, 3> analyticalFuncB = {"0.0", 
                                                              "0.0",
                                                              "0.0"};

        // Project B to a primal two form
        const int nVar = GEMPIC_SPACEDIM + 1; //x, y, z, t
        amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB; 
        amrex::Parser parser;

        for (int i=0; i<3; ++i)
        {
            parser.define(analyticalFuncB[i]);
            parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
            funcB[i] = parser.compile<nVar>();
        }

        deRham -> projection(funcB, 0.0, B);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in
        // Particle iteration ... over one particle. Hopefully.

        bool particleLoopRun{false};
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[spec], 0); pti.isValid(); ++pti)
        {
            particleLoopRun = true;

            MockSpline<degX, degY, degZ> mockSpline({1., 1., 1.}, {1., 1., 1.}, {1., 1., 1., 1.});
            // EXPECT_CALL(mockHSZigZagC2, push_v_efield).Times(Exactly(1));

            const long np{pti.numParticles()};
            EXPECT_EQ(1, np); // Only one particle added by addSingleParticles

            amrex::Particle<0, 0>* AMREX_RESTRICT particles =
                &(pti.GetArrayOfStructs()[0]);

            amrex::GpuArray<amrex::Array4<amrex::Real>, 4> jArray;
            for (int cc{0}; cc < vDim; cc++) jArray[cc] = (J.data[cc])[pti].array();


            amrex::GpuArray<amrex::Array4<amrex::Real>, 3> bArray;
            for (int cc{0}; cc < vDim; cc++) bArray[cc] = (B.data[cc])[pti].array();

            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
            for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
                position[d] = particles[0].pos(d);

            auto particle_attributes = &pti.GetStructOfArrays();
            amrex::ParticleReal* const AMREX_RESTRICT velx =
                particle_attributes->GetRealData(0).data();
            amrex::ParticleReal* const AMREX_RESTRICT vely =
                particle_attributes->GetRealData(1).data();
            amrex::ParticleReal* const AMREX_RESTRICT velz =
                particle_attributes->GetRealData(2).data();
            amrex::GpuArray<amrex::Real, 4> vel{0, 0, 0, 0};

            amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

            OperatorHamilton<4, degX, degY, degZ, degmw> operatorHamilton;

            operatorHamilton.template apply_H_p_i<0, MockSpline<degX, degY, degZ>>(
                position,
                vel,
                infra,
                mockSpline,
                bfields,
                infra.dx,
                jArray,
                bArray,
                1,
                1,
                1);

            amrex::GpuArray<amrex::Real, 3> efield({1, 1, 1});

            EXPECT_EQ(0, velx[0]);
            EXPECT_EQ(0, vely[0]);
            EXPECT_EQ(0, velz[0]);
        }
        ASSERT_TRUE(particleLoopRun);
    }
}