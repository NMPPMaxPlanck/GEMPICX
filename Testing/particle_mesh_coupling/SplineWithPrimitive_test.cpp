#include <AMReX.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>
#include <GEMPIC_particle_groups.H>
#include "gtest/gtest.h"
#include "test_utils/GEMPIC_test_utils.H"
#include "GEMPIC_Spline_Class.H"
#include "gmock/gmock.h"

using namespace Particles;
using namespace GEMPIC_FDDeRhamComplex;
using namespace GEMPIC_Fields;
using ::testing::Mock;

namespace {
    template<int degX, int degY, int degZ>
    class MockSpline : public Spline::SplineWithPrimitive<degX, degY, degZ> {
        public:
        MockSpline(amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const position,
                   amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const &plo,
                   amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> const &dxInverse) :
                   Spline::SplineWithPrimitive<degX, degY, degZ>(position, plo, dxInverse) {}

        MOCK_METHOD(amrex::Real, initBSplinesAtPositions, (
            (amrex::Real position),
            (amrex::Real plo),
            (amrex::Real dxInverse)));
    };

    // Test fixture. Sets up clean environment before each test.
    class SplineWithPrimitiveTest : public testing::Test {
        protected:

        // Degree of splines in each direction
        static const int degX{1};
        static const int degY{1};
        static const int degZ{1};
        static const int vDim{3};
        static const int numSpec{1};
        static const int spec{0};
        const int Nghost{GEMPIC_TestUtils::initNGhost(degX, degY, degZ)};
        Parameters params;

        double charge{1};
        double mass{1};

        computational_domain infra;
        amrex::GpuArray<std::unique_ptr<particle_groups<vDim>>, numSpec> particleGroup;

        void SetUp() override {
            /* Initialize the infrastructure */
            const amrex::RealBox realBox({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                            {AMREX_D_DECL(10.0, 10.0, 10.0)});
            const amrex::IntVect nCell{AMREX_D_DECL(10, 10, 10)};
            const amrex::IntVect maxGridSize{AMREX_D_DECL(10, 10, 10)};
            const amrex::Array<int, GEMPIC_SPACEDIM> isPeriodic{1, 1, 1};
            const int hodgeDegree{2};

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

    TEST_F(SplineWithPrimitiveTest, SplineConstructorTest) {
        // Adding particle to one cell
        const int numParticles{1};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{*infra.geom.ProbLo()};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in

        bool particleLoopRun{false};
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[spec], 0); pti.isValid(); ++pti)
        {
            particleLoopRun = true;

            const auto& particles{pti.GetArrayOfStructs()};
            const auto partData{particles().data()};

            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
            for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
                position[d] = partData[0].pos(d);
            Spline::SplineWithPrimitive<degX, degY, degZ> spline(position, infra.plo, infra.dxi);

            EXPECT_EQ(1., spline.splineCell[0][0]);
            EXPECT_EQ(1., spline.splineCell[1][0]);
            EXPECT_EQ(1., spline.splineCell[2][0]);

            EXPECT_EQ(1., spline.splineNode[0][0]);
            EXPECT_EQ(0., spline.splineNode[0][1]);
            EXPECT_EQ(1., spline.splineNode[1][0]);
            EXPECT_EQ(0., spline.splineNode[1][1]);
            EXPECT_EQ(1., spline.splineNode[2][0]);
            EXPECT_EQ(0., spline.splineNode[2][1]);

            EXPECT_EQ(1., spline.primitiveNew[0][0]);
            EXPECT_EQ(1., spline.primitiveNew[1][0]);
            EXPECT_EQ(1., spline.primitiveNew[2][0]);
        }
        ASSERT_TRUE(particleLoopRun);
    }
    
    TEST_F(SplineWithPrimitiveTest, SplineUpdate1DPrimitiveTest) {
        // Adding particle to one cell
        const int numParticles{1};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{*infra.geom.ProbLo()};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in

        bool particleLoopRun{false};
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[spec], 0); pti.isValid(); ++pti)
        {
            particleLoopRun = true;

            const auto& particles{pti.GetArrayOfStructs()};
            const auto partData{particles().data()};

            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
            for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
                position[d] = partData[0].pos(d);
            MockSpline<degX, degY, degZ> spline(position, infra.plo, infra.dxi);

            EXPECT_CALL(spline, initBSplinesAtPositions(1, 1, 1)).WillRepeatedly(::testing::Return(1));

            amrex::Real result = spline.initBSplinesAtPositions(1, 1, 1);
            EXPECT_EQ(1., result);

            spline.template update1DPrimitive<0, 1>(1, 1, 1);
            EXPECT_EQ(1., spline.primitiveNew[0][0]);

            spline.template update1DPrimitive<1, 1>(1, 1, 1);
            EXPECT_EQ(1., spline.primitiveNew[1][0]);

            spline.template update1DPrimitive<2, 1>(1, 1, 1);
            EXPECT_EQ(1., spline.primitiveNew[2][0]);
        }
        ASSERT_TRUE(particleLoopRun);
    }
    
    TEST_F(SplineWithPrimitiveTest, SplineComputePrimitiveDifferenceTest) {
        // Adding particle to one cell
        const int numParticles{1};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{*infra.geom.ProbLo()};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in

        bool particleLoopRun{false};
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[spec], 0); pti.isValid(); ++pti)
        {
            particleLoopRun = true;

            const auto& particles{pti.GetArrayOfStructs()};
            const auto partData{particles().data()};

            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
            for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
                position[d] = partData[0].pos(d);
            Spline::SplineWithPrimitive<degX, degY, degZ> spline(position, infra.plo, infra.dxi);

            amrex::Real primitiveDifference = spline.template computePrimitiveDifference<0, 0>({1, 1, 1}, 0);
            EXPECT_EQ(-1, primitiveDifference);

            primitiveDifference = spline.template computePrimitiveDifference<1, 0>({1, 1, 1}, 0);
            EXPECT_EQ(-1, primitiveDifference);

            primitiveDifference = spline.template computePrimitiveDifference<0, 1>({1, 1, 1}, 0);
            EXPECT_EQ(-1, primitiveDifference);

            primitiveDifference = spline.template computePrimitiveDifference<1, 1>({1, 1, 1}, 0);
            EXPECT_EQ(-1, primitiveDifference);

            primitiveDifference = spline.template computePrimitiveDifference<1, 1>({1, 1, 1}, 1);
            EXPECT_EQ(-1, primitiveDifference);

            primitiveDifference = spline.template computePrimitiveDifference<1, 1>({0.1, 0.1, 0.1}, 0);
            EXPECT_EQ(-0.1, primitiveDifference);

        }
        ASSERT_TRUE(particleLoopRun);
    }

    TEST_F(SplineWithPrimitiveTest, PrimitiveEvalTest) {
        // Adding particle to one cell
        const int numParticles{1};
        amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{*infra.geom.ProbLo()};
        amrex::Array<amrex::Real, numParticles> weights{1};
        GEMPIC_TestUtils::addSingleParticles<vDim, numSpec, numParticles>(particleGroup, infra, weights, positions);

        particleGroup[0]->Redistribute();  // assign particles to the tile they are in

        bool particleLoopRun{false};
        for (amrex::ParIter<0, 0, vDim + 1, 0> pti(*particleGroup[spec], 0); pti.isValid(); ++pti)
        {
            particleLoopRun = true;

            const auto& particles{pti.GetArrayOfStructs()};
            const auto partData{particles().data()};

            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
            for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
                position[d] = partData[0].pos(d);
            Spline::SplineWithPrimitive<degX, degY, degZ> spline(position, infra.plo, infra.dxi);

            amrex::Real factor;

            amrex::GpuArray<amrex::Real, 1> sMinusOne;
            spline.primitiveEval<-1, 1>(0, sMinusOne);
            EXPECT_EQ(1, sMinusOne[0]);

            amrex::GpuArray<amrex::Real, 1> sZero;
            spline.primitiveEval<0, 1>(0.5, sZero);
            EXPECT_EQ(0.5, sZero[0]);

            amrex::GpuArray<amrex::Real, 2> sOne;
            spline.primitiveEval<1, 2>(0, sOne);
            EXPECT_EQ(0, sOne[0]);
            EXPECT_EQ(0.5, sOne[1]);

            amrex::GpuArray<amrex::Real, 3> sTwo;
            factor = (1./6.);
            spline.primitiveEval<2, 3>(0, sTwo);
            EXPECT_EQ(0, sTwo[0]);
            EXPECT_EQ(1./6., sTwo[1]);
            EXPECT_EQ(-1. * factor + 1., sTwo[2]);

            amrex::GpuArray<amrex::Real, 4> sThree;
            factor = (1./24.);
            spline.primitiveEval<3, 4>(0, sThree);
            EXPECT_EQ(0, sThree[0]);
            EXPECT_EQ(1./24., sThree[1]);
            EXPECT_EQ(0.5, sThree[2]);
            EXPECT_EQ(-factor + 1., sThree[3]);

            amrex::GpuArray<amrex::Real, 5> sFour;
            factor = (1./120.);
            spline.primitiveEval<4, 5>(0, sFour);
            EXPECT_EQ(0, sFour[0]);
            EXPECT_EQ(1. * factor, sFour[1]);
            EXPECT_EQ(27. * factor, sFour[2]);
            EXPECT_EQ(93. * factor, sFour[3]);
            EXPECT_EQ(-factor + 1., sFour[4]);

            amrex::GpuArray<amrex::Real, 6> sFive;
            factor = (1./720.);
            spline.primitiveEval<5, 6>(0, sFive);
            EXPECT_EQ(0, sFive[0]);
            EXPECT_EQ(1./720., sFive[1]);
            EXPECT_EQ(58. * factor, sFive[2]);
            EXPECT_EQ(360. * factor, sFive[3]);
            EXPECT_EQ(662. * factor, sFive[4]);
            EXPECT_EQ(719. * factor, sFive[5]);
        }
        ASSERT_TRUE(particleLoopRun);
    }
}