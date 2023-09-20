#include <AMReX.H>
#include <GEMPIC_FDDeRhamComplex.H>
#include <GEMPIC_Fields.H>
#include <GEMPIC_Params.H>
#include "GEMPIC_particle_groups.H"
#include "gtest/gtest.h"
#include "test_utils/GEMPIC_test_utils.H"
#include "GEMPIC_Spline_Class.H"
#include "gmock/gmock.h"

using namespace Particles;
using namespace GEMPIC_FDDeRhamComplex;
using namespace GEMPIC_Fields;
using ::testing::Mock;

namespace {
    /**
     * @brief Provide mocks for the initBSplinesAtPositions and evalBSpline methods
    */
    template<int degX, int degY, int degZ>
    class MockSpline : public Spline::SplineBase<degX, degY, degZ> {
        public:
        MockSpline(amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const &position,
                   amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const &plo,
                   amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> const &dxInverse) :
                   Spline::SplineBase<degX, degY, degZ>(position, plo, dxInverse) {}

        MOCK_METHOD(amrex::Real, initBSplinesAtPositions, (
            (amrex::Real position),
            (amrex::Real plo),
            (amrex::Real dxInverse)));

        MOCK_METHOD(amrex::Real, evalBSpline, (
            (int i),
            (int j),
            (int k)));
    };
    
    /**
     * @brief Test fixture. Sets up clean environment before each test of the SplineBase class
     */
    class SplineBaseTest : public testing::Test {
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

    /**
     * @brief Test the constructor of the SplineBase class
     * @details Make sure that all values are initialized
    */
    TEST_F(SplineBaseTest, SplineConstructorTest) {
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
            Spline::SplineBase<degX, degY, degZ> spline(position, infra.plo, infra.dxi);

            EXPECT_EQ(1., spline.splineCell[0][0]);
            EXPECT_EQ(1., spline.splineCell[1][0]);
            EXPECT_EQ(1., spline.splineCell[2][0]);

            EXPECT_EQ(1., spline.splineNode[0][0]);
            EXPECT_EQ(0., spline.splineNode[0][1]);
            EXPECT_EQ(1., spline.splineNode[1][0]);
            EXPECT_EQ(0., spline.splineNode[1][1]);
            EXPECT_EQ(1., spline.splineNode[2][0]);
            EXPECT_EQ(0., spline.splineNode[2][1]);
        }
        ASSERT_TRUE(particleLoopRun);
    }

    /**
     * @brief Test the constructor of the SplineBase class on a non-unit grid
     * @details Make sure that all values are initialized
    */
    TEST_F(SplineBaseTest, SplineConstructorScalingTest) {
        /* Initialize the infrastructure */
        const amrex::RealBox realBox({AMREX_D_DECL(0.0, 0.0, 0.0)},
                                        {AMREX_D_DECL(10.0, 10.0, 10.0)});
        const amrex::IntVect nCell{AMREX_D_DECL(8, 4, 2)};
        const amrex::IntVect maxGridSize{AMREX_D_DECL(8, 4, 2)};
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
            Spline::SplineBase<degX, degY, degZ> spline(position, infra.plo, infra.dxi);

            EXPECT_EQ(infra.dxi[0], spline.splineCell[0][0]);
            EXPECT_EQ(infra.dxi[1], spline.splineCell[1][0]);
            EXPECT_EQ(infra.dxi[2], spline.splineCell[2][0]);

            EXPECT_EQ(1., spline.splineNode[0][0]);
            EXPECT_EQ(0., spline.splineNode[0][1]);
            EXPECT_EQ(1., spline.splineNode[1][0]);
            EXPECT_EQ(0., spline.splineNode[1][1]);
            EXPECT_EQ(1., spline.splineNode[2][0]);
            EXPECT_EQ(0., spline.splineNode[2][1]);
        }
        ASSERT_TRUE(particleLoopRun);
    }
    /**
     * @brief Test the initBSplinesAtPositions method
     * @details Verify values for different degrees and positions
    */
    TEST_F(SplineBaseTest, SplineInitBSplinesAtPositionsTest) {
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
            Spline::SplineBase<degX, degY, degZ> spline(position, infra.plo, infra.dxi);

            amrex::Real xintOne = spline.initBSplinesAtPositions<0, 1>(0, 0, 1);
            EXPECT_EQ(1., xintOne);
            EXPECT_EQ(0., spline.span[0]);

            amrex::Real xintTwo = spline.initBSplinesAtPositions<0, 2>(0, 0, 1);
            EXPECT_EQ(0.5, xintTwo);
            EXPECT_EQ(-1., spline.span[0]);

            amrex::Real xintThree = spline.initBSplinesAtPositions<0, 2>(0.75, 0, 1);
            EXPECT_EQ(0.75, xintThree);
            EXPECT_EQ(0., spline.span[0]);
        }
        ASSERT_TRUE(particleLoopRun);
    }
    
    /**
     * @brief Test the update1DSplines method
     * @details Verify that splineCell and splineNode get assigned the correct values
    */
    TEST_F(SplineBaseTest, SplineUpdate1DSplinesTest) {
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

            EXPECT_CALL(spline, initBSplinesAtPositions(1, 1, 1)).WillOnce(::testing::Return(1));

            amrex::Real result = spline.initBSplinesAtPositions(1, 1, 1);
            spline.template update1DSplines<0, 1>(1, 1, 1);

            EXPECT_EQ(1., result);
            EXPECT_EQ(1., spline.splineCell[0][0]);
            EXPECT_EQ(1., spline.splineNode[0][0]);
            EXPECT_EQ(0., spline.splineNode[0][1]);
        }
        ASSERT_TRUE(particleLoopRun);
    }
    
    /**
     * @brief Test the evalBSpline method
     * @details Verify that sCoeff's get assigned the correct values
    */
    TEST_F(SplineBaseTest, SplineEvalBSplineTest) {
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
            Spline::SplineBase<degX, degY, degZ> spline(position, infra.plo, infra.dxi);

            amrex::Real sCoeff;

            sCoeff = spline.template evalBSpline<0, 1>(0, 0, 0);
            EXPECT_EQ(1., sCoeff);
            sCoeff = spline.template evalBSpline<1, 1>(0, 0, 0);
            EXPECT_EQ(1., sCoeff);
            sCoeff = spline.template evalBSpline<2, 1>(0, 0, 0);
            EXPECT_EQ(1., sCoeff);

            sCoeff = spline.template evalBSpline<0, 2>(0, 0, 0);
            EXPECT_EQ(1., sCoeff);
            sCoeff = spline.template evalBSpline<1, 2>(0, 0, 0);
            EXPECT_EQ(1., sCoeff);
            sCoeff = spline.template evalBSpline<2, 2>(0, 0, 0);
            EXPECT_EQ(1., sCoeff);
        }
        ASSERT_TRUE(particleLoopRun);
    }
    
    /**
     * @brief Test the splineEval method
     * @details Verify that splineEval returns the correct values
    */
    TEST_F(SplineBaseTest, SplineSplineEvalTest) {
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
            Spline::SplineBase<degX, degY, degZ> spline(position, infra.plo, infra.dxi);

            amrex::GpuArray<amrex::Real, 1> sZero;
            spline.splineEval<0, 1>(0, sZero);
            EXPECT_EQ(1, sZero[0]);

            amrex::GpuArray<amrex::Real, 2> sOne;
            spline.splineEval<1, 2>(0.5, sOne);
            EXPECT_EQ(0.5, sOne[0]);
            EXPECT_EQ(0.5, sOne[1]);

            amrex::GpuArray<amrex::Real, 3> sTwo;
            spline.splineEval<2, 3>(0, sTwo);
            EXPECT_EQ(0, sTwo[0]);
            EXPECT_EQ(0.5, sTwo[1]);
            EXPECT_EQ(0.5, sTwo[2]);

            amrex::GpuArray<amrex::Real, 4> sThree;
            amrex::Real factor = 1./6.;
            spline.splineEval<3, 4>(0, sThree);
            EXPECT_EQ(0, sThree[0]);
            EXPECT_EQ(1.0 * factor, sThree[1]);
            EXPECT_EQ(4.0 * factor, sThree[2]);
            EXPECT_EQ(1.0 * factor, sThree[3]);

            amrex::GpuArray<amrex::Real, 5> sFour;
            factor = 1./24.;
            spline.splineEval<4, 5>(0, sFour);
            EXPECT_EQ(0, sFour[0]);
            EXPECT_EQ(1.0 * factor, sFour[1]);
            EXPECT_EQ(11.0 * factor, sFour[2]);
            EXPECT_EQ(11.0 * factor, sFour[3]);
            EXPECT_EQ(1.0 * factor, sFour[4]);

            amrex::GpuArray<amrex::Real, 6> sFive;
            factor = 1./120.;
            spline.splineEval<5, 6>(0, sFive);
            EXPECT_EQ(0, sFive[0]);
            EXPECT_EQ(1.0 * factor, sFive[1]);
            EXPECT_EQ(26.0 * factor, sFive[2]);
            EXPECT_EQ(66.0 * factor, sFive[3]);
            EXPECT_EQ(26.0 * factor, sFive[4]);
            EXPECT_EQ(1.0 * factor, sFive[5]);
        }
        ASSERT_TRUE(particleLoopRun);
    }
}