#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_SplineClass.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;
using ::testing::Mock;

namespace
{
/**
 * @brief Provide mocks for the initBSplinesAtPositions and evalBSpline methods
 */
template <int degX, int degY, int degZ>
class MockSpline : public ParticleMeshCoupling::SplineBase<degX, degY, degZ>
{
public:
    MockSpline(amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const &position,
               amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const &plo,
               amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> const &dxInverse) :
        ParticleMeshCoupling::SplineBase<degX, degY, degZ>(position, plo, dxInverse)
    {
    }

    MOCK_METHOD(amrex::Real,
                initBSplinesAtPositions,
                ((amrex::Real position), (amrex::Real plo), (amrex::Real dxInverse)));

    MOCK_METHOD(amrex::Real, evalBSpline, ((int i), (int j), (int k)));
};

/**
 * @brief Test fixture. Sets up clean environment before each test of the SplineBase class
 */
class SplineBaseTest : public testing::Test
{
protected:
    // Degree of splines in each direction
    static const int s_degX{1};
    static const int s_degY{1};
    static const int s_degZ{1};
    static const int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

    static const int s_hodgeDegree{2};
    static const int s_vDim{3};
    static const int s_numSpec{1};
    static const int s_spec{0};
    const int m_nghost{Gempic::Test::Utils::init_n_ghost(s_degX, s_degY, s_degZ)};

    ComputationalDomain m_infra{false};  // "uninitialized" computational domain
    amrex::GpuArray<std::unique_ptr<ParticleGroups<s_vDim>>, s_numSpec> m_particleGroup;

    static void SetUpTestSuite ()
    {
        /* Initialize the infrastructure */
        // const amrex::RealBox realBox({AMREX_D_DECL(0.0, 0.0, 0.0)},
        //                              {AMREX_D_DECL(10.0, 10.0, 10.0)});
        amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        //
        amrex::Vector<amrex::Real> k{AMREX_D_DECL(0.2 * M_PI, 0.2 * M_PI, 0.2 * M_PI)};
        const amrex::Vector<int> nCell{AMREX_D_DECL(10, 10, 10)};
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(10, 10, 10)};
        const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};

        amrex::ParmParse pp;
        pp.addarr("domainLo", domainLo);
        pp.addarr("k", k);
        pp.addarr("nCellVector", nCell);
        pp.addarr("maxGridSizeVector", maxGridSize);
        pp.addarr("isPeriodicVector", isPeriodic);

        // particle settings
        double charge{1};
        double mass{1};

        pp.add("Particle.species0.charge", charge);
        pp.add("Particle.species0.mass", mass);
    }

    // virtual void SetUp() will be called before each test is run.
    void SetUp () override
    {
        // Parameters initialized here so that different tests can have different parameters
        Io::Parameters parameters{};
        /* Initialize the infrastructure */
        m_infra = ComputationalDomain{};

        // particles
        for (int spec{0}; spec < s_numSpec; spec++)
        {
            m_particleGroup[spec] = std::make_unique<ParticleGroups<s_vDim>>(spec, m_infra);
        }
    }
};

class SplineBaseTestCustomInfrastructure : public SplineBaseTest
{
    void SetUp () override {}
};

/**
 * @brief Test the constructor of the SplineBase class
 * @details Make sure that all values are initialized
 */
TEST_F(SplineBaseTest, SplineConstructorTest)
{
    // Adding particle to one cell
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in

    bool particleLoopRun{false};
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        particleLoopRun = true;

        const auto &particles{pti.GetArrayOfStructs()};
        const auto *const partData{particles().data()};

        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
        for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
        {
            position[d] = partData[0].pos(d);
        }
        ParticleMeshCoupling::SplineBase<s_degX, s_degY, s_degZ> spline(position, m_infra.m_plo,
                                                                        m_infra.m_dxi);
        AMREX_D_TERM(EXPECT_EQ(1., spline.m_cellSplineVals[xDir][0]);
                     , EXPECT_EQ(1., spline.m_cellSplineVals[yDir][0]);
                     , EXPECT_EQ(1., spline.m_cellSplineVals[zDir][0]);)

        AMREX_D_TERM(EXPECT_EQ(1., spline.m_nodeSplineVals[xDir][0]);
                     EXPECT_EQ(0., spline.m_nodeSplineVals[xDir][1]);
                     , EXPECT_EQ(1., spline.m_nodeSplineVals[yDir][0]);
                     EXPECT_EQ(0., spline.m_nodeSplineVals[yDir][1]);
                     , EXPECT_EQ(1., spline.m_nodeSplineVals[zDir][0]);
                     EXPECT_EQ(0., spline.m_nodeSplineVals[zDir][1]);)
    }
    ASSERT_TRUE(particleLoopRun);
}

/**
 * @brief Test the constructor of the SplineBase class on a non-unit grid
 * @details Make sure that all values are initialized
 */
TEST_F(SplineBaseTestCustomInfrastructure, SplineConstructorScalingTest)
{
    /* Initialize the infrastructure */
    const amrex::Vector<int> nCell{AMREX_D_DECL(8, 4, 2)};
    const amrex::Vector<int> maxGridSize{AMREX_D_DECL(8, 4, 2)};

    Io::Parameters parameters{};
    parameters.set("nCellVector", nCell);
    parameters.set("maxGridSizeVector", maxGridSize);

    ComputationalDomain infra;

    // particles
    for (int species{0}; species < s_numSpec; species++)
    {
        m_particleGroup[species] = std::make_unique<ParticleGroups<s_vDim>>(species, infra);
    }

    // Adding particle to one cell
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{*infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, infra, weights, positions);

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in

    bool particleLoopRun{false};
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        particleLoopRun = true;

        const auto &particles{pti.GetArrayOfStructs()};
        const auto *const partData{particles().data()};

        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
        for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
        {
            position[d] = partData[0].pos(d);
        }
        ParticleMeshCoupling::SplineBase<s_degX, s_degY, s_degZ> spline(position, infra.m_plo,
                                                                        infra.m_dxi);

        AMREX_D_TERM(EXPECT_EQ(infra.m_dxi[xDir], spline.m_cellSplineVals[xDir][0]);
                     , EXPECT_EQ(infra.m_dxi[yDir], spline.m_cellSplineVals[yDir][0]);
                     , EXPECT_EQ(infra.m_dxi[zDir], spline.m_cellSplineVals[zDir][0]);)

        AMREX_D_TERM(EXPECT_EQ(1., spline.m_nodeSplineVals[xDir][0]);
                     EXPECT_EQ(0., spline.m_nodeSplineVals[xDir][1]);
                     , EXPECT_EQ(1., spline.m_nodeSplineVals[yDir][0]);
                     EXPECT_EQ(0., spline.m_nodeSplineVals[yDir][1]);
                     , EXPECT_EQ(1., spline.m_nodeSplineVals[zDir][0]);
                     EXPECT_EQ(0., spline.m_nodeSplineVals[zDir][1]);)
    }
    ASSERT_TRUE(particleLoopRun);
}
/**
 * @brief Test the initBSplinesAtPositions method
 * @details Verify values for different degrees and positions
 */
TEST_F(SplineBaseTest, SplineInitBSplinesAtPositionsTest)
{
    // Adding particle to one cell
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in

    bool particleLoopRun{false};
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        particleLoopRun = true;

        const auto &particles{pti.GetArrayOfStructs()};
        const auto *const partData{particles().data()};

        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
        for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
        {
            position[d] = partData[0].pos(d);
        }
        ParticleMeshCoupling::SplineBase<s_degX, s_degY, s_degZ> spline(position, m_infra.m_plo,
                                                                        m_infra.m_dxi);

        amrex::Real xintOne = spline.init_b_splines_at_positions<xDir>(0, 0, 1);
        EXPECT_EQ(1., xintOne);
        EXPECT_EQ(0., spline.m_firstIndex[xDir]);

        ParticleMeshCoupling::SplineBase<2, s_degY, s_degZ> splineDeg2(position, m_infra.m_plo,
                                                                       m_infra.m_dxi);
        amrex::Real xintTwo = splineDeg2.init_b_splines_at_positions<xDir>(0, 0, 1);
        EXPECT_EQ(0.5, xintTwo);
        EXPECT_EQ(-1., splineDeg2.m_firstIndex[xDir]);

        amrex::Real xintThree = splineDeg2.init_b_splines_at_positions<xDir>(0.75, 0, 1);
        EXPECT_EQ(0.75, xintThree);
        EXPECT_EQ(0., splineDeg2.m_firstIndex[xDir]);
    }
    ASSERT_TRUE(particleLoopRun);
}

/**
 * @brief Test the update1DSplines method
 * @details Verify that cellSplineVals and nodeSplineVals get assigned the correct values
 */
TEST_F(SplineBaseTest, SplineUpdate1DSplinesTest)
{
    // Adding particle to one cell
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in

    bool particleLoopRun{false};
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        particleLoopRun = true;

        const auto &particles{pti.GetArrayOfStructs()};
        const auto *const partData{particles().data()};

        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
        for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
        {
            position[d] = partData[0].pos(d);
        }
        MockSpline<s_degX, s_degY, s_degZ> spline(position, m_infra.m_plo, m_infra.m_dxi);

        EXPECT_CALL(spline, initBSplinesAtPositions(1, 1, 1)).WillOnce(::testing::Return(1));

        amrex::Real result = spline.initBSplinesAtPositions(1, 1, 1);
        spline.template update1_d_splines<xDir>(1, 1, 1);

        EXPECT_EQ(1., result);
        EXPECT_EQ(1., spline.m_cellSplineVals[xDir][0]);
        EXPECT_EQ(1., spline.m_nodeSplineVals[xDir][0]);
        EXPECT_EQ(0., spline.m_nodeSplineVals[xDir][1]);
    }
    ASSERT_TRUE(particleLoopRun);
}

/**
 * @brief Test the evalBSpline method
 * @details Verify that sCoeff's get assigned the correct values
 */
TEST_F(SplineBaseTest, SplineEvalBSplineTest)
{
    // Adding particle to one cell
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in

    bool particleLoopRun{false};
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        particleLoopRun = true;

        const auto &particles{pti.GetArrayOfStructs()};
        const auto *const partData{particles().data()};

        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
        for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
        {
            position[d] = partData[0].pos(d);
        }
        ParticleMeshCoupling::SplineBase<s_degX, s_degY, s_degZ> spline(position, m_infra.m_plo,
                                                                        m_infra.m_dxi);

        amrex::Real sCoeff;

        sCoeff = spline.template eval_b_spline<Field::DualTwoForm, xDir>(0, 0, 0);
        EXPECT_EQ(1., sCoeff);
        sCoeff = spline.template eval_b_spline<Field::DualTwoForm, yDir>(0, 0, 0);
        EXPECT_EQ(1., sCoeff);
        sCoeff = spline.template eval_b_spline<Field::DualTwoForm, zDir>(0, 0, 0);
        EXPECT_EQ(1., sCoeff);

        sCoeff = spline.template eval_b_spline<Field::DualOneForm, xDir>(0, 0, 0);
        EXPECT_EQ(1., sCoeff);
        sCoeff = spline.template eval_b_spline<Field::DualOneForm, yDir>(0, 0, 0);
        EXPECT_EQ(1., sCoeff);
        sCoeff = spline.template eval_b_spline<Field::DualOneForm, zDir>(0, 0, 0);
        EXPECT_EQ(1., sCoeff);
    }
    ASSERT_TRUE(particleLoopRun);
}

/**
 * @brief Test the splineEval method
 * @details Verify that splineEval returns the correct values
 */
TEST_F(SplineBaseTest, SplineSplineEvalTest)
{
    // Adding particle to one cell
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in

    bool particleLoopRun{false};
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        particleLoopRun = true;

        const auto &particles{pti.GetArrayOfStructs()};
        const auto *const partData{particles().data()};

        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
        for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
        {
            position[d] = partData[0].pos(d);
        }
        ParticleMeshCoupling::SplineBase<s_degX, s_degY, s_degZ> spline(position, m_infra.m_plo,
                                                                        m_infra.m_dxi);

        amrex::GpuArray<amrex::Real, 1> sZero;
        spline.spline_eval<0, 1>(0, sZero);
        EXPECT_EQ(1, sZero[0]);

        amrex::GpuArray<amrex::Real, 2> sOne;
        spline.spline_eval<1, 2>(0.5, sOne);
        EXPECT_EQ(0.5, sOne[0]);
        EXPECT_EQ(0.5, sOne[1]);

        amrex::GpuArray<amrex::Real, 3> sTwo;
        spline.spline_eval<2, 3>(0, sTwo);
        EXPECT_EQ(0, sTwo[0]);
        EXPECT_EQ(0.5, sTwo[1]);
        EXPECT_EQ(0.5, sTwo[2]);

        amrex::GpuArray<amrex::Real, 4> sThree;
        amrex::Real factor = 1. / 6.;
        spline.spline_eval<3, 4>(0, sThree);
        EXPECT_EQ(0, sThree[0]);
        EXPECT_EQ(1.0 * factor, sThree[1]);
        EXPECT_EQ(4.0 * factor, sThree[2]);
        EXPECT_EQ(1.0 * factor, sThree[3]);

        amrex::GpuArray<amrex::Real, 5> sFour;
        factor = 1. / 24.;
        spline.spline_eval<4, 5>(0, sFour);
        EXPECT_EQ(0, sFour[0]);
        EXPECT_EQ(1.0 * factor, sFour[1]);
        EXPECT_EQ(11.0 * factor, sFour[2]);
        EXPECT_EQ(11.0 * factor, sFour[3]);
        EXPECT_EQ(1.0 * factor, sFour[4]);

        amrex::GpuArray<amrex::Real, 6> sFive;
        factor = 1. / 120.;
        spline.spline_eval<5, 6>(0, sFive);
        EXPECT_EQ(0, sFive[0]);
        EXPECT_EQ(1.0 * factor, sFive[1]);
        EXPECT_EQ(26.0 * factor, sFive[2]);
        EXPECT_EQ(66.0 * factor, sFive[3]);
        EXPECT_EQ(26.0 * factor, sFive[4]);
        EXPECT_EQ(1.0 * factor, sFive[5]);
    }
    ASSERT_TRUE(particleLoopRun);
}
}  // namespace