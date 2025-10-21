#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>

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
    MockSpline(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& position,
               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxInverse) :
        ParticleMeshCoupling::SplineBase<degX, degY, degZ>(position, plo, dxInverse)
    {
    }

    MOCK_METHOD(amrex::Real,
                initBSplinesAtPositions,
                ((amrex::Real position), (amrex::Real plo), (amrex::Real dxInverse)));

    MOCK_METHOD(amrex::Real, evalBSpline, ((int i), (int j), (int k)));
};

/* Initialize the infrastructure */
inline ComputationalDomain get_compdom (amrex::IntVect const& nCell,
                                        amrex::IntVect const& maxGridSize)
{
    amrex::Array<amrex::Real, AMREX_SPACEDIM> const domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    amrex::Array<amrex::Real, AMREX_SPACEDIM> const domainHi{AMREX_D_DECL(10, 10, 10)};
    amrex::Array<int, AMREX_SPACEDIM> const isPeriodic{AMREX_D_DECL(1, 1, 1)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic,
                               amrex::CoordSys::cartesian);
}

/**
 * @brief Test fixture. Sets up clean environment before each test of the SplineBase class
 */
class SplineBaseTest : public testing::Test
{
protected:
    // Degree of splines in each direction
    static int const s_degX{1};
    static int const s_degY{1};
    static int const s_degZ{1};

    static int const s_vDim{3};
    static int const s_numSpec{1};
    static int const s_spec{0};
    int const m_nghost{Gempic::Test::Utils::init_n_ghost(s_degX, s_degY, s_degZ)};

    ComputationalDomain m_infra;
    std::vector<std::unique_ptr<ParticleGroups<s_vDim>>> m_particleGroup;

    SplineBaseTest() :
        m_infra{get_compdom(amrex::IntVect{AMREX_D_DECL(10, 10, 10)},
                            amrex::IntVect{AMREX_D_DECL(10, 10, 10)})}
    {
        amrex::Vector<amrex::Real> k{AMREX_D_DECL(0.0, 0.0, 0.0)};
        for (int dir{0}; dir < AMREX_SPACEDIM; ++dir)
        {
            k[dir] = m_infra.geometry_data().ProbHi(dir);
        }

        // Parameters initialized here so that different tests can have different parameters
        Gempic::Io::Parameters parameters;
        // particle settings
        double charge{1};
        double mass{1};

        parameters.set("Particle.species0.charge", charge);
        parameters.set("Particle.species0.mass", mass);

        // particles
        m_particleGroup.resize(s_numSpec);
        for (int spec{0}; spec < s_numSpec; spec++)
        {
            m_particleGroup[spec] = std::make_unique<ParticleGroups<s_vDim>>(spec, m_infra);
        }
    }
};

class SplineBaseTestCustomInfrastructure : public testing::Test
{
protected:
    Io::Parameters m_parameters{};
    // Degree of splines in each direction
    static int const s_degX{1};
    static int const s_degY{1};
    static int const s_degZ{1};

    static int const s_vDim{3};
    static int const s_numSpec{1};
    static int const s_spec{0};
    int const m_nghost{Gempic::Test::Utils::init_n_ghost(s_degX, s_degY, s_degZ)};
    ComputationalDomain m_infra;
    std::vector<std::unique_ptr<ParticleGroups<s_vDim>>> m_particleGroup;

    SplineBaseTestCustomInfrastructure() :
        m_infra{get_compdom(amrex::IntVect{AMREX_D_DECL(10, 10, 10)},
                            amrex::IntVect{AMREX_D_DECL(10, 10, 10)})}
    {
        amrex::Vector<amrex::Real> k{AMREX_D_DECL(0.0, 0.0, 0.0)};
        for (int dir{0}; dir < AMREX_SPACEDIM; ++dir)
        {
            k[dir] = m_infra.geometry_data().ProbHi(dir);
        }

        m_parameters.set("k", k);

        // particle settings
        double charge{1};
        double mass{1};

        m_parameters.set("Particle.species0.charge", charge);
        m_parameters.set("Particle.species0.mass", mass);
    }
};

/**
 * @brief Test the constructor of the SplineBase class
 * @details Make sure that all values are initialized
 */
TEST_F(SplineBaseTest, SplineConstructorTest)
{
    // Adding particle to one cell
    int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), m_infra, weights,
                                              positions);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in

    bool particleLoopRun{false};
    for (auto& particleGrid : *m_particleGroup[s_spec])
    {
        particleLoopRun = true;

        auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
        auto const ii = m_particleGroup[s_spec]->get_data_indices();
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{AMREX_D_DECL(
            ptd.rdata(ii.m_iposx)[0], ptd.rdata(ii.m_iposy)[0], ptd.rdata(ii.m_iposz)[0])};

        ParticleMeshCoupling::SplineBase<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());
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
    amrex::IntVect const nCell{AMREX_D_DECL(8, 4, 2)};
    amrex::IntVect const maxGridSize{AMREX_D_DECL(8, 4, 2)};

    ComputationalDomain infra = get_compdom(nCell, maxGridSize);

    // particles
    m_particleGroup.resize(s_numSpec);
    m_particleGroup[0] = std::make_unique<ParticleGroups<s_vDim>>(0, infra);

    // Adding particle to one cell
    int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{*infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), infra, weights, positions);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in

    bool particleLoopRun{false};
    for (auto& particleGrid : *m_particleGroup[s_spec])
    {
        particleLoopRun = true;

        auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
        auto const ii = m_particleGroup[s_spec]->get_data_indices();
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{AMREX_D_DECL(
            ptd.rdata(ii.m_iposx)[0], ptd.rdata(ii.m_iposy)[0], ptd.rdata(ii.m_iposz)[0])};

        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxi = infra.geometry().InvCellSizeArray();
        ParticleMeshCoupling::SplineBase<s_degX, s_degY, s_degZ> spline(
            position, infra.geometry().ProbLoArray(), dxi);

        AMREX_D_TERM(EXPECT_EQ(dxi[xDir], spline.m_cellSplineVals[xDir][0]);
                     , EXPECT_EQ(dxi[yDir], spline.m_cellSplineVals[yDir][0]);
                     , EXPECT_EQ(dxi[zDir], spline.m_cellSplineVals[zDir][0]);)

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
    int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), m_infra, weights,
                                              positions);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in

    bool particleLoopRun{false};
    for (auto& particleGrid : *m_particleGroup[s_spec])
    {
        particleLoopRun = true;

        auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
        auto const ii = m_particleGroup[s_spec]->get_data_indices();
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{AMREX_D_DECL(
            ptd.rdata(ii.m_iposx)[0], ptd.rdata(ii.m_iposy)[0], ptd.rdata(ii.m_iposz)[0])};
        ParticleMeshCoupling::SplineBase<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

        amrex::Real xintOne = spline.init_b_splines_at_positions<xDir>(0, 0, 1);
        EXPECT_EQ(1., xintOne);
        EXPECT_EQ(0., spline.m_firstIndex[xDir]);

        ParticleMeshCoupling::SplineBase<2, s_degY, s_degZ> splineDeg2(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());
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
    int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), m_infra, weights,
                                              positions);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in

    bool particleLoopRun{false};
    for (auto& particleGrid : *m_particleGroup[s_spec])
    {
        particleLoopRun = true;

        auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
        auto const ii = m_particleGroup[s_spec]->get_data_indices();
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{AMREX_D_DECL(
            ptd.rdata(ii.m_iposx)[0], ptd.rdata(ii.m_iposy)[0], ptd.rdata(ii.m_iposz)[0])};
        MockSpline<s_degX, s_degY, s_degZ> spline(position, m_infra.geometry().ProbLoArray(),
                                                  m_infra.geometry().InvCellSizeArray());

        EXPECT_CALL(spline, initBSplinesAtPositions(1, 1, 1)).WillOnce(::testing::Return(1));

        amrex::Real result = spline.initBSplinesAtPositions(1, 1, 1);
        spline.template update_1d_splines<xDir>(1, 1, 1);

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
    int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), m_infra, weights,
                                              positions);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in

    bool particleLoopRun{false};
    for (auto& particleGrid : *m_particleGroup[s_spec])
    {
        particleLoopRun = true;

        auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
        auto const ii = m_particleGroup[s_spec]->get_data_indices();
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{AMREX_D_DECL(
            ptd.rdata(ii.m_iposx)[0], ptd.rdata(ii.m_iposy)[0], ptd.rdata(ii.m_iposz)[0])};
        ParticleMeshCoupling::SplineBase<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

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
    int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), m_infra, weights,
                                              positions);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in

    bool particleLoopRun{false};
    for (auto& particleGrid : *m_particleGroup[s_spec])
    {
        particleLoopRun = true;

        auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
        auto const ii = m_particleGroup[s_spec]->get_data_indices();
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{AMREX_D_DECL(
            ptd.rdata(ii.m_iposx)[0], ptd.rdata(ii.m_iposy)[0], ptd.rdata(ii.m_iposz)[0])};
        ParticleMeshCoupling::SplineBase<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

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
} // namespace
