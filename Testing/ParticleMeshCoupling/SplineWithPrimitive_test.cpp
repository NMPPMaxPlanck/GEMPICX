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
 * @brief Provide mocks for the initBSplinesAtPositions method
 */
template <int degX, int degY, int degZ>
class MockSpline : public ParticleMeshCoupling::SplineWithPrimitive<degX, degY, degZ>
{
public:
    MockSpline(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const position,
               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxInverse) :
        ParticleMeshCoupling::SplineWithPrimitive<degX, degY, degZ>(position, plo, dxInverse)
    {
    }

    MOCK_METHOD(amrex::Real,
                initBSplinesAtPositions,
                ((amrex::Real position), (amrex::Real plo), (amrex::Real dxInverse)));
};

ComputationalDomain get_compdom ()
{
    std::array<amrex::Real, AMREX_SPACEDIM> const domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    std::array<amrex::Real, AMREX_SPACEDIM> const domainHi{AMREX_D_DECL(10.0, 10.0, 10.0)};
    amrex::IntVect const nCell{AMREX_D_DECL(20, 20, 20)};
    amrex::IntVect const maxGridSize{AMREX_D_DECL(20, 20, 20)};
    std::array<int, AMREX_SPACEDIM> const isPeriodic{AMREX_D_DECL(1, 1, 1)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic);
}

/**
 * @brief Test fixture. Sets up clean environment before each test of the SplineWithPrimitive class
 */
class SplineWithPrimitiveTest : public testing::Test
{
protected:
    // Degree of splines in each direction
    static int const s_degX{1};
    static int const s_degY{1};
    static int const s_degZ{1};

    static int const s_vDim{3};
    static int const s_numSpec{1};
    static int const s_spec{0};

    ComputationalDomain m_infra;
    std::vector<std::unique_ptr<ParticleGroups<s_vDim>>> m_particleGroup;

    SplineWithPrimitiveTest() : m_infra{get_compdom()}
    {
        // Parameters initialized here so that different tests can have different parameters
        Io::Parameters parameters{};

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

/**
 * @brief Test the constructor of the SplineWithPrimitive class
 * @details Make sure that all values are initialized
 */
TEST_F(SplineWithPrimitiveTest, SplineConstructorTest)
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

        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxi = m_infra.geometry().InvCellSizeArray();
        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), dxi);

        AMREX_D_TERM(EXPECT_EQ(dxi[xDir], spline.m_cellSplineVals[xDir][0]);
                     , EXPECT_EQ(dxi[yDir], spline.m_cellSplineVals[yDir][0]);
                     , EXPECT_EQ(dxi[zDir], spline.m_cellSplineVals[zDir][0]);)

        AMREX_D_TERM(EXPECT_EQ(1., spline.m_nodeSplineVals[xDir][0]);
                     EXPECT_EQ(0., spline.m_nodeSplineVals[xDir][1]);
                     , EXPECT_EQ(1., spline.m_nodeSplineVals[yDir][0]);
                     EXPECT_EQ(0., spline.m_nodeSplineVals[yDir][1]);
                     , EXPECT_EQ(1., spline.m_nodeSplineVals[zDir][0]);
                     EXPECT_EQ(0., spline.m_nodeSplineVals[zDir][1]);)

        AMREX_D_TERM(EXPECT_EQ(1., spline.m_primitiveNew[xDir][0]);
                     , EXPECT_EQ(1., spline.m_primitiveNew[yDir][0]);
                     , EXPECT_EQ(1., spline.m_primitiveNew[zDir][0]);)
    }
    ASSERT_TRUE(particleLoopRun);
}

/**
 * @brief Test the update1DPrimitive method
 * @details Verify values for different directions
 */
TEST_F(SplineWithPrimitiveTest, SplineUpdate1DPrimitiveTest)
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

        EXPECT_CALL(spline, initBSplinesAtPositions(1, 1, 1)).WillRepeatedly(::testing::Return(1));

        amrex::Real result = spline.initBSplinesAtPositions(1, 1, 1);
        EXPECT_EQ(1., result);

        spline.template update_1d_splines<xDir>(1, 1, 1);
        EXPECT_EQ(1., spline.m_primitiveNew[xDir][0]);
        if constexpr (AMREX_SPACEDIM > 1)
        {
            spline.template update_1d_splines<yDir>(1, 1, 1);
            EXPECT_EQ(1., spline.m_primitiveNew[yDir][0]);
        }
        if constexpr (AMREX_SPACEDIM == 3)
        {
            spline.template update_1d_splines<zDir>(1, 1, 1);
            EXPECT_EQ(1., spline.m_primitiveNew[zDir][0]);
        }
    }
    ASSERT_TRUE(particleLoopRun);
}

/**
 * @brief Test the compute_primitive_difference method for direction xDir and degree one
 * @details Verify values for different dx and indices
 */
TEST_F(SplineWithPrimitiveTest, SplineComputePrimitiveDifferencexDirTest)
{
    // Adding one particle at lower left corner of domain
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
        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

        EXPECT_EQ(0, spline.m_firstIndexOld[xDir]);
        EXPECT_EQ(0, spline.m_firstIndex[xDir]);
        EXPECT_EQ(0, spline.m_firstIndexOld[xDir] - spline.m_firstIndex[xDir]);

        amrex::Real primitiveDifference = spline.template compute_primitive_difference<xDir>(
            m_infra.geometry().CellSizeArray(), 0);
        EXPECT_EQ(-1, primitiveDifference);

        primitiveDifference = spline.template compute_primitive_difference<xDir>(
            m_infra.geometry().CellSizeArray(), 1);
        EXPECT_EQ(0, primitiveDifference);
        spline.template update_1d_splines<xDir>(
            m_infra.geometry().ProbLo(xDir) + 0.5 * m_infra.geometry().CellSize(xDir),
            m_infra.geometry().ProbLo(xDir), m_infra.geometry().InvCellSize(xDir));

        EXPECT_EQ(0, spline.m_firstIndexOld[xDir]);
        EXPECT_EQ(0, spline.m_firstIndex[xDir]);

        primitiveDifference = spline.template compute_primitive_difference<xDir>(
            m_infra.geometry().CellSizeArray(), 1);
        EXPECT_EQ(0, primitiveDifference);

        primitiveDifference = spline.template compute_primitive_difference<xDir>(
            m_infra.geometry().CellSizeArray(), 0);
        EXPECT_EQ(0.5, primitiveDifference);
    }
    ASSERT_TRUE(particleLoopRun);
}

/**
 * @brief Test the compute_primitive_difference method for direction yDir and degree one
 * @details Verify values for different dx and indices
 */
#if AMREX_SPACEDIM > 1
TEST_F(SplineWithPrimitiveTest, SplineComputePrimitiveDifferenceyDirTest)
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
        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

        EXPECT_EQ(0, spline.m_firstIndexOld[yDir]);
        EXPECT_EQ(0, spline.m_firstIndex[yDir]);
        EXPECT_EQ(0, spline.m_firstIndexOld[yDir] - spline.m_firstIndex[yDir]);

        amrex::Real primitiveDifference = spline.template compute_primitive_difference<yDir>(
            m_infra.geometry().CellSizeArray(), 1);
        EXPECT_EQ(0, primitiveDifference);

        primitiveDifference = spline.template compute_primitive_difference<yDir>(
            m_infra.geometry().CellSizeArray(), 0);
        EXPECT_EQ(-1, primitiveDifference);

        spline.template update_1d_splines<yDir>(
            m_infra.geometry().ProbLo(xDir) + 0.5 * m_infra.geometry().CellSize(xDir),
            m_infra.geometry().ProbLo(yDir), m_infra.geometry().InvCellSize(yDir));

        EXPECT_EQ(0, spline.m_firstIndexOld[yDir]);
        EXPECT_EQ(0, spline.m_firstIndex[yDir]);

        primitiveDifference = spline.template compute_primitive_difference<yDir>(
            m_infra.geometry().CellSizeArray(), 1);
        EXPECT_EQ(0, primitiveDifference);

        primitiveDifference = spline.template compute_primitive_difference<yDir>(
            m_infra.geometry().CellSizeArray(), 0);
        EXPECT_EQ(0.5, primitiveDifference);
    }
    ASSERT_TRUE(particleLoopRun);
}
#endif
/**
 * @brief Test the compute_primitive_difference method for direction xDir and degree two
 * @details Verify values for different dx and indices
 */
#if AMREX_SPACEDIM == 3
TEST_F(SplineWithPrimitiveTest, SplineComputePrimitiveDifferenceDegreeTwoTest)
{
    // Adding particle at lower left corner of computational domain
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
        ParticleMeshCoupling::SplineWithPrimitive<2, 2, 2> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

        EXPECT_EQ(0, spline.m_firstIndexOld[xDir]);
        EXPECT_EQ(-1, spline.m_firstIndex[xDir]);
        EXPECT_EQ(1, spline.m_firstIndexOld[xDir] - spline.m_firstIndex[xDir]);

        amrex::Real primitiveDifference = spline.template compute_primitive_difference<xDir>(
            m_infra.geometry().CellSizeArray(), 0);
        EXPECT_EQ(-0.125, primitiveDifference);

        primitiveDifference = spline.template compute_primitive_difference<xDir>(
            m_infra.geometry().CellSizeArray(), 2);
        EXPECT_EQ(-1, primitiveDifference);

        spline.template update_1d_splines<xDir>(
            m_infra.geometry().ProbLo(xDir) + 0.5 * m_infra.geometry().CellSize(xDir),
            m_infra.geometry().ProbLo(xDir), m_infra.geometry().InvCellSize(xDir));

        EXPECT_EQ(-1, spline.m_firstIndexOld[xDir]);
        EXPECT_EQ(0, spline.m_firstIndex[xDir]);
        EXPECT_EQ(-1, spline.m_firstIndexOld[xDir] - spline.m_firstIndex[xDir]);

        primitiveDifference = spline.template compute_primitive_difference<xDir>(
            m_infra.geometry().CellSizeArray(), 1);
        EXPECT_EQ(0.375, primitiveDifference);

        primitiveDifference = spline.template compute_primitive_difference<xDir>(
            m_infra.geometry().CellSizeArray(), 0);
        EXPECT_EQ(0.125, primitiveDifference);

        primitiveDifference = spline.template compute_primitive_difference<xDir>(
            m_infra.geometry().CellSizeArray(), 2);
        EXPECT_EQ(0, primitiveDifference);
    }
    ASSERT_TRUE(particleLoopRun);
}
#endif

/**
 * @brief Test the primitiveEval method
 * @details Verify that primitiveEval returns the correct values
 */
TEST_F(SplineWithPrimitiveTest, SplinePrimitiveEvalTest)
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
        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

        amrex::Real factor;

        amrex::GpuArray<amrex::Real, 1> sZero;
        spline.primitive_eval<0>(0.5, sZero);
        EXPECT_DOUBLE_EQ(0.5, sZero[0]);

        amrex::GpuArray<amrex::Real, 2> sOne;
        spline.primitive_eval<1>(0, sOne);
        EXPECT_DOUBLE_EQ(0, sOne[0]);
        EXPECT_DOUBLE_EQ(0.5, sOne[1]);

        amrex::GpuArray<amrex::Real, 3> sTwo;
        factor = (1. / 6.);
        spline.primitive_eval<2>(0, sTwo);
        EXPECT_DOUBLE_EQ(0, sTwo[0]);
        EXPECT_DOUBLE_EQ(1. / 6., sTwo[1]);
        EXPECT_DOUBLE_EQ(-1. * factor + 1., sTwo[2]);

        amrex::GpuArray<amrex::Real, 4> sThree;
        factor = (1. / 24.);
        spline.primitive_eval<3>(0, sThree);
        EXPECT_DOUBLE_EQ(0, sThree[0]);
        EXPECT_DOUBLE_EQ(1. / 24., sThree[1]);
        EXPECT_DOUBLE_EQ(0.5, sThree[2]);
        EXPECT_DOUBLE_EQ(-factor + 1., sThree[3]);

        amrex::GpuArray<amrex::Real, 5> sFour;
        factor = (1. / 120.);
        spline.primitive_eval<4>(0, sFour);
        EXPECT_DOUBLE_EQ(0, sFour[0]);
        EXPECT_DOUBLE_EQ(1. * factor, sFour[1]);
        EXPECT_DOUBLE_EQ(27. * factor, sFour[2]);
        EXPECT_DOUBLE_EQ(93. * factor, sFour[3]);
        EXPECT_DOUBLE_EQ(-factor + 1., sFour[4]);

        amrex::GpuArray<amrex::Real, 6> sFive;
        factor = (1. / 720.);
        spline.primitive_eval<5>(0, sFive);
        EXPECT_DOUBLE_EQ(0, sFive[0]);
        EXPECT_DOUBLE_EQ(1. / 720., sFive[1]);
        EXPECT_DOUBLE_EQ(58. * factor, sFive[2]);
        EXPECT_DOUBLE_EQ(360. * factor, sFive[3]);
        EXPECT_DOUBLE_EQ(662. * factor, sFive[4]);
        EXPECT_DOUBLE_EQ(719. * factor, sFive[5]);
    }
    ASSERT_TRUE(particleLoopRun);
}
} // namespace
