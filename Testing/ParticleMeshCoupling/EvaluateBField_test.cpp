/** Testing for evaluate_bfield function
 */

#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_SplineClass.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;

// Basics first
namespace
{
// When using amrex::ParallelFor you have to create a standalone helper function that does the
// execution on GPU and call that function from the unit test because of how GTest creates tests
// within a TEST_F fixture.
template <int vDim, int degX, int degY, int degZ>
amrex::GpuArray<amrex::Real, vDim>* update_b_field_parallel_for (
    amrex::ParIter<0, 0, vDim + 1, 0>& pti,
    DeRhamField<Grid::primal, Space::edge>& B,
    ComputationalDomain& infra)
{
    const long np{pti.numParticles()};
    const auto& particles{pti.GetArrayOfStructs()};
    const auto partData{particles().data()};
    amrex::AsyncArray<amrex::GpuArray<amrex::Real, vDim>> bfieldsArr(2);
    amrex::GpuArray<amrex::Real, vDim>* bfields = bfieldsArr.data();

    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> bArray;
    for (int cc{0}; cc < vDim; cc++) bArray[cc] = (B.m_data[cc])[pti].array();
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{infra.geometry().ProbLoArray()};

    amrex::ParallelFor(np,
                       [=] AMREX_GPU_DEVICE(long pp)
                       {
                           amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position;
                           for (unsigned int d{0}; d < AMREX_SPACEDIM; ++d)
                           {
                               position[d] = partData[0].pos(d);
                           }
                           ParticleMeshCoupling::SplineBase<degX, degY, degZ> spline(
                               position, plo, infra.inv_cell_size_array());

                           bfields[pp] =
                               spline.template eval_spline_field<Field::PrimalTwoForm>(bArray);
                       });

    return bfields;
}

ComputationalDomain get_compdom ()
{
    const std::array<amrex::Real, AMREX_SPACEDIM> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    const std::array<amrex::Real, AMREX_SPACEDIM> domainHi{AMREX_D_DECL(10.0, 10.0, 10.0)};
    const amrex::IntVect nCell{AMREX_D_DECL(10, 10, 10)};
    const amrex::IntVect maxGridSize{AMREX_D_DECL(10, 10, 10)};
    const std::array<int, AMREX_SPACEDIM> isPeriodic{AMREX_D_DECL(1, 1, 1)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic);
}

// Test fixture
class EvaluateBFieldTest : public testing::Test
{
protected:
    static const int s_degX{1};
    static const int s_degY{1};
    static const int s_degZ{1};
    inline static const int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

    inline static const int s_hodgeDegree{2};
    static const int s_numSpec{1};
    static const int s_vDim{3};
    static const int s_spec{0};
    Io::Parameters m_parameters{};

    ComputationalDomain m_infra;
    std::vector<std::unique_ptr<ParticleGroups<s_vDim>>> m_particleGroup;
    std::shared_ptr<FDDeRhamComplex> m_deRham;

    EvaluateBFieldTest() : m_infra{get_compdom()}
    {
        // particle settings
        double charge{1};
        double mass{1};

        m_parameters.set("Particle.species0.charge", charge);
        m_parameters.set("Particle.species0.mass", mass);

        // Initialize the De Rham Complex
        m_deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                     HodgeScheme::FDHodge);

        // particles
        m_particleGroup.resize(s_numSpec);
        for (int spec{0}; spec < s_numSpec; spec++)
        {
            m_particleGroup[spec] = std::make_unique<ParticleGroups<s_vDim>>(spec, m_infra);
        }
    }
};

TEST_F(EvaluateBFieldTest, NullTest)
{
    // Adding particle to one cell
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), m_infra, weights,
                                              positions);

    // (default) charge correctly transferred from addSingleParticles
    EXPECT_EQ(1, m_particleGroup[0]->get_charge());

    // Parse analytical fields and initialize parserEval. Has to be the same as Bx,By,Bz
    const amrex::Array<std::string, 3> analyticalFuncB = {"0.0", "0.0", "0.0"};

    const int nVar{AMREX_SPACEDIM + 1}; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncB[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcB[i] = parser[i].compile<nVar>();
    }

    DeRhamField<Grid::primal, Space::edge> B(m_deRham, funcB);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in
    // Particle iteration ... over one particle. Hopefully.

    bool particleLoopRun{false};
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        particleLoopRun = true;

        const long np{pti.numParticles()};
        EXPECT_EQ(1, np); // Only one particle added by addSingleParticles

        const auto& particles{pti.GetArrayOfStructs()};
        const auto* const partData{particles().data()};

        amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> bArray;
        for (int cc{0}; cc < s_vDim; cc++) bArray[cc] = (B.m_data[cc])[pti].array();

        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position;
        for (unsigned int d{0}; d < AMREX_SPACEDIM; ++d)
        {
            position[d] = partData[0].pos(d);
        }
        ParticleMeshCoupling::SplineBase<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.inv_cell_size_array());

        amrex::GpuArray<amrex::Real, s_vDim> bfield =
            spline.template eval_spline_field<Field::PrimalTwoForm>(bArray);

        EXPECT_EQ(bfield[xDir], 0);
        EXPECT_EQ(bfield[yDir], 0);
        EXPECT_EQ(bfield[zDir], 0);
    }
    ASSERT_TRUE(particleLoopRun);
}

TEST_F(EvaluateBFieldTest, SingleParticleNode)
{
    // Adding particle to one cell
    const int numParticles{1};
    // Particle at position (0,0,0) in box (0,0,0)
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), m_infra, weights,
                                              positions);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in

    // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
    // Ey, Ez
    const amrex::Array<std::string, 3> analyticalFuncB{"1.0", "1.0", "1.0"};

    const int nVar{AMREX_SPACEDIM + 1}; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncB[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcB[i] = parser[i].compile<nVar>();
    }

    DeRhamField<Grid::primal, Space::edge> B(m_deRham, funcB);

    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[0], 0); pti.isValid(); ++pti)
    {
        const long np{pti.numParticles()};
        EXPECT_EQ(numParticles, np);

        amrex::GpuArray<amrex::Real, s_vDim>* bfields =
            update_b_field_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(pti, B, m_infra);

        EXPECT_EQ(bfields[0][xDir], 1.0);
        EXPECT_EQ(bfields[0][yDir], 1.0);
        EXPECT_EQ(bfields[0][zDir], 1.0);
    }
}

TEST_F(EvaluateBFieldTest, SingleParticleMiddle)
{
    // Adding particle to one cell
    const int numParticles{1};
    auto dx = m_infra.geometry().CellSizeArray();
    // Add particle in the middle of final cell to check periodic boundary conditions
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(m_infra.m_geom.ProbHi(xDir) - 1.5 * dx[xDir],
                       m_infra.m_geom.ProbHi(yDir) - 1.5 * dx[yDir],
                       m_infra.m_geom.ProbHi(zDir) - 1.5 * dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), m_infra, weights,
                                              positions);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in

    // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
    // Ey, Ez
    const amrex::Array<std::string, 3> analyticalFuncB{"1.0", "1.0", "1.0"};

    const int nVar{AMREX_SPACEDIM + 1}; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncB[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcB[i] = parser[i].compile<nVar>();
    }

    DeRhamField<Grid::primal, Space::edge> B(m_deRham, funcB);

    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[0], 0); pti.isValid(); ++pti)
    {
        const long np{pti.numParticles()};
        EXPECT_EQ(numParticles, np);

        amrex::GpuArray<amrex::Real, s_vDim>* bfields =
            update_b_field_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(pti, B, m_infra);

        EXPECT_EQ(bfields[0][xDir], 1.0);
        EXPECT_EQ(bfields[0][yDir], 1.0);
        EXPECT_EQ(bfields[0][zDir], 1.0);
    }
}

TEST_F(EvaluateBFieldTest, SingleParticleUnevenNodeSplit)
{
    // Adding particle to one cell
    const int numParticles{1};
    auto dx = m_infra.geometry().CellSizeArray();
    // Add particle in the middle of final cell to check periodic boundary conditions
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(m_infra.m_geom.ProbHi(xDir) - 1.25 * dx[xDir],
                       m_infra.m_geom.ProbHi(yDir) - 1.25 * dx[yDir],
                       m_infra.m_geom.ProbHi(zDir) - 1.25 * dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), m_infra, weights,
                                              positions);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in

    // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
    // Ey, Ez
    const amrex::Array<std::string, 3> analyticalFuncB{"1.0", "1.0", "1.0"};

    const int nVar{AMREX_SPACEDIM + 1}; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncB[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcB[i] = parser[i].compile<nVar>();
    }

    DeRhamField<Grid::primal, Space::edge> B(m_deRham, funcB);

    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[0], 0); pti.isValid(); ++pti)
    {
        const long np{pti.numParticles()};
        EXPECT_EQ(numParticles, np);

        amrex::GpuArray<amrex::Real, s_vDim>* bfields =
            update_b_field_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(pti, B, m_infra);

        EXPECT_EQ(bfields[0][xDir], 1.0);
        EXPECT_EQ(bfields[0][yDir], 1.0);
        EXPECT_EQ(bfields[0][zDir], 1.0);
    }
}

TEST_F(EvaluateBFieldTest, DoubleParticleSeparate)
{
    const int numParticles{2};
    auto dx = m_infra.geometry().CellSizeArray();
    // Particles in different cells to check that they don't interfere with each other
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(0, 0, 0)},
         {AMREX_D_DECL(m_infra.m_geom.ProbLo(xDir) + 5.5 * dx[xDir],
                       m_infra.m_geom.ProbLo(yDir) + 5.5 * dx[yDir],
                       m_infra.m_geom.ProbLo(zDir) + 5.5 * dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> weights{1, 1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), m_infra, weights,
                                              positions);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in

    // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
    // Ey, Ez
    const amrex::Array<std::string, 3> analyticalFuncB{"1.0", "1.0", "1.0"};

    const int nVar{AMREX_SPACEDIM + 1}; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncB[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcB[i] = parser[i].compile<nVar>();
    }

    DeRhamField<Grid::primal, Space::edge> B(m_deRham, funcB);

    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[0], 0); pti.isValid(); ++pti)
    {
        const long np{pti.numParticles()};
        EXPECT_EQ(numParticles, np);

        amrex::GpuArray<amrex::Real, s_vDim>* bfields =
            update_b_field_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(pti, B, m_infra);

        EXPECT_EQ(bfields[0][xDir], 1.0);
        EXPECT_EQ(bfields[0][yDir], 1.0);
        EXPECT_EQ(bfields[0][zDir], 1.0);

        EXPECT_EQ(bfields[1][xDir], 1.0);
        EXPECT_EQ(bfields[1][yDir], 1.0);
        EXPECT_EQ(bfields[1][zDir], 1.0);
    }
}

TEST_F(EvaluateBFieldTest, DoubleParticleOverlap)
{
    const int numParticles{2};
    auto dx = m_infra.geometry().CellSizeArray();
    // Particles in different cells to check that they don't interfere with each other
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(0, 0, 0)},
         {AMREX_D_DECL(m_infra.m_geom.ProbLo(xDir) + 0.5 * dx[xDir],
                       m_infra.m_geom.ProbLo(yDir) + 0.5 * dx[yDir],
                       m_infra.m_geom.ProbLo(zDir) + 0.5 * dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> weights{1, 1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), m_infra, weights,
                                              positions);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in

    // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
    // Ey, Ez
    const amrex::Array<std::string, 3> analyticalFuncB{"1.0", "1.0", "1.0"};

    const int nVar{AMREX_SPACEDIM + 1}; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncB[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcB[i] = parser[i].compile<nVar>();
    }

    DeRhamField<Grid::primal, Space::edge> B(m_deRham, funcB);

    amrex::MFItInfo mfii{};
    mfii.do_tiling = amrex::TilingIfNotGPU();

    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[0], 0, mfii); pti.isValid();
         ++pti)
    {
        const long np{pti.numParticles()};
        EXPECT_EQ(numParticles, np);

        amrex::GpuArray<amrex::Real, s_vDim>* bfields =
            update_b_field_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(pti, B, m_infra);

        EXPECT_EQ(bfields[0][xDir], 1.0);
        EXPECT_EQ(bfields[0][yDir], 1.0);
        EXPECT_EQ(bfields[0][zDir], 1.0);

        EXPECT_EQ(bfields[1][xDir], 1.0);
        EXPECT_EQ(bfields[1][yDir], 1.0);
        EXPECT_EQ(bfields[1][zDir], 1.0);
    }
}

TEST_F(EvaluateBFieldTest, TestingForLiterallyAnythingOtherThanUnity)
{
    GTEST_SKIP() << "Such advanced tests have not yet been implemented!";
}
} // namespace
