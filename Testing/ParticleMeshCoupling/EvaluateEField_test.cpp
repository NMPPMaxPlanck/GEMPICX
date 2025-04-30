/** Testing for evaluate_efield function
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
void update_e_field_parallel_for (amrex::ParIter<0, 0, vDim + 1, 0>& particleGrid,
                                  DeRhamField<Grid::primal, Space::edge>& E,
                                  ComputationalDomain& infra)
{
    long const np{particleGrid.numParticles()};
    auto const& particles{particleGrid.GetArrayOfStructs()};
    auto const partData{particles().data()};
    amrex::AsyncArray<amrex::GpuArray<amrex::Real, vDim>> efieldsPtr(2);
    amrex::GpuArray<amrex::Real, vDim>* efields = efieldsPtr.data();

    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> eArray;
    for (int cc{0}; cc < vDim; cc++) eArray[cc] = (E.m_data[cc])[particleGrid].array();
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

                           efields[pp] =
                               spline.template eval_spline_field<Field::PrimalOneForm>(eArray);
                       });

    EXPECT_NEAR(efields[0][xDir], 1.0, 1e-12);
    EXPECT_NEAR(efields[0][yDir], 1.0, 1e-12);
    EXPECT_NEAR(efields[0][zDir], 1.0, 1e-12);

    if (np == 2)
    {
        EXPECT_NEAR(efields[1][xDir], 1.0, 1e-12);
        EXPECT_NEAR(efields[1][yDir], 1.0, 1e-12);
        EXPECT_NEAR(efields[1][zDir], 1.0, 1e-12);
    }
}

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

// Test fixture
class EvaluateEFieldTest : public testing::Test
{
protected:
    static int const s_degX{1};
    static int const s_degY{1};
    static int const s_degZ{1};
    inline static int const s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

    inline static int const s_hodgeDegree{2};

    static int const s_numSpec{1};
    static int const s_vDim{3};
    static int const s_spec{0};

    ComputationalDomain m_infra;
    std::vector<std::unique_ptr<ParticleGroups<s_vDim>>> m_particleGroup;
    std::shared_ptr<FDDeRhamComplex> m_deRham;

    EvaluateEFieldTest() :
        m_infra{get_compdom(amrex::IntVect{AMREX_D_DECL(10, 10, 10)},
                            amrex::IntVect{AMREX_D_DECL(10, 10, 10)})}
    {
        // Parameters initialized here so that different tests can have different parameters
        Io::Parameters parameters;
        // particle settings
        double charge{1};
        double mass{1};

        parameters.set("Particle.species0.charge", charge);
        parameters.set("Particle.species0.mass", mass);

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

TEST_F(EvaluateEFieldTest, NullTest)
{
    // Adding particle to one cell
    int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), m_infra, weights,
                                              positions);

    // (default) charge correctly transferred from addSingleParticles
    EXPECT_EQ(1, m_particleGroup[0]->get_charge());

    // Parse analytical fields and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
    // Ey, Ez
    amrex::Array<std::string, 3> const analyticalFuncE{"0.0", "0.0", "0.0"};

    int const nVar{AMREX_SPACEDIM + 1}; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcE;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncE[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcE[i] = parser[i].compile<nVar>();
    }

    DeRhamField<Grid::primal, Space::edge> E(m_deRham, funcE);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in
    // Particle iteration ... over one particle. Hopefully.

    bool particleLoopRun{false};
    for (auto& particleGrid : *m_particleGroup[s_spec])
    {
        particleLoopRun = true;

        long const np{particleGrid.numParticles()};
        EXPECT_EQ(1, np); // Only one particle added by addSingleParticles

        auto const& particles{particleGrid.GetArrayOfStructs()};
        auto const* const partData{particles().data()};

        amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> eArray;
        for (int cc{0}; cc < s_vDim; cc++) eArray[cc] = (E.m_data[cc])[particleGrid].array();

        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position;
        for (unsigned int d{0}; d < AMREX_SPACEDIM; ++d)
        {
            position[d] = partData[0].pos(d);
        }
        ParticleMeshCoupling::SplineBase<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.inv_cell_size_array());

        amrex::GpuArray<amrex::Real, s_vDim> efield =
            spline.template eval_spline_field<Field::PrimalOneForm>(eArray);

        EXPECT_EQ(efield[xDir], 0);
        EXPECT_EQ(efield[yDir], 0);
        EXPECT_EQ(efield[zDir], 0);
    }
    ASSERT_TRUE(particleLoopRun);
}

TEST_F(EvaluateEFieldTest, SingleParticleNode)
{
    // Adding particle to one cell
    int const numParticles{1};
    // Particle at position (0,0,0) in box (0,0,0)
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(0, 0, 0)}}};
    EXPECT_EQ(*m_infra.m_geom.ProbLo(), 0.0);
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), m_infra, weights,
                                              positions);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in

    // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
    // Ey, Ez
    amrex::Array<std::string, 3> const analyticalFuncE{"1.0", "1.0", "1.0"};

    int const nVar{AMREX_SPACEDIM + 1}; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcE;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncE[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcE[i] = parser[i].compile<nVar>();
    }

    DeRhamField<Grid::primal, Space::edge> E(m_deRham, funcE);

    for (auto& particleGrid : *m_particleGroup[s_spec])
    {
        long const np{particleGrid.numParticles()};
        EXPECT_EQ(numParticles, np);

        update_e_field_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(particleGrid, E, m_infra);
    }
}

TEST_F(EvaluateEFieldTest, SingleParticleMiddle)
{
    // Adding particle to one cell
    int const numParticles{1};
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
    amrex::Array<std::string, 3> const analyticalFuncE{"1.0", "1.0", "1.0"};

    int const nVar{AMREX_SPACEDIM + 1}; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcE;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncE[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcE[i] = parser[i].compile<nVar>();
    }

    DeRhamField<Grid::primal, Space::edge> E(m_deRham, funcE);

    for (auto& particleGrid : *m_particleGroup[0])
    {
        long const np{particleGrid.numParticles()};
        EXPECT_EQ(numParticles, np);

        update_e_field_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(particleGrid, E, m_infra);
    }
}

TEST_F(EvaluateEFieldTest, SingleParticleUnevenNodeSplit)
{
    // Adding particle to one cell
    int const numParticles{1};
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
    amrex::Array<std::string, 3> const analyticalFuncE{"1.0", "1.0", "1.0"};

    int const nVar{AMREX_SPACEDIM + 1}; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcE;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncE[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcE[i] = parser[i].compile<nVar>();
    }

    DeRhamField<Grid::primal, Space::edge> E(m_deRham, funcE);

    for (auto& particleGrid : *m_particleGroup[0])
    {
        long const np{particleGrid.numParticles()};
        EXPECT_EQ(numParticles, np);

        update_e_field_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(particleGrid, E, m_infra);
    }
}

TEST_F(EvaluateEFieldTest, DoubleParticleSeparate)
{
    int const numParticles{2};
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
    amrex::Array<std::string, 3> const analyticalFuncE{"1.0", "1.0", "1.0"};

    int const nVar{AMREX_SPACEDIM + 1}; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcE;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncE[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcE[i] = parser[i].compile<nVar>();
    }

    DeRhamField<Grid::primal, Space::edge> E(m_deRham, funcE);

    for (auto& particleGrid : *m_particleGroup[0])
    {
        long const np{particleGrid.numParticles()};
        EXPECT_EQ(numParticles, np);

        update_e_field_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(particleGrid, E, m_infra);
    }
}

TEST_F(EvaluateEFieldTest, DoubleParticleOverlap)
{
    int const numParticles{2};
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
    amrex::Array<std::string, 3> const analyticalFuncE{"1.0", "1.0", "1.0"};

    int const nVar{AMREX_SPACEDIM + 1}; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcE;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncE[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcE[i] = parser[i].compile<nVar>();
    }

    DeRhamField<Grid::primal, Space::edge> E(m_deRham, funcE);

    for (auto& particleGrid : *m_particleGroup[0])
    {
        long const np{particleGrid.numParticles()};
        EXPECT_EQ(numParticles, np);

        update_e_field_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(particleGrid, E, m_infra);
    }
}

TEST_F(EvaluateEFieldTest, TestingForLiterallyAnythingOtherThanUnity)
{
    GTEST_SKIP() << "Such advanced tests have not yet been implemented!";
}

TEST_F(EvaluateEFieldTest, Scaling)
{
    Io::Parameters parameters{};

    /* Initialize the infrastructure with cell sizes different from 1*/
    amrex::IntVect const nCell{AMREX_D_DECL(8, 4, 2)};
    amrex::IntVect const maxGridSize{AMREX_D_DECL(8, 4, 2)};
    int const hodgeDegree{2};

    ComputationalDomain infra = get_compdom(nCell, maxGridSize);

    // Initialize the De Rham Complex
    auto deRham{std::make_shared<FDDeRhamComplex>(infra, hodgeDegree, s_maxSplineDegree,
                                                  HodgeScheme::FDHodge)};

    // particle groups
    m_particleGroup.resize(s_numSpec);
    m_particleGroup[0] = std::make_unique<ParticleGroups<s_vDim>>(0, infra);

    // Adding particle to one cell
    int const numParticles{1};
    auto dx = infra.geometry().CellSizeArray();
    // Add particle in the middle of final cell to check periodic boundary conditions
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(infra.geometry().ProbHi(xDir) - 1.25 * dx[xDir],
                       infra.geometry().ProbHi(yDir) - 1.25 * dx[yDir],
                       infra.geometry().ProbHi(zDir) - 1.25 * dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), infra, weights, positions);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in

    // Parse analytical fields and and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
    // Ey, Ez
    amrex::Array<std::string, 3> const analyticalFuncE{"1.0", "1.0", "1.0"};

    int const nVar{AMREX_SPACEDIM + 1}; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcE;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncE[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcE[i] = parser[i].compile<nVar>();
    }

    DeRhamField<Grid::primal, Space::edge> E(deRham, funcE);

    for (auto& particleGrid : *m_particleGroup[0])
    {
        long const np{particleGrid.numParticles()};
        EXPECT_EQ(numParticles, np);

        update_e_field_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(particleGrid, E, infra);
    }
}
} // namespace
