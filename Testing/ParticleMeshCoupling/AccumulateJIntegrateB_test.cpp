#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_SplineClass.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;

namespace
{
// When using amrex::ParallelFor you have to create a standalone helper function that does the
// execution on GPU and call that function from the unit test because of how GTest creates tests
// within a TEST_F fixture.
template <Direction pDir, int degX, int degY, int degZ, unsigned int vDim>
void accumulate_j_update_v_c2_parallel_for (
    amrex::ParIterSoA<AMREX_SPACEDIM + vDim + 1, 0>& particleGrid,
    DeRhamField<Grid::primal, Space::face>& B,
    DeRhamField<Grid::dual, Space::face>& J,
    ComputationalDomain& infra,
    amrex::Real weight,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const dx,
    amrex::GpuArray<amrex::Real, 2>& bfields)
{
    long const np{particleGrid.numParticles()};

    auto const partData = particleGrid.GetParticleTile().getParticleTileData();

    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> jA;
    for (int cc = 0; cc < vDim; cc++) jA[cc] = (J.m_data[cc])[particleGrid].array();

    amrex::AsyncArray aaBfields(&bfields, 1);
    auto* bfieldsGPU = aaBfields.data();

    amrex::GpuArray<amrex::Array4<amrex::Real>, int(vDim / 2.5) * 2 + 1> bA;
    for (int cc = 0; cc < (int(vDim / 2.5) * 2 + 1); cc++)
    {
        bA[cc] = (B.m_data[cc])[particleGrid].array();
    }
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{infra.geometry().ProbLoArray()};

    amrex::ParallelFor(np,
                       [=] AMREX_GPU_DEVICE(long pp)
                       {
                           amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> posStart;
                           for (unsigned int d{0}; d < AMREX_SPACEDIM; ++d)
                           {
                               posStart[d] = partData[pp * 0].pos(d);
                           }

                           amrex::Real xEnd = 0;

                           ParticleMeshCoupling::SplineWithPrimitive<degX, degY, degZ> spline{
                               posStart, plo, infra.inv_cell_size_array()};

                           spline.template update_1d_splines<pDir>(
                               xEnd, infra.geometry_data().ProbLo(xDir),
                               1.0 / infra.geometry_data().CellSize(xDir));

                           ParticleMeshCoupling::accumulate_j_integrate_b<pDir>(*bfieldsGPU, spline,
                                                                                weight, dx, bA, jA);
                       });

    aaBfields.copyToHost(&bfields, 1);
}

ComputationalDomain get_compdom ()
{
    std::array<amrex::Real, AMREX_SPACEDIM> const domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    std::array<amrex::Real, AMREX_SPACEDIM> const domainHi{AMREX_D_DECL(10.0, 10.0, 10.0)};
    amrex::IntVect const nCell{AMREX_D_DECL(10, 10, 10)};
    amrex::IntVect const maxGridSize{AMREX_D_DECL(10, 10, 10)};
    std::array<int, AMREX_SPACEDIM> const isPeriodic{AMREX_D_DECL(1, 1, 1)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic);
}

class AccumulateJUpdateVC2Test : public testing::Test
{
protected:
    static int const s_degX{1};
    static int const s_degY{1};
    static int const s_degZ{1};
    inline static int const s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

    inline static int const s_hodgeDegree{2};
    static unsigned int const s_numSpec{1};
    static unsigned int const s_vDim{3};
    static unsigned int const s_spec{0};
    Io::Parameters m_parameters{};

    amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> m_jA;
    amrex::GpuArray<amrex::Array4<amrex::Real>, int(s_vDim / 2.5) * 2 + 1> m_bA;

    ComputationalDomain m_infra;
    std::vector<std::unique_ptr<ParticleGroups<s_vDim>>> m_particleGroup;
    std::shared_ptr<FDDeRhamComplex> m_deRham;

    static Direction const s_pDim{yDir};

    amrex::Real m_weight = 1.0;

    amrex::GpuArray<amrex::Real, 2> m_bfields{0., 0.};
    amrex::GpuArray<amrex::Real, std::max(s_degX, std::max(s_degY, s_degZ)) + 4> m_primitive;

    AccumulateJUpdateVC2Test() : m_infra(get_compdom())
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
        for (int species{0}; species < s_numSpec; species++)
        {
            m_particleGroup[species] = std::make_unique<ParticleGroups<s_vDim>>(species, m_infra);
        }
    }

    void SetUp () override
    {
        if constexpr (AMREX_SPACEDIM != 3)
        {
            GTEST_SKIP() << "This function barely works in 3D, let alone lower dimensions.";
        }
    }
};

TEST_F(AccumulateJUpdateVC2Test, NullTest)
{
    // Adding particle to one cell
    unsigned int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), m_infra, weights,
                                              positions);

    // (default) charge correctly transferred from addSingleParticles
    EXPECT_EQ(1, m_particleGroup[0]->get_charge());

    amrex::Array<std::string, 3> const analyticalFuncB = {"0.0", "0.0", "0.0"};

    amrex::Array<std::string, 3> const analyticalFuncJ = {"0.0", "0.0", "0.0"};

    int const nVar{4}; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, AMREX_SPACEDIM> funcB;
    amrex::Array<amrex::ParserExecutor<nVar>, AMREX_SPACEDIM> funcJ;
    amrex::Array<amrex::Parser, 6> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncB[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        funcB[i] = parser[i].compile<4>();
    }

    for (int i{0}; i < 3; ++i)
    {
        parser[i + 3].define(analyticalFuncJ[i]);
        parser[i + 3].registerVariables({"x", "y", "z", "t"});
        funcJ[i] = parser[i + 3].compile<4>();
    }

    DeRhamField<Grid::primal, Space::face> B(m_deRham, funcB);

    DeRhamField<Grid::dual, Space::face> J(m_deRham, funcJ);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in
    // Particle iteration ... over one particle. Hopefully.

    for (auto& particleGrid : *m_particleGroup[s_spec])
    {
        long const np{particleGrid.numParticles()};
        EXPECT_EQ(1, np); // Only one particle added by addSingleParticles

        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        accumulate_j_update_v_c2_parallel_for<s_pDim, s_degX, s_degY, s_degZ, s_vDim>(
            particleGrid, B, J, m_infra, m_weight, m_infra.cell_size_array(), bfields);

        EXPECT_EQ(bfields[0], 0);
        EXPECT_EQ(bfields[1], 0);

        // Expect all nodes to be 0
        CHECK_FIELD((J.m_data[xDir]).array(particleGrid), particleGrid.validbox(), {}, {}, 0);
        CHECK_FIELD((J.m_data[yDir]).array(particleGrid), particleGrid.validbox(), {}, {}, 0);
        CHECK_FIELD((J.m_data[zDir]).array(particleGrid), particleGrid.validbox(), {}, {}, 0);
    }
}

TEST_F(AccumulateJUpdateVC2Test, SingleParticleMiddle)
{
    // Adding particle to one cell
    int const numParticles{1};
    auto dx = m_infra.geometry().CellSizeArray();
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(m_infra.m_geom.ProbHi(xDir) - 5.5 * dx[xDir],
                       m_infra.m_geom.ProbHi(yDir) - 5.5 * dx[yDir],
                       m_infra.m_geom.ProbHi(zDir) - 5.5 * dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), m_infra, weights,
                                              positions);

    // (default) charge correctly transferred from addSingleParticles
    EXPECT_EQ(1, m_particleGroup[0]->get_charge());

    amrex::Array<std::string, 3> const analyticalFuncB = {"1.0", "1.0", "1.0"};

    amrex::Array<std::string, 3> const analyticalFuncJ = {"1.0", "1.0", "1.0"};

    int const nVar{4}; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, AMREX_SPACEDIM> funcB;
    amrex::Array<amrex::ParserExecutor<nVar>, AMREX_SPACEDIM> funcJ;
    amrex::Array<amrex::Parser, 6> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncB[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        funcB[i] = parser[i].compile<4>();
    }

    for (int i{0}; i < 3; ++i)
    {
        parser[i + 3].define(analyticalFuncJ[i]);
        parser[i + 3].registerVariables({"x", "y", "z", "t"});
        funcJ[i] = parser[i + 3].compile<4>();
    }

    DeRhamField<Grid::primal, Space::face> B(m_deRham, funcB);

    DeRhamField<Grid::dual, Space::face> J(m_deRham, funcJ);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in
    // Particle iteration ... over one particle. Hopefully.

    for (auto& particleGrid : *m_particleGroup[s_spec])
    {
        long const np{particleGrid.numParticles()};
        EXPECT_EQ(1, np); // Only one particle added by addSingleParticles

        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        accumulate_j_update_v_c2_parallel_for<s_pDim, s_degX, s_degY, s_degZ, s_vDim>(
            particleGrid, B, J, m_infra, m_weight, m_infra.cell_size_array(), bfields);

        EXPECT_NEAR(bfields[0], -4.5, 1e-15);
        EXPECT_NEAR(bfields[1], -4.5, 1e-15);

        CHECK_FIELD(
            (J.m_data[s_pDim]).array(particleGrid), particleGrid.validbox(),
            // Expect the eight nearest nodes (4/5, 4/5, 4/5) to be non-zero
            {[] (AMREX_D_DECL(int a, int b, int c))
             { return AMREX_D_TERM((a == 4 || a == 5), &&b == 4, &&(c == 4 || c == 5)); },
             [] (AMREX_D_DECL(int a, int b, int c))
             { return AMREX_D_TERM((a == 4 || a == 5), &&b <= 3, &&(c == 4 || c == 5)); }},
            // getting an eight of the particle weight times the primitive, plus the original 1
            {1 - 1. / 8, 1 - 0.25},
            // with the remaining entries being 1
            1);
        CHECK_FIELD((J.m_data[(s_pDim + 1) % 3]).array(particleGrid), particleGrid.validbox(), {},
                    {}, 1);
        CHECK_FIELD((J.m_data[(s_pDim + 2) % 3]).array(particleGrid), particleGrid.validbox(), {},
                    {}, 1);
    }
}

TEST_F(AccumulateJUpdateVC2Test, SingleParticleUnevenNodeSplit)
{
    // Adding particle to one cell
    int const numParticles{1};
    auto dx = m_infra.geometry().CellSizeArray();
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(m_infra.m_geom.ProbHi(xDir) - 5.25 * dx[xDir],
                       m_infra.m_geom.ProbHi(yDir) - 5.25 * dx[yDir],
                       m_infra.m_geom.ProbHi(zDir) - 5.25 * dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup[0].get(), m_infra, weights,
                                              positions);

    // (default) charge correctly transferred from addSingleParticles
    EXPECT_EQ(1, m_particleGroup[0]->get_charge());

    amrex::Array<std::string, 3> const analyticalFuncB = {"1.0", "1.0", "1.0"};

    amrex::Array<std::string, 3> const analyticalFuncJ = {"1.0", "1.0", "1.0"};

    int const nVar{4}; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, AMREX_SPACEDIM> funcB;
    amrex::Array<amrex::ParserExecutor<nVar>, AMREX_SPACEDIM> funcJ;
    amrex::Array<amrex::Parser, 6> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncB[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        funcB[i] = parser[i].compile<4>();
    }

    for (int i{0}; i < 3; ++i)
    {
        parser[i + 3].define(analyticalFuncJ[i]);
        parser[i + 3].registerVariables({"x", "y", "z", "t"});
        funcJ[i] = parser[i + 3].compile<4>();
    }

    DeRhamField<Grid::primal, Space::face> B(m_deRham, funcB);

    DeRhamField<Grid::dual, Space::face> J(m_deRham, funcJ);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in
    // Particle iteration ... over one particle. Hopefully.

    for (auto& particleGrid : *m_particleGroup[s_spec])
    {
        long const np{particleGrid.numParticles()};
        EXPECT_EQ(1, np); // Only one particle added by addSingleParticles

        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        accumulate_j_update_v_c2_parallel_for<s_pDim, s_degX, s_degY, s_degZ, s_vDim>(
            particleGrid, B, J, m_infra, m_weight, m_infra.geometry().CellSizeArray(), bfields);

        EXPECT_NEAR(bfields[0], -4.75, 1e-15);
        EXPECT_NEAR(bfields[1], -4.75, 1e-15);

        CHECK_FIELD(
            (J.m_data[s_pDim]).array(particleGrid), particleGrid.validbox(),
            // Expect the eight nearest nodes (4/5, 4/5, 4/5) to be non-zero
            {[] (AMREX_D_DECL(int a, int b, int c))
             { return AMREX_D_TERM(a == 4, &&b == 4, &&c == 4); },
             [] (AMREX_D_DECL(int a, int b, int c))
             { return AMREX_D_TERM(a == 4, &&b <= 3, &&c == 4); },
             [] (AMREX_D_DECL(int a, int b, int c))
             { return AMREX_D_TERM(a == 5, &&b == 4, &&c == 5); },
             [] (AMREX_D_DECL(int a, int b, int c))
             { return AMREX_D_TERM(a == 5, &&b <= 3, &&c == 5); },
             [] (AMREX_D_DECL(int a, int b, int c))
             { return AMREX_D_TERM((a == 4 || a == 5), &&b == 4, &&(c == 4 || c == 5) && c != a); },
             [] (AMREX_D_DECL(int a, int b, int c))
             {
                 return AMREX_D_TERM((a == 4 || a == 5), &&b <= 3, &&(c == 4 || c == 5) && c != a);
             }},
            // getting an eight of the particle weight times the primitive, plus the original 1
            {1 - 3. / 64, 1 - 1. / 16, 1 - 27. / 64, 1 - 9. / 16, 1 - 9. / 64, 1 - 3. / 16},
            //{},{},
            // with the remaining entries being 1
            1);
        CHECK_FIELD((J.m_data[(s_pDim + 1) % 3]).array(particleGrid), particleGrid.validbox(), {},
                    {}, 1);
        CHECK_FIELD((J.m_data[(s_pDim + 2) % 3]).array(particleGrid), particleGrid.validbox(), {},
                    {}, 1);
    }
}

TEST_F(AccumulateJUpdateVC2Test, DoubleParticleSeparate)
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

    // (default) charge correctly transferred from addSingleParticles
    EXPECT_EQ(1, m_particleGroup[0]->get_charge());

    amrex::Array<std::string, 3> const analyticalFuncB = {"1.0", "1.0", "1.0"};

    amrex::Array<std::string, 3> const analyticalFuncJ = {"1.0", "1.0", "1.0"};

    int const nVar{4}; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, AMREX_SPACEDIM> funcB;
    amrex::Array<amrex::ParserExecutor<nVar>, AMREX_SPACEDIM> funcJ;
    amrex::Array<amrex::Parser, 6> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncB[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        funcB[i] = parser[i].compile<4>();
    }

    for (int i{0}; i < 3; ++i)
    {
        parser[i + 3].define(analyticalFuncJ[i]);
        parser[i + 3].registerVariables({"x", "y", "z", "t"});
        funcJ[i] = parser[i + 3].compile<4>();
    }

    DeRhamField<Grid::primal, Space::face> B(m_deRham, funcB);

    DeRhamField<Grid::dual, Space::face> J(m_deRham, funcJ);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in
    // Particle iteration ... over one particle. Hopefully.

    for (auto& particleGrid : *m_particleGroup[s_spec])
    {
        long const np{particleGrid.numParticles()};
        EXPECT_EQ(2, np); // Only one particle added by addSingleParticles

        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        accumulate_j_update_v_c2_parallel_for<s_pDim, s_degX, s_degY, s_degZ, s_vDim>(
            particleGrid, B, J, m_infra, m_weight, m_infra.geometry().CellSizeArray(), bfields);

        EXPECT_EQ(bfields[0], 0);
        EXPECT_EQ(bfields[1], 0);

        // Expect all nodes to be 1
        CHECK_FIELD((J.m_data[s_pDim]).array(particleGrid), particleGrid.validbox(), {}, {}, 1);
        CHECK_FIELD((J.m_data[(s_pDim + 1) % 3]).array(particleGrid), particleGrid.validbox(), {},
                    {}, 1);
        CHECK_FIELD((J.m_data[(s_pDim + 2) % 3]).array(particleGrid), particleGrid.validbox(), {},
                    {}, 1);
    }
}

TEST_F(AccumulateJUpdateVC2Test, DoubleParticleOverlap)
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

    // (default) charge correctly transferred from addSingleParticles
    EXPECT_EQ(1, m_particleGroup[0]->get_charge());

    amrex::Array<std::string, 3> const analyticalFuncB = {"1.0", "1.0", "1.0"};

    amrex::Array<std::string, 3> const analyticalFuncJ = {"1.0", "1.0", "1.0"};

    int const nVar{4}; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, AMREX_SPACEDIM> funcB;
    amrex::Array<amrex::ParserExecutor<nVar>, AMREX_SPACEDIM> funcJ;
    amrex::Array<amrex::Parser, 6> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncB[i]);
        parser[i].registerVariables({"x", "y", "z", "t"});
        funcB[i] = parser[i].compile<4>();
    }

    for (int i{0}; i < 3; ++i)
    {
        parser[i + 3].define(analyticalFuncJ[i]);
        parser[i + 3].registerVariables({"x", "y", "z", "t"});
        funcJ[i] = parser[i + 3].compile<4>();
    }

    DeRhamField<Grid::primal, Space::face> B(m_deRham, funcB);

    DeRhamField<Grid::dual, Space::face> J(m_deRham, funcJ);

    m_particleGroup[0]->Redistribute(); // assign particles to the tile they are in
    // Particle iteration ... over one particle. Hopefully.

    for (auto& particleGrid : *m_particleGroup[s_spec])
    {
        long const np{particleGrid.numParticles()};
        EXPECT_EQ(2, np); // Only one particle added by addSingleParticles

        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        accumulate_j_update_v_c2_parallel_for<s_pDim, s_degX, s_degY, s_degZ, s_vDim>(
            particleGrid, B, J, m_infra, m_weight, m_infra.geometry().CellSizeArray(), bfields);

        EXPECT_EQ(bfields[0], 0);
        EXPECT_EQ(bfields[1], 0);

        // Expect all nodes to be 1
        CHECK_FIELD((J.m_data[s_pDim]).array(particleGrid), particleGrid.validbox(), {}, {}, 1);
        CHECK_FIELD((J.m_data[(s_pDim + 1) % 3]).array(particleGrid), particleGrid.validbox(), {},
                    {}, 1);
        CHECK_FIELD((J.m_data[(s_pDim + 2) % 3]).array(particleGrid), particleGrid.validbox(), {},
                    {}, 1);
    }
}

} // namespace
