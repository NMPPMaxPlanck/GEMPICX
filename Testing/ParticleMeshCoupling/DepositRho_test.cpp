/** Testing for deposit_rho function
 *  \todo: Consider mocking particles.
 */

#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;

namespace
{
// When using amrex::ParallelFor you have to create a standalone helper function that does the
// execution on GPU and call that function from the unit test because of how GTest creates tests
// within a TEST_F fixture.
template <int vDim, int degX, int degY, int degZ>
void update_rho_parallel_for (amrex::ParIterSoA<AMREX_SPACEDIM + vDim + 1, 0>& particleGrid,
                              ComputationalDomain& infra,
                              amrex::MultiFab& rho,
                              amrex::Real charge)
{
    long const np{particleGrid.numParticles()};
    auto& tile = particleGrid.GetParticleTile();
    // note 2025-10-01:
    // Ideally we would use the ParticleSpecies->get_data_indices method,
    // but we don't have access to a ParticleSpecies object.
    auto const partData = tile.getParticleTileData();
    amrex::Real* xx = nullptr;
    amrex::Real* yy = nullptr;
    amrex::Real* zz = nullptr;
    xx = particleGrid.GetStructOfArrays().GetRealData(0).data();
    if (AMREX_SPACEDIM > 1)
    {
        yy = particleGrid.GetStructOfArrays().GetRealData(1).data();
    }
    if (AMREX_SPACEDIM > 2)
    {
        zz = particleGrid.GetStructOfArrays().GetRealData(2).data();
    }
    auto const weight{particleGrid.GetStructOfArrays().GetRealData(AMREX_SPACEDIM + vDim).data()};
    amrex::Array4<amrex::Real> const& rhoarr{rho[particleGrid].array()};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo{infra.geometry().ProbLoArray()};

    amrex::ParallelFor(np,
                       [=] AMREX_GPU_DEVICE(long pp)
                       {
                           amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position;
                           AMREX_D_EXPR(position[xDir] = xx[pp], position[yDir] = yy[pp],
                                        position[zDir] = zz[pp]);
                           ParticleMeshCoupling::SplineBase<degX, degY, degZ> spline(
                               position, plo, infra.inv_cell_size_array());
                           // Needs at least max(degX, degY, degZ) ghost cells
                           ParticleMeshCoupling::deposit_rho(rhoarr, spline, charge * weight[pp]);
                       });
    amrex::Gpu::Device::synchronize();
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

// Test fixture. Sets up clean environment before each test.
class DepositRhoTest : public testing::Test
{
protected:
    // Degree of splines in each direction
    static int const s_degX{1};
    // static const int degY{AMREX_D_PICK(0, 1, 1)};
    // static const int degZ{AMREX_D_PICK(0, 0, 1)};
    static int const s_degY{1};
    static int const s_degZ{1};

    // Number of species (second species only used for DoubleParticleMultipleSpecies)
    static int const s_numSpec{2};
    // Number of velocity dimensions.
    static int const s_vDim{0};
    // Number of ghost cells in mesh
    int const m_nghost{Gempic::Test::Utils::init_n_ghost(s_degX, s_degY, s_degZ)};
    amrex::IntVect const m_nghosts{AMREX_D_DECL(m_nghost, m_nghost, m_nghost)};
    amrex::IntVect const m_dstNGhosts{AMREX_D_DECL(0, 0, 0)};

    Io::Parameters m_params{};

    amrex::Array<amrex::Real, s_numSpec> m_charge{1, -1};
    amrex::Array<amrex::Real, s_numSpec> m_mass{1, 0.1};

    ComputationalDomain m_infra;
    std::vector<std::unique_ptr<ParticleSpecies<s_vDim>>> m_particles;
    std::shared_ptr<FDDeRhamComplex> m_deRham;
    std::unique_ptr<DeRhamField<Grid::dual, Space::cell>> m_rhoPtr;

    DepositRhoTest() : m_infra{get_compdom()}
    {
        // particle settings
        m_params.set("Particle.species0.charge", m_charge[0]);
        m_params.set("Particle.species0.mass", m_mass[0]);
        m_params.set("Particle.species1.charge", m_charge[1]);
        m_params.set("Particle.species1.mass", m_mass[1]);

        int const hodgeDegree{2};
        int const maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

        // Initialize the De Rham Complex
        m_deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, maxSplineDegree);
        m_rhoPtr = std::make_unique<DeRhamField<Grid::dual, Space::cell>>(m_deRham);

        // particles
        m_particles.resize(s_numSpec);
        for (int spec{0}; spec < s_numSpec; spec++)
        {
            m_particles[spec] = std::make_unique<ParticleSpecies<s_vDim>>(spec, m_infra);
        }
    }
};

/** Single particle tests. The only reason most of these maneuvres are necessary is because of
 *  amrex::Array4<amrex::Real> const& rhoarr{rho_ptr->data[particleGrid].array()};
 *  which is required for the connection between MultiFab rho and deposit_rho function. This in
 *  turn requires the particle iterator, which means actual particles must be added, instead of
 *  simply supplying positions directly.
 */

// Adds a particle with 0 weight. Checks that rho is unchanged.
TEST_F(DepositRhoTest, NullTest)
{
    // Adding particle to one cell
    int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};

    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particles[0].get(), m_infra, weights, positions);

    // (default) charge correctly transferred from Gempic::TestUtils::addSingleParticles
    EXPECT_EQ(1, m_particles[0]->get_charge());

    // rho_ptr->data unchanged by Gempic::Test::Utils::addSingleParticles
    EXPECT_EQ(0, m_rhoPtr->m_data.norm2(0, m_infra.m_geom.periodicity()));

    m_particles[0]->Redistribute(); // assign particles to the tile they are in
    amrex::Gpu::Device::synchronize();
    // Particle iteration ... over one particle.
    bool particleLoopRun{false};
    for (auto& particleGrid : *m_particles[0])
    {
        particleLoopRun = true;

        long const np{particleGrid.numParticles()};
        EXPECT_EQ(numParticles,
                  np); // Only one particle added by Gempic::Test::Utils::addSingleParticles

        auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
        auto const ii = m_particles[0]->get_data_indices();
        EXPECT_EQ(1, ptd.rdata(ii.m_iweight)[0]);

        amrex::Array4<amrex::Real> const& rhoarr{m_rhoPtr->m_data[particleGrid].array()};
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position;
        AMREX_D_EXPR(position[xDir] = ptd.rdata(ii.m_iposx)[0],
                     position[yDir] = ptd.rdata(ii.m_iposy)[0],
                     position[zDir] = ptd.rdata(ii.m_iposz)[0]);

        ParticleMeshCoupling::SplineBase<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

        ParticleMeshCoupling::deposit_rho(rhoarr, spline, 0);
    }
    ASSERT_TRUE(particleLoopRun);

    EXPECT_EQ(0, m_rhoPtr->m_data.norm2(0, m_infra.m_geom.periodicity()));
}

// Adds one particle exactly between two nodes
TEST_F(DepositRhoTest, SingleParticleMiddle)
{
    ASSERT_EQ(0, m_rhoPtr->m_data.norm2(0, m_infra.m_geom.periodicity()));
    int const numParticles{1};
    auto dx = m_infra.geometry().CellSizeArray();

    // Add particle in the middle of final cell to check periodic boundary conditions
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(m_infra.m_geom.ProbHi(xDir) - 0.5 * dx[xDir],
                       m_infra.m_geom.ProbHi(yDir) - 0.5 * dx[yDir],
                       m_infra.m_geom.ProbHi(zDir) - 0.5 * dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> weights{3};
    // Expect the 2^AMREX_SPACEDIM nearest nodes of rho_ptr->dataarr (9/10, 9/10, 9/10) to be
    // non-zero and receiving 1/2^AMREX_SPACEDIM the weight of the particle (3)
    auto const charge{m_particles[0]->get_charge()};
    amrex::Real expectedVal{charge * weights[0] / m_infra.cell_volume() * pow(0.5, AMREX_SPACEDIM)};

    Gempic::Test::Utils::add_single_particles(m_particles[0].get(), m_infra, weights, positions);
    m_particles[0]->WritePlotFile("ptest", "particles");
    // Particle iteration ... over one particle.
    for (auto& particleGrid : *m_particles[0])
    {
        long const np{particleGrid.numParticles()};
        EXPECT_EQ(numParticles, np); // Only one particle added

        update_rho_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(
            particleGrid, m_infra, m_rhoPtr->m_data, m_particles[0]->get_charge());

        amrex::Gpu::streamSynchronize();
        // Expect the eight nearest nodes of rho_ptr->dataarr (9/10, 9/10, 9/10) to be non-zero and
        // receiving 1/8 the weight of the particle (3)
        CHECK_FIELD(
            m_rhoPtr->m_data[particleGrid].array(), particleGrid.validbox(),
            // Expect the eight nearest nodes of rho_ptr->dataarr (9/10, 9/10, 9/10) to be non-zero
            {[] (AMREX_D_DECL(int a, int b, int c))
             { return AMREX_D_TERM(a >= 9, &&b >= 9, &&c >= 9); }},
            // and receiving 1/8 the weight of the particle (3)
            {expectedVal},
            // with the remaining entries being 0
            0);
    }
    m_rhoPtr->post_particle_loop_sync();

    // Maximum occurs evenly split between 2^AMREX_SPACEDIM nodes. The sum is still 1.
    EXPECT_EQ(expectedVal, m_rhoPtr->m_data.norm0());
    EXPECT_EQ(weights[0], m_rhoPtr->m_data.norm1(0, m_infra.m_geom.periodicity()));
}

// Adds one particle closer to on node than the other
TEST_F(DepositRhoTest, SingleParticleUnevenNodeSplit)
{
    ASSERT_EQ(0, m_rhoPtr->m_data.norm2(0, m_infra.m_geom.periodicity()));
    int const numParticles{1};
    auto dx = m_infra.geometry().CellSizeArray();

    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(m_infra.m_geom.ProbLo(xDir) + 0.25 * dx[xDir],
                       m_infra.m_geom.ProbLo(yDir) + 0.25 * dx[yDir],
                       m_infra.m_geom.ProbLo(zDir) + 0.25 * dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> weights{1};

    Gempic::Test::Utils::add_single_particles(m_particles[0].get(), m_infra, weights, positions);
    // Particle iteration ... over one particle.
    for (auto& particleGrid : *m_particles[0])
    {
        long const np{particleGrid.numParticles()};
        EXPECT_EQ(numParticles, np); // Only one particle added

        update_rho_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(
            particleGrid, m_infra, m_rhoPtr->m_data, m_particles[0]->get_charge());

        amrex::Gpu::streamSynchronize();
        // Expect the 2^AMREX_SPACEDIM nearest nodes of rho_ptr->dataarr (0/1, 0/1, 0/1) to be
        // non-zero and  0 nodes receiving (3/4) and 1 nodes receiving (1/4) the weight of the
        // particle (1)
        CHECK_FIELD(m_rhoPtr->m_data[particleGrid].array(), particleGrid.validbox(),
                    // Expect the 2^SPACEDIM nearest nodes of rho_ptr->dataarr (0/1, 0/1, 0/1) to be
                    // non-zero
                    {[] (AMREX_D_DECL(int a, int b, int c))
                     { return AMREX_D_TERM(a == 0, &&b == 0, &&c == 0); },
                     [] (AMREX_D_DECL(int a, int b, int c))
                     { return AMREX_D_TERM(a == 1, &&b == 1, &&c == 1); },
                     [] (AMREX_D_DECL(int a, int b, int c)) { return GEMPIC_D_ADD(a, b, c) == 1; },
                     [] (AMREX_D_DECL(int a, int b, int c))
                     { return (GEMPIC_D_ADD(a * a, b * b, c * c) == 2); }},
                    // and  0 nodes receiving (3/4) and 1 nodes receiving (1/4) the weight of the
                    // particle (1)
                    {pow(0.75, AMREX_SPACEDIM), pow(0.25, AMREX_SPACEDIM),
                     0.25 * pow(0.75, AMREX_SPACEDIM - 1), pow(0.25, AMREX_SPACEDIM - 1) * 0.75},
                    // with the remaining entries being 0
                    0);
    }
    m_rhoPtr->post_particle_loop_sync();

    // Maximum occurs on node (0, 0, 0) with value (3/4)^AMREX_SPACEDIM. The sum is still 1.
    EXPECT_EQ(pow(0.75, AMREX_SPACEDIM), m_rhoPtr->m_data.norm0());
    EXPECT_EQ(1, m_rhoPtr->m_data.norm1(0, m_infra.m_geom.periodicity()));
}

// Adds two particles in different cells to check that they don't interfere with each other
TEST_F(DepositRhoTest, DoubleParticleSeparate)
{
    int const numParticles{2};
    auto dx = m_infra.geometry().CellSizeArray();
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(0, 0, 0)},
         {AMREX_D_DECL(m_infra.m_geom.ProbLo(xDir) + 5.5 * dx[xDir],
                       m_infra.m_geom.ProbLo(yDir) + 5.5 * dx[yDir],
                       m_infra.m_geom.ProbLo(zDir) + 5.5 * dx[zDir])}}};

    amrex::Array<amrex::Real, numParticles> weights{1, 3};
    amrex::Real expectedValA{1}, expectedValB{3 * pow(0.5, AMREX_SPACEDIM)};

    Gempic::Test::Utils::add_single_particles(m_particles[0].get(), m_infra, weights, positions);

    // Particle iteration ... over two distant particles.
    for (auto& particleGrid : *m_particles[0])
    {
        long const np{particleGrid.numParticles()};
        EXPECT_EQ(numParticles, np); // Two particles added

        update_rho_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(
            particleGrid, m_infra, m_rhoPtr->m_data, m_particles[0]->get_charge());

        amrex::Gpu::streamSynchronize();
        // See SingleParticle test for explanation of expectations
        CHECK_FIELD(m_rhoPtr->m_data[particleGrid].array(), particleGrid.validbox(),
                    {[] (AMREX_D_DECL(int a, int b, int c))
                     { return AMREX_D_TERM(a == 0, &&b == 0, &&c == 0); },
                     [] (AMREX_D_DECL(int a, int b, int c))
                     {
                         return AMREX_D_TERM((a == 5 || a == 6), &&(b == 5 || b == 6),
                                             &&(c == 5 || c == 6));
                     }},
                    {expectedValA, expectedValB},
                    // with the remaining entries being 0
                    0);
    }
    m_rhoPtr->post_particle_loop_sync();

    // The maximum expectedVal depends on the dimension on the problem
    EXPECT_EQ(std::max(expectedValA, expectedValB), m_rhoPtr->m_data.norm0());
    // Total charge added is the sum of each weight*charge, here 1 + 3
    EXPECT_EQ(4, m_rhoPtr->m_data.norm1(0, m_infra.m_geom.periodicity()));
}

// Adds particles in the same cell to check that they add up correctly
TEST_F(DepositRhoTest, DoubleParticleOverlap)
{
    int const numParticles{2};
    auto dx = m_infra.geometry().CellSizeArray();
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(0, 0, 0)},
         {AMREX_D_DECL(m_infra.m_geom.ProbLo(xDir) + 0.5 * dx[xDir],
                       m_infra.m_geom.ProbLo(yDir) + 0.5 * dx[yDir],
                       m_infra.m_geom.ProbLo(zDir) + 0.5 * dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> weights{1, 3};

    amrex::Real expectedValA{1 + 3 * pow(0.5, AMREX_SPACEDIM)};
    amrex::Real expectedValB{3 * pow(0.5, AMREX_SPACEDIM)};

    Gempic::Test::Utils::add_single_particles(m_particles[0].get(), m_infra, weights, positions);

    // Particle iteration ... over two close particles.
    for (auto& particleGrid : *m_particles[0])
    {
        long const np{particleGrid.numParticles()};
        EXPECT_EQ(numParticles, np); // Two particles added

        update_rho_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(
            particleGrid, m_infra, m_rhoPtr->m_data, m_particles[0]->get_charge());

        amrex::Gpu::streamSynchronize();
        // See SingleParticle test for explanation of expectations
        CHECK_FIELD(m_rhoPtr->m_data[particleGrid].array(), particleGrid.validbox(),
                    {[] (AMREX_D_DECL(int a, int b, int c))
                     { return AMREX_D_TERM(a == 0, &&b == 0, &&c == 0); },
                     [] (AMREX_D_DECL(int a, int b, int c))
                     { return AMREX_D_TERM(a <= 1, &&b <= 1, &&c <= 1); }},
                    {expectedValA, expectedValB}, 0);
    }
    m_rhoPtr->post_particle_loop_sync();

    EXPECT_EQ(expectedValA, m_rhoPtr->m_data.norm0());
    EXPECT_EQ(4, m_rhoPtr->m_data.norm1(0, m_infra.m_geom.periodicity()));
}

// Adds particles of different species in the same cell
TEST_F(DepositRhoTest, DoubleParticleMultipleSpecies)
{
    int const numParticles{1};
    auto dx = m_infra.geometry().CellSizeArray();
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> pPos{
        {{AMREX_D_DECL(0, 0, 0)}}};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> ePos{{{AMREX_D_DECL(
        m_infra.m_geom.ProbLo(xDir) + 0.5 * dx[xDir], m_infra.m_geom.ProbLo(yDir) + 0.5 * dx[yDir],
        m_infra.m_geom.ProbLo(zDir) + 0.5 * dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> pWeights{1};
    amrex::Array<amrex::Real, numParticles> eWeights{3};
    int pSpec{0}, eSpec{1};

    Gempic::Test::Utils::add_single_particles(m_particles[pSpec].get(), m_infra, pWeights, pPos);
    Gempic::Test::Utils::add_single_particles(m_particles[eSpec].get(), m_infra, eWeights, ePos);

    auto const pCharge{m_particles[pSpec]->get_charge()};
    auto const eCharge{m_particles[eSpec]->get_charge()};

    amrex::Real expectedValA{pCharge + eCharge * 3 * pow(0.5, AMREX_SPACEDIM)};
    amrex::Real expectedValB{eCharge * 3 * pow(0.5, AMREX_SPACEDIM)};

    for (int spec{0}; spec < s_numSpec; spec++)
    {
        m_particles[spec]->Redistribute(); // assign particles to the tile they are in
        amrex::Gpu::Device::synchronize();
        auto const charge{m_particles[spec]->get_charge()};
        // Particle iteration
        for (auto& particleGrid : *m_particles[spec])
        {
            long const np{particleGrid.numParticles()};
            EXPECT_EQ(numParticles, np); // Two particles added

            update_rho_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(particleGrid, m_infra,
                                                                    m_rhoPtr->m_data, charge);

            if (spec == s_numSpec - 1)
            {
                amrex::Gpu::streamSynchronize();
                // See SingleParticle test for explanation of expectations
                CHECK_FIELD(m_rhoPtr->m_data[particleGrid].array(), particleGrid.validbox(),
                            {[] (AMREX_D_DECL(int a, int b, int c))
                             { return AMREX_D_TERM(a == 0, &&b == 0, &&c == 0); },
                             [] (AMREX_D_DECL(int a, int b, int c))
                             { return AMREX_D_TERM(a <= 1, &&b <= 1, &&c <= 1); }},
                            {expectedValA, expectedValB}, 0);
            }
        }
    }
    m_rhoPtr->post_particle_loop_sync();

    // The maximum expectedVal depends on the dimension on the problem
    EXPECT_EQ(std::max(std::abs(expectedValA), std::abs(expectedValB)), m_rhoPtr->m_data.norm0());

    // Probably not GPU safe. Second argument of sum_unique is bool local, which decides if parallel
    // reduction is done
    EXPECT_EQ(pCharge * pWeights[0] + eCharge * eWeights[0],
              m_rhoPtr->m_data.sum_unique(0, 0, m_infra.m_geom.periodicity()));
}
} // namespace
