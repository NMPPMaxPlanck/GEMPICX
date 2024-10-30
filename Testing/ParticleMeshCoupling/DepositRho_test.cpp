/** Testing for deposit_rho function
 *  \todo: Consider mocking particles.
 */

#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_Config.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;

#define check_field(...) Gempic::Test::Utils::check_field(__FILE__, __LINE__, __VA_ARGS__)

namespace
{
// When using amrex::ParallelFor you have to create a standalone helper function that does the
// execution on GPU and call that function from the unit test because of how GTest creates tests
// within a TEST_F fixture.
template <int vDim, int degX, int degY, int degZ>
void update_rho_parallel_for (amrex::ParIter<0, 0, vDim + 1, 0>& pti,
                              ComputationalDomain& infra,
                              amrex::MultiFab& rho,
                              amrex::Real charge)
{
    const long np{pti.numParticles()};
    const auto& particles{pti.GetArrayOfStructs()};
    const auto partData{particles().data()};
    const auto weight{pti.GetStructOfArrays().GetRealData(vDim).data()};
    amrex::Array4<amrex::Real> const& rhoarr{rho[pti].array()};

    amrex::ParallelFor(np,
                       [=] AMREX_GPU_DEVICE(long pp)
                       {
                           amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
                           for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
                           {
                               position[d] = partData[pp].pos(d);
                           }
                           ParticleMeshCoupling::SplineBase<degX, degY, degZ> spline(
                               position, infra.m_plo, infra.m_dxi);
                           // Needs at least max(degX, degY, degZ) ghost cells
                           ParticleMeshCoupling::gempic_deposit_rho(rhoarr, spline,
                                                                    charge * weight[pp]);
                       });
}

// Test fixture. Sets up clean environment before each test.
class DepositRhoTest : public testing::Test
{
protected:
    // Degree of splines in each direction
    static const int s_degX{1};
    // static const int degY{AMREX_D_PICK(0, 1, 1)};
    // static const int degZ{AMREX_D_PICK(0, 0, 1)};
    static const int s_degY{1};
    static const int s_degZ{1};

    // Number of species (second species only used for DoubleParticleMultipleSpecies)
    static const int s_numSpec{2};
    // Number of velocity dimensions.
    static const int s_vDim{0};
    // Number of ghost cells in mesh
    const int m_nghost{Gempic::Test::Utils::init_n_ghost(s_degX, s_degY, s_degZ)};
    const amrex::IntVect m_nghosts{AMREX_D_DECL(m_nghost, m_nghost, m_nghost)};
    const amrex::IntVect m_dstNGhosts{AMREX_D_DECL(0, 0, 0)};

    Io::Parameters m_params{};

    amrex::Array<amrex::Real, s_numSpec> m_charge{1, -1};
    amrex::Array<amrex::Real, s_numSpec> m_mass{1, 0.1};

    ComputationalDomain m_infra{false};  // "uninitialized" computational domain
    std::vector<std::unique_ptr<ParticleGroups<s_vDim>>> m_particleGroup;
    std::shared_ptr<FDDeRhamComplex> m_deRham;
    // amrex::MultiFab rho;
    std::unique_ptr<DeRhamField<Grid::dual, Space::cell>> m_rhoPtr;

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
        /* Initialize the infrastructure */
        m_infra = ComputationalDomain{};

        const int hodgeDegree{2};
        const int maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

        // Initialize the De Rham Complex
        m_deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, maxSplineDegree);
        m_rhoPtr = std::make_unique<DeRhamField<Grid::dual, Space::cell>>(m_deRham);

        // particles
        m_particleGroup.resize(s_numSpec);
        for (int spec{0}; spec < s_numSpec; spec++)
        {
            m_particleGroup[spec] = std::make_unique<ParticleGroups<s_vDim>>(spec, m_infra);
        }
    }
};

/** Single particle tests. The only reason most of these maneuvres are necessary is because of
 *  amrex::Array4<amrex::Real> const& rhoarr{rho_ptr->data[pti].array()};
 *  which is required for the connection between MultiFab rho and deposit_rho function. This in
 *  turn requires the pti iterator, which means actual particles must be added, instead of
 *  simply supplying positions directly.
 */

// Adds a particle with 0 weight. Checks that rho is unchanged.
TEST_F(DepositRhoTest, NullTest)
{
    // Adding particle to one cell
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};

    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    // (default) charge correctly transferred from Gempic::TestUtils::addSingleParticles
    EXPECT_EQ(1, m_particleGroup[0]->get_charge());

    // rho_ptr->data unchanged by Gempic::Test::Utils::addSingleParticles
    EXPECT_EQ(0, m_rhoPtr->m_data.norm2(0, m_infra.m_geom.periodicity()));

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in
    // Particle iteration ... over one particle.
    bool particleLoopRun{false};
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[0], 0); pti.isValid(); ++pti)
    {
        particleLoopRun = true;

        const long np{pti.numParticles()};
        EXPECT_EQ(numParticles,
                  np);  // Only one particle added by Gempic::Test::Utils::addSingleParticles

        const auto& particles{pti.GetArrayOfStructs()};
        const auto* const partData{particles().data()};
        auto* const weight{pti.GetStructOfArrays().GetRealData(s_vDim).data()};
        // weight correctly transferred from Gempic::Test::Utils::addSingleParticles
        EXPECT_EQ(1, weight[0]);

        amrex::Array4<amrex::Real> const& rhoarr{m_rhoPtr->m_data[pti].array()};
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
        for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
        {
            position[d] = partData[0].pos(d);
        }

        ParticleMeshCoupling::SplineBase<s_degX, s_degY, s_degZ> spline(position, m_infra.m_plo,
                                                                        m_infra.m_dxi);

        ParticleMeshCoupling::gempic_deposit_rho(rhoarr, spline, 0);
    }
    ASSERT_TRUE(particleLoopRun);

    EXPECT_EQ(0, m_rhoPtr->m_data.norm2(0, m_infra.m_geom.periodicity()));
}

// Adds one particle exactly between two nodes
TEST_F(DepositRhoTest, SingleParticleMiddle)
{
    ASSERT_EQ(0, m_rhoPtr->m_data.norm2(0, m_infra.m_geom.periodicity()));
    const int numParticles{1};

    // Add particle in the middle of final cell to check periodic boundary conditions
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(m_infra.m_geom.ProbHi(xDir) - 0.5 * m_infra.m_dx[xDir],
                       m_infra.m_geom.ProbHi(yDir) - 0.5 * m_infra.m_dx[yDir],
                       m_infra.m_geom.ProbHi(zDir) - 0.5 * m_infra.m_dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> weights{3};
    // Expect the 2^GEMPIC_SPACEDIM nearest nodes of rho_ptr->dataarr (9/10, 9/10, 9/10) to be
    // non-zero and receiving 1/2^GEMPIC_SPACEDIM the weight of the particle (3)
    const auto charge{m_particleGroup[0]->get_charge()};
    amrex::Real expectedVal{charge * m_infra.m_dxi[GEMPIC_SPACEDIM] * weights[0] *
                            pow(0.5, GEMPIC_SPACEDIM)};

    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);
    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in
    // Particle iteration ... over one particle.
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[0], 0); pti.isValid(); ++pti)
    {
        const long np{pti.numParticles()};
        EXPECT_EQ(numParticles, np);  // Only one particle added

        update_rho_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(pti, m_infra, m_rhoPtr->m_data,
                                                                m_particleGroup[0]->get_charge());

        // Expect the eight nearest nodes of rho_ptr->dataarr (9/10, 9/10, 9/10) to be non-zero and
        // receiving 1/8 the weight of the particle (3)
        check_field(
            m_rhoPtr->m_data[pti].array(), m_infra.m_nCell.dim3(),
            // Expect the eight nearest nodes of rho_ptr->dataarr (9/10, 9/10, 9/10) to be non-zero
            {[] (AMREX_D_DECL(int a, int b, int c))
             { return AMREX_D_TERM(a >= 9, &&b >= 9, &&c >= 9); }},
            // and receiving 1/8 the weight of the particle (3)
            {expectedVal},
            // with the remaining entries being 0
            0);
    }
    m_rhoPtr->post_particle_loop_sync();

    // Maximum occurs evenly split between 2^GEMPIC_SPACEDIM nodes. The sum is still 1.
    EXPECT_EQ(expectedVal, m_rhoPtr->m_data.norm0());
    EXPECT_EQ(weights[0], m_rhoPtr->m_data.norm1(0, m_infra.m_geom.periodicity()));
}

// Adds one particle closer to on node than the other
TEST_F(DepositRhoTest, SingleParticleUnevenNodeSplit)
{
    ASSERT_EQ(0, m_rhoPtr->m_data.norm2(0, m_infra.m_geom.periodicity()));
    const int numParticles{1};

    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(m_infra.m_geom.ProbLo(xDir) + 0.25 * m_infra.m_dx[xDir],
                       m_infra.m_geom.ProbLo(yDir) + 0.25 * m_infra.m_dx[yDir],
                       m_infra.m_geom.ProbLo(zDir) + 0.25 * m_infra.m_dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> weights{1};

    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);
    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in
    // Particle iteration ... over one particle.
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[0], 0); pti.isValid(); ++pti)
    {
        const long np{pti.numParticles()};
        EXPECT_EQ(numParticles, np);  // Only one particle added

        update_rho_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(pti, m_infra, m_rhoPtr->m_data,
                                                                m_particleGroup[0]->get_charge());

        // Expect the 2^GEMPIC_SPACEDIM nearest nodes of rho_ptr->dataarr (0/1, 0/1, 0/1) to be
        // non-zero and  0 nodes receiving (3/4) and 1 nodes receiving (1/4) the weight of the
        // particle (1)
        check_field(m_rhoPtr->m_data[pti].array(), m_infra.m_nCell.dim3(),
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
                    {pow(0.75, GEMPIC_SPACEDIM), pow(0.25, GEMPIC_SPACEDIM),
                     0.25 * pow(0.75, GEMPIC_SPACEDIM - 1), pow(0.25, GEMPIC_SPACEDIM - 1) * 0.75},
                    // with the remaining entries being 0
                    0);
    }
    m_rhoPtr->post_particle_loop_sync();

    // Maximum occurs on node (0, 0, 0) with value (3/4)^GEMPIC_SPACEDIM. The sum is still 1.
    EXPECT_EQ(pow(0.75, GEMPIC_SPACEDIM), m_rhoPtr->m_data.norm0());
    EXPECT_EQ(1, m_rhoPtr->m_data.norm1(0, m_infra.m_geom.periodicity()));
}

// Adds two particles in different cells to check that they don't interfere with each other
TEST_F(DepositRhoTest, DoubleParticleSeparate)
{
    const int numParticles{2};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(0, 0, 0)},
         {AMREX_D_DECL(m_infra.m_geom.ProbLo(xDir) + 5.5 * m_infra.m_dx[xDir],
                       m_infra.m_geom.ProbLo(yDir) + 5.5 * m_infra.m_dx[yDir],
                       m_infra.m_geom.ProbLo(zDir) + 5.5 * m_infra.m_dx[zDir])}}};

    amrex::Array<amrex::Real, numParticles> weights{1, 3};
    amrex::Real expectedValA{1}, expectedValB{3 * pow(0.5, GEMPIC_SPACEDIM)};

    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in
    // Particle iteration ... over two distant particles.
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[0], 0); pti.isValid(); ++pti)
    {
        const long np{pti.numParticles()};
        EXPECT_EQ(numParticles, np);  // Two particles added

        update_rho_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(pti, m_infra, m_rhoPtr->m_data,
                                                                m_particleGroup[0]->get_charge());

        // See SingleParticle test for explanation of expectations
        check_field(m_rhoPtr->m_data[pti].array(), m_infra.m_nCell.dim3(),
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
    const int numParticles{2};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(0, 0, 0)},
         {AMREX_D_DECL(m_infra.m_geom.ProbLo(xDir) + 0.5 * m_infra.m_dx[xDir],
                       m_infra.m_geom.ProbLo(yDir) + 0.5 * m_infra.m_dx[yDir],
                       m_infra.m_geom.ProbLo(zDir) + 0.5 * m_infra.m_dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> weights{1, 3};

    amrex::Real expectedValA{1 + 3 * pow(0.5, GEMPIC_SPACEDIM)};
    amrex::Real expectedValB{3 * pow(0.5, GEMPIC_SPACEDIM)};

    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in
    // Particle iteration ... over two close particles.
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[0], 0); pti.isValid(); ++pti)
    {
        const long np{pti.numParticles()};
        EXPECT_EQ(numParticles, np);  // Two particles added

        update_rho_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(pti, m_infra, m_rhoPtr->m_data,
                                                                m_particleGroup[0]->get_charge());

        // See SingleParticle test for explanation of expectations
        check_field(m_rhoPtr->m_data[pti].array(), m_infra.m_nCell.dim3(),
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
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> pPos{
        {{AMREX_D_DECL(0, 0, 0)}}};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> ePos{
        {{AMREX_D_DECL(m_infra.m_geom.ProbLo(xDir) + 0.5 * m_infra.m_dx[xDir],
                       m_infra.m_geom.ProbLo(yDir) + 0.5 * m_infra.m_dx[yDir],
                       m_infra.m_geom.ProbLo(zDir) + 0.5 * m_infra.m_dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> pWeights{1};
    amrex::Array<amrex::Real, numParticles> eWeights{3};
    int pSpec{0}, eSpec{1};

    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, pWeights, pPos, pSpec);
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, eWeights, ePos, eSpec);

    const auto pCharge{m_particleGroup[pSpec]->get_charge()};
    const auto eCharge{m_particleGroup[eSpec]->get_charge()};

    amrex::Real expectedValA{pCharge + eCharge * 3 * pow(0.5, GEMPIC_SPACEDIM)};
    amrex::Real expectedValB{eCharge * 3 * pow(0.5, GEMPIC_SPACEDIM)};

    for (int spec{0}; spec < s_numSpec; spec++)
    {
        m_particleGroup[spec]->Redistribute();  // assign particles to the tile they are in
        const auto charge{m_particleGroup[spec]->get_charge()};
        // Particle iteration
        for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[spec], 0); pti.isValid();
             ++pti)
        {
            const long np{pti.numParticles()};
            EXPECT_EQ(numParticles, np);  // Two particles added

            update_rho_parallel_for<s_vDim, s_degX, s_degY, s_degZ>(pti, m_infra, m_rhoPtr->m_data,
                                                                    charge);

            if (spec == s_numSpec - 1)
            {
                // See SingleParticle test for explanation of expectations
                check_field(m_rhoPtr->m_data[pti].array(), m_infra.m_nCell.dim3(),
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
}  // namespace