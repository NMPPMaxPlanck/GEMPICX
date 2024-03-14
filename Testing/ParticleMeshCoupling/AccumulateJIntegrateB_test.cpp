#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_SplineClass.H"
#include "TestUtils/GEMPIC_TestUtils.H"

#define check_field(...) Gempic::Test::Utils::check_field(__FILE__, __LINE__, __VA_ARGS__)

using namespace Gempic;
using namespace Forms;
using namespace Particle;

namespace
{
// When using amrex::ParallelFor you have to create a standalone helper function that does the
// execution on GPU and call that function from the unit test because of how GTest creates tests
// within a TEST_F fixture.
template <Direction pDir, int degX, int degY, int degZ, unsigned int vDim>
void accumulate_j_update_v_c2_parallel_for (amrex::ParIter<0, 0, vDim + 1, 0>& pti,
                                            DeRhamField<Grid::primal, Space::face>& B,
                                            DeRhamField<Grid::dual, Space::face>& J,
                                            ComputationalDomain& infra,
                                            amrex::Real weight,
                                            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const dx,
                                            amrex::GpuArray<amrex::Real, 2>& bfields)
{
    const long np{pti.numParticles()};

    const auto& particles{pti.GetArrayOfStructs()};
    const auto partData{particles().data()};

    amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> jA;
    for (int cc = 0; cc < vDim; cc++) jA[cc] = (J.m_data[cc])[pti].array();

    amrex::AsyncArray aaBfields(&bfields, 1);
    auto* bfieldsGPU = aaBfields.data();

    amrex::GpuArray<amrex::Array4<amrex::Real>, int(vDim / 2.5) * 2 + 1> bA;
    for (int cc = 0; cc < (int(vDim / 2.5) * 2 + 1); cc++) bA[cc] = (B.m_data[cc])[pti].array();

    amrex::ParallelFor(
        np,
        [=] AMREX_GPU_DEVICE(long pp)
        {
            amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> posStart;
            for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
            {
                posStart[d] = partData[0].pos(d);
            }

            amrex::Real xEnd = 0;

            ParticleMeshCoupling::SplineWithPrimitive<degX, degY, degZ> spline(
                posStart, infra.m_plo, infra.m_dxi);

            spline.template update1_d_splines<pDir>(xEnd, infra.m_plo[xDir], infra.m_dxi[xDir]);
            spline.template update1_d_primitive<pDir>(xEnd, infra.m_plo[xDir], infra.m_dxi[xDir]);

            ParticleMeshCoupling::accumulate_j_integrate_b<pDir>(spline, weight, dx, bA, jA,
                                                                 *bfieldsGPU);
        });

    aaBfields.copyToHost(&bfields, 1);
}

class AccumulateJUpdateVC2Test : public testing::Test
{
protected:
    static const int s_degX{1};
    static const int s_degY{1};
    static const int s_degZ{1};
    inline static const int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

    inline static const int s_hodgeDegree{2};
    static const unsigned int s_numSpec{1};
    static const unsigned int s_vDim{3};
    static const unsigned int s_spec{0};
    Io::Parameters m_parameters{};

    amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> m_jA;
    amrex::GpuArray<amrex::Array4<amrex::Real>, int(s_vDim / 2.5) * 2 + 1> m_bA;

    ComputationalDomain m_infra{false};  // "uninitialized" computational domain
    amrex::GpuArray<std::unique_ptr<ParticleGroups<s_vDim>>, s_numSpec> m_particleGroup;
    std::shared_ptr<FDDeRhamComplex> m_deRham;

    static const int s_degP{1};
    static const int s_degP1{1};
    static const int s_degP2{1};

    static const unsigned int s_numParticles{1};

    static const Direction s_pDim{yDir};

    amrex::Real m_weight = 1.0;

    amrex::GpuArray<amrex::Real, 2> m_bfields{0., 0.};
    amrex::GpuArray<amrex::Real, std::max(s_degX, std::max(s_degY, s_degZ)) + 4> m_primitive;

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

    void SetUp () override
    {
        if constexpr (GEMPIC_SPACEDIM != 3)
        {
            GTEST_SKIP() << "This function barely works in 3D, let alone lower dimensions.";
        }

        m_infra = ComputationalDomain{};

        // Initialize the De Rham Complex
        m_deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                     HodgeScheme::FDHodge);

        // particles
        for (int species{0}; species < s_numSpec; species++)
        {
            m_particleGroup[species] = std::make_unique<ParticleGroups<s_vDim>>(species, m_infra);
        }
    }
};

TEST_F(AccumulateJUpdateVC2Test, NullTest)
{
    // Adding particle to one cell
    const unsigned int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    // (default) charge correctly transferred from addSingleParticles
    EXPECT_EQ(1, m_particleGroup[0]->get_charge());

    const amrex::Array<std::string, 3> analyticalFuncB = {"0.0", "0.0", "0.0"};

    const amrex::Array<std::string, 3> analyticalFuncJ = {"0.0", "0.0", "0.0"};

    const int nVar{4};  // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcB;
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcJ;
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

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in
    // Particle iteration ... over one particle. Hopefully.

    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        const long np{pti.numParticles()};
        EXPECT_EQ(1, np);  // Only one particle added by addSingleParticles

        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        accumulate_j_update_v_c2_parallel_for<s_pDim, s_degX, s_degY, s_degZ, s_vDim>(
            pti, B, J, m_infra, m_weight, m_infra.m_dx, bfields);

        EXPECT_EQ(bfields[0], 0);
        EXPECT_EQ(bfields[1], 0);

        // Expect all nodes to be 0
        check_field((J.m_data[xDir]).array(pti), m_infra.m_nCell.dim3(), {}, {}, 0);
        check_field((J.m_data[yDir]).array(pti), m_infra.m_nCell.dim3(), {}, {}, 0);
        check_field((J.m_data[zDir]).array(pti), m_infra.m_nCell.dim3(), {}, {}, 0);
    }
}

TEST_F(AccumulateJUpdateVC2Test, SingleParticleMiddle)
{
    // Adding particle to one cell
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(m_infra.m_geom.ProbHi(xDir) - 5.5 * m_infra.m_dx[xDir],
                       m_infra.m_geom.ProbHi(yDir) - 5.5 * m_infra.m_dx[yDir],
                       m_infra.m_geom.ProbHi(zDir) - 5.5 * m_infra.m_dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    // (default) charge correctly transferred from addSingleParticles
    EXPECT_EQ(1, m_particleGroup[0]->get_charge());

    const amrex::Array<std::string, 3> analyticalFuncB = {"1.0", "1.0", "1.0"};

    const amrex::Array<std::string, 3> analyticalFuncJ = {"1.0", "1.0", "1.0"};

    const int nVar{4};  // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcB;
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcJ;
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

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in
    // Particle iteration ... over one particle. Hopefully.

    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        const long np{pti.numParticles()};
        EXPECT_EQ(1, np);  // Only one particle added by addSingleParticles

        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        accumulate_j_update_v_c2_parallel_for<s_pDim, s_degX, s_degY, s_degZ, s_vDim>(
            pti, B, J, m_infra, m_weight, m_infra.m_dx, bfields);

        EXPECT_NEAR(bfields[0], -4.5, 1e-15);
        EXPECT_NEAR(bfields[1], -4.5, 1e-15);

        check_field(
            (J.m_data[s_pDim]).array(pti), m_infra.m_nCell.dim3(),
            // Expect the eight nearest nodes (4/5, 4/5, 4/5) to be non-zero
            {[] (AMREX_D_DECL(int a, int b, int c))
             { return AMREX_D_TERM((a == 4 || a == 5), &&b == 4, &&(c == 4 || c == 5)); },
             [] (AMREX_D_DECL(int a, int b, int c))
             { return AMREX_D_TERM((a == 4 || a == 5), &&b <= 3, &&(c == 4 || c == 5)); }},
            // getting an eight of the particle weight times the primitive, plus the original 1
            {1 - 1. / 8, 1 - 0.25},
            // with the remaining entries being 1
            1);
        check_field((J.m_data[(s_pDim + 1) % 3]).array(pti), m_infra.m_nCell.dim3(), {}, {}, 1);
        check_field((J.m_data[(s_pDim + 2) % 3]).array(pti), m_infra.m_nCell.dim3(), {}, {}, 1);
    }
}

TEST_F(AccumulateJUpdateVC2Test, SingleParticleUnevenNodeSplit)
{
    // Adding particle to one cell
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(m_infra.m_geom.ProbHi(xDir) - 5.25 * m_infra.m_dx[xDir],
                       m_infra.m_geom.ProbHi(yDir) - 5.25 * m_infra.m_dx[yDir],
                       m_infra.m_geom.ProbHi(zDir) - 5.25 * m_infra.m_dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    // (default) charge correctly transferred from addSingleParticles
    EXPECT_EQ(1, m_particleGroup[0]->get_charge());

    const amrex::Array<std::string, 3> analyticalFuncB = {"1.0", "1.0", "1.0"};

    const amrex::Array<std::string, 3> analyticalFuncJ = {"1.0", "1.0", "1.0"};

    const int nVar{4};  // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcB;
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcJ;
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

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in
    // Particle iteration ... over one particle. Hopefully.

    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        const long np{pti.numParticles()};
        EXPECT_EQ(1, np);  // Only one particle added by addSingleParticles

        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        accumulate_j_update_v_c2_parallel_for<s_pDim, s_degX, s_degY, s_degZ, s_vDim>(
            pti, B, J, m_infra, m_weight, m_infra.m_dx, bfields);

        EXPECT_NEAR(bfields[0], -4.75, 1e-15);
        EXPECT_NEAR(bfields[1], -4.75, 1e-15);

        check_field(
            (J.m_data[s_pDim]).array(pti), m_infra.m_nCell.dim3(),
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
        check_field((J.m_data[(s_pDim + 1) % 3]).array(pti), m_infra.m_nCell.dim3(), {}, {}, 1);
        check_field((J.m_data[(s_pDim + 2) % 3]).array(pti), m_infra.m_nCell.dim3(), {}, {}, 1);
    }
}

TEST_F(AccumulateJUpdateVC2Test, DoubleParticleSeparate)
{
    const int numParticles{2};
    // Particles in different cells to check that they don't interfere with each other
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(0, 0, 0)},
         {AMREX_D_DECL(m_infra.m_geom.ProbLo(xDir) + 5.5 * m_infra.m_dx[xDir],
                       m_infra.m_geom.ProbLo(yDir) + 5.5 * m_infra.m_dx[yDir],
                       m_infra.m_geom.ProbLo(zDir) + 5.5 * m_infra.m_dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> weights{1, 1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    // (default) charge correctly transferred from addSingleParticles
    EXPECT_EQ(1, m_particleGroup[0]->get_charge());

    const amrex::Array<std::string, 3> analyticalFuncB = {"1.0", "1.0", "1.0"};

    const amrex::Array<std::string, 3> analyticalFuncJ = {"1.0", "1.0", "1.0"};

    const int nVar{4};  // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcB;
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcJ;
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

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in
    // Particle iteration ... over one particle. Hopefully.

    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        const long np{pti.numParticles()};
        EXPECT_EQ(2, np);  // Only one particle added by addSingleParticles

        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        accumulate_j_update_v_c2_parallel_for<s_pDim, s_degX, s_degY, s_degZ, s_vDim>(
            pti, B, J, m_infra, m_weight, m_infra.m_dx, bfields);

        EXPECT_EQ(bfields[0], 0);
        EXPECT_EQ(bfields[1], 0);

        // Expect all nodes to be 1
        check_field((J.m_data[s_pDim]).array(pti), m_infra.m_nCell.dim3(), {}, {}, 1);
        check_field((J.m_data[(s_pDim + 1) % 3]).array(pti), m_infra.m_nCell.dim3(), {}, {}, 1);
        check_field((J.m_data[(s_pDim + 2) % 3]).array(pti), m_infra.m_nCell.dim3(), {}, {}, 1);
    }
}

TEST_F(AccumulateJUpdateVC2Test, DoubleParticleOverlap)
{
    const int numParticles{2};
    // Particles in different cells to check that they don't interfere with each other
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(0, 0, 0)},
         {AMREX_D_DECL(m_infra.m_geom.ProbLo(xDir) + 0.5 * m_infra.m_dx[xDir],
                       m_infra.m_geom.ProbLo(yDir) + 0.5 * m_infra.m_dx[yDir],
                       m_infra.m_geom.ProbLo(zDir) + 0.5 * m_infra.m_dx[zDir])}}};
    amrex::Array<amrex::Real, numParticles> weights{1, 1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    // (default) charge correctly transferred from addSingleParticles
    EXPECT_EQ(1, m_particleGroup[0]->get_charge());

    const amrex::Array<std::string, 3> analyticalFuncB = {"1.0", "1.0", "1.0"};

    const amrex::Array<std::string, 3> analyticalFuncJ = {"1.0", "1.0", "1.0"};

    const int nVar{4};  // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcB;
    amrex::Array<amrex::ParserExecutor<nVar>, GEMPIC_SPACEDIM> funcJ;
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

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in
    // Particle iteration ... over one particle. Hopefully.

    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        const long np{pti.numParticles()};
        EXPECT_EQ(2, np);  // Only one particle added by addSingleParticles

        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        accumulate_j_update_v_c2_parallel_for<s_pDim, s_degX, s_degY, s_degZ, s_vDim>(
            pti, B, J, m_infra, m_weight, m_infra.m_dx, bfields);

        EXPECT_EQ(bfields[0], 0);
        EXPECT_EQ(bfields[1], 0);

        // Expect all nodes to be 1
        check_field((J.m_data[s_pDim]).array(pti), m_infra.m_nCell.dim3(), {}, {}, 1);
        check_field((J.m_data[(s_pDim + 1) % 3]).array(pti), m_infra.m_nCell.dim3(), {}, {}, 1);
        check_field((J.m_data[(s_pDim + 2) % 3]).array(pti), m_infra.m_nCell.dim3(), {}, {}, 1);
    }
}

}  // namespace