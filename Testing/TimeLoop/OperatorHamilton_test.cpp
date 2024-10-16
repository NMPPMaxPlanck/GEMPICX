#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_HsZigzag.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_SplineClass.H"
#include "GEMPIC_Splitting.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;
using namespace TimeLoop;

using ::testing::_;
using ::testing::Exactly;
using ::testing::Mock;

namespace Gempic::ParticleMeshCoupling
{

template <>
AMREX_GPU_HOST_DEVICE void push_v_efield<4>(amrex::GpuArray<amrex::Real, 4> &vel,
                                            amrex::Real dt,
                                            amrex::Real chargemass,  // charge/mass
                                            amrex::GpuArray<amrex::Real, 4> const &ep)
{
    for (int i = 0; i < 4; i++)
    {
        vel[i] = 1;
    }
}
}  // namespace Gempic::ParticleMeshCoupling

namespace
{
template <unsigned int vDim, int degX, int degY, int degZ, int hodgeDegree, unsigned int ndata>
class MockHSZigZagC2 : public HSZigZagC2<vDim, degX, degY, degZ, hodgeDegree, ndata>
{
public:
};

template <int degX, int degY, int degZ>
class MockSpline : public ParticleMeshCoupling::SplineWithPrimitive<degX, degY, degZ>
{
public:
    MockSpline(amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const &position,
               amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const &plo,
               amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM + 1> const &dxInverse) :
        ParticleMeshCoupling::SplineWithPrimitive<degX, degY, degZ>(position, plo, dxInverse)
    {
    }

    template <Field form, unsigned int vDim>
    AMREX_GPU_HOST_DEVICE amrex::GpuArray<amrex::Real, vDim> eval_spline_field (
        const amrex::GpuArray<amrex::Array4<amrex::Real>, vDim> /*fieldArray*/) const
    {
        amrex::GpuArray<amrex::Real, vDim> fields;
        for (int comp = 0; comp < vDim; comp++)
        {
            fields[comp] = 0.;
        }

        return fields;
    }
};

class OperatorHamiltonTest : public testing::Test
{
protected:
    static const int s_degX{1};
    static const int s_degY{1};
    static const int s_degZ{1};
    inline static const int s_hodgeDegree{2};
    inline static const int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

    static const int s_numSpec{1};
    static const int s_vDim{3};
    static const int s_spec{0};
    Io::Parameters m_parameters{};

    ComputationalDomain m_infra{false};  // "uninitialized" computational domain
    std::vector<std::unique_ptr<ParticleGroups<s_vDim>>> m_particleGroup;
    std::shared_ptr<FDDeRhamComplex> m_deRham;

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
}  // namespace

namespace Gempic::ParticleMeshCoupling
{
// You cannot do partial template specialization for functions, so here is an explicit
// specialization for a special case
template <>
AMREX_GPU_HOST_DEVICE void accumulate_j_integrate_b<xDir, MockSpline<1, 1, 1>, 4>(
    MockSpline<1, 1, 1> &spline,
    amrex::Real const weight,
    amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> const &dx,
    amrex::GpuArray<amrex::Array4<amrex::Real>, int(4 / 2.5) * 2 + 1> const &bArray,
    amrex::GpuArray<amrex::Array4<amrex::Real>, 4> const &jArray,
    amrex::GpuArray<amrex::Real, 2> &fields)
{
}
}  // namespace Gempic::ParticleMeshCoupling

namespace
{
TEST_F(OperatorHamiltonTest, ApplyHEParticleTest)
{
    // Adding particle to one cell
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    // (default) charge correctly transferred from addSingleParticles
    EXPECT_EQ(1, m_particleGroup[0]->get_charge());

    // Parse analytical fields and initialize parserEval. Has to be the same as Bx,By,Bz and Ex,
    // Ey, Ez
    const amrex::Array<std::string, 3> analyticalFuncE{"0.0", "0.0", "0.0"};

    const int nVar{GEMPIC_SPACEDIM + 1};  // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcE;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i{0}; i < 3; ++i)
    {
        parser[i].define(analyticalFuncE[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcE[i] = parser[i].compile<nVar>();
    }

    DeRhamField<Grid::primal, Space::edge> E(m_deRham, funcE);

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in
    // Particle iteration ... over one particle. Hopefully.

    bool particleLoopRun{false};
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        particleLoopRun = true;

        MockSpline<s_degX, s_degY, s_degZ> mockSpline(
            {AMREX_D_DECL(1., 1., 1.)}, {AMREX_D_DECL(1., 1., 1.)}, {AMREX_D_DECL(1., 1., 1.), 1.});
        // EXPECT_CALL(mockHSZigZagC2, push_v_efield).Times(Exactly(1));

        const long np{pti.numParticles()};
        EXPECT_EQ(1, np);  // Only one particle added by addSingleParticles

        amrex::Particle<0, 0> *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

        amrex::GpuArray<amrex::Array4<amrex::Real>, 4> eArray;
        for (int cc{0}; cc < s_vDim; cc++) eArray[cc] = (E.m_data[cc])[pti].array();

        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
        for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
        {
            position[d] = particles[0].pos(d);
        }

        auto *particleAttributes = &pti.GetStructOfArrays();
        amrex::ParticleReal *const AMREX_RESTRICT velx =
            particleAttributes->GetRealData(xDir).data();
        amrex::ParticleReal *const AMREX_RESTRICT vely =
            particleAttributes->GetRealData(yDir).data();
        amrex::ParticleReal *const AMREX_RESTRICT velz =
            particleAttributes->GetRealData(zDir).data();
        amrex::GpuArray<amrex::Real, 4> vel{0, 0, 0, 0};

        OperatorHamilton<4, s_degX, s_degY, s_degZ, s_hodgeDegree> operatorHamilton;

        operatorHamilton.template apply_h_e_particle<MockSpline<s_degX, s_degY, s_degZ>>(
            eArray, mockSpline, vel, velx, vely, velz, 1, 1, 0);

        amrex::GpuArray<amrex::Real, 3> efield({1, 1, 1});
        ASSERT_THAT(efield, ::testing::ElementsAreArray({1, 1, 1}));

        EXPECT_EQ(1, velx[0]);
        EXPECT_EQ(1, vely[0]);
        EXPECT_EQ(1, velz[0]);
    }
    ASSERT_TRUE(particleLoopRun);
}

TEST_F(OperatorHamiltonTest, ApplyHpiTest)
{
    // Adding particle to one cell
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    // (default) charge correctly transferred from addSingleParticles
    EXPECT_EQ(1, m_particleGroup[0]->get_charge());

    DeRhamField<Grid::dual, Space::face> J(m_deRham);
    DeRhamField<Grid::primal, Space::face> B(m_deRham);

    const amrex::Array<std::string, 3> analyticalFuncB = {"0.0", "0.0", "0.0"};

    // Project B to a primal two form
    const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
    amrex::Array<amrex::Parser, 3> parser;

    for (int i = 0; i < 3; ++i)
    {
        parser[i].define(analyticalFuncB[i]);
        parser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcB[i] = parser[i].compile<nVar>();
    }

    m_deRham->projection(funcB, 0.0, B);

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in
    // Particle iteration ... over one particle. Hopefully.

    bool particleLoopRun{false};
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        particleLoopRun = true;

        MockSpline<s_degX, s_degY, s_degZ> mockSpline(
            {AMREX_D_DECL(1., 1., 1.)}, {AMREX_D_DECL(1., 1., 1.)}, {AMREX_D_DECL(1., 1., 1.), 1.});
        // EXPECT_CALL(mockHSZigZagC2, push_v_efield).Times(Exactly(1));

        const long np{pti.numParticles()};
        EXPECT_EQ(1, np);  // Only one particle added by addSingleParticles

        amrex::Particle<0, 0> *AMREX_RESTRICT particles = &(pti.GetArrayOfStructs()[0]);

        amrex::GpuArray<amrex::Array4<amrex::Real>, 4> jArray;
        for (int cc{0}; cc < s_vDim; cc++) jArray[cc] = (J.m_data[cc])[pti].array();

        amrex::GpuArray<amrex::Array4<amrex::Real>, 3> bArray;
        for (int cc{0}; cc < s_vDim; cc++) bArray[cc] = (B.m_data[cc])[pti].array();

        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position;
        for (unsigned int d{0}; d < GEMPIC_SPACEDIM; ++d)
        {
            position[d] = particles[0].pos(d);
        }

        auto *particleAttributes = &pti.GetStructOfArrays();
        amrex::ParticleReal *const AMREX_RESTRICT velx =
            particleAttributes->GetRealData(xDir).data();
        amrex::ParticleReal *const AMREX_RESTRICT vely =
            particleAttributes->GetRealData(yDir).data();
        amrex::ParticleReal *const AMREX_RESTRICT velz =
            particleAttributes->GetRealData(zDir).data();
        amrex::GpuArray<amrex::Real, 4> vel{0, 0, 0, 0};

        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        OperatorHamilton<4, s_degX, s_degY, s_degZ, s_hodgeDegree> operatorHamilton;

        operatorHamilton.template apply_h_p_i<xDir>(position, vel, m_infra, mockSpline, bfields,
                                                    m_infra.m_dx, jArray, bArray, 1, 1, 1);

        amrex::GpuArray<amrex::Real, 3> efield({1, 1, 1});
        ASSERT_THAT(efield, ::testing::ElementsAreArray({1, 1, 1}));

        EXPECT_EQ(0, velx[0]);
        EXPECT_EQ(0, vely[0]);
        EXPECT_EQ(0, velz[0]);
    }
    ASSERT_TRUE(particleLoopRun);
}
}  // namespace