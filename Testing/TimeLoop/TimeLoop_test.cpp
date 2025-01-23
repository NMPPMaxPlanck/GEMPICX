#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_ParticleGroups.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_SplineClass.H"
#include "GEMPIC_Splitting.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;
using ::testing::Mock;

#define check_field(...) Gempic::Test::Utils::check_field(__FILE__, __LINE__, __VA_ARGS__)
#define compare_fields(...) Gempic::Test::Utils::compare_fields(__FILE__, __LINE__, __VA_ARGS__)

namespace
{
/**
 * @brief Test fixture. Sets up clean environment before each test of the SplineWithPrimitive class
 */
class HamiltonianSplittingTest : public testing::Test
{
protected:
    // Degree of splines in each direction
    static const int s_degX{2};  // to match degree 2 in script
    static const int s_degY{2};
    static const int s_degZ{2};

    static const int s_vDim{3};
    static const int s_numSpec{1};
    static const int s_spec{0};

    inline static const int s_maxSplineDegree{
        AMREX_D_PICK(s_degX, std::max(s_degX, s_degY), std::max(std::max(s_degX, s_degY), s_degZ))};
    inline static const int s_hodgeDegree{2};

    Gempic::Io::Parameters m_parameters{};

    ComputationalDomain m_infra{false};  // "unitialized" computational domain
    std::vector<std::unique_ptr<ParticleGroups<s_vDim>>> m_particleGroup;

    static void SetUpTestSuite ()
    {
        /* Initialize the infrastructure */
        amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        amrex::Vector<amrex::Real> k{AMREX_D_DECL(1.0, 1.0, 1.0)};
        // Dimension needs to be big enough to cover the full support of the splines
        const amrex::Vector<int> nCell{AMREX_D_DECL(20, 20, 20)};
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(20, 20, 20)};
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
        m_particleGroup.resize(s_numSpec);
        for (int spec{0}; spec < s_numSpec; spec++)
        {
            m_particleGroup[spec] = std::make_unique<ParticleGroups<s_vDim>>(spec, m_infra);
        }
    }
};

TEST_F(HamiltonianSplittingTest, AccumulateJTest)
{
    // Adding particle to one cell
    const unsigned int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in

    // Setting testing parameters in the field far away from the border to ignore boundary
    // conditions
    amrex::Real chargeWeight = 2.0;
    AMREX_D_TERM(amrex::Real xPosOld = 2.5;, amrex::Real yPosOld = 3.7;, amrex::Real zPosOld = 2.2;)
    AMREX_D_TERM(amrex::Real xPosNew = 2.6;, amrex::Real yPosNew = 4.2;, amrex::Real zPosNew = 2.0;)

    amrex::Real primDiffRef = 0;
    amrex::Real yNodeVal = 0;
    amrex::Real zNodeVal = 0;

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::face> B(deRham);
    DeRhamField<Grid::dual, Space::face> J(deRham);

    // TEST FOR X DIRECTION
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        // set random positions
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};
        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> bA, jA;

        for (int cc = 0; cc < 3; cc++)
        {
            jA[cc] = (J.m_data[cc])[pti].array();
            bA[cc] = (B.m_data[cc])[pti].array();
        }

        // initialization: oldSpline = newSpline -> first PrimDiff = 0
        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.m_plo, m_infra.m_dxi);

        // set position after movement in x-direction
        position = {AMREX_D_DECL(xPosNew, yPosOld, zPosOld)};

        spline.template update_1d_splines<xDir>(position[xDir], m_infra.m_plo[xDir],
                                                m_infra.m_dxi[xDir]);
        ParticleMeshCoupling::accumulate_j_integrate_b<xDir>(bfields, spline, chargeWeight,
                                                             m_infra.m_dx, bA, jA);

        // setup is in a way, that the first spline is in the "bothSplines" region
        amrex::Real primitiveDifference = spline.template compute_primitive_difference<
            xDir, Gempic::ParticleMeshCoupling::Impl::PrimitiveDifferenceRegion::bothSplines>(
            m_infra.m_dx, 0);
        primDiffRef = primitiveDifference;

        yNodeVal = spline.m_nodeSplineVals[yDir][0];
        zNodeVal = spline.m_nodeSplineVals[zDir][0];
    }
    for (amrex::MFIter mfi(J.m_data[0]); mfi.isValid(); ++mfi)
    {
        // Checking only the first index !=0, not the full field. Indices according to the current
        // setup!
        check_field(J.m_data[0].array(mfi), m_infra.m_nCell.dim3(),
                    {[] (AMREX_D_DECL(int i, int j, int k))
                     { return AMREX_D_TERM(i == 7, &&j == 11, &&k == 6); }},
                    {chargeWeight * GEMPIC_D_MULT(primDiffRef, yNodeVal, zNodeVal)}, {}, 1e-12);
    }

    // TEST FOR Y DIRECTION
#if GEMPIC_SPACEDIM > 1
    amrex::Real xNodeVal = 0;
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        // set random positions
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};
        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> bA, jA;

        for (int cc = 0; cc < 3; cc++)
        {
            jA[cc] = (J.m_data[cc])[pti].array();
            bA[cc] = (B.m_data[cc])[pti].array();
        }

        // initialization: oldSpline = newSpline -> first PrimDiff = 0
        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.m_plo, m_infra.m_dxi);

        // set position after movement in y-direction
        position = {AMREX_D_DECL(xPosOld, yPosNew, zPosOld)};

        spline.template update_1d_splines<yDir>(position[yDir], m_infra.m_plo[yDir],
                                                m_infra.m_dxi[yDir]);
        ParticleMeshCoupling::accumulate_j_integrate_b<yDir>(bfields, spline, chargeWeight,
                                                             m_infra.m_dx, bA, jA);

        // setup is in a way, that the second index is in the "bothSplines" region
        amrex::Real primitiveDifference = spline.template compute_primitive_difference<
            yDir, Gempic::ParticleMeshCoupling::Impl::PrimitiveDifferenceRegion::bothSplines>(
            m_infra.m_dx, 1);
        primDiffRef = primitiveDifference;

        xNodeVal = spline.m_nodeSplineVals[xDir][0];
        zNodeVal = spline.m_nodeSplineVals[zDir][0];
    }
    for (amrex::MFIter mfi(J.m_data[1]); mfi.isValid(); ++mfi)
    {
        check_field(J.m_data[1].array(mfi), m_infra.m_nCell.dim3(),
                    {[] (AMREX_D_DECL(int i, int j, int k))
                     { return AMREX_D_TERM(i == 7, &&j == 12, &&k == 6); }},
                    {chargeWeight * GEMPIC_D_MULT(xNodeVal, primDiffRef, zNodeVal)}, {}, 1e-12);
    }

    // TEST FOR Z DIRECTION
#if GEMPIC_SPACEDIM > 2
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        // set random positions
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};
        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> bA, jA;

        for (int cc = 0; cc < 3; cc++)
        {
            jA[cc] = (J.m_data[cc])[pti].array();
            bA[cc] = (B.m_data[cc])[pti].array();
        }

        // initialization: oldSpline = newSpline -> first PrimDiff = 0
        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.m_plo, m_infra.m_dxi);

        // set position after movement in y-direction
        position = {AMREX_D_DECL(xPosOld, yPosOld, zPosNew)};

        spline.template update_1d_splines<zDir>(position[zDir], m_infra.m_plo[zDir],
                                                m_infra.m_dxi[zDir]);
        ParticleMeshCoupling::accumulate_j_integrate_b<zDir>(bfields, spline, chargeWeight,
                                                             m_infra.m_dx, bA, jA);

        // setup is in a way, that the first spline is in the "firstSpline" region
        amrex::Real primitiveDifference = spline.template compute_primitive_difference<
            zDir, Gempic::ParticleMeshCoupling::Impl::PrimitiveDifferenceRegion::firstSpline>(
            m_infra.m_dx, 0);
        primDiffRef = primitiveDifference;

        xNodeVal = spline.m_nodeSplineVals[xDir][0];
        yNodeVal = spline.m_nodeSplineVals[yDir][0];
    }

    for (amrex::MFIter mfi(J.m_data[2]); mfi.isValid(); ++mfi)
    {
        check_field(J.m_data[2].array(mfi), m_infra.m_nCell.dim3(),
                    {[] (AMREX_D_DECL(int i, int j, int k))
                     { return AMREX_D_TERM(i == 7, &&j == 11, &&k == 5); }},
                    {chargeWeight * xNodeVal * yNodeVal * primDiffRef}, {}, 1e-12);
    }
#endif
#endif
}

#if GEMPIC_SPACEDIM < 3
TEST_F(HamiltonianSplittingTest, AccumulateJEulerTest)
{
    // Adding particle to one cell
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in

    // Setting testing parameters in the field far away from the border to ignore boundary
    // conditions
    amrex::Real chargeWeight = 2.0;
    amrex::Real dt = 0.001;
    AMREX_D_TERM(amrex::Real xPosOld = 2.5;, amrex::Real yPosOld = 3.7;, amrex::Real zPosOld = 0.0;)

    // make sure particle doesn't go out of computational domain (+ ghost cells)
    amrex::Real cfl{dt / m_infra.m_dx[0]};
    amrex::Real vx{1 * cfl}, vy{3.1 * cfl}, vz{-2.5 * cfl};
    amrex::GpuArray<amrex::Real, s_vDim> vel{vx, vy, vz};

    AMREX_D_TERM(amrex::Real xNodeVal = 0;, amrex::Real yNodeVal = 0;, )
    amrex::Real dtv = 0;

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::face> B(deRham);
    DeRhamField<Grid::dual, Space::face> J(deRham);

    // TEST FOR X DIRECTION
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        // set random positions
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};
        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> bA, jA;

        for (int cc = 0; cc < 3; cc++)
        {
            jA[cc] = (J.m_data[cc])[pti].array();
            bA[cc] = (B.m_data[cc])[pti].array();
        }

        // initialization: oldSpline = newSpline -> first PrimDiff = 0
        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.m_plo, m_infra.m_dxi);

        dtv = dt * vel[yDir];
        xNodeVal = spline.m_nodeSplineVals[xDir][0];

#if GEMPIC_SPACEDIM == 1
        ParticleMeshCoupling::accumulate_j_integrate_b_euler_y(bfields, spline, dtv, chargeWeight,
                                                               bA, jA);
        ParticleMeshCoupling::accumulate_j_integrate_b_euler_z(bfields, spline, dtv, chargeWeight,
                                                               bA, jA);
#elif GEMPIC_SPACEDIM == 2
        yNodeVal = spline.m_nodeSplineVals[yDir][0];
        ParticleMeshCoupling::accumulate_j_integrate_b_euler_z(bfields, spline, dtv, chargeWeight,
                                                               bA, jA);
#endif
    }

    for (int dir{GEMPIC_SPACEDIM}; dir < 3; ++dir)
    {
        // check the first non-zero entry of the J field in xDir
        for (amrex::MFIter mfi(J.m_data[dir]); mfi.isValid(); ++mfi)
        {
            check_field(J.m_data[dir].array(mfi), m_infra.m_nCell.dim3(),
                        {[] (AMREX_D_DECL(int i, int j, int k))
                         { return AMREX_D_TERM(i == 7, &&j == 11, &&k == 0); }},
                        {chargeWeight * dtv * GEMPIC_D_MULT(xNodeVal, yNodeVal, 1.0)}, {}, 1e-12);
        }
    }
}
#endif

// Test if accumulate J indeed preserves the Gauss law
// it $rho^{n+1} = rho^n - div integratedJ$
TEST_F(HamiltonianSplittingTest, GaussTest)
{
    // Adding a few particles
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in

    amrex::Real chargeWeight = 1.0;
    AMREX_D_TERM(amrex::Real xPosOld = 2.5;, amrex::Real yPosOld = 3.7;, amrex::Real zPosOld = 2.2;)
    amrex::Real xPosNew = 3.2;
    amrex::Real valBx = 2.0;
    amrex::Real valBy = 3.0;
    amrex::Real valBz = 4.0;
    amrex::Real valJx = 5.0;
    amrex::Real valJy = 6.0;
    amrex::Real valJz = 7.0;

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::face> B(deRham);
    DeRhamField<Grid::dual, Space::face> J(deRham);
    DeRhamField<Grid::dual, Space::cell> divJ(deRham);
    DeRhamField<Grid::dual, Space::cell> rhoOld(deRham);
    DeRhamField<Grid::dual, Space::cell> rho(deRham);
    DeRhamField<Grid::dual, Space::cell> rhoMinJ(deRham);

    // Deposit initial random field values to test for something different than 0
    rhoOld.m_data.setVal(6.0);
    rho.m_data.setVal(6.0);
    B.m_data[xDir].setVal(valBx);
    J.m_data[xDir].setVal(valJx);
    B.m_data[yDir].setVal(valBy);
    J.m_data[yDir].setVal(valJy);
    B.m_data[zDir].setVal(valBz);
    J.m_data[zDir].setVal(valJz);

    // TEST FOR X DIRECTION

    // Particle iteration ... over one particle.
    bool particleLoopRun{false};
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[0], 0); pti.isValid(); ++pti)
    {
        // set random positions (positions of particles in particle group never used here)
        // Check if this is OK especially when there are several particle tiles
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};
        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> bA, jA;
        amrex::Array4<amrex::Real> const& rhoOldarr = rhoOld.m_data[pti].array();
        amrex::Array4<amrex::Real> const& rhoarr = rho.m_data[pti].array();

        for (int cc = 0; cc < 3; cc++)
        {
            jA[cc] = (J.m_data[cc])[pti].array();
            bA[cc] = (B.m_data[cc])[pti].array();
        }

        particleLoopRun = true;

        // compute: rho^n
        ParticleMeshCoupling::SplineBase<s_degX, s_degY, s_degZ> splineRhoOld(
            position, m_infra.m_plo, m_infra.m_dxi);

        ParticleMeshCoupling::gempic_deposit_rho(rhoOldarr, splineRhoOld, chargeWeight);

        // compute: integratedJ
        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.m_plo, m_infra.m_dxi);

        // set position after movement in x-direction
        position = {AMREX_D_DECL(xPosNew, yPosOld, zPosOld)};

        spline.template update_1d_splines<xDir>(position[xDir], m_infra.m_plo[xDir],
                                                m_infra.m_dxi[xDir]);
        ParticleMeshCoupling::accumulate_j_integrate_b<xDir>(bfields, spline, chargeWeight,
                                                             m_infra.m_dx, bA, jA);

        // compute: rho^n+1
        ParticleMeshCoupling::SplineBase<s_degX, s_degY, s_degZ> splineRho(position, m_infra.m_plo,
                                                                           m_infra.m_dxi);
        gempic_deposit_rho(rhoarr, splineRho, chargeWeight);
    }
    ASSERT_TRUE(particleLoopRun);

    // compute: div integratedJ
    deRham->div(divJ, J);

    // compute: rho^n - div integratedJ
    rhoMinJ = rhoOld - divJ;

    amrex::Real tol = 1e-1;
    for (amrex::MFIter mfi(rhoMinJ.m_data); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        compare_fields(rhoMinJ.m_data.array(mfi), rho.m_data.array(mfi), bx, tol);
    }
}

TEST_F(HamiltonianSplittingTest, IntegrateBTest)
{
    // Adding particle to one cell
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in

    amrex::Real chargeWeight = 2.0;

    AMREX_D_TERM(amrex::Real xPosOld = 2.5;, amrex::Real yPosOld = 1.7;, amrex::Real zPosOld = 2.2;)
    AMREX_D_TERM(amrex::Real xPosNew = 3.2;, amrex::Real yPosNew = 2.1;, amrex::Real zPosNew = 0.0;)

    amrex::Real valBx = 3.0;
    amrex::Real valBy = 4.0;
    amrex::Real valBz = 5.0;

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::face> B(deRham);
    DeRhamField<Grid::dual, Space::face> J(deRham);

    amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> bA;
    amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> jA;

    amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

    amrex::GpuArray<amrex::Real, 3> dxdy{GEMPIC_D_MULT(1.0, m_infra.m_dx[1], m_infra.m_dx[2]),
                                         GEMPIC_D_MULT(m_infra.m_dx[0], 1.0, m_infra.m_dx[2]),
                                         GEMPIC_D_MULT(m_infra.m_dx[0], m_infra.m_dx[1], 1.0)};

    // fill the B field with the value 3 in xDirection (multiply by area for 2-form dofs)
    amrex::Real Bx{valBx};
    B.m_data[xDir].setVal(Bx * dxdy[0]);

    // fill the B field with the value 4 in yDirection (multiply by area for 2-form dofs)
    amrex::Real By{valBy};
    B.m_data[yDir].setVal(By * dxdy[1]);

    // fill the B field with the value 5 in zDirection (multiply by area for 2-form dofs)
    amrex::Real Bz{valBz};
    B.m_data[zDir].setVal(Bz * dxdy[2]);

    // TEST FOR X DIRECTION
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        // set random initial positions
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> positionOld{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};
        // set position after movement
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position{
            AMREX_D_DECL(xPosNew, yPosNew, zPosNew)};

        for (int cc = 0; cc < 3; cc++)
        {
            jA[cc] = (J.m_data[cc])[pti].array();
            bA[cc] = (B.m_data[cc])[pti].array();
        }

        // initialization: oldSpline = newSpline -> first PrimDiff = 0
        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            positionOld, m_infra.m_plo, m_infra.m_dxi);

        // update the splines in x direction
        spline.template update_1d_splines<xDir>(position[xDir], m_infra.m_plo[xDir],
                                                m_infra.m_dxi[xDir]);
        ParticleMeshCoupling::accumulate_j_integrate_b<xDir>(bfields, spline, chargeWeight,
                                                             m_infra.m_dx, bA, jA);
        // For a constant B, primitive difference is B * (position - positionOld)
        EXPECT_NEAR(Bz * (position[xDir] - positionOld[xDir]), bfields[0], 1e-12);
        EXPECT_NEAR(By * (position[xDir] - positionOld[xDir]), bfields[1], 1e-12);

#if GEMPIC_SPACEDIM > 1
        // update the splines in y direction
        spline.template update_1d_splines<yDir>(position[yDir], m_infra.m_plo[yDir],
                                                m_infra.m_dxi[yDir]);
        ParticleMeshCoupling::accumulate_j_integrate_b<yDir>(bfields, spline, chargeWeight,
                                                             m_infra.m_dx, bA, jA);
        EXPECT_NEAR(Bx * (position[yDir] - positionOld[yDir]), bfields[0], 1e-12);
        EXPECT_NEAR(Bz * (position[yDir] - positionOld[yDir]), bfields[1], 1e-12);

#endif
#if GEMPIC_SPACEDIM == 3
        // update the splines in z direction
        spline.template update_1d_splines<zDir>(position[zDir], m_infra.m_plo[zDir],
                                                m_infra.m_dxi[zDir]);
        ParticleMeshCoupling::accumulate_j_integrate_b<zDir>(bfields, spline, chargeWeight,
                                                             m_infra.m_dx, bA, jA);
        EXPECT_NEAR(By * (position[zDir] - positionOld[zDir]), bfields[0], 1e-12);
        EXPECT_NEAR(Bx * (position[zDir] - positionOld[zDir]), bfields[1], 1e-12);

#endif
    }
}

#if GEMPIC_SPACEDIM < 3
TEST_F(HamiltonianSplittingTest, IntegrateBEulerTest)
{
    // Adding particle to one cell
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in

    amrex::Real chargeWeight = 2.0;
    amrex::Real dt = 0.001;

    // make sure particle doesn't go out of computational domain (+ ghost cells)
    amrex::Real cfl{dt / m_infra.m_dx[0]};
    amrex::Real vx{1 * cfl}, vy{3.1 * cfl}, vz{-2.5 * cfl};
    amrex::GpuArray<amrex::Real, s_vDim> vel{vx, vy, vz};

    AMREX_D_TERM(amrex::Real xPosOld = 2.5;, amrex::Real yPosOld = 1.7;, amrex::Real zPosOld = 2.2;)
    AMREX_D_TERM(amrex::Real xPosNew = xPosOld + dt * vel[0];
                 , amrex::Real yPosNew = yPosOld + dt * vel[1];
                 , amrex::Real zPosNew = zPosOld + dt * vel[2];)

    amrex::Real valBx = 3.0;
    amrex::Real valBy = 4.0;
    amrex::Real valBz = 5.0;

    amrex::Real dtv = 0;
    AMREX_D_TERM(amrex::Real nodeSumX = 0; amrex::Real cellSumXhalf = 0; amrex::Real cellSumX = 0;
                 , amrex::Real nodeSumY = 0; amrex::Real cellSumYhalf = 0; amrex::Real cellSumY = 0;
                 , )

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::face> B(deRham);
    DeRhamField<Grid::dual, Space::face> J(deRham);

    amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> bA;
    amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> jA;

    amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

    amrex::GpuArray<amrex::Real, 3> dxdy{GEMPIC_D_MULT(1.0, m_infra.m_dx[1], m_infra.m_dx[2]),
                                         GEMPIC_D_MULT(m_infra.m_dx[0], 1.0, m_infra.m_dx[2]),
                                         GEMPIC_D_MULT(m_infra.m_dx[0], m_infra.m_dx[1], 1.0)};

    // fill the B field with the value 3 in xDirection (multiply by area for 2-form dofs)
    amrex::Real Bx{valBx};
    B.m_data[xDir].setVal(Bx * dxdy[0]);

    // fill the B field with the value 4 in yDirection (multiply by area for 2-form dofs)
    amrex::Real By{valBy};
    B.m_data[yDir].setVal(By * dxdy[1]);

    // fill the B field with the value 5 in zDirection (multiply by area for 2-form dofs)
    amrex::Real Bz{valBz};
    B.m_data[zDir].setVal(Bz * dxdy[2]);

    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        // set random initial positions
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> positionOld{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};
        // set position after movement
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position{
            AMREX_D_DECL(xPosNew, yPosNew, zPosNew)};

        for (int cc = 0; cc < 3; cc++)
        {
            jA[cc] = (J.m_data[cc])[pti].array();
            bA[cc] = (B.m_data[cc])[pti].array();
        }

        // initialization: oldSpline = newSpline -> first PrimDiff = 0
        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            positionOld, m_infra.m_plo, m_infra.m_dxi);

        // update the splines in x direction
        spline.template update_1d_splines<xDir>(position[xDir], m_infra.m_plo[xDir],
                                                m_infra.m_dxi[xDir]);

        for (int i = 0; i <= s_degX; ++i)
        {
            nodeSumX += spline.m_nodeSplineVals[xDir][i];
            cellSumX += spline.m_cellSplineVals[xDir][i];
            if (i < s_degX)
            {
                cellSumXhalf += spline.m_cellSplineVals[xDir][i];
            }
        }

#if GEMPIC_SPACEDIM == 1
        // TEST FOR Y DIRECTION
        ParticleMeshCoupling::accumulate_j_integrate_b_euler_y(bfields, spline, dtv, chargeWeight,
                                                               bA, jA);

        // B field is constant
        EXPECT_NEAR(nodeSumX * valBx, bfields[0], 1e-12);
        EXPECT_NEAR(cellSumX * dxdy[2] * valBz, bfields[1], 1e-12);

        // TEST FOR Z DIRECTION
        bfields[0] = 0;
        bfields[1] = 0;

        ParticleMeshCoupling::accumulate_j_integrate_b_euler_z(bfields, spline, dtv, chargeWeight,
                                                               bA, jA);

        // B field is constant
        EXPECT_NEAR(cellSumX * dxdy[1] * valBy, bfields[0], 1e-12);
        EXPECT_NEAR(nodeSumX * valBx, bfields[1], 1e-12);

#elif GEMPIC_SPACEDIM == 2
        for (int j = 0; j <= s_degY; ++j)
        {
            nodeSumY += spline.m_nodeSplineVals[yDir][j];
            cellSumY += spline.m_cellSplineVals[yDir][j];
            if (j < s_degY)
            {
                cellSumYhalf += spline.m_cellSplineVals[yDir][j];
            }
        }

        ParticleMeshCoupling::accumulate_j_integrate_b_euler_z(bfields, spline, dtv, chargeWeight,
                                                               bA, jA);

        // B field is constant
        EXPECT_NEAR(cellSumXhalf * nodeSumY * dxdy[1] * valBy, bfields[0], 1e-12);
        EXPECT_NEAR(nodeSumX * cellSumYhalf * dxdy[0] * valBx, bfields[1], 1e-12);

#endif
    }
}
#endif

TEST_F(HamiltonianSplittingTest, ApplyHpiTest)
{
    Gempic::TimeLoop::OperatorHamilton<s_vDim, s_degX, s_degY, s_degZ> operatorHamilton;

    // Adding particle to one cell
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in

    amrex::Real chargeWeight = 2.0;
    amrex::Real dt = 0.1;  // set time step to 1.0 for testing purposes
    amrex::Real chargeOverMass = 0.6;
    AMREX_D_TERM(amrex::Real xPosOld = 2.5;, amrex::Real yPosOld = 3.2;, amrex::Real zPosOld = 2.2;)

    // make sure particle doesn't go out of computational domain (+ ghost cells)
    amrex::Real cfl{dt / m_infra.m_dx[0]};
    amrex::Real vx{1 * cfl}, vy{3.1 * cfl}, vz{-2.5 * cfl};
    amrex::GpuArray<amrex::Real, s_vDim> vel{vx, vy, vz};

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::face> B(deRham);
    DeRhamField<Grid::dual, Space::face> J(deRham);

    amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> bA;
    amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> jA;

    amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

    amrex::GpuArray<amrex::Real, 3> dxdy{GEMPIC_D_MULT(1.0, m_infra.m_dx[1], m_infra.m_dx[2]),
                                         GEMPIC_D_MULT(m_infra.m_dx[0], 1.0, m_infra.m_dx[2]),
                                         GEMPIC_D_MULT(m_infra.m_dx[0], m_infra.m_dx[1], 1.0)};

    // fill the B field with the value 3 in xDirection (multiply by area for 2-form dofs)
    amrex::Real Bx{3.0};
    B.m_data[xDir].setVal(Bx * dxdy[0]);

    // fill the B field with the value 4 in yDirection (multiply by area for 2-form dofs)
    amrex::Real By{4.0};
    B.m_data[yDir].setVal(By * dxdy[1]);

    // fill the B field with the value 5 in zDirection (multiply by area for 2-form dofs)
    amrex::Real Bz{5.0};
    B.m_data[zDir].setVal(Bz * dxdy[2]);

    // TEST FOR ALL DIRECTIONS
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        // set random positions and velocities
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};

        for (int cc = 0; cc < 3; cc++)
        {
            jA[cc] = (J.m_data[cc])[pti].array();
            bA[cc] = (B.m_data[cc])[pti].array();
        }

        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.m_plo, m_infra.m_dxi);

        // apply h_p_i in x direction
        operatorHamilton.template apply_h_p_i<xDir>(position, vel, m_infra, spline, bfields,
                                                    m_infra.m_dx, jA, bA, chargeOverMass,
                                                    chargeWeight, dt);
        // Compare to the exact solution for constant magnetic field
        EXPECT_NEAR(xPosOld + dt * vel[xDir], position[0], 1e-12);
        EXPECT_NEAR(vy - dt * chargeOverMass * vel[xDir] * Bz, vel[yDir], 1e-12);
        EXPECT_NEAR(vz + dt * chargeOverMass * vel[xDir] * By, vel[zDir], 1e-12);

#if GEMPIC_SPACEDIM == 3
        // apply h_p_i in y direction
        // reset random positions
        position[yDir] = yPosOld;

        // reset velocities
        vel[xDir] = vx;
        vel[yDir] = vy;
        vel[zDir] = vz;

        operatorHamilton.template apply_h_p_i<yDir>(position, vel, m_infra, spline, bfields,
                                                    m_infra.m_dx, jA, bA, chargeOverMass,
                                                    chargeWeight, dt);
        // Compare to the exact solution for constant magnetic field
        EXPECT_NEAR(yPosOld + dt * vel[yDir], position[yDir], 1e-12);
        EXPECT_NEAR(vx + dt * chargeOverMass * vel[yDir] * Bz, vel[xDir], 1e-12);
        EXPECT_NEAR(vz - dt * chargeOverMass * vel[yDir] * Bx, vel[zDir], 1e-12);

        position[zDir] = zPosOld;

        // reset velocities
        vel[xDir] = vx;
        vel[yDir] = vy;
        vel[zDir] = vz;

        operatorHamilton.template apply_h_p_i<zDir>(position, vel, m_infra, spline, bfields,
                                                    m_infra.m_dx, jA, bA, chargeOverMass,
                                                    chargeWeight, dt);
        // Compare to the exact solution for constant magnetic field
        EXPECT_NEAR(zPosOld + dt * vel[zDir], position[zDir], 1e-12);
        EXPECT_NEAR(vx - dt * chargeOverMass * vel[zDir] * By, vel[xDir], 1e-12);
        EXPECT_NEAR(vy + dt * chargeOverMass * vel[zDir] * Bx, vel[yDir], 1e-12);

#elif GEMPIC_SPACEDIM == 2
        amrex::Real cellSumX = 0.0;
        amrex::Real cellSumY = 0.0;
        amrex::Real cellSumXhalf = 0.0;
        amrex::Real cellSumYhalf = 0.0;
        amrex::Real nodeSumX = 0.0;
        amrex::Real nodeSumY = 0.0;
        // apply h_p_i in y direction
        // reset random positions
        position[yDir] = yPosOld;

        // reset velocities
        vel[xDir] = vx;
        vel[yDir] = vy;
        vel[zDir] = vz;

        operatorHamilton.template apply_h_p_i<yDir>(position, vel, m_infra, spline, bfields,
                                                    m_infra.m_dx, jA, bA, chargeOverMass,
                                                    chargeWeight, dt);
        // Compare to the exact solution for constant magnetic field
        EXPECT_NEAR(yPosOld + dt * vel[yDir], position[yDir], 1e-12);
        EXPECT_NEAR(vx + dt * chargeOverMass * vel[yDir] * Bz, vel[xDir], 1e-12);
        EXPECT_NEAR(vz - dt * chargeOverMass * vel[yDir] * Bx, vel[zDir], 1e-12);

        // apply h_p_i in z direction
        // reset velocities
        vel[xDir] = vx;
        vel[yDir] = vy;
        vel[zDir] = vz;

        for (int i = 0; i <= s_degX; ++i)
        {
            nodeSumX += spline.m_nodeSplineVals[xDir][i];
            cellSumX += spline.m_cellSplineVals[xDir][i];
            if (i < s_degX)
            {
                cellSumXhalf += spline.m_cellSplineVals[xDir][i];
            }
        }

        for (int j = 0; j <= s_degY; ++j)
        {
            nodeSumY += spline.m_nodeSplineVals[yDir][j];
            cellSumY += spline.m_cellSplineVals[yDir][j];
            if (j < s_degY)
            {
                cellSumYhalf += spline.m_cellSplineVals[yDir][j];
            }
        }

        operatorHamilton.template apply_h_p_i_euler<zDir>(
            vel, m_infra, spline, bfields, m_infra.m_dx, jA, bA, chargeOverMass, chargeWeight, dt);
        EXPECT_NEAR(vx - dt * chargeOverMass * vel[zDir] * By * dxdy[1] * cellSumXhalf * nodeSumY,
                    vel[xDir], 1e-12);
        EXPECT_NEAR(vy + dt * chargeOverMass * vel[zDir] * Bx * dxdy[0] * nodeSumX * cellSumYhalf,
                    vel[yDir], 1e-12);

#elif GEMPIC_SPACEDIM == 1
        amrex::Real cellSumX = 0.0;
        amrex::Real cellSumXhalf = 0.0;
        amrex::Real nodeSumX = 0.0;
        // apply h_p_i in y direction
        // reset velocities
        vel[xDir] = vx;
        vel[yDir] = vy;
        vel[zDir] = vz;

        for (int i = 0; i <= s_degX; ++i)
        {
            nodeSumX += spline.m_nodeSplineVals[xDir][i];
            cellSumX += spline.m_cellSplineVals[xDir][i];
            if (i < s_degX)
            {
                cellSumXhalf += spline.m_cellSplineVals[xDir][i];
            }
        }

        operatorHamilton.template apply_h_p_i_euler<yDir>(
            vel, m_infra, spline, bfields, m_infra.m_dx, jA, bA, chargeOverMass, chargeWeight, dt);
        EXPECT_NEAR(vx + dt * chargeOverMass * vel[yDir] * Bz * dxdy[2] * cellSumX, vel[xDir],
                    1e-12);
        EXPECT_NEAR(vz - dt * chargeOverMass * vel[yDir] * Bx * nodeSumX, vel[zDir], 1e-12);

        // apply h_p_i in z direction
        // reset velocities
        vel[xDir] = vx;
        vel[yDir] = vy;
        vel[zDir] = vz;

        operatorHamilton.template apply_h_p_i_euler<zDir>(
            vel, m_infra, spline, bfields, m_infra.m_dx, jA, bA, chargeOverMass, chargeWeight, dt);
        EXPECT_NEAR(vx - dt * chargeOverMass * vel[zDir] * By * dxdy[1] * cellSumX, vel[xDir],
                    1e-12);
        EXPECT_NEAR(vy + dt * chargeOverMass * vel[zDir] * Bx * nodeSumX, vel[yDir], 1e-12);

#endif
    }
}

// check for a constant E field
TEST_F(HamiltonianSplittingTest, ApplyHeParticleTest)
{
    Gempic::TimeLoop::OperatorHamilton<s_vDim, s_degX, s_degY, s_degZ> operatorHamilton;

    // Adding particle to one cell
    const int numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM>, numParticles> positions{
        {{*m_infra.m_geom.ProbLo()}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particleGroup, m_infra, weights, positions);

    m_particleGroup[0]->Redistribute();  // assign particles to the tile they are in

    amrex::Real dt = 0.1;  // set time step to 0.1 for testing purposes
    amrex::Real chargeOverMass = 0.6;

    AMREX_D_TERM(amrex::Real xPosOld = 2.5;, amrex::Real yPosOld = 3.7;, amrex::Real zPosOld = 2.2;)
    amrex::Real valEx = 3.0;
    amrex::Real valEy = 4.0;
    amrex::Real valEz = 5.0;
    amrex::GpuArray<amrex::Real, 3> v{-3.9, 1.8, 0.9};

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::edge> E(deRham);

    amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> eA;

    amrex::GpuArray<amrex::Real, 3> dx{
        GEMPIC_D_PAD_ONE(m_infra.m_dx[0], m_infra.m_dx[1], m_infra.m_dx[2])};

    amrex::Real Ex{valEx}, Ey{valEy}, Ez{valEz};
    // fill the E field with the value 3 in xDirection (need to multiply by dx for corresponding
    // dof)
    E.m_data[xDir].setVal(Ex * dx[0]);

    // fill the E field with the value 4 in yDirection (need to multiply by dx for corresponding
    // dof)
    E.m_data[yDir].setVal(Ey * dx[1]);

    // fill the E field with the value 5 in zDirection (need to multiply by dx for corresponding
    // dof)
    E.m_data[zDir].setVal(Ez * dx[2]);

    // TEST FOR ALL DIRECTIONS
    for (amrex::ParIter<0, 0, s_vDim + 1, 0> pti(*m_particleGroup[s_spec], 0); pti.isValid(); ++pti)
    {
        // set random positions
        amrex::GpuArray<amrex::Real, GEMPIC_SPACEDIM> position{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};

        auto* const velx = pti.GetStructOfArrays().GetRealData(0).data();
        auto* const vely = pti.GetStructOfArrays().GetRealData(1).data();
        auto* const velz = pti.GetStructOfArrays().GetRealData(2).data();
        long pp = 0;

        velx[0] = v[0] * m_infra.m_dx[0];
        vely[0] = v[1] * m_infra.m_dx[1];
        velz[0] = v[2] * m_infra.m_dx[2];

        amrex::GpuArray<amrex::Real, s_vDim> vel{velx[pp], vely[pp], velz[pp]};

        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.m_plo, m_infra.m_dxi);

        for (int cc = 0; cc < 3; cc++)
        {
            eA[cc] = (E.m_data[cc])[pti].array();
        }
        operatorHamilton.apply_h_e_particle(vel, eA, spline, chargeOverMass, dt);

        EXPECT_NEAR(velx[0] + dt * chargeOverMass * Ex, vel[0], 1e-12);
        EXPECT_NEAR(vely[0] + dt * chargeOverMass * Ey, vel[1], 1e-12);
        EXPECT_NEAR(velz[0] + dt * chargeOverMass * Ez, vel[2], 1e-12);
    }
}

}  // namespace