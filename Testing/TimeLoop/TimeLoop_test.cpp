/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_Particle.H"
#include "GEMPIC_ParticleMeshCoupling.H"
#include "GEMPIC_SplineClass.H"
#include "GEMPIC_Splitting.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Forms;
using namespace Particle;
using ::testing::Mock;

namespace
{
ComputationalDomain get_compdom ()
{
    std::array<amrex::Real, AMREX_SPACEDIM> const domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    std::array<amrex::Real, AMREX_SPACEDIM> const domainHi{
        AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
    amrex::IntVect const nCell{AMREX_D_DECL(20, 20, 20)};
    amrex::IntVect const maxGridSize{AMREX_D_DECL(20, 20, 20)};
    std::array<int, AMREX_SPACEDIM> const isPeriodic{AMREX_D_DECL(1, 1, 1)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic);
}

/**
 * @brief Test fixture. Sets up clean environment before each test of the SplineWithPrimitive class
 */
class HamiltonianSplittingTest : public testing::Test
{
protected:
    // Degree of splines in each direction
    static int const s_degX{2}; // to match degree 2 in script
    static int const s_degY{2};
    static int const s_degZ{2};

    static int const s_vDim{3};
    static int const s_numSpec{1};
    static int const s_spec{0};

    static int const s_nVar = AMREX_SPACEDIM + 1; // x, y, z, t

    inline static int const s_maxSplineDegree{
        AMREX_D_PICK(s_degX, std::max(s_degX, s_degY), std::max(std::max(s_degX, s_degY), s_degZ))};
    inline static int const s_hodgeDegree{2};

    Gempic::Io::Parameters m_parameters{};

    ComputationalDomain m_infra;
    std::vector<std::unique_ptr<ParticleSpecies<s_vDim>>> m_particles;

    HamiltonianSplittingTest() : m_infra{get_compdom()}
    {
        // particle settings
        double charge{1};
        double mass{1};

        m_parameters.set("Particle.species0.charge", charge);
        m_parameters.set("Particle.species0.mass", mass);

        // particles
        m_particles.resize(s_numSpec);
        for (int spec{0}; spec < s_numSpec; spec++)
        {
            m_particles[spec] = std::make_unique<ParticleSpecies<s_vDim>>(spec, m_infra);
        }
    }
};

TEST_F(HamiltonianSplittingTest, AccumulateJTest)
{
    // Setting testing parameters in the field far away from the border to ignore boundary
    // conditions
    amrex::Real chargeWeight = 2.0;
    AMREX_D_TERM(amrex::Real xPosOld = 2.5;, amrex::Real yPosOld = 3.7;, amrex::Real zPosOld = 2.2;)
    AMREX_D_TERM(amrex::Real xPosNew = 2.6;, amrex::Real yPosNew = 4.2;, amrex::Real zPosNew = 2.0;)
    AMREX_D_TERM(amrex::Real primDiffRef = 0;, amrex::Real yNodeVal = 0;, amrex::Real zNodeVal = 0;)

    // Adding particle to one cell
    unsigned int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(xPosOld, yPosOld, zPosOld)}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particles[0].get(), m_infra, weights, positions);

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::face> B(deRham);
    DeRhamField<Grid::dual, Space::face> J(deRham);

    // TEST FOR X DIRECTION
    for (auto& particleGrid : *m_particles[s_spec])
    {
        // set random positions
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};

        amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> bA, jA;

        for (int cc = 0; cc < 3; cc++)
        {
            jA[cc] = (J.m_data[cc])[particleGrid].array();
            bA[cc] = (B.m_data[cc])[particleGrid].array();
        }

        // initialization: oldSpline = newSpline -> first PrimDiff = 0
        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

        // set position after movement in x-direction
        position = {AMREX_D_DECL(xPosNew, yPosOld, zPosOld)};

        spline.template update_1d_splines<xDir>(position[xDir], m_infra.geometry().ProbLo(xDir),
                                                m_infra.geometry().InvCellSize(xDir));
        ParticleMeshCoupling::accumulate_j<xDir>(spline, chargeWeight,
                                                 m_infra.geometry().CellSizeArray(), jA);

        // setup is in a way, that the first spline is in the "bothSplines" region
        amrex::Real primitiveDifference = spline.template compute_primitive_difference<
            xDir, Gempic::ParticleMeshCoupling::Impl::PrimitiveDifferenceRegion::bothSplines>(0);
        primDiffRef = primitiveDifference;

        AMREX_D_TERM(, yNodeVal = spline.m_nodeSplineVals[yDir][0];
                     , zNodeVal = spline.m_nodeSplineVals[zDir][0];)
    }
    for (amrex::MFIter mfi(J.m_data[0]); mfi.isValid(); ++mfi)
    {
        // Checking only the first index !=0, not the full field. Indices according to the current
        // setup!
        CHECK_FIELD(J.m_data[0].array(mfi), mfi.validbox(),
                    {[] (AMREX_D_DECL(int i, int j, int k))
                     { return AMREX_D_TERM(i == 7, &&j == 11, &&k == 6); }},
                    {chargeWeight * GEMPIC_D_MULT(primDiffRef, yNodeVal, zNodeVal)}, {}, 1e-12);
    }

    // TEST FOR Y DIRECTION
#if AMREX_SPACEDIM > 1
    amrex::Real xNodeVal = 0;
    for (auto& particleGrid : *m_particles[s_spec])
    {
        // set random positions
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};

        amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> bA, jA;

        for (int cc = 0; cc < 3; cc++)
        {
            jA[cc] = (J.m_data[cc])[particleGrid].array();
            bA[cc] = (B.m_data[cc])[particleGrid].array();
        }

        // initialization: oldSpline = newSpline -> first PrimDiff = 0
        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

        // set position after movement in y-direction
        position = {AMREX_D_DECL(xPosOld, yPosNew, zPosOld)};

        spline.template update_1d_splines<yDir>(position[yDir], m_infra.geometry().ProbLo(yDir),
                                                m_infra.geometry().InvCellSize(yDir));
        ParticleMeshCoupling::accumulate_j<yDir>(spline, chargeWeight,
                                                 m_infra.geometry().CellSizeArray(), jA);

        // setup is in a way, that the second index is in the "bothSplines" region
        amrex::Real primitiveDifference = spline.template compute_primitive_difference<
            yDir, Gempic::ParticleMeshCoupling::Impl::PrimitiveDifferenceRegion::bothSplines>(1);
        primDiffRef = primitiveDifference;

        AMREX_D_TERM(xNodeVal = spline.m_nodeSplineVals[xDir][0];
                     , , zNodeVal = spline.m_nodeSplineVals[zDir][0];)
    }
    for (amrex::MFIter mfi(J.m_data[1]); mfi.isValid(); ++mfi)
    {
        CHECK_FIELD(J.m_data[1].array(mfi), mfi.validbox(),
                    {[] (AMREX_D_DECL(int i, int j, int k))
                     { return AMREX_D_TERM(i == 7, &&j == 12, &&k == 6); }},
                    {chargeWeight * GEMPIC_D_MULT(xNodeVal, primDiffRef, zNodeVal)}, {}, 1e-12);
    }

    // TEST FOR Z DIRECTION
#if AMREX_SPACEDIM > 2
    for (auto& particleGrid : *m_particles[s_spec])
    {
        // set random positions
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};

        amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> bA, jA;

        for (int cc = 0; cc < 3; cc++)
        {
            jA[cc] = (J.m_data[cc])[particleGrid].array();
            bA[cc] = (B.m_data[cc])[particleGrid].array();
        }

        // initialization: oldSpline = newSpline -> first PrimDiff = 0
        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

        // set position after movement in y-direction
        position = {AMREX_D_DECL(xPosOld, yPosOld, zPosNew)};

        spline.template update_1d_splines<zDir>(position[zDir], m_infra.geometry().ProbLo(zDir),
                                                m_infra.geometry().InvCellSize(zDir));
        ParticleMeshCoupling::accumulate_j<zDir>(spline, chargeWeight,
                                                 m_infra.geometry().CellSizeArray(), jA);

        // setup is in a way, that the first spline is in the "firstSpline" region
        amrex::Real primitiveDifference = spline.template compute_primitive_difference<
            zDir, Gempic::ParticleMeshCoupling::Impl::PrimitiveDifferenceRegion::firstSpline>(0);
        primDiffRef = primitiveDifference;

        xNodeVal = spline.m_nodeSplineVals[xDir][0];
        yNodeVal = spline.m_nodeSplineVals[yDir][0];
    }

    for (amrex::MFIter mfi(J.m_data[2]); mfi.isValid(); ++mfi)
    {
        CHECK_FIELD(J.m_data[2].array(mfi), mfi.validbox(),
                    {[] (AMREX_D_DECL(int i, int j, int k))
                     { return AMREX_D_TERM(i == 7, &&j == 11, &&k == 5); }},
                    {chargeWeight * xNodeVal * yNodeVal * primDiffRef}, {}, 1e-12);
    }
#endif
#endif
}

#if AMREX_SPACEDIM < 3
// tests euler for both non-relativistic as well as relativistic implementation
TEST_F(HamiltonianSplittingTest, AccumulateJEulerTest)
{
    // set the particle positions to a random position within the box
    AMREX_D_TERM(amrex::Real xPosOld = 2.5;, amrex::Real yPosOld = 3.7;, amrex::Real zPosOld = 0.0;)
    // Adding particle to one cell
    int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(xPosOld, yPosOld, zPosOld)}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particles[0].get(), m_infra, weights, positions);

    // Setting testing parameters in the field far away from the border to ignore boundary
    // conditions
    amrex::Real chargeWeight = 2.0;
    amrex::Real dt = 0.001;

    // make sure particle doesn't go out of computational domain (+ ghost cells)
    amrex::Real cfl{dt / m_infra.geometry().CellSize(xDir)};
    amrex::Real vx{1 * cfl}, vy{3.1 * cfl}, vz{-2.5 * cfl};
    amrex::GpuArray<amrex::Real, s_vDim> vel{vx, vy, vz};

    AMREX_D_TERM(amrex::Real xNodeVal = 0;, amrex::Real yNodeVal = 0;, )
    amrex::Real dtv = 0;

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::face> B(deRham);
    DeRhamField<Grid::dual, Space::face> J(deRham);
    DeRhamField<Grid::dual, Space::face> jrel(deRham);

    // TEST FOR X DIRECTION
    for (auto& particleGrid : *m_particles[s_spec])
    {
        // set random positions
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};
        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> bA, jA, jArel;

        for (int cc = 0; cc < 3; cc++)
        {
            bA[cc] = (B.m_data[cc])[particleGrid].array();
            jA[cc] = (J.m_data[cc])[particleGrid].array();
            jArel[cc] = (jrel.m_data[cc])[particleGrid].array();
        }

        // initialization: oldSpline = newSpline -> first PrimDiff = 0
        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

        dtv = dt * vel[yDir];
        xNodeVal = spline.m_nodeSplineVals[xDir][0];

#if AMREX_SPACEDIM == 1
        ParticleMeshCoupling::accumulate_j_integrate_b_euler_y(bfields, spline, dtv, chargeWeight,
                                                               bA, jA);
        ParticleMeshCoupling::accumulate_j_integrate_b_euler_z(bfields, spline, dtv, chargeWeight,
                                                               bA, jA);
        ParticleMeshCoupling::accumulate_j_euler_y(spline, dtv, chargeWeight, jArel);
        ParticleMeshCoupling::accumulate_j_euler_z(spline, dtv, chargeWeight, jArel);
#elif AMREX_SPACEDIM == 2
        yNodeVal = spline.m_nodeSplineVals[yDir][0];
        ParticleMeshCoupling::accumulate_j_integrate_b_euler_z(bfields, spline, dtv, chargeWeight,
                                                               bA, jA);
        ParticleMeshCoupling::accumulate_j_euler_z(spline, dtv, chargeWeight, jArel);
#endif
    }

    for (int dir{AMREX_SPACEDIM}; dir < 3; ++dir)
    {
        // check the first non-zero entry of the J field in xDir
        for (amrex::MFIter mfi(J.m_data[dir]); mfi.isValid(); ++mfi)
        {
            CHECK_FIELD(J.m_data[dir].array(mfi), mfi.validbox(),
                        {[] (AMREX_D_DECL(int i, int j, int k))
                         { return AMREX_D_TERM(i == 7, &&j == 11, &&k == 0); }},
                        {chargeWeight * dtv * GEMPIC_D_MULT(xNodeVal, yNodeVal, 1.0)}, {}, 1e-12);
        }

        for (amrex::MFIter mfi(jrel.m_data[dir]); mfi.isValid(); ++mfi)
        {
            CHECK_FIELD(jrel.m_data[dir].array(mfi), mfi.validbox(),
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
    // set the particle positions to a random position within the box
    AMREX_D_TERM(amrex::Real xPosOld = 2.5;, amrex::Real yPosOld = 3.7;, amrex::Real zPosOld = 2.2;)
    // Adding a few particles
    int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(xPosOld, yPosOld, zPosOld)}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particles[0].get(), m_infra, weights, positions);

    amrex::Real chargeWeight = 1.0;
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
    for (auto& particleGrid : *m_particles[s_spec])
    {
        // set random positions (positions of particles never used here)
        // Check if this is OK especially when there are several particle tiles
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};
        amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

        amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> bA, jA;
        amrex::Array4<amrex::Real> const& rhoOldarr = rhoOld.m_data[particleGrid].array();
        amrex::Array4<amrex::Real> const& rhoarr = rho.m_data[particleGrid].array();

        for (int cc = 0; cc < 3; cc++)
        {
            jA[cc] = (J.m_data[cc])[particleGrid].array();
            bA[cc] = (B.m_data[cc])[particleGrid].array();
        }

        particleLoopRun = true;

        // compute: rho^n
        ParticleMeshCoupling::SplineBase<s_degX, s_degY, s_degZ> splineRhoOld(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

        ParticleMeshCoupling::deposit_rho(rhoOldarr, splineRhoOld, chargeWeight);

        // compute: integratedJ
        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

        // set position after movement in x-direction
        position = {AMREX_D_DECL(xPosNew, yPosOld, zPosOld)};

        spline.template update_1d_splines<xDir>(position[xDir], m_infra.geometry().ProbLo(xDir),
                                                m_infra.geometry().InvCellSize(xDir));
        ParticleMeshCoupling::accumulate_j_integrate_b<xDir>(
            bfields, spline, chargeWeight, m_infra.geometry().CellSizeArray(), bA, jA);

        // compute: rho^n+1
        ParticleMeshCoupling::SplineBase<s_degX, s_degY, s_degZ> splineRho(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());
        deposit_rho(rhoarr, splineRho, chargeWeight);
    }
    ASSERT_TRUE(particleLoopRun);

    // compute: div integratedJ
    div(divJ, J);

    // compute: rho^n - div integratedJ
    rhoMinJ = rhoOld - divJ;

    amrex::Real tol = 1e-1;
    for (amrex::MFIter mfi(rhoMinJ.m_data); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.tilebox();
        COMPARE_FIELDS(rhoMinJ.m_data.array(mfi), rho.m_data.array(mfi), bx, tol);
    }
}

TEST_F(HamiltonianSplittingTest, IntegrateBTest)
{
    // Setting testing parameters in the field far away from the border to ignore boundary
    // conditions
    amrex::Real chargeWeight = 2.0;
    AMREX_D_TERM(amrex::Real xPosOld = 2.5;, amrex::Real yPosOld = 1.7;, amrex::Real zPosOld = 2.2;)
    AMREX_D_TERM(amrex::Real xPosNew = 3.2;, amrex::Real yPosNew = 2.1;, amrex::Real zPosNew = 0.0;)

    // Adding particle to one cell
    int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(xPosOld, yPosOld, zPosOld)}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particles[0].get(), m_infra, weights, positions);

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

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = m_infra.geometry().CellSizeArray();
    amrex::GpuArray<amrex::Real, 3> dxdy{GEMPIC_D_MULT(1.0, dx[1], dx[2]),
                                         GEMPIC_D_MULT(dx[0], 1.0, dx[2]),
                                         GEMPIC_D_MULT(dx[0], dx[1], 1.0)};

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
    for (auto& particleGrid : *m_particles[s_spec])
    {
        // set random initial positions
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionOld{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};
        // set position after movement
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{
            AMREX_D_DECL(xPosNew, yPosNew, zPosNew)};

        for (int cc = 0; cc < 3; cc++)
        {
            jA[cc] = (J.m_data[cc])[particleGrid].array();
            bA[cc] = (B.m_data[cc])[particleGrid].array();
        }

        // initialization: oldSpline = newSpline -> first PrimDiff = 0
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dxi = m_infra.geometry().InvCellSizeArray();
        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            positionOld, m_infra.geometry().ProbLoArray(), dxi);

        // update the splines in x direction
        spline.template update_1d_splines<xDir>(position[xDir], m_infra.geometry().ProbLo(xDir),
                                                dxi[xDir]);
        ParticleMeshCoupling::accumulate_j_integrate_b<xDir>(
            bfields, spline, chargeWeight, m_infra.geometry().CellSizeArray(), bA, jA);
        // For a constant B, primitive difference is B * (position - positionOld)
        EXPECT_NEAR(Bz * (position[xDir] - positionOld[xDir]), bfields[0], 1e-12);
        EXPECT_NEAR(By * (position[xDir] - positionOld[xDir]), bfields[1], 1e-12);

#if AMREX_SPACEDIM > 1
        // update the splines in y direction
        spline.template update_1d_splines<yDir>(position[yDir], m_infra.geometry().ProbLo(yDir),
                                                dxi[yDir]);
        ParticleMeshCoupling::accumulate_j_integrate_b<yDir>(
            bfields, spline, chargeWeight, m_infra.geometry().CellSizeArray(), bA, jA);
        EXPECT_NEAR(Bx * (position[yDir] - positionOld[yDir]), bfields[0], 1e-12);
        EXPECT_NEAR(Bz * (position[yDir] - positionOld[yDir]), bfields[1], 1e-12);

#endif
#if AMREX_SPACEDIM == 3
        // update the splines in z direction
        spline.template update_1d_splines<zDir>(position[zDir], m_infra.geometry().ProbLo(zDir),
                                                dxi[zDir]);
        ParticleMeshCoupling::accumulate_j_integrate_b<zDir>(
            bfields, spline, chargeWeight, m_infra.geometry().CellSizeArray(), bA, jA);
        EXPECT_NEAR(By * (position[zDir] - positionOld[zDir]), bfields[0], 1e-12);
        EXPECT_NEAR(Bx * (position[zDir] - positionOld[zDir]), bfields[1], 1e-12);

#endif
    }
}

#if AMREX_SPACEDIM < 3
TEST_F(HamiltonianSplittingTest, IntegrateBEulerTest)
{
    // Setting testing parameters in the field far away from the border to ignore boundary
    // conditions
    amrex::Real chargeWeight = 2.0;
    amrex::Real dt = 0.001;
    // make sure particle doesn't go out of computational domain (+ ghost cells)
    amrex::Real cfl{dt / m_infra.geometry().CellSize(xDir)};
    amrex::Real vx{1 * cfl}, vy{3.1 * cfl}, vz{-2.5 * cfl};
    amrex::GpuArray<amrex::Real, s_vDim> vel{vx, vy, vz};
    AMREX_D_TERM(amrex::Real xPosOld = 2.5;, amrex::Real yPosOld = 1.7;, amrex::Real zPosOld = 2.2;)
    AMREX_D_TERM(amrex::Real xPosNew = xPosOld + dt * vel[0];
                 , amrex::Real yPosNew = yPosOld + dt * vel[1];
                 , amrex::Real zPosNew = zPosOld + dt * vel[2];)

    // Adding particle to one cell
    int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(xPosOld, yPosOld, zPosOld)}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particles[0].get(), m_infra, weights, positions);

    amrex::Real valBx = 3.0;
    amrex::Real valBy = 4.0;
    amrex::Real valBz = 5.0;

    amrex::Real dtv = 0;
    AMREX_D_TERM(amrex::Real nodeSumX = 0; amrex::Real cellSumXhalf = 0;
                 [[maybe_unused]] amrex::Real cellSumX = 0;, amrex::Real nodeSumY = 0;
                 amrex::Real cellSumYhalf = 0; [[maybe_unused]] amrex::Real cellSumY = 0;, )

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::face> B(deRham);
    DeRhamField<Grid::dual, Space::face> J(deRham);

    amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> bA;
    amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> jA;

    amrex::GpuArray<amrex::Real, 2> bfields{0., 0.};

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = m_infra.geometry().CellSizeArray();
    amrex::GpuArray<amrex::Real, 3> dxdy{GEMPIC_D_MULT(1.0, dx[1], dx[2]),
                                         GEMPIC_D_MULT(dx[0], 1.0, dx[2]),
                                         GEMPIC_D_MULT(dx[0], dx[1], 1.0)};

    // fill the B field with the value 3 in xDirection (multiply by area for 2-form dofs)
    amrex::Real Bx{valBx};
    B.m_data[xDir].setVal(Bx * dxdy[0]);

    // fill the B field with the value 4 in yDirection (multiply by area for 2-form dofs)
    amrex::Real By{valBy};
    B.m_data[yDir].setVal(By * dxdy[1]);

    // fill the B field with the value 5 in zDirection (multiply by area for 2-form dofs)
    amrex::Real Bz{valBz};
    B.m_data[zDir].setVal(Bz * dxdy[2]);

    for (auto& particleGrid : *m_particles[s_spec])
    {
        // set random initial positions
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> positionOld{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};
        // set position after movement
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{
            AMREX_D_DECL(xPosNew, yPosNew, zPosNew)};

        for (int cc = 0; cc < 3; cc++)
        {
            jA[cc] = (J.m_data[cc])[particleGrid].array();
            bA[cc] = (B.m_data[cc])[particleGrid].array();
        }

        // initialization: oldSpline = newSpline -> first PrimDiff = 0
        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            positionOld, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

        // update the splines in x direction
        spline.template update_1d_splines<xDir>(position[xDir], m_infra.geometry().ProbLo(xDir),
                                                m_infra.geometry().InvCellSize(xDir));

        for (int i = 0; i <= s_degX; ++i)
        {
            nodeSumX += spline.m_nodeSplineVals[xDir][i];
            if (i < s_degX)
            {
                cellSumX += spline.m_cellSplineVals[xDir][i];
                cellSumXhalf += spline.m_cellSplineVals[xDir][i];
            }
        }

#if AMREX_SPACEDIM == 1
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
        EXPECT_NEAR(cellSumXhalf * dxdy[1] * valBy, bfields[0], 1e-12);
        EXPECT_NEAR(nodeSumX * valBx, bfields[1], 1e-12);

#elif AMREX_SPACEDIM == 2
        for (int j = 0; j <= s_degY; ++j)
        {
            nodeSumY += spline.m_nodeSplineVals[yDir][j];
            if (j < s_degY)
            {
                cellSumY += spline.m_cellSplineVals[yDir][j];
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

    // Setting testing parameters in the field far away from the border to ignore boundary
    // conditions
    AMREX_D_TERM(amrex::Real xPosOld = 2.5;, amrex::Real yPosOld = 3.2;, amrex::Real zPosOld = 2.2;)

    // Adding particle to one cell
    int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(xPosOld, yPosOld, zPosOld)}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particles[0].get(), m_infra, weights, positions);

    amrex::Real chargeWeight = 2.0;
    amrex::Real dt = 0.1; // set time step to 1.0 for testing purposes
    amrex::Real chargeOverMass = 0.6;

    // make sure particle doesn't go out of computational domain (+ ghost cells)
    amrex::Real cfl{dt / m_infra.geometry().CellSize(0)};
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

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = m_infra.geometry().CellSizeArray();
    amrex::GpuArray<amrex::Real, 3> dxdy{GEMPIC_D_MULT(1.0, dx[1], dx[2]),
                                         GEMPIC_D_MULT(dx[0], 1.0, dx[2]),
                                         GEMPIC_D_MULT(dx[0], dx[1], 1.0)};

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
    for (auto& particleGrid : *m_particles[s_spec])
    {
        // set random positions and velocities
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};

        for (int cc = 0; cc < 3; cc++)
        {
            jA[cc] = (J.m_data[cc])[particleGrid].array();
            bA[cc] = (B.m_data[cc])[particleGrid].array();
        }

        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

        // apply h_p_i in x direction
        operatorHamilton.template apply_h_p_i<xDir>(position, vel, m_infra, spline, bfields,
                                                    m_infra.geometry().CellSizeArray(), jA, bA,
                                                    chargeOverMass, chargeWeight, dt);
        // Compare to the exact solution for constant magnetic field
        EXPECT_NEAR(xPosOld + dt * vel[xDir], position[0], 1e-12);
        EXPECT_NEAR(vy - dt * chargeOverMass * vel[xDir] * Bz, vel[yDir], 1e-12);
        EXPECT_NEAR(vz + dt * chargeOverMass * vel[xDir] * By, vel[zDir], 1e-12);

#if AMREX_SPACEDIM == 3
        // apply h_p_i in y direction
        // reset random positions
        position[yDir] = yPosOld;

        // reset velocities
        vel[xDir] = vx;
        vel[yDir] = vy;
        vel[zDir] = vz;

        operatorHamilton.template apply_h_p_i<yDir>(position, vel, m_infra, spline, bfields,
                                                    m_infra.geometry().CellSizeArray(), jA, bA,
                                                    chargeOverMass, chargeWeight, dt);
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
                                                    m_infra.geometry().CellSizeArray(), jA, bA,
                                                    chargeOverMass, chargeWeight, dt);
        // Compare to the exact solution for constant magnetic field
        EXPECT_NEAR(zPosOld + dt * vel[zDir], position[zDir], 1e-12);
        EXPECT_NEAR(vx - dt * chargeOverMass * vel[zDir] * By, vel[xDir], 1e-12);
        EXPECT_NEAR(vy + dt * chargeOverMass * vel[zDir] * Bx, vel[yDir], 1e-12);

#elif AMREX_SPACEDIM == 2
        [[maybe_unused]] amrex::Real cellSumX = 0.0;
        [[maybe_unused]] amrex::Real cellSumY = 0.0;
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
                                                    m_infra.geometry().CellSizeArray(), jA, bA,
                                                    chargeOverMass, chargeWeight, dt);
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
            if (i < s_degX)
            {
                cellSumX += spline.m_cellSplineVals[xDir][i];
                cellSumXhalf += spline.m_cellSplineVals[xDir][i];
            }
        }

        for (int j = 0; j <= s_degY; ++j)
        {
            nodeSumY += spline.m_nodeSplineVals[yDir][j];
            if (j < s_degY)
            {
                cellSumY += spline.m_cellSplineVals[yDir][j];
                cellSumYhalf += spline.m_cellSplineVals[yDir][j];
            }
        }

        operatorHamilton.template apply_h_p_i_euler<zDir>(vel, m_infra, spline, bfields,
                                                          m_infra.geometry().CellSizeArray(), jA,
                                                          bA, chargeOverMass, chargeWeight, dt);
        EXPECT_NEAR(vx - dt * chargeOverMass * vel[zDir] * By * dxdy[1] * cellSumXhalf * nodeSumY,
                    vel[xDir], 1e-12);
        EXPECT_NEAR(vy + dt * chargeOverMass * vel[zDir] * Bx * dxdy[0] * nodeSumX * cellSumYhalf,
                    vel[yDir], 1e-12);

#elif AMREX_SPACEDIM == 1
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
            if (i < s_degX)
            {
                cellSumX += spline.m_cellSplineVals[xDir][i];
                cellSumXhalf += spline.m_cellSplineVals[xDir][i];
            }
        }

        operatorHamilton.template apply_h_p_i_euler<yDir>(vel, m_infra, spline, bfields,
                                                          m_infra.geometry().CellSizeArray(), jA,
                                                          bA, chargeOverMass, chargeWeight, dt);
        EXPECT_NEAR(vx + dt * chargeOverMass * vel[yDir] * Bz * dxdy[2] * cellSumX, vel[xDir],
                    1e-12);
        EXPECT_NEAR(vz - dt * chargeOverMass * vel[yDir] * Bx * nodeSumX, vel[zDir], 1e-12);

        // apply h_p_i in z direction
        // reset velocities
        vel[xDir] = vx;
        vel[yDir] = vy;
        vel[zDir] = vz;

        operatorHamilton.template apply_h_p_i_euler<zDir>(vel, m_infra, spline, bfields,
                                                          m_infra.geometry().CellSizeArray(), jA,
                                                          bA, chargeOverMass, chargeWeight, dt);
        EXPECT_NEAR(vx - dt * chargeOverMass * vel[zDir] * By * dxdy[1] * cellSumXhalf, vel[xDir],
                    1e-12);
        EXPECT_NEAR(vy + dt * chargeOverMass * vel[zDir] * Bx * nodeSumX, vel[yDir], 1e-12);

#endif
    }
}

// check for a constant E field
TEST_F(HamiltonianSplittingTest, ApplyHeParticleTest)
{
    Gempic::TimeLoop::OperatorHamilton<s_vDim, s_degX, s_degY, s_degZ> operatorHamilton;

    // Setting testing parameters in the field far away from the border to ignore boundary
    // conditions
    AMREX_D_TERM(amrex::Real xPosOld = 2.5;, amrex::Real yPosOld = 3.7;, amrex::Real zPosOld = 2.2;)
    // Adding particle to one cell
    int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(xPosOld, yPosOld, zPosOld)}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particles[0].get(), m_infra, weights, positions);

    amrex::Real dt = 0.1; // set time step to 0.1 for testing purposes
    amrex::Real chargeOverMass = 0.6;

    amrex::Real valEx = 3.0;
    amrex::Real valEy = 4.0;
    amrex::Real valEz = 5.0;
    amrex::GpuArray<amrex::Real, 3> v{-3.9, 1.8, 0.9};

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::edge> E(deRham);

    amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> eA;

    // These code sections would be much simpler if we consider only 3-D implementations and
    // design away special cases.
    amrex::GpuArray<amrex::Real, 3> dx{GEMPIC_D_PAD_ONE(m_infra.geometry().CellSize(xDir),
                                                        m_infra.geometry().CellSize(yDir),
                                                        m_infra.geometry().CellSize(zDir))};

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
    for (auto& particleGrid : *m_particles[s_spec])
    {
        // set random positions
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};

        auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
        auto const ii = m_particles[s_spec]->get_data_indices();

        ptd.rdata(ii.m_ivelx)[0] = v[xDir] * dx[xDir];
        ptd.rdata(ii.m_ively)[0] = v[yDir] * dx[yDir];
        ptd.rdata(ii.m_ivelz)[0] = v[zDir] * dx[zDir];

        amrex::GpuArray<amrex::Real, s_vDim> vel{ptd.rdata(ii.m_ivelx)[0], ptd.rdata(ii.m_ively)[0],
                                                 ptd.rdata(ii.m_ivelz)[0]};

        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

        for (int cc = 0; cc < 3; cc++)
        {
            eA[cc] = (E.m_data[cc])[particleGrid].array();
        }
        operatorHamilton.apply_h_e_particle(vel, eA, spline, chargeOverMass, dt);

        EXPECT_NEAR(ptd.rdata(ii.m_ivelx)[0] + dt * chargeOverMass * Ex, vel[0], 1e-12);
        EXPECT_NEAR(ptd.rdata(ii.m_ively)[0] + dt * chargeOverMass * Ey, vel[1], 1e-12);
        EXPECT_NEAR(ptd.rdata(ii.m_ivelz)[0] + dt * chargeOverMass * Ez, vel[2], 1e-12);
    }
}

// For alpha = 2pi, check if the rotation ends at the starting position,
// i.e., if we have the matrix vector multiplication
// [1 0 0]   [ux]
// [0 1 0] * [uy]
// [0 0 1]   [uv]
TEST_F(HamiltonianSplittingTest, ApplyHpireluPositionTest)
{
    Gempic::TimeLoop::OperatorHamilton<s_vDim, s_degX, s_degY, s_degZ> operatorHamilton;

    // set the particle positions to a random position within the box
    AMREX_D_TERM(amrex::Real xPosOld = 2.5;, amrex::Real yPosOld = 3.7;, amrex::Real zPosOld = 2.2;)
    // Adding particle to one cell using these position
    int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(xPosOld, yPosOld, zPosOld)}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particles[0].get(), m_infra, weights, positions);

    // setup constant random values for the B field
    amrex::Real Bx{3.0};
    amrex::Real By{4.0};
    amrex::Real Bz{5.0};

    amrex::Real dt = 0.1; // set time step to 0.1 for testing purposes

    // start with random velocities
    amrex::GpuArray<amrex::Real, 3> v{-3.9, 1.8, 0.9};

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::face> B(deRham);

    amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> bA;

    // These code sections would be much simpler if we consider only 3-D implementations and
    // design away special cases.
    amrex::GpuArray<amrex::Real, 3> dx{GEMPIC_D_PAD_ONE(m_infra.geometry().CellSize(xDir),
                                                        m_infra.geometry().CellSize(yDir),
                                                        m_infra.geometry().CellSize(zDir))};

    amrex::GpuArray<amrex::Real, 3> dxdy{GEMPIC_D_MULT(1.0, dx[1], dx[2]),
                                         GEMPIC_D_MULT(dx[0], 1.0, dx[2]),
                                         GEMPIC_D_MULT(dx[0], dx[1], 1.0)};

    // fill the B-field with constant values
    B.m_data[xDir].setVal(Bx * dxdy[0]);
    B.m_data[yDir].setVal(By * dxdy[1]);
    B.m_data[zDir].setVal(Bz * dxdy[2]);

    for (auto& particleGrid : *m_particles[s_spec])
    {
        // set random positions
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};

        auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
        auto const ii = m_particles[s_spec]->get_data_indices();

        ptd.rdata(ii.m_ivelx)[0] = v[xDir] * dx[xDir];
        ptd.rdata(ii.m_ively)[0] = v[yDir] * dx[yDir];
        ptd.rdata(ii.m_ivelz)[0] = v[zDir] * dx[zDir];

        amrex::Real gamma = sqrt(1 + (ptd.rdata(ii.m_ivelx)[0] * ptd.rdata(ii.m_ivelx)[0] +
                                      ptd.rdata(ii.m_ively)[0] * ptd.rdata(ii.m_ively)[0] +
                                      ptd.rdata(ii.m_ivelz)[0] * ptd.rdata(ii.m_ivelz)[0]) /
                                         1.0);

        // set q/m = 2pi*gamma/(dt*|B|) to make sure there is exactly one rotation
        amrex::Real chargeOverMass = 2 * M_PI * gamma / (dt * sqrt(Bx * Bx + By * By + Bz * Bz));
        amrex::GpuArray<amrex::Real, s_vDim> vel{ptd.rdata(ii.m_ivelx)[0], ptd.rdata(ii.m_ively)[0],
                                                 ptd.rdata(ii.m_ivelz)[0]};

        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

        for (int cc = 0; cc < 3; cc++)
        {
            bA[cc] = (B.m_data[cc])[particleGrid].array();
        }

        operatorHamilton.apply_h_p_rel_u(vel, spline, bA, chargeOverMass, dt, gamma);

        // the velocity is the same after one rotation
        EXPECT_NEAR(ptd.rdata(ii.m_ivelx)[0], vel[0], 1e-12);
        EXPECT_NEAR(ptd.rdata(ii.m_ively)[0], vel[1], 1e-12);
        EXPECT_NEAR(ptd.rdata(ii.m_ivelz)[0], vel[2], 1e-12);
    }
}

// check if, for a constant magnetic field of one, the original rotation matrices are recovered
// by the apply_h_p_rel_u function, i.e.,
// in x-Dir:
// [1     0           0     ]
// [0 cos(alpha) -sin(alpha)]
// [0 sin(alpha)  cos(alpha)]
// in y-Dir:
// [ cos(alpha)  0 sin(alpha)]
// [     0       1     0     ]
// [-sin(alpha)] 0 cos(alpha)]
// in z-Dir:
// [cos(alpha) -sin(alpha) 0]
// [sin(alpha)  cos(alpha) 0]
// [    0           0      1]
TEST_F(HamiltonianSplittingTest, ApplyHpireluRotationMatrixTest)
{
    Gempic::TimeLoop::OperatorHamilton<s_vDim, s_degX, s_degY, s_degZ> operatorHamilton;

    // set the particle positions to a random position within the box
    AMREX_D_TERM(amrex::Real xPosOld = 2.5;, amrex::Real yPosOld = 3.7;, amrex::Real zPosOld = 2.2;)
    // Adding particle to one cell
    int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(xPosOld, yPosOld, zPosOld)}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particles[0].get(), m_infra, weights, positions);

    amrex::Real dt = 0.1;             // set time step to 0.1 for testing purposes
    amrex::Real chargeOverMass = 0.6; // random number for testing purposes

    // make sure that the velocity we multiply with is 1, so that we just get the rotation
    amrex::GpuArray<amrex::Real, 3> v{-3.9, 1.8, 0.9};

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::face> B(deRham);

    amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> bA;

    // These code sections would be much simpler if we consider only 3-D implementations and
    // design away special cases.
    amrex::GpuArray<amrex::Real, 3> dx{GEMPIC_D_PAD_ONE(m_infra.geometry().CellSize(xDir),
                                                        m_infra.geometry().CellSize(yDir),
                                                        m_infra.geometry().CellSize(zDir))};

    amrex::GpuArray<amrex::Real, 3> dxdy{GEMPIC_D_MULT(1.0, dx[yDir], dx[zDir]),
                                         GEMPIC_D_MULT(dx[xDir], 1.0, dx[zDir]),
                                         GEMPIC_D_MULT(dx[xDir], dx[yDir], 1.0)};

    amrex::GpuArray<amrex::Real, 3> velold{v[xDir] * dx[xDir], v[yDir] * dx[yDir],
                                           v[zDir] * dx[zDir]};

    auto rotate = [&] (amrex::Real Bx, amrex::Real By, amrex::Real Bz)
    {
        // fill the B-field with constant values
        B.m_data[xDir].setVal(Bx * dxdy[0]);
        B.m_data[yDir].setVal(By * dxdy[1]);
        B.m_data[zDir].setVal(Bz * dxdy[2]);

        // set random positions
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};

        amrex::GpuArray<amrex::Real, 3> velocity{v[xDir] * dx[xDir], v[yDir] * dx[yDir],
                                                 v[zDir] * dx[zDir]};

        amrex::Real gamma =
            sqrt(1 + (velocity[xDir] * velocity[xDir] + velocity[yDir] * velocity[yDir] +
                      velocity[zDir] * velocity[zDir]) /
                         1.0);

        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());
        for (auto& particleGrid : *m_particles[s_spec])
        {
            for (int cc = 0; cc < 3; cc++)
            {
                bA[cc] = (B.m_data[cc])[particleGrid].array();
            }
        }
        operatorHamilton.apply_h_p_rel_u(velocity, spline, bA, chargeOverMass, dt, gamma);

        amrex::Real alpha = -chargeOverMass * dt / gamma;

        return std::tuple{velocity, alpha};
    };

    // TEST FOR ROTATION MATRIX AROUND THE X-AXIS
    amrex::Real Bx{1.0};
    amrex::Real By{0.0};
    amrex::Real Bz{0.0};
    auto [velRotX, alphaX] = rotate(Bx, By, Bz);
    // first row of rotation matrix around the x axis times (vx, vy, vz)'
    EXPECT_NEAR(1 * velold[xDir], velRotX[xDir], 1e-12);
    // second row of rotation matrix around the x axis times (vx, vy, vz)'
    EXPECT_NEAR(velold[yDir] * cos(alphaX) - velold[zDir] * sin(alphaX), velRotX[yDir], 1e-12);
    // third row of rotation matrix around the x axis times (vx, vy, vz)'
    EXPECT_NEAR(velold[yDir] * sin(alphaX) + velold[zDir] * cos(alphaX), velRotX[zDir], 1e-12);

    // TEST FOR ROTATION MATRIX AROUND THE Y-AXIS
    Bx = 0.0;
    By = 1.0;
    Bz = 0.0;
    auto [velRotY, alphaY] = rotate(Bx, By, Bz);
    // first row of rotation matrix around the y axis times (vx, vy, vz)'
    EXPECT_NEAR(velold[xDir] * cos(alphaY) + velold[zDir] * sin(alphaY), velRotY[xDir], 1e-12);
    // second row of rotation matrix around the y axis times (vx, vy, vz)'
    EXPECT_NEAR(1 * velold[yDir], velRotY[yDir], 1e-12);
    // third row of rotation matrix around the y axis  times (vx, vy, vz)'
    EXPECT_NEAR(-velold[xDir] * sin(alphaY) + velold[zDir] * cos(alphaY), velRotY[zDir], 1e-12);

    // TEST FOR ROTATION MATRIX AROUND THE z-AXIS
    Bx = 0.0;
    By = 0.0;
    Bz = 1.0;
    auto [velRotZ, alphaZ] = rotate(Bx, By, Bz);
    // first row of rotation matrix around the z axis times (vx, vy, vz)'
    EXPECT_NEAR(velold[xDir] * cos(alphaZ) - velold[yDir] * sin(alphaZ), velRotZ[xDir], 1e-12);
    // second row of rotation matrix around the z axis times (vx, vy, vz)'
    EXPECT_NEAR(velold[xDir] * sin(alphaZ) + velold[yDir] * cos(alphaZ), velRotZ[yDir], 1e-12);
    // third row of rotation matrix around the z axis times (vx, vy, vz)'
    EXPECT_NEAR(1 * velold[zDir], velRotZ[zDir], 1e-12);
}

#if AMREX_SPACEDIM == 3
// check whether the rotation is moving the correct direction
// by comparing the rotation result with an explicit euler scheme:
//
// vy -= dt * q/m * vx * bz * |B|
// vz += dt * q/m * vx * by * |B|
//
// vx += dt * q/m * vy * bz * |B|
// vz -= dt * q/m * vy * bx * |B|
//
// vx -= dt * q/m * vz * by * |B|
// vy += dt * q/m * vz * bx * |B|
TEST_F(HamiltonianSplittingTest, ApplyHpireluRotationDirectionTest)
{
    Gempic::TimeLoop::OperatorHamilton<s_vDim, s_degX, s_degY, s_degZ> operatorHamilton;

    // set the particle positions to a random position within the box
    AMREX_D_TERM(amrex::Real xPosOld = 2.5;, amrex::Real yPosOld = 3.7;, amrex::Real zPosOld = 2.2;)
    // Adding particle to one cell
    int const numParticles{1};
    amrex::Array<amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>, numParticles> positions{
        {{AMREX_D_DECL(xPosOld, yPosOld, zPosOld)}}};
    amrex::Array<amrex::Real, numParticles> weights{1};
    Gempic::Test::Utils::add_single_particles(m_particles[0].get(), m_infra, weights, positions);

    amrex::Array<std::string, 3> const analyticalB = {
        "sqrt(3)*cos(x+y+z-sqrt(3.0))",
        "cos(x+y+z)",
        "-cos(x+y+z-sqrt(3.0))",
    };

    amrex::Array<amrex::ParserExecutor<s_nVar>, 3> funcB;
    amrex::Array<amrex::Parser, 3> parserB;
    for (int i = 0; i < 3; ++i)
    {
        parserB[i].define(analyticalB[i]);
        parserB[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcB[i] = parserB[i].compile<s_nVar>();
    }

    amrex::GpuArray<amrex::Real, 3> bfield = {0.0, 0.0, 0.0};
    bfield[0] = sqrt(3) * cos(xPosOld + yPosOld + zPosOld - sqrt(3.0));
    bfield[1] = cos(xPosOld + yPosOld + zPosOld);
    bfield[2] = -cos(xPosOld + yPosOld + zPosOld - sqrt(3.0));

    amrex::Real normBfield = 0;
    normBfield = sqrt(bfield[0] * bfield[0] + bfield[1] * bfield[1] + bfield[2] * bfield[2]);

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::face> B(deRham, funcB);

    amrex::GpuArray<amrex::Real, 3> v{-3.9, 1.8, 0.9};
    amrex::Real dt = 0.1;             // set time step to 0.1 for testing purposes
    amrex::Real chargeOverMass = 0.6; // random number for testing purposes

    amrex::GpuArray<amrex::Array4<amrex::Real>, s_vDim> bA;

    for (auto& particleGrid : *m_particles[s_spec])
    {
        // set random positions
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> position{
            AMREX_D_DECL(xPosOld, yPosOld, zPosOld)};

        auto const ptd = particleGrid.GetParticleTile().getParticleTileData();
        auto const ii = m_particles[s_spec]->get_data_indices();

        ptd.rdata(ii.m_ivelx)[0] = v[0] * m_infra.geometry().CellSize(xDir);
        ptd.rdata(ii.m_ively)[0] = v[1] * m_infra.geometry().CellSize(yDir);
        ptd.rdata(ii.m_ivelz)[0] = v[2] * m_infra.geometry().CellSize(zDir);

        // perform Euler step for reference solution
        amrex::Real vxEuler = ptd.rdata(ii.m_ivelx)[0];
        amrex::Real vyEuler = ptd.rdata(ii.m_ively)[0];
        amrex::Real vzEuler = ptd.rdata(ii.m_ivelz)[0];

        vyEuler -= dt * chargeOverMass * vxEuler * normBfield * bfield[2];
        vzEuler += dt * chargeOverMass * vxEuler * normBfield * bfield[1];

        vxEuler += dt * chargeOverMass * vyEuler * normBfield * bfield[2];
        vzEuler -= dt * chargeOverMass * vyEuler * normBfield * bfield[0];

        vxEuler -= dt * chargeOverMass * vzEuler * normBfield * bfield[1];
        vyEuler += dt * chargeOverMass * vzEuler * normBfield * bfield[0];

        // perform function call
        amrex::GpuArray<amrex::Real, s_vDim> vel{ptd.rdata(ii.m_ivelx)[0], ptd.rdata(ii.m_ively)[0],
                                                 ptd.rdata(ii.m_ivelz)[0]};

        ParticleMeshCoupling::SplineWithPrimitive<s_degX, s_degY, s_degZ> spline(
            position, m_infra.geometry().ProbLoArray(), m_infra.geometry().InvCellSizeArray());

        for (int cc = 0; cc < 3; cc++)
        {
            bA[cc] = (B.m_data[cc])[particleGrid].array();
        }

        operatorHamilton.apply_h_p_rel_u(vel, spline, bA, chargeOverMass, dt, 1.0);

        EXPECT_NEAR(vxEuler, vel[0], 5e-2);
        EXPECT_NEAR(vyEuler, vel[1], 5e-2);
        EXPECT_NEAR(vzEuler, vel[2], 5e-2);
    }
}
#endif

} // namespace