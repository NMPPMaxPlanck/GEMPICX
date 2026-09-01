/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_PlotFileUtil.H>

#include "GEMPIC_Amr.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Gempic::Amr;

namespace
{
class AMRConformingProjectionTest : public testing::Test
{
protected:
    AMRConformingProjectionTest()
    {
        m_parameters.set("ComputationalDomain.domainLo", m_probLo);
        m_parameters.set("ComputationalDomain.domainHi", m_probHi);
        m_parameters.set("ComputationalDomain.isPeriodic", m_isPeriodic);
        m_parameters.set("ComputationalDomain.nCell", m_nCell);
        m_parameters.set("ComputationalDomain.maxGridSize", m_maxGridSize);

        m_parameters.set("MeshRefinement.maxLevel", m_maxLevel);
        m_parameters.set("MeshRefinement.taggingLevels", m_taggingLevels);
        m_parameters.set("MeshRefinement.taggingFunction", m_taggingFunc);
        m_parameters.set("MeshRefinement.blockingFactor", m_blockingFactor);
        m_parameters.set("MeshRefinement.nErrorBuff", m_nErrorBuff);
        m_parameters.set("MeshRefinement.gridEff", m_gridEff);
        m_parameters.set("MeshRefinement.nProper", m_nProper);

        m_amr = std::make_shared<GempicAmrCore>(m_parameters);
        m_amr->InitFromScratch(0.);
    }
    std::shared_ptr<GempicAmrCore> m_amr;

    Gempic::Io::Parameters m_parameters;
    amrex::Vector<amrex::Real> const m_probLo{AMREX_D_DECL(-1, -1, -1)};
    amrex::Vector<amrex::Real> const m_probHi{AMREX_D_DECL(1, 1, 1)};
    amrex::Vector<int> const m_isPeriodic{AMREX_D_DECL(1, 1, 1)};
    amrex::Vector<int> const m_nCell{AMREX_D_DECL(16, 16, 16)};
    amrex::Vector<int> const m_maxGridSize{AMREX_D_DECL(12, 12, 12)};

    int const m_maxLevel{1};
    amrex::Vector<int> const m_blockingFactor{AMREX_D_DECL(4, 4, 4)};
    amrex::Vector<int> const m_nErrorBuff{AMREX_D_DECL(0, 0, 0)};
    amrex::Real const m_gridEff{1.0};
    int const m_nProper{1};

    amrex::Vector<amrex::Real> const m_taggingLevels{0.5};
    // cross-shaped refinement to have different kinds of corners
#if AMREX_SPACEDIM == 1
    std::string const m_taggingFunc{"if(x<-0.7,0.,if(x<0.7,1.,0.))"};
#elif AMREX_SPACEDIM == 2
    std::string const m_taggingFunc{
        "if(x<-0.7,0.,if(x<0.7,1.,0.))*if(y<-0.2,0.,if(y<0.2,1.,0.)) + "
        "if(x<-0.2,0.,if(x<0.2,1.,0.))*if(y<-0.7,0.,if(y<0.7,1.,0.))"};
#else
    std::string const m_taggingFunc{
        "if(x<-0.7,0.,if(x<0.7,1.,0.))*if(y<-0.2,0.,if(y<0.2,1.,0.))*if(z<-0.2,0.,if(z<0.2,1.,0.)) "
        "+ "
        "if(x<-0.2,0.,if(x<0.2,1.,0.))*if(y<-0.7,0.,if(y<0.7,1.,0.))*if(z<-0.2,0.,if(z<0.2,1.,0.)) "
        "+ "
        "if(x<-0.2,0.,if(x<0.2,1.,0.))*if(y<-0.2,0.,if(y<0.2,1.,0.))*if(z<-0.7,0.,if(z<0.7,1.,0.)"
        ")"};
#endif
};

void init_conforming_field (MLPrimalZeroForm& f)
{
    auto fillFunc = [=] AMREX_GPU_HOST_DEVICE(AMREX_D_DECL(double x, double y, double z))
    { return GEMPIC_D_MULT(x, y, z); };
    project(f, fillFunc);
}

void init_conforming_field (MLPrimalOneForm& vf)
{
    auto fillFunc = [=] AMREX_GPU_HOST_DEVICE(
                        Direction dir, AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z))
    {
        switch (dir)
        {
            case Direction::xDir:
                return AMREX_D_PICK(1.0, y, y * z);
            case Direction::yDir:
                return AMREX_D_PICK(x, x, x * z);
            case Direction::zDir:
                return AMREX_D_PICK(x, x * y, x * y);
            default:
                AMREX_ALWAYS_ASSERT(false);
                return 0.0;
        };
    };
    project(vf, fillFunc);
}

void init_conforming_field (MLPrimalTwoForm& vf)
{
    auto fillFunc = [=] AMREX_GPU_HOST_DEVICE(
                        Direction dir, AMREX_D_DECL(amrex::Real x, amrex::Real y, amrex::Real z))
    {
        switch (dir)
        {
            case Direction::xDir:
                return x;
            case Direction::yDir:
                return AMREX_D_PICK(x, y, y);
            case Direction::zDir:
                return AMREX_D_PICK(x, y, z);
            default:
                AMREX_ALWAYS_ASSERT(false);
                return 0.0;
        };
    };
    project(vf, fillFunc);
}

TEST_F(AMRConformingProjectionTest, IsProjection)
{
    MLFiniteDifferenceDeRhamSpaces mLdeRham{m_parameters, m_amr};
    std::array<Gempic::Impl::BoundaryConditionConfiguration, 3> bcConfig{};

    auto zeroA{
        mLdeRham.create_primal_zero_form("", Gempic::Impl::BoundaryConditionConfiguration())};
    auto zeroB{
        mLdeRham.create_primal_zero_form("", Gempic::Impl::BoundaryConditionConfiguration())};
    auto oneA{mLdeRham.create_primal_one_form("", bcConfig)};
    auto oneB{mLdeRham.create_primal_one_form("", bcConfig)};
    auto twoA{mLdeRham.create_primal_two_form("", bcConfig)};
    auto twoB{mLdeRham.create_primal_two_form("", bcConfig)};
    auto threeA{
        mLdeRham.create_primal_three_form("", Gempic::Impl::BoundaryConditionConfiguration())};
    auto threeB{
        mLdeRham.create_primal_three_form("", Gempic::Impl::BoundaryConditionConfiguration())};

    amrex::InitRandom(1);

    // For each primal form test that
    // 1. conforming field is not altered
    // 2. second application of projection is identity

    // scalar 0-form
    init_conforming_field(zeroA);
    copy(zeroB, zeroA);
    conforming_projection(zeroB);
    EXPECT_NEAR(l_inf_error(zeroA, zeroB), 0, 1e-15);

    fill_random(zeroA);
    conforming_projection(zeroA);
    copy(zeroB, zeroA);
    conforming_projection(zeroB);
    EXPECT_NEAR(l_inf_error(zeroA, zeroB), 0, 1e-15);

    // vector 1-form
    init_conforming_field(oneA);
    copy(oneB, oneA);
    conforming_projection(oneB);
    EXPECT_NEAR(l_inf_error(oneA, oneB), 0, 1e-15);

    fill_random(oneA);
    conforming_projection(oneA);
    copy(oneB, oneA);
    conforming_projection(oneB);
    EXPECT_NEAR(l_inf_error(oneA, oneB), 0, 1e-15);

    // vector 2-form
    init_conforming_field(twoA);
    copy(twoB, twoA);
    conforming_projection(twoB);
    EXPECT_NEAR(l_inf_error(twoA, twoB), 0, 1e-15);

    fill_random(twoA);
    conforming_projection(twoA);
    copy(twoB, twoA);
    conforming_projection(twoB);
    EXPECT_NEAR(l_inf_error(twoA, twoB), 0, 1e-15);

    // scalar 3-form is discontinuous anyways, projection reduces to identity
    fill_random(threeA);
    copy(threeB, threeA);
    conforming_projection(threeB);
    EXPECT_NEAR(l_inf_error(threeA, threeB), 0, 1e-15);
}

TEST_F(AMRConformingProjectionTest, P0AreAdjoints)
{
    // test adjointness with random vectors
    MLFiniteDifferenceDeRhamSpaces mLdeRham{m_parameters, m_amr};
    Gempic::Impl::BoundaryConditionConfiguration bcConfig{};

    auto phi{mLdeRham.create_primal_zero_form("", bcConfig)};
    auto projPhi{mLdeRham.create_primal_zero_form("", bcConfig)};
    auto rho{mLdeRham.create_dual_three_form("", bcConfig)};
    auto projRho{mLdeRham.create_dual_three_form("", bcConfig)};

    amrex::InitRandom(2);
    fill_random(phi);
    copy(projPhi, phi);
    conforming_projection(projPhi);
    fill_random(rho);
    copy(projRho, rho);
    conforming_projection(projRho);

    amrex::Real lhs = dot_product(projPhi, rho);
    amrex::Real rhs = dot_product(phi, projRho);

    EXPECT_NEAR(lhs, rhs, std::min(lhs, rhs) * 1e-15);
}

TEST_F(AMRConformingProjectionTest, P1AreAdjoints)
{
    // test adjointness with random vectors
    MLFiniteDifferenceDeRhamSpaces mLdeRham{m_parameters, m_amr};
    std::array<Gempic::Impl::BoundaryConditionConfiguration, 3> bcConfig{};

    auto E{mLdeRham.create_primal_one_form("", bcConfig)};
    auto projE{mLdeRham.create_primal_one_form("", bcConfig)};
    auto D{mLdeRham.create_dual_two_form("", bcConfig)};
    auto projD{mLdeRham.create_dual_two_form("", bcConfig)};

    amrex::InitRandom(3);
    fill_random(E);
    copy(projE, E);
    conforming_projection(projE);
    fill_random(D);
    copy(projD, D);
    conforming_projection(projD);

    amrex::Real lhs = dot_product(projE, D);
    amrex::Real rhs = dot_product(E, projD);

    EXPECT_NEAR(lhs, rhs, std::min(lhs, rhs) * 1e-15);
}

TEST_F(AMRConformingProjectionTest, P2AreAdjoints)
{
    // test adjointness with random vectors
    MLFiniteDifferenceDeRhamSpaces mLdeRham{m_parameters, m_amr};
    std::array<Gempic::Impl::BoundaryConditionConfiguration, 3> bcConfig{};

    auto B{mLdeRham.create_primal_two_form("", bcConfig)};
    auto projB{mLdeRham.create_primal_two_form("", bcConfig)};
    auto H{mLdeRham.create_dual_one_form("", bcConfig)};
    auto projH{mLdeRham.create_dual_one_form("", bcConfig)};

    amrex::InitRandom(4);
    fill_random(B);
    copy(projB, B);
    conforming_projection(projB);
    fill_random(H);
    copy(projH, H);
    conforming_projection(projH);

    amrex::Real lhs = dot_product(projB, H);
    amrex::Real rhs = dot_product(B, projH);

    EXPECT_NEAR(lhs, rhs, std::min(lhs, rhs) * 1e-15);
}
} //namespace