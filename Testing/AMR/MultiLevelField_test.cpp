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
class MultiLevelFieldTest : public testing::Test
{
protected:
    MultiLevelFieldTest()
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

TEST_F(MultiLevelFieldTest, SerializeDeserializeUnity)
{
    MLFiniteDifferenceDeRhamSpaces mLdeRham{m_parameters, m_amr};

    auto field1{
        mLdeRham.create_primal_zero_form("a", Gempic::Impl::BoundaryConditionConfiguration())};
    auto field2{
        mLdeRham.create_primal_zero_form("a", Gempic::Impl::BoundaryConditionConfiguration())};

    fill_random(field1);
    fill_all(field2, 0);

    EXPECT_GT(l_inf_error(field1, field2), 0.5);

    DiscreteTime simTime{0.1, 3, 0};
    serialize(field1, simTime);
    deserialize(field2, simTime);

    EXPECT_EQ(l_inf_error(field1, field2), 0.0);
}

TEST_F(MultiLevelFieldTest, HodgeCorrection)
{
    MLFiniteDifferenceDeRhamSpaces mLdeRham{m_parameters, m_amr};

    auto nodal{
        mLdeRham.create_primal_zero_form("nodal", Gempic::Impl::BoundaryConditionConfiguration())};
    auto cell{
        mLdeRham.create_dual_three_form("cell", Gempic::Impl::BoundaryConditionConfiguration())};

    fill_all(nodal, 1);
    zero_covered_coarse(nodal);
    hodge(cell, nodal);

    // sum should be equal to domain volume
    amrex::Real integral = sum_unique(cell);
    EXPECT_DOUBLE_EQ(integral, 1 << AMREX_SPACEDIM);
}

} //namespace
