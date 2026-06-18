/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_AmrLevelInterface.H"

using namespace Gempic;
using namespace Gempic::Amr;

namespace
{

std::array<int, 6> count_box_types (Gempic::Amr::Impl::LevelInterface const& li)
{
    std::array<int, 6> count;
    count.fill(0);
    for (int i = 0; i < li.m_baFine.size(); i++)
    {
        amrex::IntVect length = li.m_baFine[i].length();
        auto dimension{std::count_if(length.begin(), length.end(), [] (int i) { return i > 1; })};
        GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(dimension < 3, "Interface box cannot be 3-dimensional");
        count[dimension]++;
    }
    for (int i = 0; i < li.m_baCoarse.size(); i++)
    {
        amrex::IntVect length = li.m_baCoarse[i].length();
        auto dimension{std::count_if(length.begin(), length.end(), [] (int i) { return i > 1; })};
        GEMPIC_ALWAYS_ASSERT_WITH_MESSAGE(dimension < 3, "Interface box cannot be 3-dimensional");
        count[dimension + 3]++;
    }
    return count;
}

amrex::Geometry build_geom (int nCells)
{
    amrex::Box domain(amrex::IntVect::TheZeroVector(),
                      amrex::IntVect(AMREX_D_DECL(nCells - 1, nCells - 1, nCells - 1)));
    amrex::RealBox realBox({AMREX_D_DECL(0, 0, 0)}, {AMREX_D_DECL(1, 1, 1)});
    amrex::Array<int, AMREX_SPACEDIM> isPeriodic{AMREX_D_DECL(0, 0, 0)};

    return amrex::Geometry(domain, realBox, 0, isPeriodic);
}

class AMRLevelInterfaceParameterTest : public testing::TestWithParam<amrex::IntVect>
{
};

TEST_P(AMRLevelInterfaceParameterTest, CrossConfiguration)
{
    amrex::IntVect idxType = GetParam();
    int nCells = 25;
    amrex::IntVect refRatio{AMREX_D_DECL(2, 2, 2)};

    amrex::Geometry geom = build_geom(nCells);
    geom.refine(refRatio);

    amrex::BoxArray baCoarse(
        amrex::Box(amrex::IntVect::TheZeroVector(),
                   amrex::IntVect(AMREX_D_DECL(nCells - 1, nCells - 1, nCells - 1))));

    amrex::BoxList bl;
    // safe way to create a nested fine boxArray: Create coarse and then refine
    bl.push_back(amrex::Box(amrex::IntVect(AMREX_D_DECL(5, 10, 10)),
                            amrex::IntVect(AMREX_D_DECL(9, 14, 14))));
    bl.push_back(amrex::Box(amrex::IntVect(AMREX_D_DECL(10, 10, 10)),
                            amrex::IntVect(AMREX_D_DECL(14, 14, 14))));
    bl.push_back(amrex::Box(amrex::IntVect(AMREX_D_DECL(15, 10, 10)),
                            amrex::IntVect(AMREX_D_DECL(19, 14, 14))));
#if AMREX_SPACEDIM >= 2
    bl.push_back(amrex::Box(amrex::IntVect(AMREX_D_DECL(10, 5, 10)),
                            amrex::IntVect(AMREX_D_DECL(14, 9, 14))));
    bl.push_back(amrex::Box(amrex::IntVect(AMREX_D_DECL(10, 15, 10)),
                            amrex::IntVect(AMREX_D_DECL(14, 19, 14))));
#endif
#if AMREX_SPACEDIM == 3
    bl.push_back(amrex::Box(amrex::IntVect(AMREX_D_DECL(10, 10, 5)),
                            amrex::IntVect(AMREX_D_DECL(14, 14, 9))));
    bl.push_back(amrex::Box(amrex::IntVect(AMREX_D_DECL(10, 10, 15)),
                            amrex::IntVect(AMREX_D_DECL(14, 14, 19))));
#endif
    amrex::BoxArray baFine(std::move(bl));
    baFine.refine(refRatio);

    baCoarse.convert(idxType);
    baFine.convert(idxType);

    amrex::DistributionMapping dmCoarse(baCoarse);
    amrex::DistributionMapping dmFine(baFine);
    // build MultiFabs
    int nGhostMF = 0;
    amrex::MultiFab mfCoarse(baCoarse, dmCoarse, 1, nGhostMF);
    amrex::MultiFab mfFine(baFine, dmFine, 1, nGhostMF);

    Gempic::Amr::Impl::LevelInterface levelInterface(mfFine, mfCoarse, refRatio, geom);
    std::array<int, 6> count = count_box_types(levelInterface);

    int dofType = 0;
    for (auto dir : {AMREX_D_DECL(xDir, yDir, zDir)})
    {
        dofType += idxType[dir];
    }

    // fine perspective can have overlap on vertices and edges, coarse perspective is
    // non-intersecting (inside each coarse box). That leads to the different counts in some cases.
#if AMREX_SPACEDIM == 1
    switch (dofType)
    {
        case 0: // cells
            EXPECT_EQ(count, (std::array<int, 6>{0, 0, 0, 0, 0, 0}));
            break;
        case 1: // vertices
            EXPECT_EQ(count, (std::array<int, 6>{2, 0, 0, 2, 0, 0}));
            break;
        default:
            GEMPIC_ERROR("Encountered 2D box in 1D");
    }
#elif AMREX_SPACEDIM == 2
    switch (dofType)
    {
        case 0: // cells
            EXPECT_EQ(count, (std::array<int, 6>{0, 0, 0, 0, 0, 0}));
            break;
        case 1: // edges
            EXPECT_EQ(count, (std::array<int, 6>{0, 6, 0, 0, 6, 0}));
            break;
        case 2: // vertices
            EXPECT_EQ(count, (std::array<int, 6>{20, 12, 0, 12, 12, 0}));
            break;
        default:
            GEMPIC_ERROR("Encountered 3D box in 2D");
    }
#else
    switch (dofType)
    {
        case 0: // cells
            EXPECT_EQ(count, (std::array<int, 6>{0, 0, 0, 0, 0, 0}));
            break;
        case 1: // faces
            EXPECT_EQ(count, (std::array<int, 6>{0, 0, 10, 0, 0, 10}));
            break;
        case 2: // edges
            EXPECT_EQ(count, (std::array<int, 6>{0, 28, 20, 0, 20, 20}));
            break;
        case 3: // vertices
            EXPECT_EQ(count, (std::array<int, 6>{56, 84, 30, 32, 60, 30}));
            break;
        default:
            GEMPIC_ERROR("Impossible to reach");
    }
#endif
}

std::vector<amrex::IntVect> make_index_types ()
{
    std::vector<amrex::IntVect> v;
    for (int i = 0; i < (1 << AMREX_SPACEDIM); ++i)
    {
        v.emplace_back(AMREX_D_DECL((i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1));
    }
    return v;
}

INSTANTIATE_TEST_SUITE_P(IndexTypes,
                         AMRLevelInterfaceParameterTest,
                         testing::ValuesIn(make_index_types()));

class AMRLevelInterfaceTest : public testing::Test
{
public:
    AMRLevelInterfaceTest() {}
};

// If the refinement interface aligns with coarse box boundaries (in a node-centered direction),
// coarse level interface boxes are needed on both sides of the boundary
TEST_F(AMRLevelInterfaceTest, CoarseDuplication)
{
    amrex::IntVect idxType = amrex::IntVect::TheUnitVector();
    int nCells = 25;
    amrex::IntVect refRatio{AMREX_D_DECL(2, 2, 2)};

    amrex::Geometry geom = build_geom(nCells);
    geom.refine(refRatio);

    amrex::BoxList blCoarse;
    blCoarse.push_back(amrex::Box(amrex::IntVect::TheZeroVector(),
                                  amrex::IntVect(AMREX_D_DECL(10, nCells - 1, nCells - 1))));
    blCoarse.push_back(
        amrex::Box(amrex::IntVect(AMREX_D_DECL(11, 0, 0)),
                   amrex::IntVect(AMREX_D_DECL(nCells - 1, nCells - 1, nCells - 1))));
    amrex::BoxArray baCoarse(std::move(blCoarse));

    amrex::BoxArray baFine(amrex::Box(amrex::IntVect(AMREX_D_DECL(11, 5, 5)),
                                      amrex::IntVect(AMREX_D_DECL(19, 19, 19))));
    baFine.refine(refRatio);

    baCoarse.convert(idxType);
    baFine.convert(idxType);

    amrex::DistributionMapping dmCoarse(baCoarse);
    amrex::DistributionMapping dmFine(baFine);
    int nGhostMF = 0;
    amrex::MultiFab mfCoarse(baCoarse, dmCoarse, 1, nGhostMF);
    amrex::MultiFab mfFine(baFine, dmFine, 1, nGhostMF);

    Gempic::Amr::Impl::LevelInterface levelInterface(mfFine, mfCoarse, refRatio, geom);
    std::array<int, 6> count = count_box_types(levelInterface);

#if AMREX_SPACEDIM == 1
    EXPECT_EQ(count, (std::array<int, 6>{2, 0, 0, 3, 0, 0}));
#elif AMREX_SPACEDIM == 2
    EXPECT_EQ(count, (std::array<int, 6>{4, 4, 0, 6, 5, 0}));
#else
    EXPECT_EQ(count, (std::array<int, 6>{8, 12, 6, 12, 16, 7}));
#endif
}

#if AMREX_SPACEDIM == 3
// In 3D, some edges have to be split further according to the neighboring fine boxes
TEST_F(AMRLevelInterfaceTest, EdgeSplitting)
{
    amrex::IntVect idxType = amrex::IntVect::TheUnitVector();
    int nCells = 25;
    amrex::IntVect refRatio{AMREX_D_DECL(2, 2, 2)};

    amrex::Geometry geom = build_geom(nCells);
    geom.refine(refRatio);

    amrex::BoxArray baCoarse(
        amrex::Box({AMREX_D_DECL(0, 0, 0)}, {AMREX_D_DECL(nCells - 1, nCells - 1, nCells - 1)}));

    amrex::BoxList bl;
    bl.push_back(amrex::Box({AMREX_D_DECL(5, 5, 5)}, {AMREX_D_DECL(19, 19, 9)}));
    bl.push_back(amrex::Box({AMREX_D_DECL(10, 5, 10)}, {AMREX_D_DECL(14, 14, 14)}));

    amrex::BoxArray baFine(std::move(bl));
    baFine.refine(refRatio);

    baCoarse.convert(idxType);
    baFine.convert(idxType);

    amrex::DistributionMapping dmCoarse(baCoarse);
    amrex::DistributionMapping dmFine(baFine);
    int nGhostMF = 0;
    amrex::MultiFab mfCoarse(baCoarse, dmCoarse, 1, nGhostMF);
    amrex::MultiFab mfFine(baFine, dmFine, 1, nGhostMF);

    Gempic::Amr::Impl::LevelInterface levelInterface(mfFine, mfCoarse, refRatio, geom);
    std::array<int, 6> count = count_box_types(levelInterface);

    EXPECT_EQ(count, (std::array<int, 6>{20, 28, 13, 16, 24, 13}));
}
#endif
} //namespace
