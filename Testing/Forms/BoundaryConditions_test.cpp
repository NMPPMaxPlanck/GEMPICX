/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Parameters.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Forms;

namespace
{

using condLambda = bool (*)(AMREX_D_DECL(int, int, int));
bool func_cond_x (AMREX_D_DECL(int i, int /*j*/, int /*k*/)) { return (i == 0 || i == 3); }
bool func_cond_y (AMREX_D_DECL(int /*i*/, int j, int /*k*/))
{
#if AMREX_SPACEDIM >= 2
    return (j == 0 || j == 3);
#else
    return false;
#endif
}
bool func_cond_z (AMREX_D_DECL(int /*i*/, int /*j*/, int k))
{
#if AMREX_SPACEDIM == 3
    return (k == 0 || k == 3);
#else
    return false;
#endif
}
condLambda condX = func_cond_x;
condLambda condY = func_cond_y;
condLambda condZ = func_cond_z;

ComputationalDomain get_compdom ()
{
    // Cells of size 1x1x1
    std::array<amrex::Real, AMREX_SPACEDIM> const domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
    std::array<amrex::Real, AMREX_SPACEDIM> const domainHi{AMREX_D_DECL(3.0, 3.0, 3.0)};
    amrex::IntVect const nCell{AMREX_D_DECL(3, 3, 3)};
    amrex::IntVect const maxGridSize{AMREX_D_DECL(10, 10, 10)};
    std::array<int, AMREX_SPACEDIM> const isPeriodic{AMREX_D_DECL(0, 0, 0)};

    return ComputationalDomain(domainLo, domainHi, nCell, maxGridSize, isPeriodic);
}

class BoundaryConditionTest : public testing::Test
{
protected:
    // Linear splines is ok, and lower dimension Hodge is good enough
    // Spline degreesx
    static int const s_degX{1};
    static int const s_degY{1};
    static int const s_degZ{1};
    inline static int const s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

    Io::Parameters m_parameters{};
    ComputationalDomain m_infra;

    BoundaryConditionTest() : m_infra{get_compdom()}
    {
        amrex::Vector<std::string> const bcVec(6, "PerfectlyConducting");
        // Not checking particles
        int const nGhostExtra{0};

        m_parameters.set("nGhostExtra", nGhostExtra);
        m_parameters.set("BoundaryCondition.Default", bcVec);
    }
};

TEST_F(BoundaryConditionTest, PerfectlyConductingPrimal)
{
    constexpr int hodgeDegree{2};

    std::string const oneFuncScalar = "1.0";
    amrex::Array<std::string, 3> const oneFuncVector = {
        "1.0",
        "1.0",
        "1.0",
    };

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t

    amrex::ParserExecutor<nVar> funcScalar;
    amrex::Parser parserScalar;
    parserScalar.define(oneFuncScalar);
    parserScalar.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    funcScalar = parserScalar.compile<nVar>();

    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcVector;
    amrex::Array<amrex::Parser, 3> parserVector;
    for (int i = 0; i < 3; ++i)
    {
        parserVector[i].define(oneFuncVector[i]);
        parserVector[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcVector[i] = parserVector[i].compile<nVar>();
    }

    // Initialize the De Rham Complex with deg 2
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::node> zeroForm(deRham, funcScalar);
    DeRhamField<Grid::primal, Space::edge> oneForm(deRham, funcVector);
    DeRhamField<Grid::primal, Space::face> twoForm(deRham, funcVector);
    DeRhamField<Grid::primal, Space::cell> threeForm(deRham, funcScalar);

    zeroForm.apply_bc();
    oneForm.apply_bc();
    twoForm.apply_bc();
    threeForm.apply_bc();

    // Zero form -> expect all boundary nodes to be zero
    bool loopRun{false};
    for (amrex::MFIter mfi(zeroForm.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        CHECK_FIELD((zeroForm.m_data[mfi]).array(), mfi.validbox(), {condX, condY, condZ},
                    {0, 0, 0}, 1);
    }
    ASSERT_TRUE(loopRun);

    // One form -> expect edges tangential to boundary to be zero
    loopRun = false;
    for (amrex::MFIter mfi(oneForm.m_data[xDir]); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        CHECK_FIELD((oneForm.m_data[xDir][mfi]).array(), mfi.validbox(), {condY, condZ}, {0, 0}, 1);
    }
    ASSERT_TRUE(loopRun);
    loopRun = false;
    for (amrex::MFIter mfi(oneForm.m_data[yDir]); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        CHECK_FIELD((oneForm.m_data[yDir][mfi]).array(), mfi.validbox(), {condX, condZ}, {0, 0}, 1);
    }
    ASSERT_TRUE(loopRun);
    loopRun = false;
    for (amrex::MFIter mfi(oneForm.m_data[zDir]); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        CHECK_FIELD((oneForm.m_data[zDir][mfi]).array(), mfi.validbox(), {condX, condY}, {0, 0}, 1);
    }
    ASSERT_TRUE(loopRun);

    // Two form -> expect fluxes normal to boundary to be zero
    loopRun = false;
    for (amrex::MFIter mfi(twoForm.m_data[xDir]); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        CHECK_FIELD((twoForm.m_data[xDir][mfi]).array(), mfi.validbox(), {condX}, {0}, 1);
    }
    ASSERT_TRUE(loopRun);
    loopRun = false;
    for (amrex::MFIter mfi(twoForm.m_data[yDir]); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        CHECK_FIELD((twoForm.m_data[yDir][mfi]).array(), mfi.validbox(), {condY}, {0}, 1);
    }
    ASSERT_TRUE(loopRun);
    loopRun = false;
    for (amrex::MFIter mfi(twoForm.m_data[zDir]); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        CHECK_FIELD((twoForm.m_data[zDir][mfi]).array(), mfi.validbox(), {condZ}, {0}, 1);
    }
    ASSERT_TRUE(loopRun);

    // Three form -> expect no change
    loopRun = false;
    for (amrex::MFIter mfi(threeForm.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        CHECK_FIELD((threeForm.m_data[mfi]).array(), mfi.validbox(), {}, {}, 1);
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(BoundaryConditionTest, PerfectlyConductingDual)
{
    constexpr int hodgeDegree{2};

    std::string const oneFuncScalar = "1.0";
    amrex::Array<std::string, 3> const oneFuncVector = {
        "1.0",
        "1.0",
        "1.0",
    };

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t

    amrex::ParserExecutor<nVar> funcScalar;
    amrex::Parser parserScalar;
    parserScalar.define(oneFuncScalar);
    parserScalar.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    funcScalar = parserScalar.compile<nVar>();

    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcVector;
    amrex::Array<amrex::Parser, 3> parserVector;
    for (int i = 0; i < 3; ++i)
    {
        parserVector[i].define(oneFuncVector[i]);
        parserVector[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcVector[i] = parserVector[i].compile<nVar>();
    }

    // Initialize the De Rham Complex with deg 2
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::dual, Space::node> zeroForm(deRham, funcScalar);
    DeRhamField<Grid::dual, Space::edge> oneForm(deRham, funcVector);
    DeRhamField<Grid::dual, Space::face> twoForm(deRham, funcVector);
    DeRhamField<Grid::dual, Space::cell> threeForm(deRham, funcScalar);

    zeroForm.apply_bc();
    oneForm.apply_bc();
    twoForm.apply_bc();
    threeForm.apply_bc();

    // All dual forms should not change, i.e., ones for all dofs
    // Zero form
    bool loopRun{false};
    for (amrex::MFIter mfi(zeroForm.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        CHECK_FIELD((zeroForm.m_data[mfi]).array(), mfi.validbox(), {}, {}, 1);
    }
    ASSERT_TRUE(loopRun);

    // One and two forms
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        loopRun = false;
        for (amrex::MFIter mfi(oneForm.m_data[dir]); mfi.isValid(); ++mfi)
        {
            loopRun = true;
            CHECK_FIELD((oneForm.m_data[dir][mfi]).array(), mfi.validbox(), {}, {}, 1);
        }
        ASSERT_TRUE(loopRun);
        loopRun = false;
        for (amrex::MFIter mfi(twoForm.m_data[dir]); mfi.isValid(); ++mfi)
        {
            loopRun = true;
            CHECK_FIELD((twoForm.m_data[dir][mfi]).array(), mfi.validbox(), {}, {}, 1);
        }
        ASSERT_TRUE(loopRun);
    }

    // Three form
    loopRun = false;
    for (amrex::MFIter mfi(threeForm.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        CHECK_FIELD((threeForm.m_data[mfi]).array(), mfi.validbox(), {}, {}, 1);
    }
    ASSERT_TRUE(loopRun);
}
} // namespace
