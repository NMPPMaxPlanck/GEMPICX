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
class FieldMultiplyByFunctionTest : public testing::Test
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
    amrex::Real m_tol{1e-11};

    FieldMultiplyByFunctionTest() : m_infra{Gempic::Test::Utils::get_default_compdom()}
    {
        // Not checking particles
        int const nGhostExtra{0};

        m_parameters.set("nGhostExtra", nGhostExtra);
    }
};

TEST_F(FieldMultiplyByFunctionTest, ZeroThreeForms)
{
    constexpr int hodgeDegree{2};

#if (AMREX_SPACEDIM == 1)
    std::string const analyticalWeightFunc = "x";
#endif

#if (AMREX_SPACEDIM == 2)
    std::string const analyticalWeightFunc = "x * y";
#endif

#if (AMREX_SPACEDIM == 3)
    std::string const analyticalWeightFunc = "x * y * z";
#endif

    std::string const analyticalOneFunc = "1.0";

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::ParserExecutor<nVar> weightFunc;
    amrex::ParserExecutor<nVar> oneFunc;
    amrex::Parser weightParser;
    amrex::Parser oneParser;

    weightParser.define(analyticalWeightFunc);
    oneParser.define(analyticalOneFunc);
    weightParser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    oneParser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    weightFunc = weightParser.compile<nVar>();
    oneFunc = oneParser.compile<nVar>();

    // Initialize the De Rham Complex with deg 2
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::node> A(deRham, oneFunc);
    DeRhamField<Grid::primal, Space::node> aResult(deRham, weightFunc);
    DeRhamField<Grid::primal, Space::cell> D(deRham, oneFunc);
    DeRhamField<Grid::primal, Space::cell> dResult(deRham, weightFunc);

    A *= weightFunc;
    D *= weightFunc;

    bool loopRun{false};

    for (amrex::MFIter mfi(A.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        amrex::Box const& bx = mfi.validbox();
        COMPARE_FIELDS(A.m_data[mfi].array(), aResult.m_data[mfi].array(), bx, m_tol);
    }
    ASSERT_TRUE(loopRun);

    loopRun = false;

    for (amrex::MFIter mfi(D.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        amrex::Box const& bx = mfi.validbox();
        COMPARE_FIELDS(D.m_data[mfi].array(), dResult.m_data[mfi].array(), bx, m_tol);
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(FieldMultiplyByFunctionTest, OneTwoForms)
{
    constexpr int hodgeDegree{2};

#if (AMREX_SPACEDIM == 1)
    std::string const analyticalWeightFunc = "x";
    amrex::Array<std::string, 3> const analyticalResultFunc = {
        "x",
        "x",
        "x",
    };
#endif

#if (AMREX_SPACEDIM == 2)
    std::string const analyticalWeightFunc = "x * y";
    amrex::Array<std::string, 3> const analyticalResultFunc = {
        "x * y",
        "x * y",
        "x * y",
    };
#endif

#if (AMREX_SPACEDIM == 3)
    std::string const analyticalWeightFunc = "x * y * z";
    amrex::Array<std::string, 3> const analyticalResultFunc = {
        "x * y * z",
        "x * y * z",
        "x * y * z",
    };
#endif

    amrex::Array<std::string, 3> const analyticalOneFunc = {
        "1",
        "1",
        "1",
    };

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::ParserExecutor<nVar> weightFunc;
    amrex::Parser weightParser;

    weightParser.define(analyticalWeightFunc);
    weightParser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    weightFunc = weightParser.compile<nVar>();

    amrex::Array<amrex::ParserExecutor<nVar>, 3> oneFunc;
    amrex::Array<amrex::ParserExecutor<nVar>, 3> resultFunc;
    amrex::Array<amrex::Parser, 3> oneParser;
    amrex::Array<amrex::Parser, 3> resultParser;
    for (int i = 0; i < 3; ++i)
    {
        oneParser[i].define(analyticalOneFunc[i]);
        oneParser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        oneFunc[i] = oneParser[i].compile<nVar>();
        resultParser[i].define(analyticalResultFunc[i]);
        resultParser[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        resultFunc[i] = resultParser[i].compile<nVar>();
    }

    // Initialize the De Rham Complex with deg 2
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::edge> B(deRham, oneFunc);
    DeRhamField<Grid::primal, Space::edge> bResult(deRham, resultFunc);
    DeRhamField<Grid::primal, Space::face> C(deRham, oneFunc);
    DeRhamField<Grid::primal, Space::face> cResult(deRham, resultFunc);

    B *= weightFunc;
    C *= weightFunc;

    bool loopRun{false};

    for (int comp = 0; comp < 3; ++comp)
    {
        loopRun = false;
        for (amrex::MFIter mfi(B.m_data[comp]); mfi.isValid(); ++mfi)
        {
            loopRun = true;
            amrex::Box const& bx = mfi.validbox();
            COMPARE_FIELDS((B.m_data[comp])[mfi].array(), (bResult.m_data[comp])[mfi].array(), bx,
                           m_tol);
        }
        ASSERT_TRUE(loopRun);
    }

    for (int comp = 0; comp < 3; ++comp)
    {
        loopRun = false;
        for (amrex::MFIter mfi(C.m_data[comp]); mfi.isValid(); ++mfi)
        {
            loopRun = true;
            amrex::Box const& bx = mfi.validbox();
            COMPARE_FIELDS((C.m_data[comp])[mfi].array(), (cResult.m_data[comp])[mfi].array(), bx,
                           m_tol);
        }
        ASSERT_TRUE(loopRun);
    }
}
} // namespace
