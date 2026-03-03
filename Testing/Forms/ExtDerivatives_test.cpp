/**************************************************************************************************
 * Copyright (c) 2021 GEMPICX                                                                     *
 * SPDX-License-Identifier: BSD-3-Clause                                                          *
 **************************************************************************************************/
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_Parameters.H"
#include "TestUtils/GEMPIC_TestUtils.H"

namespace
{
using namespace Gempic;
using namespace Forms;

// Test fixture. Sets up clean environment before each test.
class ExtDerivativesTest : public testing::Test
{
protected:
    static int const s_degX{1};
    static int const s_degY{1};
    static int const s_degZ{1};
    inline static int const s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

    inline static int const s_hodgeDegree = 2;

    // Initialize computational_domain and parameters
    Io::Parameters m_parameters{};
    ComputationalDomain m_infra;

// Define analytical fields as string
#if (AMREX_SPACEDIM == 1)
    std::string const m_analyticalScalarField = "sin(x)";
    amrex::Array<std::string, 3> const m_analyticalField = {
        "-cos(x)",
        "sin(x)",
        "-sin(x)",
    };
#elif (AMREX_SPACEDIM == 2)
    std::string const m_analyticalScalarField = "sin(x + 2*y)";
    amrex::Array<std::string, 3> const m_analyticalField = {
        "-cos(x)*cos(y)",
        "sin(x)*cos(y)",
        "-sin(x)*sin(y)",
    };
#elif (AMREX_SPACEDIM == 3)
    std::string const m_analyticalScalarField = "sin(x + 2*y - z)";
    amrex::Array<std::string, 3> const m_analyticalField = {
        "-cos(x)*cos(y)*sin(z)",
        "sin(x)*cos(y)*sin(z)",
        "-sin(x)*sin(y)*cos(z)",
    };
#endif

    // Analytical grad Field
#if (AMREX_SPACEDIM == 1)
    amrex::Array<std::string, 3> const m_analyticalGradField = {
        "cos(x)",
        "0.",
        "0.",
    };
#elif (AMREX_SPACEDIM == 2)
    amrex::Array<std::string, 3> const m_analyticalGradField = {
        "cos(x + 2*y)",
        "2*cos(x + 2*y)",
        "0.",
    };
#elif (AMREX_SPACEDIM == 3)
    amrex::Array<std::string, 3> const m_analyticalGradField = {
        "cos(x + 2*y - z)",
        "2*cos(x + 2*y - z)",
        "-cos(x + 2*y - z)",
    };
#endif

// Analytical curlField
#if (AMREX_SPACEDIM == 1)
    amrex::Array<std::string, 3> const m_analyticalCurlField = {
        "0.",
        "cos(x)",
        "cos(x)",
    };
#elif (AMREX_SPACEDIM == 2)
    amrex::Array<std::string, 3> const m_analyticalCurlField = {
        "-sin(x)*cos(y)",
        "cos(x)*sin(y)",
        "cos(x)*cos(y) - cos(x)*sin(y)",
    };
#elif (AMREX_SPACEDIM == 3)
    amrex::Array<std::string, 3> const m_analyticalCurlField = {
        "-2*sin(x)*cos(y)*cos(z)",
        "-cos(x)*cos(y)*cos(z) + cos(x)*sin(y)*cos(z)",
        "cos(x)*cos(y)*sin(z) - cos(x)*sin(y)*sin(z)",
    };
#endif

    // Analytical divField
#if (AMREX_SPACEDIM == 1)
    std::string const m_analyticalDivField = "sin(x)";
#elif (AMREX_SPACEDIM == 2)
    std::string const m_analyticalDivField = "sin(x)*cos(y) - sin(x)*sin(y)";
#elif (AMREX_SPACEDIM == 3)
    std::string const m_analyticalDivField =
        "sin(x)*cos(y)*sin(z) - sin(x)*sin(y)*sin(z) + sin(x)*sin(y)*sin(z)";
#endif

    ExtDerivativesTest() : m_infra{Gempic::Test::Utils::get_default_compdom()} {}
};

TEST_F(ExtDerivativesTest, Grad)
{
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    // Test grad of field on primal complex
    //-------------------------------------
    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::ParserExecutor<nVar> funcField;
    amrex::Parser parserField;
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcGrad;
    amrex::Array<amrex::Parser, 3> parserGrad;

    parserField.define(m_analyticalScalarField);
    parserField.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    funcField = parserField.compile<nVar>();

    for (int i = 0; i < 3; ++i)
    {
        parserGrad[i].define(m_analyticalGradField[i]);
        parserGrad[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcGrad[i] = parserGrad[i].compile<nVar>();
    }

    DeRhamField<Grid::primal, Space::node> field(deRham, funcField);
    DeRhamField<Grid::primal, Space::edge> gradField(deRham);
    DeRhamField<Grid::primal, Space::edge> analyticalGradField(deRham, funcGrad);

    // Calculate gradField from Field
    grad(gradField, field);

    // Calculate error
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(gradField.m_data[comp]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.validbox();
            COMPARE_FIELDS(gradField.m_data[comp][mfi].array(),
                           analyticalGradField.m_data[comp][mfi].array(), bx);
        }
    }

    // Calculate gradField from Field with a_times_grad
    // Take opposite for the analytical grad field
    analyticalGradField *= -1.0;

    a_times_grad(gradField, field, -1.0);

    // Calculate error
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(gradField.m_data[comp]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.validbox();
            COMPARE_FIELDS(gradField.m_data[comp][mfi].array(),
                           analyticalGradField.m_data[comp][mfi].array(), bx);
        }
    }

    // Test grad of field on dual complex
    //-------------------------------------
    DeRhamField<Grid::dual, Space::node> fieldDual(deRham, funcField);
    DeRhamField<Grid::dual, Space::edge> gradFieldDual(deRham);
    DeRhamField<Grid::dual, Space::edge> analyticalGradFieldDual(deRham, funcGrad);

    // Calculate gradFieldDual from FieldDual
    grad(gradFieldDual, fieldDual);

    // Calculate error
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(gradField.m_data[comp]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.validbox();
            COMPARE_FIELDS(gradFieldDual.m_data[comp][mfi].array(),
                           analyticalGradFieldDual.m_data[comp][mfi].array(), bx);
        }
    }

    // Calculate gradField from Field with a_times_grad
    // Take opposite for the analytical grad field
    analyticalGradFieldDual *= -1.0;

    a_times_grad(gradFieldDual, fieldDual, -1.0);

    // Calculate error
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(gradFieldDual.m_data[comp]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.validbox();
            COMPARE_FIELDS(gradFieldDual.m_data[comp][mfi].array(),
                           analyticalGradFieldDual.m_data[comp][mfi].array(), bx);
        }
    }
}

TEST_F(ExtDerivativesTest, Curl)
{
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcField, funcCurl;
    amrex::Array<amrex::Parser, 3> parserField, parserCurl;
    for (int i = 0; i < 3; ++i)
    {
        parserField[i].define(m_analyticalField[i]);
        parserField[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcField[i] = parserField[i].compile<nVar>();
        parserCurl[i].define(m_analyticalCurlField[i]);
        parserCurl[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcCurl[i] = parserCurl[i].compile<nVar>();
    }

    // Test curl of field on primal complex
    //-------------------------------------
    DeRhamField<Grid::primal, Space::edge> field(deRham, funcField);
    DeRhamField<Grid::primal, Space::face> curlField(deRham);
    DeRhamField<Grid::primal, Space::face> analyticalCurlField(deRham, funcCurl);

    // Calculate curlField from Field
    curl(curlField, field);

    // Calculate error
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(curlField.m_data[comp]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.validbox();
            COMPARE_FIELDS(curlField.m_data[comp][mfi].array(),
                           analyticalCurlField.m_data[comp][mfi].array(), bx);
        }
    }

    // Calculate curlField from Field with add_dt_curl
    double dt{0.3}; // arbitrary dt factor
    analyticalCurlField *= dt;
    for (int comp = 0; comp < 3; ++comp)
    {
        curlField.m_data[comp].setVal(0.0);
    }

    add_dt_curl(curlField, field, dt);

    // Calculate error
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(curlField.m_data[comp]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.validbox();
            COMPARE_FIELDS(curlField.m_data[comp][mfi].array(),
                           analyticalCurlField.m_data[comp][mfi].array(), bx);
        }
    }

    // Test curl of field on dual complex
    //-------------------------------------
    DeRhamField<Grid::dual, Space::edge> fieldDual(deRham, funcField);
    DeRhamField<Grid::dual, Space::face> curlFieldDual(deRham);
    DeRhamField<Grid::dual, Space::face> analyticalCurlFieldDual(deRham, funcCurl);

    // Calculate curlFieldDual from FieldDual
    curl(curlFieldDual, fieldDual);

    // Calculate error
    for (int comp = 0; comp < 3; ++comp)
    {
        for (amrex::MFIter mfi(curlField.m_data[comp]); mfi.isValid(); ++mfi)
        {
            amrex::Box const& bx = mfi.validbox();
            COMPARE_FIELDS(curlFieldDual.m_data[comp][mfi].array(),
                           analyticalCurlFieldDual.m_data[comp][mfi].array(), bx);
        }
    }
}

TEST_F(ExtDerivativesTest, Div)
{
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, s_hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcField;
    amrex::ParserExecutor<nVar> funcDiv;
    amrex::Array<amrex::Parser, 3> parserField;
    amrex::Parser parserDiv;
    for (int i = 0; i < 3; ++i)
    {
        parserField[i].define(m_analyticalField[i]);
        parserField[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcField[i] = parserField[i].compile<nVar>();
    }
    parserDiv.define(m_analyticalDivField);
    parserDiv.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    funcDiv = parserDiv.compile<nVar>();

    // Test div of field on primal complex
    //-------------------------------------
    DeRhamField<Grid::primal, Space::face> field(deRham, funcField);
    DeRhamField<Grid::primal, Space::cell> divField(deRham);
    DeRhamField<Grid::primal, Space::cell> analyticalDivField(deRham, funcDiv);

    // Calculate DivField from Field
    div(divField, field);

    // Calculate error
    for (amrex::MFIter mfi(divField.m_data); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();
        COMPARE_FIELDS(divField.m_data[mfi].array(), analyticalDivField.m_data[mfi].array(), bx);
    }

    // Test div of field on dual complex
    //-------------------------------------
    DeRhamField<Grid::dual, Space::face> fieldDual(deRham, funcField);
    DeRhamField<Grid::dual, Space::cell> divFieldDual(deRham);
    DeRhamField<Grid::dual, Space::cell> analyticalDivFieldDual(deRham, funcDiv);

    // Calculate divFieldDual from FieldDual
    div(divFieldDual, fieldDual);

    // Calculate error
    for (amrex::MFIter mfi(divField.m_data); mfi.isValid(); ++mfi)
    {
        amrex::Box const& bx = mfi.validbox();
        COMPARE_FIELDS(divFieldDual.m_data[mfi].array(), analyticalDivFieldDual.m_data[mfi].array(),
                       bx);
    }
}

} // namespace
