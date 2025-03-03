#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

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
    static const int s_degX{1};
    static const int s_degY{1};
    static const int s_degZ{1};
    inline static const int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

    Io::Parameters m_parameters{};
    ComputationalDomain m_infra{false}; // "uninitialized" computational domain
    amrex::Real m_tol{1e-11};

    static void SetUpTestSuite ()
    {
        /* Initialize the infrastructure */
        amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        // Domain has to be small enough so x*y*z does not get too large for absolute float
        // comparison
        amrex::Vector<amrex::Real> k{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
        const amrex::Vector<int> nCell{AMREX_D_DECL(10, 10, 10)};
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(10, 10, 10)};
        const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};
        // Not checking particles
        const int nGhostExtra{0};

        amrex::ParmParse pp;
        pp.addarr("ComputationalDomain.domainLo", domainLo);
        pp.addarr("k", k);
        pp.addarr("ComputationalDomain.nCell", nCell);
        pp.addarr("ComputationalDomain.maxGridSize", maxGridSize);
        pp.addarr("ComputationalDomain.isPeriodic", isPeriodic);
        pp.add("nGhostExtra", nGhostExtra);
    }

    // virtual void SetUp() will be called before each test is run.
    void SetUp () override { m_infra = ComputationalDomain{}; }
};

TEST_F(FieldMultiplyByFunctionTest, ZeroThreeForms)
{
    constexpr int hodgeDegree{2};

#if (AMREX_SPACEDIM == 1)
    const std::string analyticalWeightFunc = "x";
#endif

#if (AMREX_SPACEDIM == 2)
    const std::string analyticalWeightFunc = "x * y";
#endif

#if (AMREX_SPACEDIM == 3)
    const std::string analyticalWeightFunc = "x * y * z";
#endif

    const std::string analyticalOneFunc = "1.0";

    const int nVar = AMREX_SPACEDIM + 1; // x, y, z, t
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
        const amrex::Box& bx = mfi.validbox();
        COMPARE_FIELDS(A.m_data[mfi].array(), aResult.m_data[mfi].array(), bx, m_tol);
    }
    ASSERT_TRUE(loopRun);

    loopRun = false;

    for (amrex::MFIter mfi(D.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        const amrex::Box& bx = mfi.validbox();
        COMPARE_FIELDS(D.m_data[mfi].array(), dResult.m_data[mfi].array(), bx, m_tol);
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(FieldMultiplyByFunctionTest, OneTwoForms)
{
    constexpr int hodgeDegree{2};

#if (AMREX_SPACEDIM == 1)
    const std::string analyticalWeightFunc = "x";
    const amrex::Array<std::string, 3> analyticalResultFunc = {
        "x",
        "x",
        "x",
    };
#endif

#if (AMREX_SPACEDIM == 2)
    const std::string analyticalWeightFunc = "x * y";
    const amrex::Array<std::string, 3> analyticalResultFunc = {
        "x * y",
        "x * y",
        "x * y",
    };
#endif

#if (AMREX_SPACEDIM == 3)
    const std::string analyticalWeightFunc = "x * y * z";
    const amrex::Array<std::string, 3> analyticalResultFunc = {
        "x * y * z",
        "x * y * z",
        "x * y * z",
    };
#endif

    const amrex::Array<std::string, 3> analyticalOneFunc = {
        "1",
        "1",
        "1",
    };

    const int nVar = AMREX_SPACEDIM + 1; // x, y, z, t
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
            const amrex::Box& bx = mfi.validbox();
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
            const amrex::Box& bx = mfi.validbox();
            COMPARE_FIELDS((C.m_data[comp])[mfi].array(), (cResult.m_data[comp])[mfi].array(), bx,
                           m_tol);
        }
        ASSERT_TRUE(loopRun);
    }
}
} // namespace
