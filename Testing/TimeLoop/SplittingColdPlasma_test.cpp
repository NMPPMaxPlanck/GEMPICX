#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Parameters.H"
#include "GEMPIC_Splitting.H"
#include "TestUtils/GEMPIC_TestUtils.H"

using namespace Gempic;
using namespace Forms;
using namespace TimeLoop;

namespace
{
class SplittingColdPlasmaTest : public testing::Test
{
protected:
    // Linear splines is ok, and lower dimension Hodge is good enough
    // Spline degreesx
    static const int s_degX{1};
    static const int s_degY{1};
    static const int s_degZ{1};
    inline static const int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

    Io::Parameters m_parameters{};
    ComputationalDomain m_infra{false};  // "uninitialized" computational domain
    amrex::Real m_tol{1e-11};

    static void SetUpTestSuite ()
    {
        /* Initialize the infrastructure */
        amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        amrex::Vector<amrex::Real> k{AMREX_D_DECL(2 * M_PI, 2 * M_PI, 2 * M_PI)};
        const amrex::Vector<int> nCell{AMREX_D_DECL(1, 1, 1)};
        const amrex::Vector<int> maxGridSize{AMREX_D_DECL(1, 1, 1)};
        const amrex::Vector<int> isPeriodic{AMREX_D_DECL(1, 1, 1)};
        // Not checking particles
        const int nGhostExtra{1};

        amrex::ParmParse pp;
        pp.addarr("domainLo", domainLo);
        pp.addarr("k", k);
        pp.addarr("nCellVector", nCell);
        pp.addarr("maxGridSizeVector", maxGridSize);
        pp.addarr("isPeriodicVector", isPeriodic);
        pp.add("nGhostExtra", nGhostExtra);
    }

    // virtual void SetUp() will be called before each test is run.
    void SetUp () override { m_infra = ComputationalDomain{}; }
};

TEST_F(SplittingColdPlasmaTest, RotationZAxis)
{
    constexpr int hodgeDegree{2};

    const amrex::Array<std::string, 3> analyticalBFunc = {
        "0",
        "0",
        "1",
    };
    const amrex::Array<std::string, 3> analyticalJFunc = {
        "1",
        "0",
        "0",
    };
    const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcJ;
    amrex::Array<amrex::Parser, 3> parserB;
    amrex::Array<amrex::Parser, 3> parserJ;
    for (int i = 0; i < 3; ++i)
    {
        parserB[i].define(analyticalBFunc[i]);
        parserB[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcB[i] = parserB[i].compile<nVar>();
        parserJ[i].define(analyticalJFunc[i]);
        parserJ[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcJ[i] = parserJ[i].compile<nVar>();
    }

    // Initialize the De Rham Complex with deg 2
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::dual, Space::face> D(deRham);
    DeRhamField<Grid::dual, Space::face> J(deRham, funcJ);

    // Rotate J in X-Y plane by quarter turn
    apply_h_j(deRham, D, J, funcB, 0.5 * M_PI);

    bool loopRun{false};

    for (amrex::MFIter mfi(J.m_data[xDir]); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        amrex::Array4<amrex::Real> const& Dx = (D.m_data[xDir])[mfi].array();
        amrex::Array4<amrex::Real> const& Dy = (D.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real> const& Dz = (D.m_data[zDir])[mfi].array();

        amrex::Array4<amrex::Real> const& Jx = (J.m_data[xDir])[mfi].array();
        amrex::Array4<amrex::Real> const& Jy = (J.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real> const& Jz = (J.m_data[zDir])[mfi].array();

        EXPECT_NEAR(Jx(0, 0, 0), 0., m_tol);
        EXPECT_NEAR(Jy(0, 0, 0), -1., m_tol);
        EXPECT_NEAR(Jz(0, 0, 0), 0., m_tol);

        EXPECT_NEAR(Dx(0, 0, 0), -1., m_tol);
        EXPECT_NEAR(Dy(0, 0, 0), 1., m_tol);
        EXPECT_NEAR(Dz(0, 0, 0), 0., m_tol);
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(SplittingColdPlasmaTest, RotationGeneralAxis)
{
    constexpr int hodgeDegree{2};

    const amrex::Array<std::string, 3> analyticalBFunc = {
        "1",
        "1",
        "1",
    };
    const amrex::Array<std::string, 3> analyticalJFunc = {
        "1",
        "-1",
        "0",
    };
    const int nVar = GEMPIC_SPACEDIM + 1;  // x, y, z, t
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcB;
    amrex::Array<amrex::ParserExecutor<nVar>, 3> funcJ;
    amrex::Array<amrex::Parser, 3> parserB;
    amrex::Array<amrex::Parser, 3> parserJ;
    for (int i = 0; i < 3; ++i)
    {
        parserB[i].define(analyticalBFunc[i]);
        parserB[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcB[i] = parserB[i].compile<nVar>();
        parserJ[i].define(analyticalJFunc[i]);
        parserJ[i].registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
        funcJ[i] = parserJ[i].compile<nVar>();
    }

    // Initialize the De Rham Complex with deg 2
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::dual, Space::face> D(deRham);
    DeRhamField<Grid::dual, Space::face> J(deRham, funcJ);

    // Rotate J around (1,1,1) by half turn, time step scaled by 1/normB
    apply_h_j(deRham, D, J, funcB, 0.5773502691896258 * M_PI);

    bool loopRun{false};

    for (amrex::MFIter mfi(J.m_data[xDir]); mfi.isValid(); ++mfi)
    {
        loopRun = true;
        amrex::Array4<amrex::Real> const& Dx = (D.m_data[xDir])[mfi].array();
        amrex::Array4<amrex::Real> const& Dy = (D.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real> const& Dz = (D.m_data[zDir])[mfi].array();

        amrex::Array4<amrex::Real> const& Jx = (J.m_data[xDir])[mfi].array();
        amrex::Array4<amrex::Real> const& Jy = (J.m_data[yDir])[mfi].array();
        amrex::Array4<amrex::Real> const& Jz = (J.m_data[zDir])[mfi].array();

        EXPECT_NEAR(Jx(0, 0, 0), -1., m_tol);
        EXPECT_NEAR(Jy(0, 0, 0), 1., m_tol);
        EXPECT_NEAR(Jz(0, 0, 0), 0., m_tol);

        EXPECT_NEAR(Dx(0, 0, 0), 2. / 3, m_tol);
        EXPECT_NEAR(Dy(0, 0, 0), 2. / 3, m_tol);
        EXPECT_NEAR(Dz(0, 0, 0), -4. / 3, m_tol);
    }
    ASSERT_TRUE(loopRun);
}
}  // namespace
