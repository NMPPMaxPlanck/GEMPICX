#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Parameters.H"
#include "TestUtils/GEMPIC_TestUtils.H"

// E = one form
// rho = three form
// phi = zero form

#define check_field(...) Gempic::Test::Utils::check_field(__FILE__, __LINE__, __VA_ARGS__)

using namespace Gempic;
using namespace Forms;

namespace
{
class FDDeRhamComplexTest : public testing::Test
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

    static void SetUpTestSuite ()
    {
        /* Initialize the infrastructure */
        // const amrex::RealBox realBox({AMREX_D_DECL(0.0, 0.0, 0.0)},
        //                              {AMREX_D_DECL(10.0, 10.0, 10.0)});
        amrex::Vector<amrex::Real> domainLo{AMREX_D_DECL(0.0, 0.0, 0.0)};
        //
        amrex::Vector<amrex::Real> k{AMREX_D_DECL(0.2 * M_PI, 0.2 * M_PI, 0.2 * M_PI)};
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

TEST_F(FDDeRhamComplexTest, MatrixMultTestDeg2)
{
    constexpr int hodgeDegree{2};

    const std::string analyticalFunc = "1.0";

    const int nVar = AMREX_SPACEDIM + 1;  // x, y, z, t
    amrex::ParserExecutor<nVar> func;
    amrex::Parser parser;

    parser.define(analyticalFunc);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Initialize the De Rham Complex with deg 2
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::dual, Space::cell> rho(deRham, func);
    DeRhamField<Grid::primal, Space::node> phi(deRham);

    EXPECT_NEAR(1, Utils::gempic_norm(rho.m_data, m_infra, 2), 1e-12);

    // Select stencil according to degree
    [[maybe_unused]] auto [stencilNodeToCell, stencilCellToNode] =
        get_hodge_stencils<hodgeDegree, HodgeScheme::FDHodge>();

    apply_1d_hodge<xDir, hodgeDegree>(m_infra.m_geom, stencilCellToNode, rho.m_data, phi.m_data);

    bool loopRun{false};

    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;

        // expect all nodes to be 1
        check_field((phi.m_data[mfi]).array(), m_infra.m_nCell.dim3(), {}, {}, 1);
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(FDDeRhamComplexTest, MatrixMultTestDeg4)
{
    constexpr int hodgeDegree{4};
    const std::string analyticalFunc = "1.0";

    const int nVar = AMREX_SPACEDIM + 1;  // x, y, z, t
    amrex::ParserExecutor<nVar> func;
    amrex::Parser parser;

    parser.define(analyticalFunc);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::dual, Space::cell> rho(deRham, func);
    DeRhamField<Grid::primal, Space::node> phi(deRham);

    EXPECT_NEAR(1, Utils::gempic_norm(rho.m_data, m_infra, 2), 1e-12);

    // Select stencil according to degree
    [[maybe_unused]] auto [stencilNodeToCell, stencilCellToNode] =
        get_hodge_stencils<hodgeDegree, HodgeScheme::FDHodge>();

    apply_1d_hodge<xDir, hodgeDegree>(m_infra.m_geom, stencilCellToNode, rho.m_data, phi.m_data);

    bool loopRun{false};

    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;

        // Expect all entries to be 1
        check_field((phi.m_data[mfi]).array(), m_infra.m_nCell.dim3(), {}, {}, 1);
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(FDDeRhamComplexTest, MatrixMultTestDeg6)
{
    constexpr int hodgeDegree{6};

    const std::string analyticalFunc = "1.0";

    const int nVar = AMREX_SPACEDIM + 1;  // x, y, z, t
    amrex::ParserExecutor<nVar> func;
    amrex::Parser parser;

    parser.define(analyticalFunc);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::dual, Space::cell> rho(deRham, func);
    DeRhamField<Grid::primal, Space::node> phi(deRham);

    EXPECT_NEAR(1, Utils::gempic_norm(rho.m_data, m_infra, 2), 1e-12);

    [[maybe_unused]] auto [stencilNodeToCell, stencilCellToNode] =
        get_hodge_stencils<hodgeDegree, HodgeScheme::FDHodge>();

    apply_1d_hodge<xDir, hodgeDegree>(m_infra.m_geom, stencilCellToNode, rho.m_data, phi.m_data);

    bool loopRun{false};

    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;

        // Expect all nodes to be 1
        check_field((phi.m_data[mfi]).array(), m_infra.m_nCell.dim3(), {}, {}, 1);
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(FDDeRhamComplexTest, HodgeFDThreeFormZeroFormTest)
{
    constexpr int hodgeDegree{2};

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::dual, Space::cell> rho(deRham);
    DeRhamField<Grid::primal, Space::node> phi(deRham);

    ASSERT_EQ(0, Utils::gempic_norm(rho.m_data, m_infra, 2));

    // Select stencil according to degree
    [[maybe_unused]] auto [stencilNodeToCell, stencilCellToNode] =
        get_hodge_stencils<hodgeDegree, HodgeScheme::FDHodge>();

    apply_1d_hodge<xDir, hodgeDegree>(m_infra.m_geom, stencilCellToNode, rho.m_data, phi.m_data);

    bool loopRun{false};

    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;

        // Expect all entires to be 0
        check_field((phi.m_data[mfi]).array(), m_infra.m_nCell.dim3(), {}, {}, 0);
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(FDDeRhamComplexTest, HodgeFDThreeFormZeroFormTestII)
{
    constexpr int hodgeDegree{2};

    const std::string analyticalFunc = "1.0";

    const int nVar = AMREX_SPACEDIM + 1;  // x, y, z, t
    amrex::ParserExecutor<nVar> func;
    amrex::Parser parser;

    parser.define(analyticalFunc);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::dual, Space::cell> rho(deRham, func);
    DeRhamField<Grid::primal, Space::node> phi(deRham);

    EXPECT_NEAR(1, Utils::gempic_norm(rho.m_data, m_infra, 2), 1e-12);

    // Select stencil according to degree
    [[maybe_unused]] auto [stencilNodeToCell, stencilCellToNode] =
        get_hodge_stencils<hodgeDegree, HodgeScheme::FDHodge>();

    apply_1d_hodge<xDir, hodgeDegree>(m_infra.m_geom, stencilCellToNode, rho.m_data, phi.m_data);

    bool loopRun{false};

    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;

        // Expect all entires to be 1
        check_field((phi.m_data[mfi]).array(), m_infra.m_nCell.dim3(), {}, {}, 1);
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(FDDeRhamComplexTest, HodgeFDThreeFormZeroFormTestIII)
{
    constexpr int hodgeDegree{2};

    const std::string analyticalFunc = "1.0";

    const int nVar = AMREX_SPACEDIM + 1;  // x, y, z, t
    amrex::ParserExecutor<nVar> func;
    amrex::Parser parser;

    parser.define(analyticalFunc);
    parser.registerVariables({AMREX_D_DECL("x", "y", "z"), "t"});
    func = parser.compile<nVar>();

    // Initialize the De Rham Complex
    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::dual, Space::cell> rho(deRham);
    DeRhamField<Grid::primal, Space::node> phi(deRham, func);

    //        const amrex::BoxArray &nba{amrex::convert(infra.grid,
    //        amrex::IntVect::TheNodeVector())};

    //        rho.data.define(nba, infra.distriMap, Ncomp, Nghost);

    //        phi.data.define(nba, infra.distriMap, Ncomp, Nghost);

    ASSERT_EQ(0, Utils::gempic_norm(rho.m_data, m_infra, 2));

    // Select stencil according to degree
    [[maybe_unused]] auto [stencilNodeToCell, stencilCellToNode] =
        get_hodge_stencils<hodgeDegree, HodgeScheme::FDHodge>();

    apply_1d_hodge<xDir, hodgeDegree>(m_infra.m_geom, stencilCellToNode, rho.m_data, phi.m_data);

    bool loopRun{false};

    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;

        // Expect all entires to be 0
        check_field((phi.m_data[mfi]).array(), m_infra.m_nCell.dim3(), {}, {}, 0);
    }
    ASSERT_TRUE(loopRun);
}
}  // namespace
