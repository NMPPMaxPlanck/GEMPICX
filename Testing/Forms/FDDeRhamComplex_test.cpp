#include <gtest/gtest.h>

#include <AMReX.H>

#include "GEMPIC_ComputationalDomain.H"
#include "GEMPIC_FDDeRhamComplex.H"
#include "GEMPIC_Fields.H"
#include "GEMPIC_GempicNorm.H"
#include "GEMPIC_Parameters.H"
#include "TestUtils/GEMPIC_TestUtils.H"

// E = one form
// rho = three form
// phi = zero form

using namespace Gempic;
using namespace Forms;

namespace
{
class FDDeRhamComplexTest : public testing::Test
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

    FDDeRhamComplexTest() : m_infra{Gempic::Test::Utils::get_default_compdom()}
    {
        // Not checking particles
        int const nGhostExtra{0};

        m_parameters.set("nGhostExtra", nGhostExtra);
    }
};

TEST_F(FDDeRhamComplexTest, MatrixMultTestDeg2)
{
    constexpr int hodgeDegree{2};

    std::string const analyticalFunc = "1.0";

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
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

    EXPECT_NEAR(m_infra.cell_volume(), Utils::gempic_norm(rho.m_data, m_infra, 2), 1e-12);

    // Select stencil according to degree
    [[maybe_unused]] auto [stencilNodeToCell, stencilCellToNode] =
        get_hodge_stencils<hodgeDegree, HodgeScheme::FDHodge>();

    apply_1d_hodge<xDir, hodgeDegree>(m_infra.m_geom, stencilCellToNode, rho.m_data, phi.m_data);

    bool loopRun{false};

    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;

        // expect all nodes to be equal to the cell volume
        CHECK_FIELD((phi.m_data[mfi]).array(), mfi.validbox(), {}, {}, m_infra.cell_volume());
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(FDDeRhamComplexTest, MatrixMultTestDeg4)
{
    constexpr int hodgeDegree{4};
    std::string const analyticalFunc = "1.0";

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
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

    EXPECT_NEAR(m_infra.cell_volume(), Utils::gempic_norm(rho.m_data, m_infra, 2), 1e-12);

    // Select stencil according to degree
    [[maybe_unused]] auto [stencilNodeToCell, stencilCellToNode] =
        get_hodge_stencils<hodgeDegree, HodgeScheme::FDHodge>();

    apply_1d_hodge<xDir, hodgeDegree>(m_infra.m_geom, stencilCellToNode, rho.m_data, phi.m_data);

    bool loopRun{false};

    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;

        // expect all nodes to be equal to the cell volume
        CHECK_FIELD((phi.m_data[mfi]).array(), mfi.validbox(), {}, {}, m_infra.cell_volume());
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(FDDeRhamComplexTest, MatrixMultTestDeg6)
{
    constexpr int hodgeDegree{6};

    std::string const analyticalFunc = "1.0";

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
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

    EXPECT_NEAR(m_infra.cell_volume(), Utils::gempic_norm(rho.m_data, m_infra, 2), 1e-12);

    [[maybe_unused]] auto [stencilNodeToCell, stencilCellToNode] =
        get_hodge_stencils<hodgeDegree, HodgeScheme::FDHodge>();

    apply_1d_hodge<xDir, hodgeDegree>(m_infra.m_geom, stencilCellToNode, rho.m_data, phi.m_data);

    bool loopRun{false};

    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;

        // expect all nodes to be equal to the cell volume
        CHECK_FIELD((phi.m_data[mfi]).array(), mfi.validbox(), {}, {}, m_infra.cell_volume());
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
        CHECK_FIELD((phi.m_data[mfi]).array(), mfi.validbox(), {}, {}, 0);
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(FDDeRhamComplexTest, HodgeFDThreeFormZeroFormTestII)
{
    constexpr int hodgeDegree{2};

    std::string const analyticalFunc = "1.0";

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
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

    EXPECT_NEAR(m_infra.cell_volume(), Utils::gempic_norm(rho.m_data, m_infra, 2), 1e-12);

    // Select stencil according to degree
    [[maybe_unused]] auto [stencilNodeToCell, stencilCellToNode] =
        get_hodge_stencils<hodgeDegree, HodgeScheme::FDHodge>();

    apply_1d_hodge<xDir, hodgeDegree>(m_infra.m_geom, stencilCellToNode, rho.m_data, phi.m_data);

    bool loopRun{false};

    for (amrex::MFIter mfi(phi.m_data); mfi.isValid(); ++mfi)
    {
        loopRun = true;

        // Expect all entries to be equal to the cell volume
        CHECK_FIELD((phi.m_data[mfi]).array(), mfi.validbox(), {}, {}, m_infra.cell_volume());
    }
    ASSERT_TRUE(loopRun);
}

TEST_F(FDDeRhamComplexTest, HodgeFDThreeFormZeroFormTestIII)
{
    constexpr int hodgeDegree{2};

    std::string const analyticalFunc = "1.0";

    int const nVar = AMREX_SPACEDIM + 1; // x, y, z, t
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
        CHECK_FIELD((phi.m_data[mfi]).array(), mfi.validbox(), {}, {}, 0);
    }
    ASSERT_TRUE(loopRun);
}
} // namespace
