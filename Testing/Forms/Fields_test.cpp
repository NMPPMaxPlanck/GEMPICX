#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

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
class FieldsTest : public testing::Test
{
protected:
    static const int s_degX{1};
    static const int s_degY{1};
    static const int s_degZ{1};
    inline static const int s_maxSplineDegree{std::max(std::max(s_degX, s_degY), s_degZ)};

    Io::Parameters m_parameters{};

    // Initialize computational_domain
    ComputationalDomain m_infra;

    int m_nComps{1};
    static const int s_gSize{5};

    FieldsTest() : m_infra{Gempic::Test::Utils::get_compdom(s_gSize)} {}
};

TEST_F(FieldsTest, OperatorsHodge2)
{
    constexpr int hodgeDegree{2};

    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::node> primalNode(deRham);
    DeRhamField<Grid::primal, Space::edge> primalEdge(deRham);

    DeRhamField<Grid::dual, Space::face> dualFaceOther(deRham);
    DeRhamField<Grid::dual, Space::cell> dualCellOther(deRham);

    primalNode.m_data.setVal(1);
    primalNode *= 2; // result = 2
    dualCellOther.m_data.setVal(3);

    amrex::Real dotProd = dot_product(primalNode, dualCellOther);
    EXPECT_NEAR(
        dotProd,
        2 * 3 *
            GEMPIC_D_MULT(m_infra.m_nCell[0] + 1, m_infra.m_nCell[1] + 1, m_infra.m_nCell[2] + 1),
        1e-14);

    for (int comp = 0; comp < 3; comp++)
    {
        primalEdge.m_data[comp].setVal(1);
        dualFaceOther.m_data[comp].setVal(3);
    }
    dotProd = dot_product(primalEdge, dualFaceOther);
    EXPECT_NEAR(
        dotProd,
        3 * (GEMPIC_D_MULT(m_infra.m_nCell[0], m_infra.m_nCell[1] + 1, m_infra.m_nCell[2] + 1) +
             GEMPIC_D_MULT(m_infra.m_nCell[0] + 1, m_infra.m_nCell[1], m_infra.m_nCell[2] + 1) +
             GEMPIC_D_MULT(m_infra.m_nCell[0] + 1, m_infra.m_nCell[1] + 1, m_infra.m_nCell[2])),
        1e-14);
}

TEST_F(FieldsTest, LinearCombination)
{
    constexpr int hodgeDegree{2};

    auto deRham = std::make_shared<FDDeRhamComplex>(m_infra, hodgeDegree, s_maxSplineDegree,
                                                    HodgeScheme::FDHodge);

    DeRhamField<Grid::primal, Space::node> primalNode1(deRham);
    DeRhamField<Grid::primal, Space::edge> primalEdge1(deRham);
    DeRhamField<Grid::primal, Space::face> primalFace1(deRham);

    DeRhamField<Grid::primal, Space::node> primalNode2(deRham);
    DeRhamField<Grid::primal, Space::edge> primalEdge2(deRham);
    DeRhamField<Grid::dual, Space::edge> dualEdge2(deRham);
    DeRhamField<Grid::dual, Space::cell> dualCell2(deRham);

    primalNode1.m_data.setVal(1.0);
    primalNode2.m_data.setVal(2.0);
    Gempic::Forms::linear_combination(primalNode1, 2.0, primalNode1, 3.0, primalNode2);
    EXPECT_EQ(primalNode1.m_data.min(0), 2.0 * 1.0 + 3.0 * 2.0);

    primalNode1.m_data.setVal(1.0);
    dualCell2.m_data.setVal(2.0);
    Gempic::Forms::linear_combination(primalNode1, 2.0, primalNode1, 3.0, dualCell2);
    EXPECT_EQ(primalNode1.m_data.min(0), 2.0 * 1.0 + 3.0 * 2.0);

    for (int comp = 0; comp < 3; comp++)
    {
        primalEdge1.m_data[comp].setVal(1);
        primalEdge2.m_data[comp].setVal(2);
        primalFace1.m_data[comp].setVal(1);
        dualEdge2.m_data[comp].setVal(2);
    }

    linear_combination(primalEdge1, 2.0, primalEdge1, 3.0, primalEdge2);
    linear_combination(primalFace1, 2.0, primalFace1, 3.0, dualEdge2);

    for (int comp = 0; comp < 3; comp++)
    {
        EXPECT_EQ(primalEdge1.m_data[comp].min(0), 2.0 * 1.0 + 3.0 * 2.0);
        EXPECT_EQ(primalFace1.m_data[comp].min(0), 2.0 * 1.0 + 3.0 * 2.0);
    }
}

} // namespace
